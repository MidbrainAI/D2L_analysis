"""Post-block state-transition readers and the hidden-state hooks shared by the reader and transplant runners."""

from __future__ import annotations

import contextlib
import hashlib
import json
import sys
from pathlib import Path

import torch
from einops import einsum
from torch import nn
from torch.nn import functional

sys.path.insert(0, str(Path(__file__).parent))


SEED = 124
BOTTLENECK = 32


class StateTransition(nn.Module):
    """Shared zero-initialized transition from adapted state and live LoRA effect."""

    execution_site = "post-block"

    def __init__(self, width: int, layers: int, bottleneck: int = BOTTLENECK) -> None:
        super().__init__()
        self.width = width
        self.layers = layers
        self.state_projection = nn.Linear(width, bottleneck, bias=False)
        self.effect_projection = nn.Linear(width, bottleneck, bias=False)
        self.layer_embedding = nn.Embedding(layers, bottleneck)
        self.output_projection = nn.Linear(bottleneck, width, bias=False)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, state: torch.Tensor, effect: torch.Tensor, layer: int) -> torch.Tensor:
        if state.shape != effect.shape or state.shape[-1] != self.width:
            raise ValueError("state and LoRA effect must have matching hidden-width shapes")
        state32 = functional.layer_norm(state.float(), (self.width,))
        effect32 = functional.layer_norm(effect.float(), (self.width,))
        layer_ids = torch.tensor(layer, device=state.device)
        hidden = functional.silu(
            self.state_projection(state32) + self.effect_projection(effect32) + self.layer_embedding(layer_ids)
        )
        return state + self.output_projection(hidden).to(state.dtype)


class PerLayerStateTransition(nn.Module):
    """Independent zero-initialized state transition for every transformer layer."""

    execution_site = "post-block"

    def __init__(self, width: int, layers: int, bottleneck: int = BOTTLENECK) -> None:
        super().__init__()
        self.width = width
        self.layers = layers
        self.bottleneck = bottleneck
        self.state_projections = nn.ModuleList(
            nn.Linear(width, bottleneck, bias=False) for _ in range(layers)
        )
        self.effect_projections = nn.ModuleList(
            nn.Linear(width, bottleneck, bias=False) for _ in range(layers)
        )
        self.output_projections = nn.ModuleList(
            nn.Linear(bottleneck, width, bias=False) for _ in range(layers)
        )
        for projection in self.output_projections:
            nn.init.zeros_(projection.weight)

    def forward(self, state: torch.Tensor, effect: torch.Tensor, layer: int) -> torch.Tensor:
        if state.shape != effect.shape or state.shape[-1] != self.width:
            raise ValueError("state and LoRA effect must have matching hidden-width shapes")
        if layer < 0 or layer >= self.layers:
            raise ValueError("layer index is outside the transition")
        state32 = functional.layer_norm(state.float(), (self.width,))
        effect32 = functional.layer_norm(effect.float(), (self.width,))
        hidden = functional.silu(
            self.state_projections[layer](state32) + self.effect_projections[layer](effect32)
        )
        return state + self.output_projections[layer](hidden).to(state.dtype)


class WriteSiteResidualTransition(nn.Module):
    """Identity-preserving nonlinear correction at the native LoRA write site."""

    execution_site = "write-site"

    def __init__(self, width: int, layers: int, bottleneck: int = BOTTLENECK) -> None:
        super().__init__()
        self.width = width
        self.layers = layers
        self.bottleneck = bottleneck
        self.state_projection = nn.Linear(width, bottleneck, bias=False)
        self.effect_projection = nn.Linear(width, bottleneck, bias=False)
        self.magnitude_projection = nn.Linear(1, bottleneck, bias=False)
        self.layer_embedding = nn.Embedding(layers, bottleneck)
        self.output_projection = nn.Linear(bottleneck, width, bias=False)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, state: torch.Tensor, effect: torch.Tensor, layer: int) -> torch.Tensor:
        if state.shape != effect.shape or state.shape[-1] != self.width:
            raise ValueError("state and LoRA effect must have matching hidden-width shapes")
        if layer < 0 or layer >= self.layers:
            raise ValueError("layer index is outside the transition")
        state32 = functional.layer_norm(state.float(), (self.width,))
        effect32 = functional.layer_norm(effect.float(), (self.width,))
        state_norm = torch.linalg.vector_norm(state.float(), dim=-1, keepdim=True).clamp_min(1e-6)
        relative_magnitude = torch.log1p(
            torch.linalg.vector_norm(effect.float(), dim=-1, keepdim=True) / state_norm
        )
        layer_ids = torch.tensor(layer, device=state.device)
        hidden = functional.silu(
            self.state_projection(state32)
            + self.effect_projection(effect32)
            + self.magnitude_projection(relative_magnitude)
            + self.layer_embedding(layer_ids)
        )
        correction = (self.output_projection(hidden) / self.bottleneck).to(state.dtype)
        return state + effect + correction


def make_state_transition(
    transition_type: str,
    width: int,
    layers: int,
    bottleneck: int = BOTTLENECK,
) -> nn.Module:
    if transition_type == "shared":
        return StateTransition(width, layers, bottleneck)
    if transition_type == "per-layer":
        return PerLayerStateTransition(width, layers, bottleneck)
    if transition_type == "write-site-residual":
        return WriteSiteResidualTransition(width, layers, bottleneck)
    raise ValueError(f"unknown transition type: {transition_type}")


def render_ids(backend, question: str, context: str = "", dataset: str = "squad") -> torch.Tensor:
    prompt = backend.templates[dataset].format(input=question)
    if context:
        prompt = context.strip() + "\n\n" + prompt
    return backend.tokenizer.apply_chat_template(
        [{"role": "system", "content": ""}, {"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(backend.model.device)


def tensor_bytes(value: object) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(item) for item in value)
    return 0


def tensor_digest(value: object) -> str:
    result = hashlib.sha256()

    def update(item: object, path: str) -> None:
        if isinstance(item, dict):
            for key in sorted(item):
                update(item[key], f"{path}/{key}")
            return
        if not isinstance(item, torch.Tensor):
            raise TypeError(f"unsupported digest value at {path}: {type(item)}")
        tensor = item.detach().cpu().contiguous()
        result.update(path.encode())
        result.update(str(tensor.dtype).encode())
        result.update(str(tuple(tensor.shape)).encode())
        result.update(tensor.view(torch.uint8).numpy().tobytes())

    update(value, "state")
    return result.hexdigest()


def hidden_output(output) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def factor_delta(inputs: torch.Tensor, a: torch.Tensor, b: torch.Tensor, scaling: float) -> torch.Tensor:
    """Match upstream D2L's batched einsum execution exactly."""
    if a.ndim == 2 and b.ndim == 2:
        a = a.unsqueeze(0).expand(inputs.shape[0], -1, -1)
        b = b.unsqueeze(0).expand(inputs.shape[0], -1, -1)
    elif a.ndim != 3 or b.ndim != 3 or a.shape[0] != inputs.shape[0] or b.shape[0] != inputs.shape[0]:
        raise ValueError("LoRA factors must be unbatched or match the input batch")
    latent = einsum(a, inputs.to(a.dtype), "batch rank input, batch seq input -> batch seq rank")
    delta = einsum(b, latent, "batch rank output, batch seq rank -> batch seq output")
    return (delta * scaling).to(inputs.dtype)


def replace_hidden(output, hidden: torch.Tensor):
    return (hidden, *output[1:]) if isinstance(output, tuple) else hidden


@contextlib.contextmanager
def installed_transition(
    model,
    layers,
    state: dict | list[tuple[dict, float]] | None,
    transition: StateTransition | None,
    scaling: float,
    *,
    official_backend=None,
):
    weighted_states = [] if state is None else state if isinstance(state, list) else [(state, 1.0)]
    device_states = []
    for component, weight in weighted_states:
        if set(component) != {"down_proj"}:
            raise ValueError("SHARED_POSTBLOCK_READER expects only down_proj D2L state")
        device_state = {key: value.to(model.device) for key, value in component["down_proj"].items()}
        if device_state["A"].ndim != 4 or device_state["B"].ndim != 4:
            raise ValueError("expected a batched combined D2L state")
        if device_state["A"].shape[:2] != device_state["B"].shape[:2]:
            raise ValueError("combined D2L factor batches and layers must match")
        device_states.append((device_state, weight))
    use_official_install = official_backend is not None and len(device_states) == 1 and not isinstance(state, list)
    if use_official_install:
        official_backend.model.patch_lora_forward()
        official_backend.install(
            official_backend.model.base_model,
            official_backend.model.hypernet.layer_indices,
            {"down_proj": device_states[0][0]},
            torch.ones(device_states[0][0]["A"].shape[0], device=model.device, dtype=torch.int32),
        )
    effects: dict[int, torch.Tensor] = {}
    handles = []
    for layer_index, layer in enumerate(layers):
        module = layer.mlp.down_proj

        if device_states:

            def inject(_module, inputs, output, layer_index=layer_index):
                delta = sum(
                    weight
                    * factor_delta(
                        inputs[0],
                        device_state["A"][:, layer_index],
                        device_state["B"][:, layer_index],
                        scaling,
                    )
                    for device_state, weight in device_states
                )
                effects[layer_index] = delta
                return output if use_official_install else output + delta

            handles.append(module.register_forward_hook(inject))

        if transition is not None:

            def correct(_module, _inputs, output, layer_index=layer_index):
                hidden = hidden_output(output)
                effect = effects.pop(layer_index) if device_states else torch.zeros_like(hidden)
                return replace_hidden(output, transition(hidden, effect, layer_index))

            handles.append(layer.register_forward_hook(correct))
    try:
        yield
    finally:
        effects.clear()
        for handle in handles:
            handle.remove()
        if use_official_install:
            official_backend.model.reset()


def checkpoint(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def generate_condition(
    backend,
    layers,
    state,
    transition,
    scaling,
    question: str,
    max_new_tokens: int,
    dataset: str = "squad",
) -> str:
    ids = render_ids(backend, question, dataset=dataset)
    with (
        installed_transition(
            backend.model.base_model,
            layers,
            state,
            transition,
            scaling,
            official_backend=backend,
        ),
        torch.inference_mode(),
    ):
        output = backend.model.base_model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=backend.tokenizer.pad_token_id,
        )
    return backend.tokenizer.decode(output[0, ids.shape[1] :], skip_special_tokens=True).strip()


class DirectStateTransition(nn.Module):
    """Directly map clean state and live LoRA effect to the next state."""

    def __init__(self, width: int, layers: int) -> None:
        super().__init__()
        self.width = width
        self.layers = layers
        self.input_projection = nn.Linear(2 * width, width)
        self.layer_embedding = nn.Embedding(layers, width)
        self.output_projection = nn.Linear(width, width)
        nn.init.normal_(self.layer_embedding.weight, std=0.02)

    def forward(self, state: torch.Tensor, effect: torch.Tensor, layer: int) -> torch.Tensor:
        if state.shape != effect.shape or state.shape[-1] != self.width:
            raise ValueError("state and LoRA effect must have matching hidden-width shapes")
        layer_ids = torch.tensor(layer, device=state.device)
        hidden = functional.silu(
            self.input_projection(torch.cat((state.float(), effect.float()), dim=-1)) + self.layer_embedding(layer_ids)
        )
        return self.output_projection(hidden).to(state.dtype)


@contextlib.contextmanager
def installed_direct_transition(
    model,
    layers,
    state: dict | list[tuple[dict, float]] | None,
    transition: DirectStateTransition,
    scaling: float,
):
    """Replace each clean down-projection output with G(s, BAx)."""
    weighted_states = [] if state is None else state if isinstance(state, list) else [(state, 1.0)]
    device_states = []
    for component, weight in weighted_states:
        if set(component) != {"down_proj"}:
            raise ValueError("MATCHED_READER_DIAGNOSTIC expects only down_proj D2L state")
        device_state = {key: value.to(model.device) for key, value in component["down_proj"].items()}
        if device_state["A"].ndim != 4 or device_state["B"].ndim != 4:
            raise ValueError("expected a batched combined D2L state")
        if device_state["A"].shape[:2] != device_state["B"].shape[:2]:
            raise ValueError("combined D2L factor batches and layers must match")
        if device_state["A"].shape[1] != len(layers):
            raise ValueError("D2L state does not cover every transition layer")
        device_states.append((device_state, weight))

    handles = []
    for layer_index, layer in enumerate(layers):

        def replace(_module, inputs, output, layer_index=layer_index):
            clean_state = hidden_output(output)
            if getattr(transition, "uses_rank_contributions", False):
                if not device_states:
                    effect = torch.zeros_like(clean_state)
                    rank_latent = torch.empty(
                        (*inputs[0].shape[:-1], 0),
                        device=inputs[0].device,
                        dtype=inputs[0].dtype,
                    )
                    rank_output = torch.empty(
                        (0, clean_state.shape[-1]),
                        device=inputs[0].device,
                        dtype=inputs[0].dtype,
                    )
                else:
                    rank_input = torch.cat(
                        [weight * device_state["A"][0, layer_index] for device_state, weight in device_states],
                        dim=0,
                    )
                    rank_output = torch.cat(
                        [device_state["B"][0, layer_index] for device_state, _ in device_states],
                        dim=0,
                    )
                    rank_latent = (
                        torch.einsum(
                            "bsi,ri->bsr",
                            inputs[0].to(rank_input.dtype),
                            rank_input,
                        )
                        * scaling
                    )
                    effect = torch.einsum("bsr,ro->bso", rank_latent, rank_output).to(clean_state.dtype)
                next_state = transition(
                    clean_state,
                    effect,
                    rank_latent,
                    rank_output,
                    layer_index,
                )
            elif not device_states:
                effect = torch.zeros_like(clean_state)
                next_state = transition(clean_state, effect, layer_index)
            else:
                effect = sum(
                    weight
                    * factor_delta(
                        inputs[0],
                        device_state["A"][:, layer_index],
                        device_state["B"][:, layer_index],
                        scaling,
                    )
                    for device_state, weight in device_states
                )
                next_state = transition(clean_state, effect, layer_index)
            return replace_hidden(output, next_state)

        handles.append(layer.mlp.down_proj.register_forward_hook(replace))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
