"""READOUT_LADDER primitives for matched K=1 compiled-representation readers."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch
from einops import einsum
from official_d2l_online_batch_distillation import (
    concatenate_generated_parts,
    pad_sequences,
)
from post_block_transitions import (
    factor_delta,
    hidden_output,
    replace_hidden,
    tensor_bytes,
    tensor_digest,
)
from torch import nn
from torch.nn import functional

REPRESENTATIONS = ("summed-effect", "rank-effect", "pre-head-latent", "projected-source")


@dataclass(frozen=True)
class CompiledRepresentationBatch:
    """Frozen native factors and the matched document representations."""

    factors: dict
    generated_factors: dict
    bias_factors: dict
    pre_head_latents: torch.Tensor
    projected_source: torch.Tensor | None = None
    projected_source_mask: torch.Tensor | None = None
    document_latents: torch.Tensor | None = None  # [batch, K, layers, 1, latent] per bank document (LAYER_INTERVENTIONS/GATED_READERS)


class ActiveMemoryTransition(nn.Module):
    """Shared post-block reader over token-conditioned or static memory slots."""

    execution_site = "post-block"

    def __init__(
        self,
        width: int,
        memory_width: int,
        layers: int,
        bottleneck: int = 32,
    ) -> None:
        super().__init__()
        self.width = width
        self.memory_width = memory_width
        self.layers = layers
        self.bottleneck = bottleneck
        self.state_projection = nn.Linear(width, bottleneck, bias=False)
        self.query_projection = nn.Linear(width, bottleneck, bias=False)
        self.key_projection = nn.Linear(memory_width, bottleneck, bias=False)
        self.value_projection = nn.Linear(memory_width, bottleneck, bias=False)
        self.bias_effect_projection = nn.Linear(width, bottleneck, bias=False)
        self.layer_embedding = nn.Embedding(layers, bottleneck)
        self.output_projection = nn.Linear(bottleneck, width, bias=False)
        nn.init.zeros_(self.output_projection.weight)

    def forward(
        self,
        state: torch.Tensor,
        memory: torch.Tensor,
        bias_effect: torch.Tensor,
        layer: int,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if state.shape != bias_effect.shape or state.shape[-1] != self.width:
            raise ValueError("state and bias effect must have matching hidden-width shapes")
        if memory.ndim not in (3, 4) or memory.shape[0] != state.shape[0]:
            raise ValueError("memory must contain static or tokenwise slots for every batch item")
        if memory.shape[-1] != self.memory_width:
            raise ValueError("memory width does not match the transition")
        if memory.ndim == 4 and memory.shape[1] != state.shape[1]:
            raise ValueError("tokenwise memory must match the state sequence length")
        if memory_mask is not None and (
            memory.ndim != 3
            or memory_mask.shape != memory.shape[:1] + memory.shape[1:2]
            or not memory_mask.any(dim=1).all()
        ):
            raise ValueError("memory mask must retain at least one static slot per batch item")
        if not 0 <= layer < self.layers:
            raise ValueError("layer index is outside the transition")

        state32 = functional.layer_norm(state.float(), (self.width,))
        memory32 = functional.layer_norm(memory.float(), (self.memory_width,))
        bias32 = functional.layer_norm(bias_effect.float(), (self.width,))
        query = self.query_projection(state32)
        keys = self.key_projection(memory32)
        values = self.value_projection(memory32)
        if memory.ndim == 3:
            scores = einsum(query, keys, "batch seq hidden, batch slots hidden -> batch seq slots")
            if memory_mask is not None:
                scores = scores.masked_fill(~memory_mask[:, None].bool(), torch.finfo(scores.dtype).min)
            attended = einsum(
                functional.softmax(scores / self.bottleneck**0.5, dim=-1),
                values,
                "batch seq slots, batch slots hidden -> batch seq hidden",
            )
        else:
            scores = einsum(
                query,
                keys,
                "batch seq hidden, batch seq slots hidden -> batch seq slots",
            )
            attended = einsum(
                functional.softmax(scores / self.bottleneck**0.5, dim=-1),
                values,
                "batch seq slots, batch seq slots hidden -> batch seq hidden",
            )
        layer_ids = torch.tensor(layer, device=state.device)
        hidden = functional.silu(
            self.state_projection(state32)
            + attended
            + self.bias_effect_projection(bias32)
            + self.layer_embedding(layer_ids)
        )
        return state + self.output_projection(hidden).to(state.dtype)


class NonlinearRankOperator(nn.Module):
    """Shared permutation-equivariant operator over paired ``(Ax)_r, B_r`` tokens."""

    execution_site = "down-proj"

    def __init__(
        self,
        output_width: int,
        layers: int,
        bottleneck: int = 32,
        heads: int = 2,
        feedforward_width: int = 64,
    ) -> None:
        super().__init__()
        if bottleneck % heads:
            raise ValueError("operator bottleneck must be divisible by its attention heads")
        self.output_width = output_width
        self.layers = layers
        self.bottleneck = bottleneck
        self.heads = heads
        self.feedforward_width = feedforward_width
        self.coefficient_projection = nn.Linear(1, bottleneck, bias=False)
        self.b_projection = nn.Linear(output_width, bottleneck, bias=False)
        self.layer_embedding = nn.Embedding(layers, bottleneck)
        self.rank_attention = nn.TransformerEncoderLayer(
            bottleneck,
            heads,
            dim_feedforward=feedforward_width,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.coefficient_readout = nn.Linear(bottleneck, 1, bias=False)

    def forward(
        self,
        coefficients: torch.Tensor,
        b_rows: torch.Tensor,
        layer: int | torch.Tensor,
    ) -> torch.Tensor:
        if coefficients.ndim != 3 or b_rows.ndim != 3:
            raise ValueError("coefficients and B rows must be [batch, sequence, rank] and [batch, rank, width]")
        if coefficients.shape[0] != b_rows.shape[0] or coefficients.shape[2] != b_rows.shape[1]:
            raise ValueError("coefficients and B rows must have matching batch and rank dimensions")
        if b_rows.shape[2] != self.output_width:
            raise ValueError("B width or layer index does not match the operator")
        batch, sequence, rank = coefficients.shape
        layer_ids = torch.as_tensor(layer, device=coefficients.device, dtype=torch.long)
        if layer_ids.ndim == 0:
            if not 0 <= int(layer_ids) < self.layers:
                raise ValueError("B width or layer index does not match the operator")
            layer_embedding = self.layer_embedding(layer_ids)[None, None, None]
        elif layer_ids.shape == (batch,) and bool(((layer_ids >= 0) & (layer_ids < self.layers)).all()):
            layer_embedding = self.layer_embedding(layer_ids)[:, None, None]
        else:
            raise ValueError("tensor layer IDs must have one valid ID per batch row")
        tokens = (
            self.coefficient_projection(coefficients.float().unsqueeze(-1))
            + self.b_projection(b_rows.float()).unsqueeze(1)
            + layer_embedding
        )
        transformed = self.rank_attention(tokens.reshape(batch * sequence, rank, self.bottleneck))
        transformed_coefficients = self.coefficient_readout(transformed).reshape(batch, sequence, rank)
        return einsum(
            transformed_coefficients,
            b_rows.float(),
            "batch seq rank, batch rank output -> batch seq output",
        )


class QueryConditionedLatentEncoder(nn.Module):
    """Transform bounded document latents from clean request features."""

    def __init__(
        self,
        query_width: int,
        latent_width: int,
        layers: int,
        bottleneck: int = 32,
    ) -> None:
        super().__init__()
        self.query_width = query_width
        self.latent_width = latent_width
        self.layers = layers
        self.bottleneck = bottleneck
        self.latent_projection = nn.Linear(latent_width, bottleneck, bias=False)
        self.query_projection = nn.Linear(query_width, bottleneck, bias=False)
        self.layer_embedding = nn.Embedding(layers, bottleneck)
        self.output_projection = nn.Linear(bottleneck, latent_width, bias=False)
        nn.init.zeros_(self.output_projection.weight)

    def forward(
        self,
        latents: torch.Tensor,
        query_states: torch.Tensor,
    ) -> torch.Tensor:
        if latents.ndim != 4 or latents.shape[1] != self.layers:
            raise ValueError("latents must be [batch, layers, slots, latent width]")
        if latents.shape[-1] != self.latent_width:
            raise ValueError("latent width does not match the encoder")
        if query_states.shape != (latents.shape[0], self.layers, self.query_width):
            raise ValueError("query states must be [batch, layers, query width]")

        latent32 = functional.layer_norm(latents.float(), (self.latent_width,))
        query32 = functional.layer_norm(query_states.float(), (self.query_width,))
        layer_ids = torch.arange(self.layers, device=latents.device)
        hidden = functional.silu(
            self.latent_projection(latent32)
            + self.query_projection(query32).unsqueeze(2)
            + self.layer_embedding(layer_ids)[None, :, None]
        )
        return latents + self.output_projection(hidden).to(latents.dtype)


def last_prompt_layer_states(
    hidden_states: tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor,
    layers: int,
) -> torch.Tensor:
    """Collect detached final non-padding prompt states from every model layer."""

    if len(hidden_states) != layers + 1:
        raise ValueError("hidden states must contain embeddings plus every model layer")
    if attention_mask.ndim != 2 or attention_mask.shape[0] != hidden_states[0].shape[0]:
        raise ValueError("attention mask must match the hidden-state batch")
    if attention_mask.shape[1] != hidden_states[0].shape[1] or not attention_mask.bool().any(dim=1).all():
        raise ValueError("every prompt must contain at least one unmasked token")
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    final_positions = positions.masked_fill(~attention_mask.bool(), -1).max(dim=1).values
    batch_indices = torch.arange(attention_mask.shape[0], device=attention_mask.device)
    return torch.stack(
        [state[batch_indices, final_positions] for state in hidden_states[1:]],
        dim=1,
    ).detach()


def make_representation_transition(
    representation: str,
    width: int,
    latent_width: int,
    layers: int,
    bottleneck: int = 32,
) -> ActiveMemoryTransition:
    if representation not in REPRESENTATIONS:
        raise ValueError(f"unknown representation: {representation}")
    memory_width = latent_width if representation in ("pre-head-latent", "projected-source") else width
    return ActiveMemoryTransition(width, memory_width, layers, bottleneck)


@contextlib.contextmanager
def capture_pre_head_latents(hypernet):
    """Capture normalized compiler latents without changing frozen head execution."""

    captured: list[torch.Tensor] = []

    def capture(_module, inputs):
        if len(inputs) != 1 or inputs[0].ndim != 5:
            raise ValueError("unexpected pre-LoRA-head latent layout")
        captured.append(inputs[0].detach().clone())

    handle = hypernet.head.register_forward_pre_hook(capture)
    try:
        yield captured
    finally:
        handle.remove()


@contextlib.contextmanager
def capture_projected_source(aggregator):
    """Capture per-layer source tokens after modality projection and before compression."""

    captured: list[torch.Tensor] = []

    def capture(_module, _inputs, output):
        if output.ndim != 3:
            raise ValueError("unexpected projected source layout")
        captured.append(output.detach().clone())

    handle = aggregator.perceiver.modality_projection.register_forward_hook(capture)
    try:
        yield captured
    finally:
        handle.remove()


def combine_captured_latents(
    captured: list[torch.Tensor],
    chunks_per_example: list[int],
) -> torch.Tensor:
    """Restore compiler microbatch order and enforce READOUT_LADDER's one-chunk K=1 contract."""

    if not captured:
        raise ValueError("the compiler head produced no latent capture")
    if any(count != 1 for count in chunks_per_example):
        raise ValueError("READOUT_LADDER requires exactly one compiler chunk per K=1 example")
    latents = torch.cat(captured, dim=0)
    if latents.shape[0] != len(chunks_per_example):
        raise ValueError("captured latent count does not match the compiled examples")
    if latents.shape[2] != 1:
        raise ValueError("READOUT_LADDER expects one adapted module")
    return latents[:, :, 0].detach().clone()


def combine_projected_sources(
    captured_parts: list[list[torch.Tensor]],
    masks: list[torch.Tensor],
    layers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Restore compiler microbatches as padded per-example, per-layer source memory."""

    if len(captured_parts) != len(masks) or not captured_parts:
        raise ValueError("projected source captures and masks must align")
    rows = []
    for captured, mask in zip(captured_parts, masks, strict=True):
        if len(captured) != layers:
            raise ValueError("projected source capture count does not match compiler layers")
        stacked = torch.stack(captured, dim=1)
        if stacked.shape[0] != mask.shape[0] or stacked.shape[2] != mask.shape[1]:
            raise ValueError("projected source capture does not match its token mask")
        for index in range(stacked.shape[0]):
            tokens = int(mask[index].sum())
            rows.append(stacked[index, :, :tokens].detach().clone())
    width = max(row.shape[1] for row in rows)
    projected = rows[0].new_zeros((len(rows), layers, width, rows[0].shape[-1]))
    projected_mask = torch.zeros((len(rows), width), dtype=torch.bool, device=projected.device)
    for index, row in enumerate(rows):
        projected[index, :, : row.shape[1]] = row
        projected_mask[index, : row.shape[1]] = True
    return projected, projected_mask


def compile_k1_representation_batch(
    backend,
    plans: list[dict],
    documents: dict[str, dict],
    compiler_chunk_batch_size: int,
    *,
    capture_projected: bool = False,
    source_texts: list[str] | None = None,
    source_token_ids: list[list[int]] | None = None,
    source_length_multiple: int | None = None,
) -> tuple[CompiledRepresentationBatch, dict]:
    """Compile native factors and normalized pre-head latents in one frozen pass."""

    if not plans or any(plan["k"] != 1 or len(plan["bank_document_ids"]) != 1 for plan in plans):
        raise ValueError("READOUT_LADDER accepts only nonempty K=1 batches")
    if source_texts is not None and source_token_ids is not None:
        raise ValueError("custom compiler source text and token IDs are mutually exclusive")
    if source_texts is not None and (len(source_texts) != len(plans) or any(not text.strip() for text in source_texts)):
        raise ValueError("custom compiler source texts must be nonempty and align with plans")
    if source_token_ids is not None and (
        len(source_token_ids) != len(plans) or any(not ids for ids in source_token_ids)
    ):
        raise ValueError("custom compiler source token IDs must be nonempty and align with plans")
    if source_length_multiple is not None and source_length_multiple < 1:
        raise ValueError("source length multiple must be positive")
    chunks = []
    chunks_per_example = []
    source_tokens = 0
    for index, plan in enumerate(plans):
        if source_token_ids is not None:
            ids = source_token_ids[index]
        else:
            text = (
                source_texts[index]
                if source_texts is not None
                else documents[plan["bank_document_ids"][0]]["text"]
            )
            ids = backend.tokenize_context({"context": [text]}, backend.tokenizer)["ctx_ids"][0]
        document_chunks = backend.split_context(
            {"ctx_ids": ids},
            model_name_or_path=backend.model_id,
            num_chunk_probs=None,
            max_chunk_len=backend.config["model"]["context_chunk_len"],
            min_chunk_len=0,
            max_num_split=None,
            is_train=False,
        )["ctx_ids"]
        chunks.extend(torch.tensor(chunk, dtype=torch.long) for chunk in document_chunks)
        chunks_per_example.append(len(document_chunks))
        source_tokens += len(ids)

    generated_parts = []
    latent_parts = []
    projected_parts = []
    projected_masks = []
    with torch.inference_mode():
        for start in range(0, len(chunks), compiler_chunk_batch_size):
            ctx_ids, ctx_mask = pad_sequences(
                chunks[start : start + compiler_chunk_batch_size],
                backend.tokenizer.pad_token_id,
                device=backend.model.device,
            )
            if source_length_multiple is not None:
                padding = (-ctx_ids.shape[1]) % source_length_multiple
                if padding:
                    ctx_ids = functional.pad(ctx_ids, (0, padding), value=backend.tokenizer.pad_token_id)
                    ctx_mask = functional.pad(ctx_mask, (0, padding), value=0)
            projected_context = (
                capture_projected_source(backend.model.hypernet.aggregator)
                if capture_projected
                else contextlib.nullcontext([])
            )
            with (
                capture_pre_head_latents(backend.model.hypernet) as captured,
                projected_context as projected,
            ):
                generated, layernorms = backend.model.generate_weights(ctx_ids, ctx_mask, None)
            if layernorms:
                raise ValueError("layernorm adaptation is outside the frozen READOUT_LADDER protocol")
            generated_parts.append(generated)
            latent_parts.extend(captured)
            if capture_projected:
                projected_parts.append(projected)
                projected_masks.append(ctx_mask.detach().clone())

        generated = concatenate_generated_parts(generated_parts)
        latents = combine_captured_latents(latent_parts, chunks_per_example)
        n_chunks = torch.ones(len(plans), device=backend.model.device, dtype=torch.int32)
        live_head_bias = (
            backend.model.hypernet.get_head_bias()
            if backend.model.hypernet.config.use_bias
            else None
        )
        head_bias = (
            {
                module: {key: value.to(backend.model.device) for key, value in part.items()}
                for module, part in live_head_bias.items()
            }
            if live_head_bias is not None
            else None
        )
        factors = backend.merge(generated, n_chunks, lora_bias=head_bias, bias_scaler=1.0)
        bias_factors = {
            module: {
                key: value.unsqueeze(0).expand(len(plans), -1, -1, -1)
                for key, value in part.items()
            }
            for module, part in head_bias.items()
        }
        projected_source, projected_source_mask = (None, None)
        if capture_projected:
            projected_source, projected_source_mask = combine_projected_sources(
                projected_parts,
                projected_masks,
                generated["down_proj"]["A"].shape[1],
            )

    def constants(value: dict) -> dict:
        return {
            module: {key: tensor.detach().clone() for key, tensor in part.items()}
            for module, part in value.items()
        }

    batch = CompiledRepresentationBatch(
        factors=constants(factors),
        generated_factors=constants(generated),
        bias_factors=constants(bias_factors),
        pre_head_latents=latents.detach().clone(),
        projected_source=projected_source,
        projected_source_mask=projected_source_mask,
    )
    return batch, {
        "document_occurrences": len(plans),
        "context_chunks": len(chunks),
        "source_tokens": source_tokens,
        "native_factor_bytes": tensor_bytes(batch.factors),
        "generated_factor_bytes": tensor_bytes(batch.generated_factors),
        "shared_bias_factor_bytes": tensor_bytes(batch.bias_factors),
        "pre_head_latent_bytes": tensor_bytes(batch.pre_head_latents),
        "projected_source_bytes": tensor_bytes(batch.projected_source),
        "projected_source_slots": int(batch.projected_source_mask.sum())
        if batch.projected_source_mask is not None
        else 0,
    }


def compiler_capture_parity_check(backend, plan: dict, documents: dict[str, dict]) -> dict:
    """Verify on a real compiler input that the latent hook leaves factors unchanged."""

    if plan["k"] != 1 or len(plan["bank_document_ids"]) != 1:
        raise ValueError("capture parity requires one K=1 plan")
    text = documents[plan["bank_document_ids"][0]]["text"]
    ids = backend.tokenize_context({"context": [text]}, backend.tokenizer)["ctx_ids"][0]
    chunks = backend.split_context(
        {"ctx_ids": ids},
        model_name_or_path=backend.model_id,
        num_chunk_probs=None,
        max_chunk_len=backend.config["model"]["context_chunk_len"],
        min_chunk_len=0,
        max_num_split=None,
        is_train=False,
    )["ctx_ids"]
    if len(chunks) != 1:
        raise ValueError("READOUT_LADDER capture parity requires one compiler chunk")
    ctx_ids = torch.tensor(chunks, dtype=torch.long, device=backend.model.device)
    ctx_mask = torch.ones_like(ctx_ids)
    with torch.inference_mode():
        expected, expected_layernorms = backend.model.generate_weights(ctx_ids, ctx_mask, None)
        with (
            capture_pre_head_latents(backend.model.hypernet) as captured,
            capture_projected_source(backend.model.hypernet.aggregator) as projected_captured,
        ):
            actual, actual_layernorms = backend.model.generate_weights(ctx_ids, ctx_mask, None)
    if expected_layernorms or actual_layernorms:
        raise ValueError("layernorm adaptation is outside the frozen READOUT_LADDER protocol")
    expected_digest = tensor_digest(expected)
    actual_digest = tensor_digest(actual)
    if expected_digest != actual_digest:
        raise RuntimeError("pre-head capture changed generated factors")
    latents = combine_captured_latents(captured, [1])
    projected, projected_mask = combine_projected_sources(
        [projected_captured],
        [ctx_mask],
        actual["down_proj"]["A"].shape[1],
    )
    return {
        "factors_equal": True,
        "factor_sha256": actual_digest,
        "captured_shape": list(latents.shape),
        "captured_dtype": str(latents.dtype),
        "captured_bytes": tensor_bytes(latents),
        "projected_source_shape": list(projected.shape),
        "projected_source_dtype": str(projected.dtype),
        "projected_source_bytes": tensor_bytes(projected),
        "projected_source_slots": int(projected_mask.sum()),
    }


def factor_rank_effects(
    inputs: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """Return each rank-one contribution without changing the native sum."""

    if a.ndim == 2 and b.ndim == 2:
        a = a.unsqueeze(0).expand(inputs.shape[0], -1, -1)
        b = b.unsqueeze(0).expand(inputs.shape[0], -1, -1)
    if a.ndim != 3 or b.ndim != 3 or a.shape[:2] != b.shape[:2] or a.shape[0] != inputs.shape[0]:
        raise ValueError("LoRA factors must be unbatched or match the input batch and rank")
    coefficients = einsum(a, inputs.to(a.dtype), "batch rank input, batch seq input -> batch seq rank")
    # This is an outer product, not a contraction. Broadcasting avoids the
    # generic einsum kernel timing out on long sequences on a display GPU.
    effects = coefficients.unsqueeze(-1) * b.unsqueeze(1)
    return (effects * scaling).to(inputs.dtype)


@contextlib.contextmanager
def installed_nonlinear_rank_operator(
    model,
    layers,
    compiled: CompiledRepresentationBatch,
    operator: NonlinearRankOperator,
    scaling: float,
    *,
    official_backend,
):
    """Install shared bias plus a direct nonlinear read of generated document factors."""

    compute_dtype = getattr(operator, "compute_dtype", None)
    generated = {
        module: {
            key: tensor.to(device=model.device, dtype=compute_dtype or tensor.dtype)
            for key, tensor in part.items()
        }
        for module, part in compiled.generated_factors.items()
    }
    bias = {
        module: {key: tensor.to(model.device) for key, tensor in part.items()}
        for module, part in compiled.bias_factors.items()
    }
    official_backend.model.patch_lora_forward()
    official_backend.install(
        official_backend.model.base_model,
        official_backend.model.hypernet.layer_indices,
        bias,
        torch.ones(bias["down_proj"]["A"].shape[0], device=model.device, dtype=torch.int32),
    )
    handles = []
    for layer_index, layer in enumerate(layers):
        module = layer.mlp.down_proj

        def inject(_module, inputs, output, layer_index=layer_index):
            a = generated["down_proj"]["A"][:, layer_index]
            b = generated["down_proj"]["B"][:, layer_index]
            coefficients = einsum(
                a,
                inputs[0].to(a.dtype),
                "batch rank input, batch seq input -> batch seq rank",
            )
            update = operator(coefficients, b, layer_index) * scaling
            return output + update.to(output.dtype)

        handles.append(module.register_forward_hook(inject))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        official_backend.model.reset()


@contextlib.contextmanager
def installed_representation_transition(
    model,
    layers,
    compiled: CompiledRepresentationBatch,
    transition: ActiveMemoryTransition | None,
    representation: str,
    scaling: float,
    *,
    apply_document_factors: bool = True,
    official_backend=None,
):
    """Install exact native D2L plus an optional post-block representation reader."""

    if representation not in REPRESENTATIONS:
        raise ValueError(f"unknown representation: {representation}")
    parts = (compiled.factors, compiled.generated_factors, compiled.bias_factors)
    device_parts = [
        {
            module: {key: tensor.to(model.device) for key, tensor in part.items()}
            for module, part in value.items()
        }
        for value in parts
    ]
    factors, generated, bias = device_parts
    latents = compiled.pre_head_latents.to(model.device)
    projected_source = (
        compiled.projected_source.to(model.device) if compiled.projected_source is not None else None
    )
    projected_source_mask = (
        compiled.projected_source_mask.to(model.device)
        if compiled.projected_source_mask is not None
        else None
    )
    installed_factors = factors if apply_document_factors else bias
    use_official_install = official_backend is not None
    if use_official_install:
        official_backend.model.patch_lora_forward()
        official_backend.install(
            official_backend.model.base_model,
            official_backend.model.hypernet.layer_indices,
            installed_factors,
            torch.ones(installed_factors["down_proj"]["A"].shape[0], device=model.device, dtype=torch.int32),
        )
    effects: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    handles = []
    for layer_index, layer in enumerate(layers):
        module = layer.mlp.down_proj

        def inject(_module, inputs, output, layer_index=layer_index):
            full_delta = factor_delta(
                inputs[0],
                factors["down_proj"]["A"][:, layer_index],
                factors["down_proj"]["B"][:, layer_index],
                scaling,
            )
            generated_ranks = factor_rank_effects(
                inputs[0],
                generated["down_proj"]["A"][:, layer_index],
                generated["down_proj"]["B"][:, layer_index],
                scaling,
            )
            bias_effect = factor_delta(
                inputs[0],
                bias["down_proj"]["A"][:, layer_index],
                bias["down_proj"]["B"][:, layer_index],
                scaling,
            )
            effects[layer_index] = (full_delta, generated_ranks, bias_effect)
            native_delta = full_delta if apply_document_factors else bias_effect
            return output if use_official_install else output + native_delta

        handles.append(module.register_forward_hook(inject))
        if transition is not None:

            def correct(_module, _inputs, output, layer_index=layer_index):
                hidden = hidden_output(output)
                _full, generated_ranks, bias_effect = effects.pop(layer_index)
                if representation == "summed-effect":
                    memory = generated_ranks.sum(dim=2, keepdim=True)
                elif representation == "rank-effect":
                    memory = generated_ranks
                elif representation == "projected-source":
                    if projected_source is None or projected_source_mask is None:
                        raise ValueError("projected-source representation was not compiled")
                    memory = projected_source[:, layer_index]
                else:
                    memory = latents[:, layer_index]
                memory_mask = projected_source_mask if representation == "projected-source" else None
                return replace_hidden(
                    output,
                    transition(hidden, memory, bias_effect, layer_index, memory_mask),
                )

            handles.append(layer.register_forward_hook(correct))
    try:
        yield
    finally:
        effects.clear()
        for handle in handles:
            handle.remove()
        if use_official_install:
            official_backend.model.reset()
