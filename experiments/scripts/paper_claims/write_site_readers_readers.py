"""WRITE_SITE_READERS reader arms: write-site readers over individual rank effects, and their installer.

DISTRACTOR_CURVES's post-block reader (arm A, ``ActiveMemoryTransition``) layer-norms each rank effect
``c_r B_r`` before attending to it. Since ``LN(c v) = sign(c) LN(v)``, its keys and values are
``+-LN(B_r)``: token-independent up to sign, with the match strength ``|c_r|`` and the bank's
magnitude discarded. Its correction also lives in a fixed width-32 output subspace after the
block, so it cannot shrink the native sum the way ``K**-beta`` normalization does.

The arms here act at the write site (the ``down_proj`` output, where the LoRA delta is added
and before ``post_feedforward_layernorm``), see the residual stream entering the MLP sub-block,
keep the signed strength ``m_r = scaling * c_r * ||B_r||`` of every rank, read raw full-width
effects, and can rescale the native sum. Every arm starts as an exact zero correction, so
update 0 reproduces native D2L.

* **B, shallow write-site** (``WriteSiteShallowReader``): one attention read over rank slots
  with width-32 keys, like DISTRACTOR_CURVES, but with magnitude-aware keys and the write-site output path.
* **C, factored deep rank reader** (``FactoredDeepRankReader``): token-independent slot
  embeddings from ``(A_r, B_r)`` pass through transformer blocks once per bank and layer
  (``encode_bank``); per token, a few read tokens formed from the state cross-attend to the
  slots (keys modulated by ``m_r``) through further blocks, then a nonlinear readout.
* **D, deep aggregate control**: arm C with the slots pooled into one aggregate slot before
  any processing (``aggregate=True``), so depth without rank separation.

Output contract shared by B/C/D (``WriteSiteReader.correction``): with layer-normalized read
state ``h``, native delta ``d = scaling * sum_r c_r B_r``, and ``n`` attention reads ``v_i =
scaling * sum_r alpha_ir c_r B_r`` (full-width convex combinations of raw effects, computed
without materializing ``slots x hidden``), the correction is ``W_out h + gamma(h) d + sum_i
beta_i(h) v_i``. ``W_out h`` is expressed in branch units (scaled by the native MLP output's token RMS and
1/sqrt(dim)); ``gamma`` and ``beta`` are token-wise scalars from the read state, bounded by
``tanh`` of a 1/sqrt(dim)-scaled linear head. They let the reader express normalization
(``gamma = K**-beta - 1``) and emphasis of what it read, and tie every term's step size to the
signal scale (an unscaled readout diverged after one update in the CPU smoke). There is
no document-level gate, source-ID input, mask, or top-k; selection happens inside the
attention over rank slots. ``W_out``, ``gamma``, and ``beta`` start at zero.

FLOPs per token are estimated by ``per_token_flops`` from the module shapes, so every arm's
cost is reported next to its accuracy (claim 1 of the paper).
"""

from __future__ import annotations

import contextlib
import math
from typing import Any

import torch
from distractor_curves_core import MAX_NEW_TOKENS
from einops import einsum
from torch import nn
from torch.nn import functional

ARMS = ("A", "B", "C", "D")
WRITE_SITE_ARMS = ("B", "C", "D")
MAGNITUDE_FEATURES = 2  # m and sign(m) log1p(|m|)


# -- shared pieces ----------------------------------------------------------------------
def signed_strength(coefficients: torch.Tensor, b_norms: torch.Tensor, scaling: float) -> torch.Tensor:
    """``m_r = scaling * c_r * ||B_r||`` for coefficients [batch, seq, R] and norms [batch, R]."""
    return coefficients.float() * b_norms.float().unsqueeze(1) * scaling


def magnitude_features(m: torch.Tensor) -> torch.Tensor:
    return torch.stack((m, torch.sign(m) * torch.log1p(m.abs())), dim=-1)


def effect_read(weights: torch.Tensor, coefficients: torch.Tensor, b_rows: torch.Tensor, scaling: float) -> torch.Tensor:
    """``scaling * sum_r w_r c_r B_r`` without materializing slots x hidden: [batch, seq, R] -> [batch, seq, D]."""
    return einsum((weights * coefficients.float()), b_rows.float(), "b s r, b r d -> b s d") * scaling


class WriteSiteReader(nn.Module):
    """Common install contract: ``encode_bank`` once per bank, ``correction`` per token."""

    execution_site = "write-site"
    arm = ""

    def __init__(self, width: int, hidden: int, layers: int, reads: int) -> None:
        super().__init__()
        self.width = width
        self.hidden = hidden
        self.layers = layers
        self.reads = reads
        self.read_norm = nn.LayerNorm(hidden * reads)
        self.output_projection = nn.Linear(hidden * reads, width, bias=False)
        self.aggregate_scale = nn.Linear(hidden * reads, 1, bias=False)
        self.read_scales = nn.Linear(hidden * reads, reads, bias=False)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.aggregate_scale.weight)
        nn.init.zeros_(self.read_scales.weight)

    def scalars(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Bounded token-wise multipliers (gamma for the native sum, beta_i per read) from the read state."""
        scale = 1.0 / math.sqrt(h.shape[-1])
        return torch.tanh(self.aggregate_scale(h) * scale), torch.tanh(self.read_scales(h) * scale)

    def encode_bank(self, a_rows: torch.Tensor, b_rows: torch.Tensor, layer: int) -> dict[str, torch.Tensor]:
        """Token-independent slot features for one layer; subclasses add processed embeddings."""
        return {"b_norms": b_rows.float().norm(dim=-1)}

    def read(self, state: torch.Tensor, m: torch.Tensor, bank: dict[str, torch.Tensor], layer: int,
             ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (read state [batch, seq, reads*hidden], attention weights [batch, seq, reads, R])."""
        raise NotImplementedError

    def readout(self, h: torch.Tensor, reference: torch.Tensor | None) -> torch.Tensor:
        """``W_out h`` in branch units: scaled by the native MLP output's token RMS and 1/sqrt(dim).

        The correction joins the ``down_proj`` output before ``post_feedforward_layernorm``, whose
        per-token RMS is small, so an unscaled zero-initialized readout moved by one optimizer step
        already matched the branch's own magnitude (CPU smoke: arms C/D diverged in one update).
        """
        out = self.output_projection(h) / math.sqrt(h.shape[-1])
        if reference is None:
            return out
        return out * reference.float().pow(2).mean(dim=-1, keepdim=True).sqrt().detach()

    def correction(self, state: torch.Tensor, coefficients: torch.Tensor, b_rows: torch.Tensor,
                   bank: dict[str, torch.Tensor], layer: int, scaling: float,
                   reference: torch.Tensor | None = None) -> torch.Tensor:
        if state.shape[-1] != self.width or b_rows.shape[-1] != self.width:
            raise ValueError("state and B rows must have the model width")
        if coefficients.shape[:2] != state.shape[:2] or coefficients.shape[2] != b_rows.shape[1]:
            raise ValueError("coefficients must be [batch, seq, R] matching the state and B rows")
        m = signed_strength(coefficients, bank["b_norms"], scaling)
        h, weights = self.read(state, m, bank, layer)
        h = self.read_norm(h)
        native = effect_read(torch.ones_like(coefficients, dtype=torch.float32), coefficients, b_rows, scaling)
        gamma, betas = self.scalars(h)
        out = self.readout(h, reference) + gamma * native
        for index in range(self.reads):
            out = out + betas[..., index:index + 1] * effect_read(weights[:, :, index], coefficients, b_rows, scaling)
        return out


class WriteSiteShallowReader(WriteSiteReader):
    """Arm B: DISTRACTOR_CURVES's single attention read, moved to the write site with magnitude-aware keys."""

    arm = "B"

    def __init__(self, width: int, layers: int, bottleneck: int = 32) -> None:
        super().__init__(width, bottleneck, layers, reads=1)
        self.bottleneck = bottleneck
        self.query_projection = nn.Linear(width, bottleneck, bias=False)
        self.state_projection = nn.Linear(width, bottleneck, bias=False)
        self.key_projection = nn.Linear(width, bottleneck, bias=False)
        self.magnitude_projection = nn.Linear(MAGNITUDE_FEATURES, bottleneck, bias=False)
        self.count_projection = nn.Linear(1, bottleneck, bias=False)
        self.layer_embedding = nn.Embedding(layers, bottleneck)

    def encode_bank(self, a_rows, b_rows, layer):
        bank = super().encode_bank(a_rows, b_rows, layer)
        bank["keys"] = self.key_projection(functional.layer_norm(b_rows.float(), (self.width,)))
        return bank

    def read(self, state, m, bank, layer):
        state32 = functional.layer_norm(state.float(), (self.width,))
        keys = bank["keys"].unsqueeze(1) + self.magnitude_projection(magnitude_features(m))  # [b, s, R, w]
        query = self.query_projection(state32)
        scores = einsum(query, keys, "b s w, b s r w -> b s r") / math.sqrt(self.bottleneck)
        weights = functional.softmax(scores, dim=-1)
        attended = einsum(weights, keys, "b s r, b s r w -> b s w")
        count = torch.full((*state.shape[:2], 1), math.log(m.shape[-1] / 8), device=state.device)
        layer_ids = torch.tensor(layer, device=state.device)
        h = functional.silu(self.state_projection(state32) + attended + self.count_projection(count)
                            + self.layer_embedding(layer_ids))
        return h, weights.unsqueeze(2)


def block(width: int, heads: int) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(width, heads, dim_feedforward=4 * width, dropout=0.0, batch_first=True,
                                      norm_first=True, activation="gelu")


class FactoredDeepRankReader(WriteSiteReader):
    """Arms C (``aggregate=False``) and D (``aggregate=True``)."""

    def __init__(self, width: int, input_width: int, layers: int, hidden: int = 128, reads: int = 4,
                 bank_blocks: int = 2, read_blocks: int = 2, heads: int = 4, aggregate: bool = False) -> None:
        super().__init__(width, hidden, layers, reads)
        self.arm = "D" if aggregate else "C"
        self.aggregate = aggregate
        self.input_width = input_width
        self.heads = heads
        self.b_embedding = nn.Linear(width, hidden, bias=False)
        self.a_embedding = nn.Linear(input_width, hidden, bias=False)
        self.layer_embedding = nn.Embedding(layers, hidden)
        self.bank_blocks = nn.ModuleList(block(hidden, heads) for _ in range(bank_blocks))
        self.state_projection = nn.Linear(width, hidden, bias=False)
        self.read_embedding = nn.Embedding(reads, hidden)
        self.count_projection = nn.Linear(1, hidden, bias=False)
        self.magnitude_projection = nn.Linear(MAGNITUDE_FEATURES, hidden, bias=False)
        self.cross_attention = nn.ModuleList(nn.MultiheadAttention(hidden, heads, batch_first=True)
                                             for _ in range(read_blocks))
        self.read_norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(2 * read_blocks))
        self.read_ffn = nn.ModuleList(nn.Sequential(nn.Linear(hidden, 4 * hidden), nn.GELU(),
                                                    nn.Linear(4 * hidden, hidden)) for _ in range(read_blocks))

    def encode_bank(self, a_rows, b_rows, layer):
        """Slot embeddings processed once per bank and layer; arm D pools the slots first."""
        bank = super().encode_bank(a_rows, b_rows, layer)
        z = (self.b_embedding(functional.layer_norm(b_rows.float(), (self.width,)))
             + self.a_embedding(functional.layer_norm(a_rows.float(), (self.input_width,))))
        if self.aggregate:
            z = z.mean(dim=1, keepdim=True)
        z = z + self.layer_embedding(torch.tensor(layer, device=z.device))
        for module in self.bank_blocks:
            z = module(z)
        bank["slots"] = z  # [batch, R or 1, hidden]
        return bank

    def read(self, state, m, bank, layer):
        batch, seq, _ = state.shape
        slots = bank["slots"]
        if self.aggregate:
            m = m.sum(dim=-1, keepdim=True)  # the aggregate slot carries the summed signed strength
        keys = slots.unsqueeze(1) + self.magnitude_projection(magnitude_features(m))  # [b, s, R', hidden]
        rows = keys.reshape(batch * seq, keys.shape[2], self.hidden)
        state32 = functional.layer_norm(state.float(), (self.width,))
        count = torch.full((batch, seq, 1), math.log(m.shape[-1] / 8), device=state.device)
        h = (self.state_projection(state32) + self.count_projection(count)).unsqueeze(2) \
            + self.read_embedding.weight[None, None] + self.layer_embedding(torch.tensor(layer, device=state.device))
        h = h.reshape(batch * seq, self.reads, self.hidden)
        weights = None
        for index, attention in enumerate(self.cross_attention):
            attended, weights = attention(self.read_norms[2 * index](h), rows, rows, need_weights=True,
                                          average_attn_weights=True)
            h = h + attended
            h = h + self.read_ffn[index](self.read_norms[2 * index + 1](h))
        weights = weights.reshape(batch, seq, self.reads, -1)
        if self.aggregate:
            weights = weights.expand(batch, seq, self.reads, m.shape[-1]) if m.shape[-1] == 1 else weights
        return h.reshape(batch, seq, self.reads * self.hidden), weights

    def correction(self, state, coefficients, b_rows, bank, layer, scaling, reference=None):
        if not self.aggregate:
            return super().correction(state, coefficients, b_rows, bank, layer, scaling, reference)
        # Arm D reads only the aggregate: every read weight is 1 over the summed effect.
        m = signed_strength(coefficients, bank["b_norms"], scaling)
        h, _ = self.read(state, m, bank, layer)
        h = self.read_norm(h)
        native = effect_read(torch.ones_like(coefficients, dtype=torch.float32), coefficients, b_rows, scaling)
        gamma, betas = self.scalars(h)
        return self.readout(h, reference) + (gamma + betas.sum(-1, keepdim=True)) * native


# -- construction and persistence --------------------------------------------------------
def make_reader(arm: str, model, latent_width: int, layers: int, *, bottleneck: int = 32, hidden: int = 128):
    """Arm A is DISTRACTOR_CURVES's post-block reader; B/C/D are the write-site readers above."""
    width = model.config.hidden_size
    if arm == "A":
        from official_d2l_k1_representation_ladder import make_representation_transition

        return make_representation_transition("rank-effect", width, latent_width, layers, bottleneck)
    if arm == "B":
        return WriteSiteShallowReader(width, layers, bottleneck)
    if arm in ("C", "D"):
        return FactoredDeepRankReader(width, model.config.intermediate_size, layers, hidden=hidden,
                                      aggregate=arm == "D")
    raise ValueError(f"unknown arm {arm}")


def reader_payload(reader, arm: str, selected_step: int, seed: int, recipe: str) -> dict[str, Any]:
    payload = {"transition": {k: v.detach().cpu() for k, v in reader.state_dict().items()}, "arm": arm,
               "selected_step": selected_step, "seed": seed, "recipe": recipe,
               "execution_site": reader.execution_site}
    if hasattr(reader, "payload_extras"):  # GATED_READERS gated readers describe their own body
        payload.update(reader.payload_extras())
        return payload
    if arm == "A":
        payload.update({"transition_type": "active-memory", "representation": "rank-effect",
                        "transition_bottleneck": reader.bottleneck, "memory_width": reader.memory_width})
    elif arm == "B":
        payload["transition_bottleneck"] = reader.bottleneck
    else:
        payload.update({"hidden": reader.hidden, "reads": reader.reads, "bank_blocks": len(reader.bank_blocks),
                        "read_blocks": len(reader.cross_attention), "heads": reader.heads})
    return payload


def load_write_site_readers_reader(path, backend, layers):
    """Return (arm, reader) for an WRITE_SITE_READERS checkpoint; arm A loads through DISTRACTOR_CURVES's loader."""
    model = backend.model.base_model
    saved = torch.load(path, map_location=model.device, weights_only=False)
    arm = saved.get("arm", "A")
    if arm == "A":
        from distractor_curves_core import load_reader

        return "A", load_reader(path, backend, layers)[1]
    if arm == "B":
        reader = WriteSiteShallowReader(model.config.hidden_size, len(layers), saved["transition_bottleneck"])
    else:
        reader = FactoredDeepRankReader(model.config.hidden_size, model.config.intermediate_size, len(layers),
                                        hidden=saved["hidden"], reads=saved["reads"], bank_blocks=saved["bank_blocks"],
                                        read_blocks=saved["read_blocks"], heads=saved["heads"], aggregate=arm == "D")
    reader.load_state_dict(saved["transition"])
    return arm, reader.to(model.device).eval()


# -- installation ------------------------------------------------------------------------
def mlp_input_norm(layer) -> nn.Module:
    """The normalization whose input is the residual entering the MLP sub-block: Gemma-2's
    ``pre_feedforward_layernorm`` or, on architectures without a separate pre-MLP norm (Qwen3,
    Llama), ``post_attention_layernorm``. Both receive the same quantity, the residual stream after
    the attention sub-block, so a pre-hook on either captures the write-site reader's state."""
    for name in ("pre_feedforward_layernorm", "post_attention_layernorm"):
        module = getattr(layer, name, None)
        if module is not None:
            return module
    raise ValueError("decoder layer has neither pre_feedforward_layernorm nor post_attention_layernorm")


@contextlib.contextmanager
def installed_write_site_reader(backend, layers, compiled, reader: WriteSiteReader, scaling: float):
    """Exact native D2L through the official install, plus the reader's write-site correction.

    ``compiled`` is a ``CompiledRepresentationBatch`` whose ``generated_factors["down_proj"]`` hold
    per-rank A [batch, layers, R, input] and B [batch, layers, R, width]. The residual entering
    each MLP sub-block is captured at the pre-MLP normalization (``mlp_input_norm``); the correction
    is added to the ``down_proj`` output, where the native delta already sits.
    """
    model = backend.model.base_model
    device = model.device
    factors = {m: {k: t.to(device) for k, t in part.items()} for m, part in compiled.factors.items()}
    a_all = compiled.generated_factors["down_proj"]["A"].to(device)
    b_all = compiled.generated_factors["down_proj"]["B"].to(device)
    backend.model.patch_lora_forward()
    backend.install(model, backend.model.hypernet.layer_indices, factors,
                    torch.ones(factors["down_proj"]["A"].shape[0], device=device, dtype=torch.int32))
    if getattr(reader, "uses_latents", False):
        if compiled.document_latents is None:
            raise ValueError("this reader keys on document latents but the compiled batch carries none")
        latents = compiled.document_latents.to(device)
        banks = [reader.encode_bank(a_all[:, i], b_all[:, i], i, latents=latents[:, :, i]) for i in range(len(layers))]
    else:
        banks = [reader.encode_bank(a_all[:, i], b_all[:, i], i) for i in range(len(layers))]
    states: dict[int, torch.Tensor] = {}
    handles = []
    for layer_index, layer in enumerate(layers):
        def capture(_module, inputs, layer_index=layer_index):
            states[layer_index] = inputs[0]

        def inject(_module, inputs, output, layer_index=layer_index):
            a = a_all[:, layer_index]
            coefficients = einsum(a, inputs[0].to(a.dtype), "b r i, b s i -> b s r")
            state = states.pop(layer_index)
            correction = reader.correction(state, coefficients, b_all[:, layer_index], banks[layer_index],
                                           layer_index, scaling, reference=output)
            return output + correction.to(output.dtype)

        handles.append(mlp_input_norm(layer).register_forward_pre_hook(capture))
        handles.append(layer.mlp.down_proj.register_forward_hook(inject))
    try:
        yield
    finally:
        states.clear()
        for handle in handles:
            handle.remove()
        backend.model.reset()


@contextlib.contextmanager
def installed_reader(backend, layers, compiled, arm: str, reader, scaling: float):
    """One entry point for every arm (and ``reader=None`` for native D2L)."""
    if reader is None or arm == "A":
        from official_d2l_k1_representation_ladder import installed_representation_transition

        with installed_representation_transition(backend.model.base_model, layers, compiled, reader, "rank-effect",
                                                 scaling, official_backend=backend):
            yield
    else:
        with installed_write_site_reader(backend, layers, compiled, reader, scaling):
            yield


def forward_with_reader(backend, layers, arm: str, reader, compiled, targets: dict[str, Any], scaling: float):
    """Teacher-forced logits for a gold batch under any arm."""
    backend.model.reset()
    with installed_reader(backend, layers, compiled, arm, reader, scaling):
        outputs = backend.model.base_model(targets["student_ids"], attention_mask=targets["student_mask"],
                                           use_cache=False, output_hidden_states=False, return_dict=True)
    return outputs.logits


def generate_with_reader(backend, layers, arm: str, reader, compiled, question: str, dataset: str,
                         max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Greedy batch-1 generation under any arm, with the ONLINE_BATCH_READER/READOUT_LADDER prompt."""
    from official_d2l_online_batch_distillation import render_prompt

    scaling = backend.model.base_model.peft_config["default"].lora_alpha
    prompt = render_prompt(backend, question, dataset).unsqueeze(0).to(backend.model.device)
    backend.model.reset()
    with installed_reader(backend, layers, compiled, arm, reader, scaling), torch.inference_mode():
        generated = backend.model.base_model.generate(prompt, attention_mask=torch.ones_like(prompt), do_sample=False,
                                                      max_new_tokens=max_new_tokens, use_cache=True,
                                                      pad_token_id=backend.tokenizer.pad_token_id)
    return backend.tokenizer.decode(generated[0, prompt.shape[1]:], skip_special_tokens=True).strip()


# -- accounting ---------------------------------------------------------------------------
def per_token_flops(reader, slots: int, layers: int, width: int, input_width: int) -> dict[str, float]:
    """Multiply-add-based FLOP estimate per generated token (2 FLOPs per MAC), excluding the backbone.

    The native coefficient computation ``A x`` is part of native D2L and is reported separately
    (``native_coefficients``). ``per_bank`` is the one-off cost per bank and layer for arm C/D.
    """
    linear = lambda i, o, n=1: 2 * i * o * n  # noqa: E731
    native = layers * linear(input_width, slots) * 1
    if isinstance(reader, WriteSiteShallowReader):
        w = reader.bottleneck
        per_layer = (linear(width, w) * 2 + linear(MAGNITUDE_FEATURES, w, slots) + 2 * slots * w * 2
                     + linear(w, width) + linear(w, 1) + linear(w, 1) + 2 * 2 * slots * width)
        return {"reader": layers * per_layer, "per_bank": layers * linear(width, w, slots), "native_coefficients": native}
    if isinstance(reader, FactoredDeepRankReader):
        h, r, s = reader.hidden, reader.reads, 1 if reader.aggregate else slots
        ffn = linear(h, 4 * h) * 2
        cross = len(reader.cross_attention) * (r * (linear(h, h) * 2 + 2 * s * h * 2 + ffn) + s * linear(h, h) * 2)
        keys = linear(MAGNITUDE_FEATURES, h, s)
        out = linear(r * h, width) + linear(r * h, 1 + r) + 2 * (1 + r) * slots * width
        per_layer = linear(width, h) + keys + cross + out
        per_bank = len(reader.bank_blocks) * (s * (linear(h, h) * 4 + ffn) + 2 * s * s * h * 2) \
            + s * (linear(width, h) + linear(input_width, h))
        return {"reader": layers * per_layer, "per_bank": layers * per_bank, "native_coefficients": native}
    # Arm A: DISTRACTOR_CURVES's post-block reader over layer-normed slots x hidden
    w = reader.bottleneck
    per_layer = linear(width, w) * 3 + linear(width, w, slots) * 2 + 2 * slots * w * 2 + linear(w, width) \
        + 2 * slots * width * 2
    return {"reader": layers * per_layer, "per_bank": 0, "native_coefficients": native}


def backbone_flops_per_token(parameters: int) -> int:
    return 2 * parameters


def parameter_count(reader) -> int:
    return sum(p.numel() for p in reader.parameters())
