"""ATTENUATION_SCHEDULES: ceilings of three question-conditioned attenuation schedules (no training).

LAYER_INTERVENTIONS--RESPONSE_ONLY_ORACLE closed every reader that decides relevance from the model's state at the write site
while the prompt is processed under native K: the distractors' writes enter the prompt's own
forward pass in layers 0--7, before any state carries membership (LAYER_INTERVENTIONS), and gates derived at
or after prefill cannot undo that (RESPONSE_ONLY_ORACLE). The remaining question is whether the prompt can be
processed *without* the bank and the (attenuated) bank applied afterwards. Three schedules,
each run here with the true per-rank membership mask (the ceiling of any gate on that schedule):

* **A** (clean prefill, attenuated tail): the bank is silent at every prompt position except the
  last; the masked bank acts at the last prompt position (whose logits give the first response
  token) and at every decode position, at every layer. One causal pass: the last position's
  computation equals its recomputation over the clean cache.
* **B** (late attenuation): the bank is silent below layer ``layer_from`` (default 8, LAYER_INTERVENTIONS's
  damage boundary) at every position and masked from that layer on at every position. One pass.
* **C** (clean prefill, replayed cache): the prompt is prefilled with the bank silent; then the
  masked bank's writes at every prompt position are added after the fact through the residual
  path (``replay_prompt_cache``): per layer the write changes the MLP output by
  ``post_ffn_ln(clean + write) - post_ffn_ln(clean)``, the residual entering every later layer
  shifts by the cumulative change, and that layer's keys and values are recomputed from the
  shifted residual (RoPE applied) and overwrite the cache. Attention and MLP outputs of later
  layers are not re-evaluated (first-order replay). The last prompt position is then recomputed
  exactly with the masked bank over the replayed cache, and decoding proceeds with the masked
  bank at every layer. Manual greedy loop over a ``HybridCache`` (``generate_design_c``).
* **none**: the bank silent everywhere (the shared bias alone, READOUT_LADDER's no-effect control): the
  PK-free floor every schedule must clear.

With the true mask the masked computation does not depend on the distractors, so each ceiling is
K-independent by construction (as ``oracle_from0`` was in LAYER_INTERVENTIONS); K=1 and K=8 are both run as the
check of that.
"""

from __future__ import annotations

from typing import Any

import torch
from distractor_curves_core import MAX_NEW_TOKENS, bank_representation, compile_documents, score
from write_site_readers_readers import effect_read, generate_with_reader, installed_write_site_reader
from layer_interventions_gates import RANKS_PER_DOCUMENT, membership, rank_mask
from torch import nn

SCHEDULES = ("none", "A", "B", "C")
DESIGN_B_LAYER_FROM = 8
RESPONSE_PREFIX = "schedule_"


class ScheduleReader(nn.Module):
    """Fixed per-rank mask on schedule ``none``, ``A`` or ``B`` (write-site; batch-1 generation).

    Weights are ``-1`` (bank silent: the native sum is cancelled exactly) wherever the schedule keeps
    the bank off and ``mask - 1`` (source ranks native, distractor ranks cancelled) where it is on.
    """

    execution_site = "write-site"
    arm = "schedule"
    uses_latents = False

    def __init__(self, mask: torch.Tensor, schedule: str, layer_from: int = DESIGN_B_LAYER_FROM) -> None:
        super().__init__()
        if schedule not in ("none", "A", "B"):
            raise ValueError(f"ScheduleReader handles none/A/B, not {schedule!r}")
        self.register_buffer("mask", mask.float())
        self.schedule, self.layer_from = schedule, layer_from

    def encode_bank(self, a_rows, b_rows, layer):
        return {"b_norms": b_rows.float().norm(dim=-1)}

    def weights(self, batch: int, tokens: int, layer: int, device) -> torch.Tensor:
        gate = self.mask.to(device) - 1.0
        weights = torch.full((batch, tokens, self.mask.numel()), -1.0, device=device)
        if self.schedule == "A":
            if tokens == 1:
                weights[:] = gate
            else:
                weights[:, -1] = gate
        elif self.schedule == "B" and layer >= self.layer_from:
            weights[:] = gate
        return weights

    def correction(self, state, coefficients, b_rows, bank, layer, scaling, reference=None):
        if coefficients.shape[-1] != self.mask.numel():
            raise ValueError("mask width does not match the bank's rank count")
        batch, tokens = coefficients.shape[:2]
        return effect_read(self.weights(batch, tokens, layer, coefficients.device), coefficients, b_rows, scaling)


class ReplayReader(nn.Module):
    """Design C's write-site reader: silent during the multi-token prefill while recording, per layer,
    the clean ``down_proj`` output and the masked bank's write at every prompt position; masked at
    the single-token steps (the recomputed last prompt position and the decode positions)."""

    execution_site = "write-site"
    arm = "schedule"
    uses_latents = False

    def __init__(self, mask: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mask", mask.float())
        self.records: dict[int, dict[str, torch.Tensor]] = {}

    def encode_bank(self, a_rows, b_rows, layer):
        return {"b_norms": b_rows.float().norm(dim=-1)}

    def correction(self, state, coefficients, b_rows, bank, layer, scaling, reference=None):
        if coefficients.shape[-1] != self.mask.numel():
            raise ValueError("mask width does not match the bank's rank count")
        batch, tokens = coefficients.shape[:2]
        mask = self.mask.to(coefficients.device).expand(batch, tokens, -1)
        if tokens == 1:
            return effect_read(mask - 1.0, coefficients, b_rows, scaling)
        silent = effect_read(torch.full_like(mask, -1.0), coefficients, b_rows, scaling)
        if reference is None:
            raise ValueError("the replay reader needs the down_proj output as reference")
        self.records[layer] = {"clean": (reference.float() + silent).detach(),
                               "write": effect_read(mask, coefficients, b_rows, scaling).detach()}
        return silent


def decoder_of(model) -> nn.Module:
    """The Gemma-2 decoder stack (owner of ``rotary_emb`` and ``layers``) inside a PEFT/HF wrapper."""
    for module in model.modules():
        if hasattr(module, "rotary_emb") and hasattr(module, "layers"):
            return module
    raise ValueError("no decoder with rotary embeddings found")


def replay_prompt_cache(layers, cache, residuals: dict[int, torch.Tensor], records: dict[int, dict[str, torch.Tensor]],
                        cos: torch.Tensor, sin: torch.Tensor, positions: int) -> torch.Tensor:
    """Add the recorded writes to the clean prompt cache through the residual path only; returns the
    cumulative residual shift [batch, positions, width] after the last layer.

    Layer ``i``'s keys and values are recomputed from ``input_layernorm(h_i + sum_{j<i} delta_j)``
    with ``delta_j = post_ffn_ln(clean_j + write_j) - post_ffn_ln(clean_j)`` and overwrite cache
    positions ``[0, positions)``; layer 0's inputs are the embeddings and stay unchanged."""
    from transformers.models.gemma2.modeling_gemma2 import apply_rotary_pos_emb

    delta = None
    for index, layer in enumerate(layers):
        if delta is not None:
            residual = residuals[index][:, :positions]
            normed = layer.input_layernorm(residual + delta.to(residual.dtype))
            attention = layer.self_attn
            shape = (normed.shape[0], normed.shape[1], -1, attention.head_dim)
            keys = attention.k_proj(normed).view(shape).transpose(1, 2)
            values = attention.v_proj(normed).view(shape).transpose(1, 2)
            _, keys = apply_rotary_pos_emb(keys, keys, cos, sin)
            cache.key_cache[index][:, :, :positions] = keys.to(cache.key_cache[index].dtype)
            cache.value_cache[index][:, :, :positions] = values.to(cache.value_cache[index].dtype)
        record = records[index]
        clean = record["clean"][:, :positions].to(layer.post_feedforward_layernorm.weight.dtype)
        shifted = (record["clean"][:, :positions] + record["write"][:, :positions]).to(clean.dtype)
        shift = (layer.post_feedforward_layernorm(shifted).float() - layer.post_feedforward_layernorm(clean).float())
        delta = shift if delta is None else delta + shift
    return delta


def stop_ids(model) -> set[int]:
    eos = model.generation_config.eos_token_id
    return set(eos if isinstance(eos, (list, tuple)) else [eos])


def greedy_decode(model, prompt: torch.Tensor, cache, stop: set[int], max_new_tokens: int,
                  prefill_hook=None, offset: int = 0) -> list[int]:
    """Batch-1 greedy decoding with an explicit cache: prefill ``prompt[:, :-1]`` at cache positions
    ``offset + ...``, call ``prefill_hook(cache)`` if given, then step from the last prompt token (at position
    ``offset + width - 1``, overwriting that cache slot if it was filled before). Returns the generated ids
    (stop token excluded)."""
    device = prompt.device
    width = prompt.shape[1]
    if width > 1:
        model(prompt[:, :-1], attention_mask=torch.ones(1, offset + width - 1, dtype=torch.long, device=device),
              past_key_values=cache, use_cache=True, cache_position=offset + torch.arange(width - 1, device=device))
    if prefill_hook is not None:
        prefill_hook(cache)
    generated: list[int] = []
    token = prompt[:, -1:]
    for step in range(max_new_tokens):
        position = offset + width - 1 + step
        logits = model(token, attention_mask=torch.ones(1, position + 1, dtype=torch.long, device=device),
                       past_key_values=cache, use_cache=True,
                       cache_position=torch.tensor([position], device=device)).logits[:, -1]
        next_id = int(logits.argmax(-1))
        if next_id in stop:
            break
        generated.append(next_id)
        token = torch.tensor([[next_id]], device=device)
    return generated


def generate_design_c(backend, layers, compiled, mask: torch.Tensor, question: str, dataset: str,
                      max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Design C for one question: clean prefill, residual-path replay of the masked writes into the
    cache, exact recomputation of the last prompt position, greedy decoding with the masked bank."""
    from official_d2l_online_batch_distillation import render_prompt
    from transformers.cache_utils import HybridCache

    model = backend.model.base_model
    device = model.device
    scaling = model.peft_config["default"].lora_alpha
    prompt = render_prompt(backend, question, dataset).unsqueeze(0).to(device)
    reader = ReplayReader(mask).to(device)
    residuals: dict[int, torch.Tensor] = {}
    handles = [layer.input_layernorm.register_forward_pre_hook(
        lambda _module, inputs, index=index: residuals.__setitem__(index, inputs[0].detach()))
        for index, layer in enumerate(layers)]
    decoder = decoder_of(model)

    def replay(cache) -> None:
        for handle in handles:
            handle.remove()
        positions = prompt.shape[1] - 1
        if positions == 0:
            return
        cos, sin = decoder.rotary_emb(residuals[0], torch.arange(positions, device=device).unsqueeze(0))
        replay_prompt_cache(layers, cache, residuals, reader.records, cos, sin, positions)
        residuals.clear()
        reader.records.clear()

    backend.model.reset()
    try:
        with installed_write_site_reader(backend, layers, compiled, reader, scaling), torch.inference_mode():
            cache = HybridCache(model.config, max_batch_size=1, max_cache_len=prompt.shape[1] + max_new_tokens,
                                dtype=model.dtype, device=device)
            generated = greedy_decode(model, prompt, cache, stop_ids(model), max_new_tokens, prefill_hook=replay)
    finally:
        for handle in handles:
            handle.remove()
    return backend.tokenizer.decode(generated, skip_special_tokens=True).strip()


def schedule_mask(members: list[bool], schedule: str, ranks_per_document: int) -> torch.Tensor:
    """``C-none`` is the smoke's plumbing check: Design C's machinery with every rank off must answer as ``none``."""
    mask = rank_mask(members, ranks_per_document)
    return torch.zeros_like(mask) if schedule in ("none", "C-none") else mask


def generate_schedule(backend, layers, compiled, members: list[bool], schedule: str, question: str, dataset: str, *,
                      layer_from: int = DESIGN_B_LAYER_FROM, ranks_per_document: int = RANKS_PER_DOCUMENT) -> str:
    mask = schedule_mask(members, schedule, ranks_per_document)
    if schedule in ("C", "C-none"):
        return generate_design_c(backend, layers, compiled, mask, question, dataset)
    reader = ScheduleReader(mask, schedule, layer_from).to(backend.model.device)
    return generate_with_reader(backend, layers, "schedule", reader, compiled, question, dataset)


def schedule_rows(backend, layers, suite: dict[str, Any], source: str, banks: list[dict[str, Any]], *,
                  schedules: tuple[str, ...] = SCHEDULES, ks: tuple[int, ...] = (1, 8),
                  layer_from: int = DESIGN_B_LAYER_FROM, ranks_per_document: int = RANKS_PER_DOCUMENT) -> dict[str, Any]:
    """Greedy responses of every schedule on every question of one source (DISTRACTOR_CURVES row shape; responses keyed
    ``schedule_<name>``, Design B as ``schedule_B<layer_from>``)."""
    documents = suite["documents"]
    chosen = [b for b in banks if b["k"] in ks and b["condition"] == "random"]
    needed = sorted({i for bank in chosen for i in bank["document_ids"]})
    compiled = dict(zip(needed, compile_documents(backend, [documents[i]["text"] for i in needed]), strict=True))
    questions = suite["questions"][source]
    answers = {q["id"]: q["answers"] for q in questions}
    rows = []
    for bank in chosen:
        representation = bank_representation(backend, [compiled[i] for i in bank["document_ids"]])
        members = membership(bank["document_ids"], source)
        for query in questions:
            responses = {}
            for schedule in schedules:
                name = f"{RESPONSE_PREFIX}{schedule}{layer_from if schedule == 'B' else ''}"
                responses[name] = generate_schedule(backend, layers, representation, members, schedule, query["input"],
                                                    query["dataset"], layer_from=layer_from,
                                                    ranks_per_document=ranks_per_document)
            rows.append({"query_id": query["id"], "source_id": source, "bank_id": bank["bank_id"],
                         "condition": bank["condition"], "k": bank["k"], "responses": responses,
                         "scores": {name: score(text, answers[query["id"]]) for name, text in responses.items()}})
    return {"source_id": source, "per_question": [], "rows": rows, "compiled_documents": len(needed),
            "source_tokens": {i: compiled[i]["source_tokens"] for i in needed}}


def run_schedule(backend, suite: dict[str, Any], unit: dict[str, Any]) -> dict[str, Any]:
    from ctx_to_lora.utils import get_layers

    layers = get_layers(backend.model.base_model)
    results = [schedule_rows(backend, layers, suite, source, unit["banks"][source],
                             schedules=tuple(unit.get("schedules", SCHEDULES)), ks=tuple(unit.get("ks", (1, 8))),
                             layer_from=int(unit.get("layer_from", DESIGN_B_LAYER_FROM)),
                             ranks_per_document=unit.get("ranks_per_document", RANKS_PER_DOCUMENT))
               for source in unit["sources"]]
    return {"sources": results, "suite": unit["suite"], "schedules": list(unit.get("schedules", SCHEDULES)),
            "layer_from": int(unit.get("layer_from", DESIGN_B_LAYER_FROM)),
            "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0}
