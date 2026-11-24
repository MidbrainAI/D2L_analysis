"""Attenuation schedules with gates *predicted* from the clean prefill state (the probe-derived gates of RQ2.3).

``attenuation_schedules_schedules`` measures each schedule with the true membership mask. Here the mask is
replaced by a per-rank gate computed from what the model has at serving time: the question's own
(bank-silent) state at one layer, scored against each rank's key by the clean-prefill membership
probe fit with the clean-prefill probe recipe (``layer_interventions_gates.fitted_probe`` with ``silent``), or by the parameter-free
A-side strength ``|m_r|``. Nothing is routed: every rank keeps its own weight (``pool="rank"``);
``pool="document"`` averages the scores over a document's ranks and broadcasts (several documents
may stay open; reported separately).

Gate modes: ``soft`` = sigmoid of the probe logit (strength: ``s_r / max s``), ``hard`` = 1 if the
logit is positive (strength: ``soft >= 0.5``), else 0.

Schedules with a predicted gate (``PredictedGateReader``):

* **A**: the full prompt is prefilled with the bank silent while the probe layer's states are
  recorded; the gate is computed; the last prompt position is recomputed with the gated bank over
  the clean cache (overwriting its slot) and decoding runs with the gated bank.
* **C**: as A, plus the residual-path replay of the gated writes into positions ``[0, p-1)`` before
  the recomputation (``attenuation_schedules_schedules.replay_prompt_cache``).
* **B**: one pass; silent below ``layer_from``; at the probe layer (``<= layer_from``) every prompt
  position gets its own gate (``last_prompt``: that position's state; ``prompt_mean``: the causal mean
  up to it) and layers from ``layer_from`` on use it; decode positions use the last prompt position's gate.

``probe-fit`` units fit and save the probe once per scorer recipe; ``predicted`` units load it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from distractor_curves_core import MAX_NEW_TOKENS, bank_representation, compile_documents, score
from write_site_readers_readers import effect_read, generate_with_reader, installed_write_site_reader, signed_strength
from layer_interventions_gates import RANKS_PER_DOCUMENT, STRENGTH_KEY, document_logits, load_probe, membership, probe_batch
from attenuation_schedules_schedules import DESIGN_B_LAYER_FROM, decoder_of, greedy_decode, replay_prompt_cache, stop_ids
from torch import nn

GATE_MODES = ("soft", "hard")
POOLS = ("rank", "document")
RESPONSE_PREFIX = "predicted_"


def gates_from_scores(scores: torch.Tensor, key_type: str, mode: str, pool: str,
                      ranks_per_document: int = RANKS_PER_DOCUMENT) -> torch.Tensor:
    """[..., R] scores -> [..., R] gates in [0, 1]."""
    if mode not in GATE_MODES or pool not in POOLS:
        raise ValueError(f"unknown gate mode {mode!r} or pool {pool!r}")
    if pool == "document":
        scores = document_logits(scores, ranks_per_document).repeat_interleave(ranks_per_document, dim=-1)
    if key_type == STRENGTH_KEY:
        soft = scores / scores.amax(dim=-1, keepdim=True).clamp_min(1e-6)
        return soft if mode == "soft" else (soft >= 0.5).float()
    return torch.sigmoid(scores) if mode == "soft" else (scores > 0).float()


def per_position_scores(probe, key_type: str, layer: int, captured: dict[str, torch.Tensor], query_mode: str,
                        ranks_per_document: int) -> torch.Tensor:
    """[positions, R] scores of every prompt position from its own state (``last_prompt``) or the causal mean
    of the states up to it (``prompt_mean``); batch 1."""
    state, m = captured["state"][0].float(), captured["m"][0].float()
    strength = m.abs()
    if query_mode == "prompt_mean":
        counts = torch.arange(1, state.shape[0] + 1, device=state.device, dtype=state.dtype).unsqueeze(1)
        state, m, strength = state.cumsum(0) / counts, m.cumsum(0) / counts, strength.cumsum(0) / counts
    elif query_mode != "last_prompt":
        raise ValueError(f"predicted gates take last_prompt or prompt_mean, not {query_mode!r}")
    if key_type == STRENGTH_KEY:
        return strength  # as layer_interventions_gates.absolute_strength: |m_r| per position, or its mean over the prefix
    rows = {"state": state.unsqueeze(0), "m": m.unsqueeze(0), "a_rows": captured["a_rows"], "b_rows": captured["b_rows"],
            "latents": captured["latents"]}
    return probe_batch(probe, {layer: rows}, [(0, state.shape[0])], ranks_per_document)[key_type][0]


class PredictedGateReader(nn.Module):
    """Write-site reader: silent and recording during the multi-token prefill; gated at single-token steps
    (and, for schedule B, from ``layer_from`` on during the prefill itself)."""

    execution_site = "write-site"
    arm = "predicted"
    uses_latents = True

    def __init__(self, probe, key_type: str, layer: int, query_mode: str, mode: str, pool: str, schedule: str, *,
                 layer_from: int = DESIGN_B_LAYER_FROM, ranks_per_document: int = RANKS_PER_DOCUMENT) -> None:
        super().__init__()
        if schedule not in ("A", "B", "C"):
            raise ValueError(f"predicted gates run schedules A, B, C, not {schedule!r}")
        if schedule == "B" and layer > layer_from:
            raise ValueError("schedule B needs the probe layer at or below layer_from")
        self.probe, self.key_type, self.layer, self.query_mode = probe, key_type, layer, query_mode
        self.mode, self.pool, self.schedule, self.layer_from, self.rpd = mode, pool, schedule, layer_from, ranks_per_document
        self.records: dict[int, dict[str, torch.Tensor]] = {}
        self.gate: torch.Tensor | None = None  # [R], the gate of the last prompt position
        self.prompt_gates: torch.Tensor | None = None  # [positions, R] (schedule B)
        self.scores: torch.Tensor | None = None

    def encode_bank(self, a_rows, b_rows, layer, latents=None):
        if layer == 0:
            self.records, self.gate, self.prompt_gates, self.scores = {}, None, None, None
        return {"a_rows": a_rows, "b_rows": b_rows, "latents": latents, "b_norms": b_rows.float().norm(dim=-1)}

    def compute(self, state, coefficients, bank, scaling) -> None:
        captured = {"state": state.detach(), "m": signed_strength(coefficients.detach(), bank["b_norms"], scaling),
                    "a_rows": bank["a_rows"], "b_rows": bank["b_rows"], "latents": bank["latents"]}
        with torch.no_grad():
            scores = per_position_scores(self.probe, self.key_type, self.layer, captured, self.query_mode, self.rpd)
        self.prompt_gates = gates_from_scores(scores, self.key_type, self.mode, self.pool, self.rpd)
        self.scores, self.gate = scores[-1], self.prompt_gates[-1]

    def correction(self, state, coefficients, b_rows, bank, layer, scaling, reference=None):
        batch, tokens = coefficients.shape[:2]
        if batch != 1:
            raise ValueError("predicted gates run batch-1 generation only")
        if tokens == 1:
            if self.gate is None:
                raise RuntimeError("the gate is computed during the prompt's prefill")
            return effect_read((self.gate - 1.0).expand(batch, 1, -1), coefficients, b_rows, scaling)
        if layer == self.layer:
            self.compute(state, coefficients, bank, scaling)
        if self.schedule == "B" and layer >= self.layer_from:
            return effect_read((self.prompt_gates - 1.0).unsqueeze(0), coefficients, b_rows, scaling)
        silent = effect_read(torch.full((batch, tokens, coefficients.shape[-1]), -1.0, device=coefficients.device),
                             coefficients, b_rows, scaling)
        if self.schedule == "C":
            self.records[layer] = {"clean": (reference.float() + silent).detach(), "coefficients": coefficients.detach(),
                                   "b_rows": b_rows}
        return silent

    def writes(self, scaling: float) -> dict[int, dict[str, torch.Tensor]]:
        """Schedule C's replay records with the gated writes."""
        return {layer: {"clean": r["clean"], "write": effect_read(self.gate.expand(1, r["coefficients"].shape[1], -1),
                                                                  r["coefficients"], r["b_rows"], scaling)}
                for layer, r in self.records.items()}


def generate_predicted(backend, layers, compiled, reader: PredictedGateReader, question: str, dataset: str,
                       max_new_tokens: int = MAX_NEW_TOKENS) -> tuple[str, dict[str, Any]]:
    """One question under the reader's schedule; returns the text and the gate diagnostics."""
    from official_d2l_online_batch_distillation import render_prompt
    from transformers.cache_utils import HybridCache

    model = backend.model.base_model
    device = model.device
    scaling = model.peft_config["default"].lora_alpha
    if reader.schedule == "B":
        text = generate_with_reader(backend, layers, "predicted", reader, compiled, question, dataset, max_new_tokens)
        return text, gate_record(reader)
    prompt = render_prompt(backend, question, dataset).unsqueeze(0).to(device)
    p = prompt.shape[1]
    residuals: dict[int, torch.Tensor] = {}
    handles = [layer.input_layernorm.register_forward_pre_hook(
        lambda _module, inputs, index=index: residuals.__setitem__(index, inputs[0].detach()))
        for index, layer in enumerate(layers)] if reader.schedule == "C" else []
    decoder = decoder_of(model)
    backend.model.reset()
    try:
        with installed_write_site_reader(backend, layers, compiled, reader, scaling), torch.inference_mode():
            cache = HybridCache(model.config, max_batch_size=1, max_cache_len=p + max_new_tokens, dtype=model.dtype, device=device)
            model(prompt, attention_mask=torch.ones(1, p, dtype=torch.long, device=device), past_key_values=cache,
                  use_cache=True, cache_position=torch.arange(p, device=device))
            for handle in handles:
                handle.remove()
            if reader.schedule == "C" and p > 1:
                cos, sin = decoder.rotary_emb(residuals[0], torch.arange(p - 1, device=device).unsqueeze(0))
                replay_prompt_cache(layers, cache, residuals, reader.writes(scaling), cos, sin, p - 1)
            residuals.clear()
            reader.records.clear()
            generated = greedy_decode(model, prompt[:, -1:], cache, stop_ids(model), max_new_tokens, offset=p - 1)
    finally:
        for handle in handles:
            handle.remove()
    return backend.tokenizer.decode(generated, skip_special_tokens=True).strip(), gate_record(reader)


def gate_record(reader: PredictedGateReader) -> dict[str, Any]:
    gate = reader.gate.detach().float().cpu()
    docs = gate.reshape(-1, reader.rpd).mean(-1)
    return {"gate_mean_per_document": [round(float(v), 4) for v in docs], "scores_last": [round(float(v), 4) for v in reader.scores.cpu()]}


def gate_quality(record: dict[str, Any], members: list[bool]) -> dict[str, float]:
    """Mean gate on the source's ranks and on the distractors' ranks (per document means)."""
    docs = record["gate_mean_per_document"]
    source = [g for g, m in zip(docs, members, strict=True) if m]
    others = [g for g, m in zip(docs, members, strict=True) if not m]
    return {"source_gate": sum(source) / len(source), "distractor_gate": (sum(others) / len(others)) if others else float("nan")}


def predicted_rows(backend, layers, suite: dict[str, Any], source: str, banks: list[dict[str, Any]], probes: dict[str, Any],
                   arms: list[dict[str, Any]], *, ks: tuple[int, ...] = (1, 8), layer_from: int = DESIGN_B_LAYER_FROM,
                   ranks_per_document: int = RANKS_PER_DOCUMENT) -> dict[str, Any]:
    """Each arm = {name, schedule, scorer, key_type, layer, query_mode, mode, pool[, layer_from]}; ``probes[scorer]`` is a loaded probe
    (``None`` for the strength key). DISTRACTOR_CURVES row shape with responses ``predicted_<name>`` and per-row gate records."""
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
            responses, gates = {}, {}
            for arm in arms:
                reader = PredictedGateReader(probes.get(arm["scorer"]), arm["key_type"], int(arm["layer"]), arm["query_mode"],
                                             arm["mode"], arm.get("pool", "rank"), arm["schedule"],
                                             layer_from=int(arm.get("layer_from", layer_from)),
                                             ranks_per_document=ranks_per_document).to(backend.model.device)
                name = f"{RESPONSE_PREFIX}{arm['name']}"
                responses[name], record = generate_predicted(backend, layers, representation, reader, query["input"], query["dataset"])
                gates[name] = {**record, **gate_quality(record, members)}
            rows.append({"query_id": query["id"], "source_id": source, "bank_id": bank["bank_id"], "condition": bank["condition"],
                         "k": bank["k"], "responses": responses, "gates": gates,
                         "scores": {name: score(text, answers[query["id"]]) for name, text in responses.items()}})
    return {"source_id": source, "per_question": [], "rows": rows, "compiled_documents": len(needed),
            "source_tokens": {i: compiled[i]["source_tokens"] for i in needed}}


def probe_path(side: Path, model: str, scorer: str) -> Path:
    return side / "probes" / model / f"{scorer}.pt"


def run_probe_fit(backend, payload: dict[str, Any], latents: dict[str, Any], selection: dict[str, Any], unit: dict[str, Any],
                  path: Path) -> dict[str, Any]:
    """Fit the clean-prefill probe of one scorer recipe and save it for the ``predicted`` units."""
    from ctx_to_lora.utils import get_layers
    from distractor_curves_core import hash_rank
    from layer_interventions_gates import fitted_probe, probe_recipe, save_probe

    layers = get_layers(backend.model.base_model)
    probe, train_plans, log = fitted_probe(backend, layers, payload, latents, selection, unit)
    sha = hash_rank(*(p["query"]["id"] for p in train_plans))
    save_probe(probe, path, probe_recipe(unit), {"train_plans_sha256": sha})
    return {"path": str(path), "train_plans": len(train_plans), "train_plans_sha256": sha, "fit_log_tail": log[-20:],
            "key_types": list(probe.key_types), "query_mode": unit.get("query_mode"), "silent": bool(unit.get("silent", False))}


def run_predicted(backend, suite: dict[str, Any], unit: dict[str, Any], side: Path) -> dict[str, Any]:
    from ctx_to_lora.utils import get_layers

    layers = get_layers(backend.model.base_model)
    device = backend.model.device
    probes: dict[str, Any] = {}
    for arm in unit["arms"]:
        if arm["key_type"] != STRENGTH_KEY and arm["scorer"] not in probes:
            probes[arm["scorer"]], _ = load_probe(probe_path(side, unit["model"], arm["scorer"]), device)
    results = [predicted_rows(backend, layers, suite, source, unit["banks"][source], probes, unit["arms"],
                              ks=tuple(unit.get("ks", (1, 8))), layer_from=int(unit.get("layer_from", DESIGN_B_LAYER_FROM)),
                              ranks_per_document=unit.get("ranks_per_document", RANKS_PER_DOCUMENT))
               for source in unit["sources"]]
    return {"sources": results, "suite": unit["suite"], "arms": unit["arms"],
            "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0}

