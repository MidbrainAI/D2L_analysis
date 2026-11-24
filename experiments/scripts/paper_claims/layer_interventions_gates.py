"""LAYER_INTERVENTIONS: no-training oracles and a membership probe for rank-slot gating under distractors.

The WRITE_SITE_READERS readers never learned to use slot identity: the deep rank reader (C) and the aggregate
control (D) converged to the same K=16 validation CE (1.94) from update 1,500 onward, while
KNOWN_RELEVANCE's known-relevance weights reach 0.78 F1 at K=8 on the same bank. Before training a gated
reader (GATED_READERS), this module measures two things with no learned reader:

* **Layer-prefix oracle** (``oracle_rows``): per-rank gates ``g_r = 1`` for ranks of the source
  document and ``0`` for distractor ranks, applied only from layer ``L`` onward (``layer_from``);
  layers below ``L`` keep the native sum. ``L = 0`` is exactly native K=1 (same ranks, same
  shared bias), the ceiling of document-level gating; ``L = 26`` is native K. The sweep says how
  much of the ceiling survives when early layers stay unfiltered, which is what a per-layer
  reader that cannot identify the source from an early state would face.
* **Membership probe** (``MembershipProbe``): per layer, a bilinear score between the layer-normed
  residual entering the MLP (the state the write-site readers see) and each slot's key, plus the
  slot's signed strength features, trained with BCE against "this slot belongs to the question's
  source document" on reader-training banks and evaluated by AUC on the selection suite. Three
  key types: ``b_rows`` (LN of the slot's B column, arm B's key), ``ab_rows`` (A and B rows, arm
  C's input), ``latent`` (the writer's per-layer pre-head latent of the slot itself: the head
  maps one latent to one rank, so this is the slot's own representation before the head). The
  AUC by layer and key type says whether relevance is recoverable from what the reader sees,
  and from which layer.

Everything here is a diagnostic: the gates use the bank's source label, which no served
reader has. The gated readers of GATED_READERS take their labels only at training time.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import torch
from distractor_curves_core import bank_representation, compile_documents, hash_rank, score
from distractor_curves_reader_training import bank_for, microbatches, representation_batch
from write_site_readers_readers import effect_read, forward_with_reader, generate_with_reader, magnitude_features, signed_strength
from einops import einsum
from torch import nn
from torch.nn import functional

RANKS_PER_DOCUMENT = 8
ORACLE_LAYER_FROM = (0, 4, 8, 13, 18, 22)
ORACLE_K = (4, 8, 16)
PROBE_K = (2, 4, 8, 16)
KEY_TYPES = ("b_rows", "ab_rows", "latent")
LEARNED_KEY_TYPES = KEY_TYPES
STRENGTH_KEY = "abs_strength"  # ATTENUATION_SCHEDULES: parameter-free |m_r| pooled over the query positions (the A-side scorer)
PROBE_DIM = 64
PROBE_LEARNING_RATE = 1e-3


# -- membership and gates (CPU) ----------------------------------------------------------
def membership(bank_document_ids: list[str], source_id: str) -> list[bool]:
    members = [i == source_id for i in bank_document_ids]
    if not any(members):
        raise ValueError(f"source {source_id} is not in the bank")
    return members


def rank_mask(members: list[bool], ranks_per_document: int = RANKS_PER_DOCUMENT) -> torch.Tensor:
    """Per-rank membership [R] in bank order: document ``j`` owns ranks ``j*rpd .. (j+1)*rpd-1``."""
    return torch.tensor(members, dtype=torch.float32).repeat_interleave(ranks_per_document)


def oracle_gates(members: list[bool], layers: int, layer_from: int,
                 ranks_per_document: int = RANKS_PER_DOCUMENT) -> torch.Tensor:
    """[layers, R] gates: native (ones) below ``layer_from``, source-only membership from it on."""
    if not 0 <= layer_from <= layers:
        raise ValueError("layer_from must be within 0..layers")
    mask = rank_mask(members, ranks_per_document)
    gates = torch.ones(layers, mask.numel())
    gates[layer_from:] = mask
    return gates


class FixedGateReader(nn.Module):
    """A parameter-free write-site 'reader' applying fixed per-layer, per-rank gates."""

    execution_site = "write-site"
    arm = "oracle"
    uses_latents = False

    def __init__(self, gates: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("gates", gates.float())

    def encode_bank(self, a_rows, b_rows, layer):
        return {"b_norms": b_rows.float().norm(dim=-1)}

    def correction(self, state, coefficients, b_rows, bank, layer, scaling, reference=None):
        if coefficients.shape[-1] != self.gates.shape[1]:
            raise ValueError("gate width does not match the bank's rank count")
        weights = (self.gates[layer].to(coefficients.device) - 1.0).expand(*coefficients.shape[:2], -1)
        return effect_read(weights, coefficients, b_rows, scaling)


class ResponseOnlyGateReader(nn.Module):
    """RESPONSE_ONLY_ORACLE: the source-only mask on the generated positions only. Prefill (``tokens > 1``) runs native K
    at every layer, except that ``include_last_prompt`` also masks the last prompt position, whose logits
    produce the first response token. Decode calls (``tokens == 1``) are masked at every layer. Batch-1
    greedy generation only (``generate_with_reader``)."""

    execution_site = "write-site"
    arm = "oracle"
    uses_latents = False

    def __init__(self, mask: torch.Tensor, include_last_prompt: bool) -> None:
        super().__init__()
        self.register_buffer("mask", mask.float())
        self.include_last_prompt = include_last_prompt

    def encode_bank(self, a_rows, b_rows, layer):
        return {"b_norms": b_rows.float().norm(dim=-1)}

    def correction(self, state, coefficients, b_rows, bank, layer, scaling, reference=None):
        if coefficients.shape[-1] != self.mask.numel():
            raise ValueError("mask width does not match the bank's rank count")
        batch, tokens = coefficients.shape[:2]
        weights = torch.zeros(batch, tokens, self.mask.numel(), device=coefficients.device)
        if tokens == 1:
            weights[:] = self.mask.to(coefficients.device) - 1.0
        elif self.include_last_prompt:
            weights[:, -1] = self.mask.to(coefficients.device) - 1.0
        return effect_read(weights, coefficients, b_rows, scaling)


POSITION_MODES = ("decode", "decode_last")


def position_mode_reader(members: list[bool], mode: str, ranks_per_document: int = RANKS_PER_DOCUMENT) -> ResponseOnlyGateReader:
    if mode not in POSITION_MODES:
        raise ValueError(f"unknown position mode {mode!r}")
    return ResponseOnlyGateReader(rank_mask(members, ranks_per_document), include_last_prompt=(mode == "decode_last"))


class StateCollector(nn.Module):
    """Zero correction that records, per layer, what a write-site reader would see."""

    execution_site = "write-site"
    arm = "collector"
    uses_latents = True

    def __init__(self, silent: bool = False) -> None:
        """``silent`` (ATTENUATION_SCHEDULES): cancel the native sum at every position and layer, so the recorded state is the
        bank-free (clean-prefill) state while the coefficients and strengths are still those of the bank's ranks."""
        super().__init__()
        self.silent = silent
        self.layers: dict[int, dict[str, torch.Tensor]] = {}

    def encode_bank(self, a_rows, b_rows, layer, latents=None):
        if layer == 0:
            self.layers = {}
        return {"a_rows": a_rows, "b_rows": b_rows, "latents": latents, "b_norms": b_rows.float().norm(dim=-1)}

    def correction(self, state, coefficients, b_rows, bank, layer, scaling, reference=None):
        self.layers[layer] = {"state": state.detach(), "coefficients": coefficients.detach(),
                              "m": signed_strength(coefficients.detach(), bank["b_norms"], scaling),
                              "a_rows": bank["a_rows"], "b_rows": bank["b_rows"], "latents": bank["latents"]}
        if self.silent:
            return effect_read(torch.full(coefficients.shape, -1.0, device=coefficients.device), coefficients, b_rows, scaling)
        return torch.zeros_like(state)


# -- oracle evaluation (GPU) -------------------------------------------------------------
def oracle_rows(backend, layers, suite: dict[str, Any], source: str, banks: list[dict[str, Any]], *,
                layer_from_values: tuple[int, ...] = ORACLE_LAYER_FROM, ks: tuple[int, ...] = ORACLE_K,
                ranks_per_document: int = RANKS_PER_DOCUMENT, position_modes: tuple[str, ...] = ()) -> dict[str, Any]:
    """Greedy responses of the layer-prefix oracles on every question of one source; DISTRACTOR_CURVES row shape."""
    documents = suite["documents"]
    chosen = [b for b in banks if b["k"] in ks and b["condition"] == "random"]
    needed = sorted({i for bank in chosen for i in bank["document_ids"]})
    compiled = dict(zip(needed, compile_documents(backend, [documents[i]["text"] for i in needed]), strict=True))
    questions = suite["questions"][source]
    answers = {q["id"]: q["answers"] for q in questions}
    rows = []
    for bank in chosen:
        records = [compiled[i] for i in bank["document_ids"]]
        representation = bank_representation(backend, records)
        members = membership(bank["document_ids"], source)
        readers = {f"oracle_from{L}": FixedGateReader(oracle_gates(members, len(layers), L, ranks_per_document))
                   .to(backend.model.device) for L in layer_from_values}
        readers.update({f"oracle_{mode}": position_mode_reader(members, mode, ranks_per_document).to(backend.model.device)
                        for mode in position_modes})
        for query in questions:
            responses = {name: generate_with_reader(backend, layers, "oracle", reader, representation, query["input"],
                                                    query["dataset"]) for name, reader in readers.items()}
            rows.append({"query_id": query["id"], "source_id": source, "bank_id": bank["bank_id"],
                         "condition": bank["condition"], "k": bank["k"], "responses": responses,
                         "scores": {name: score(text, answers[query["id"]]) for name, text in responses.items()}})
    return {"source_id": source, "per_question": [], "rows": rows, "compiled_documents": len(needed),
            "source_tokens": {i: compiled[i]["source_tokens"] for i in needed}}


# -- membership probe --------------------------------------------------------------------
def per_rank_latents(latents: torch.Tensor, ranks_per_document: int, latent_width: int) -> torch.Tensor:
    """[batch, K, ..., latent] -> [batch, K * rpd, latent]: the writer emits one pre-head latent per rank
    (Gemma-2 D2L: 8 x 512 per layer and document); a single per-document latent is repeated per rank."""
    flat = latents.float().reshape(latents.shape[0], latents.shape[1], -1, latent_width)
    if flat.shape[2] == 1:
        flat = flat.expand(-1, -1, ranks_per_document, -1)
    elif flat.shape[2] != ranks_per_document:
        raise ValueError(f"{flat.shape[2]} latents per document, expected 1 or {ranks_per_document}")
    return flat.reshape(flat.shape[0], -1, latent_width)


class MembershipProbe(nn.Module):
    """Per-layer bilinear slot scorers, one parameter set per key type, trained jointly."""

    def __init__(self, width: int, input_width: int, latent_width: int, layers: int, dim: int = PROBE_DIM,
                 key_types: tuple[str, ...] = KEY_TYPES) -> None:
        super().__init__()
        self.width, self.input_width, self.latent_width, self.layers, self.dim = width, input_width, latent_width, layers, dim
        self.key_types = tuple(key_types)
        unknown = set(self.key_types) - set(LEARNED_KEY_TYPES) - {STRENGTH_KEY}
        if unknown:
            raise ValueError(f"unknown key types {sorted(unknown)}")
        learned = [t for t in self.key_types if t in LEARNED_KEY_TYPES]
        scale = 1.0 / math.sqrt(width)
        self.query = nn.ParameterDict({t: nn.Parameter(torch.randn(layers, width, dim) * scale) for t in learned})
        self.key_b = nn.ParameterDict({t: nn.Parameter(torch.randn(layers, width, dim) * scale)
                                       for t in learned if t in ("b_rows", "ab_rows")})
        self.key_a = nn.ParameterDict({t: nn.Parameter(torch.randn(layers, input_width, dim) / math.sqrt(input_width))
                                       for t in learned if t == "ab_rows"})
        self.key_latent = nn.ParameterDict({t: nn.Parameter(torch.randn(layers, latent_width, dim) / math.sqrt(latent_width))
                                            for t in learned if t == "latent"})
        self.magnitude = nn.ParameterDict({t: nn.Parameter(torch.zeros(layers, 2)) for t in learned})
        self.bias = nn.ParameterDict({t: nn.Parameter(torch.zeros(layers)) for t in learned})

    @property
    def learned_key_types(self) -> tuple[str, ...]:
        return tuple(self.query.keys())

    def keys(self, key_type: str, layer: int, a_rows: torch.Tensor, b_rows: torch.Tensor,
             latents: torch.Tensor | None, ranks_per_document: int) -> torch.Tensor:
        """[batch, R, dim] token-independent slot keys of one layer."""
        if key_type in ("b_rows", "ab_rows"):
            keys = functional.layer_norm(b_rows.float(), (self.width,)) @ self.key_b[key_type][layer]
            if key_type == "ab_rows":
                keys = keys + functional.layer_norm(a_rows.float(), (self.input_width,)) @ self.key_a[key_type][layer]
            return keys
        if latents is None:
            raise ValueError("latent keys need the bank's document latents")
        return functional.layer_norm(per_rank_latents(latents, ranks_per_document, self.latent_width),
                                     (self.latent_width,)) @ self.key_latent[key_type][layer]

    def logits(self, key_type: str, layer: int, state: torch.Tensor, m: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """[batch, n, R] slot logits from states [batch, n, width], strengths [batch, n, R], keys [batch, R, dim]."""
        query = functional.layer_norm(state.float(), (self.width,)) @ self.query[key_type][layer]
        scores = einsum(query, keys, "b n d, b r d -> b n r") / math.sqrt(self.dim)
        magnitude = einsum(magnitude_features(m.float()), self.magnitude[key_type][layer], "b n r f, f -> b n r")
        return scores + magnitude + self.bias[key_type][layer]


def response_positions(prompt_lengths: list[int], response_lengths: list[int]) -> list[tuple[int, int]]:
    """(start, count) per row: the positions whose logits predict the response tokens."""
    return [(p - 1, r) for p, r in zip(prompt_lengths, response_lengths, strict=True)]


QUERY_MODES = ("response", "last_prompt", "prompt_mean")


def query_positions(mode: str, prompt_lengths: list[int], response_lengths: list[int]) -> tuple[list[tuple[int, int]], bool]:
    """Which positions feed the probe's query: the response positions (LAYER_INTERVENTIONS), the last prompt token
    alone, or the mean over all prompt positions (RESPONSE_ONLY_ORACLE; right-padded batches). Returns (positions, pooled)."""
    if mode == "response":
        return response_positions(prompt_lengths, response_lengths), False
    if mode == "last_prompt":
        return [(p - 1, 1) for p in prompt_lengths], False
    if mode == "prompt_mean":
        return [(0, p) for p in prompt_lengths], True
    raise ValueError(f"unknown query mode {mode!r}")


def document_logits(slot_logits: torch.Tensor, ranks_per_document: int = RANKS_PER_DOCUMENT) -> torch.Tensor:
    """Mean of each document's rank logits: [..., R] -> [..., K]."""
    return slot_logits.reshape(*slot_logits.shape[:-1], -1, ranks_per_document).mean(-1)


def auc(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    """Mann-Whitney AUC of 1-D scores against boolean labels (ties count half); None without both
    classes. Rank-based, O(n log n): the first version compared every positive with every negative
    and did not finish on the LAYER_INTERVENTIONS run (156 slot-level cells of up to 1e11 pairs on 4 CPU threads)."""
    scores, labels = scores.detach().double().cpu().flatten(), labels.detach().bool().cpu().flatten()
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = torch.argsort(scores)
    sorted_scores = scores[order]
    # average ranks (1-based) within tie groups
    _, inverse, counts = torch.unique_consecutive(sorted_scores, return_inverse=True, return_counts=True)
    ends = torch.cumsum(counts, 0).double()
    starts = ends - counts.double() + 1.0
    ranks = ((starts + ends) / 2.0)[inverse]
    rank_sum_pos = float(ranks[labels[order]].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def probe_plans(rows: list[dict[str, Any]], pool: list[str], count: int, seed: int,
                ks: tuple[int, ...] = PROBE_K) -> list[dict[str, Any]]:
    """``count`` K-balanced training plans over ``rows`` (one bank per question, DISTRACTOR_CURVES's sampler)."""
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    plans = []
    for index, row_index in enumerate(order[:count]):
        row, k = rows[row_index], ks[index % len(ks)]
        plans.append({"query": row["query"], "k": k, "bank_document_ids": bank_for(row, k, pool, f"probe-{seed}")})
    plans.sort(key=lambda p: p["k"])
    return plans


def suite_plans(suite: dict[str, Any], banks: dict[str, list[dict[str, Any]]], ks: tuple[int, ...]) -> list[dict[str, Any]]:
    """Every (question, random bank) of the suite at the chosen K, in the plan shape, sorted by K."""
    plans = []
    for source, group in sorted(banks.items()):
        for bank in group:
            if bank["condition"] != "random" or bank["k"] not in ks:
                continue
            for query in suite["questions"][source]:
                plans.append({"query": query, "k": bank["k"], "bank_document_ids": list(bank["document_ids"]),
                              "bank_id": bank["bank_id"]})
    plans.sort(key=lambda p: p["k"])
    return plans


def collect(backend, layers, collector: StateCollector, plans: list[dict[str, Any]], latents: dict[str, Any],
            scaling: float) -> tuple[dict[int, dict[str, torch.Tensor]], dict[str, Any], torch.Tensor]:
    """Teacher-forced native forward of K-homogeneous plans; returns per-layer captures, targets, labels [batch, R]."""
    from write_site_readers_reader_training import make_answer_batch

    targets = make_answer_batch(backend, plans)
    compiled = representation_batch(backend, plans, latents)
    with torch.no_grad():
        forward_with_reader(backend, layers, "collector", collector, compiled, targets, scaling)
    labels = torch.stack([rank_mask(membership(p["bank_document_ids"], p["query"]["document_id"])) for p in plans])
    return collector.layers, targets, labels.to(backend.model.device)


def gathered(captured: dict[str, torch.Tensor], positions: list[tuple[int, int]],
             pooled: bool = False) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Stack the selected states and strengths of every row: [n_total, ...] plus row sizes. With
    ``pooled`` each row's slice is averaged into one query position (size 1)."""
    states, strengths, sizes = [], [], []
    for row, (start, count) in enumerate(positions):
        state, strength = captured["state"][row, start:start + count], captured["m"][row, start:start + count]
        if pooled:
            state, strength = state.float().mean(0, keepdim=True), strength.float().mean(0, keepdim=True)
            count = 1
        states.append(state)
        strengths.append(strength)
        sizes.append(count)
    return torch.cat(states), torch.cat(strengths), sizes


def absolute_strength(captured: dict[str, torch.Tensor], positions: list[tuple[int, int]], pooled: bool = False) -> torch.Tensor:
    """[n_total, R] of |m_r| at the query positions (mean of |m_r| over the slice when pooled): the parameter-free
    A-side scorer, "does the rank's A row fire on the question's tokens"."""
    out = []
    for row, (start, count) in enumerate(positions):
        strength = captured["m"][row, start:start + count].float().abs()
        out.append(strength.mean(0, keepdim=True) if pooled else strength)
    return torch.cat(out)


def probe_batch(probe: MembershipProbe, captures: dict[int, dict[str, torch.Tensor]], positions: list[tuple[int, int]],
                ranks_per_document: int, pooled: bool = False) -> dict[str, torch.Tensor]:
    """Slot logits per key type: [layers, n_total, R] over the query positions of every row."""
    out: dict[str, list[torch.Tensor]] = {t: [] for t in probe.key_types}
    for layer in sorted(captures):
        captured = captures[layer]
        states, strengths, sizes = gathered(captured, positions, pooled)
        if STRENGTH_KEY in probe.key_types:
            out[STRENGTH_KEY].append(absolute_strength(captured, positions, pooled))
        for key_type in probe.learned_key_types:
            keys = probe.keys(key_type, layer, captured["a_rows"], captured["b_rows"], captured["latents"],
                              ranks_per_document)
            keys = torch.cat([keys[row].expand(size, -1, -1) for row, size in enumerate(sizes)])  # [n_total, R, dim]
            query = functional.layer_norm(states.float(), (probe.width,)) @ probe.query[key_type][layer]
            scores = einsum(query, keys, "n d, n r d -> n r") / math.sqrt(probe.dim)
            magnitude = einsum(magnitude_features(strengths.float()), probe.magnitude[key_type][layer], "n r f, f -> n r")
            out[key_type].append(scores + magnitude + probe.bias[key_type][layer])
    return {t: torch.stack(v) for t, v in out.items()}


def expand_labels(labels: torch.Tensor, sizes: list[int]) -> torch.Tensor:
    return torch.cat([labels[row].expand(size, -1) for row, size in enumerate(sizes)])


def fit_probe(backend, layers, probe: MembershipProbe, plans: list[dict[str, Any]], latents: dict[str, Any],
              scaling: float, max_document_occurrences: int, ranks_per_document: int = RANKS_PER_DOCUMENT,
              lr: float = PROBE_LEARNING_RATE, query_mode: str = "response", silent: bool = False) -> list[dict[str, Any]]:
    """One pass of AdamW over K-homogeneous microbatches; returns the per-batch BCE log."""
    collector = StateCollector(silent=silent)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.0)
    log = []
    for k in sorted({p["k"] for p in plans}):
        for batch in microbatches([p for p in plans if p["k"] == k], max_document_occurrences):
            captures, targets, labels = collect(backend, layers, collector, batch, latents, scaling)
            positions, pooled = query_positions(query_mode, targets["student_prompt_lengths"], targets["response_lengths"])
            logits = probe_batch(probe, captures, positions, ranks_per_document, pooled)
            expanded = expand_labels(labels, [1 for _ in positions] if pooled else [count for _, count in positions])
            losses = {t: functional.binary_cross_entropy_with_logits(logits[t], expanded.expand_as(logits[t]))
                      for t in probe.learned_key_types}
            if losses:  # a probe with only the parameter-free strength scorer has nothing to fit
                optimizer.zero_grad(set_to_none=True)
                sum(losses.values()).backward()
                optimizer.step()
            log.append({"k": k, "rows": len(batch), **{t: float(v) for t, v in losses.items()}})
    return log


def evaluate_probe(backend, layers, probe: MembershipProbe, plans: list[dict[str, Any]], latents: dict[str, Any],
                   scaling: float, max_document_occurrences: int,
                   ranks_per_document: int = RANKS_PER_DOCUMENT, query_mode: str = "response",
                   silent: bool = False) -> dict[str, Any]:
    """AUC by key type, layer, and K: per token at document level, per question (position mean), and per slot."""
    collector = StateCollector(silent=silent)
    store: dict[tuple[str, int, int, str], list[torch.Tensor]] = {}
    label_store: dict[tuple[str, int, int, str], list[torch.Tensor]] = {}
    bce: dict[tuple[str, int, int], list[float]] = {}

    def add(key, scores, labels):
        store.setdefault(key, []).append(scores.detach().cpu())
        label_store.setdefault(key, []).append(labels.detach().cpu())

    with torch.no_grad():
        for k in sorted({p["k"] for p in plans}):
            for batch in microbatches([p for p in plans if p["k"] == k], max_document_occurrences):
                captures, targets, labels = collect(backend, layers, collector, batch, latents, scaling)
                positions, pooled = query_positions(query_mode, targets["student_prompt_lengths"], targets["response_lengths"])
                sizes = [1 for _ in positions] if pooled else [count for _, count in positions]
                logits = probe_batch(probe, captures, positions, ranks_per_document, pooled)
                expanded = expand_labels(labels, sizes)  # [n_total, R]
                doc_labels = document_logits(labels, ranks_per_document) > 0.5  # [rows, K]
                for key_type, slot_logits in logits.items():
                    docs = document_logits(slot_logits, ranks_per_document)  # [layers, n_total, K]
                    for layer in range(slot_logits.shape[0]):
                        if key_type in probe.learned_key_types:
                            bce.setdefault((key_type, layer, k), []).append(float(
                                functional.binary_cross_entropy_with_logits(slot_logits[layer], expanded)))
                        add((key_type, layer, k, "slot"), slot_logits[layer].reshape(-1), expanded.reshape(-1))
                        add((key_type, layer, k, "token"), docs[layer].reshape(-1),
                            document_logits(expanded, ranks_per_document).reshape(-1) > 0.5)
                        pooled = torch.stack([chunk.mean(0) for chunk in docs[layer].split(sizes)])  # [rows, K]
                        add((key_type, layer, k, "question"), pooled.reshape(-1), doc_labels.reshape(-1))
    result: dict[str, Any] = {}
    for (key_type, layer, k, level), chunks in store.items():
        result.setdefault(key_type, {}).setdefault(str(layer), {}).setdefault(str(k), {})[level] = auc(
            torch.cat(chunks), torch.cat(label_store[(key_type, layer, k, level)]))
    for (key_type, layer, k), values in bce.items():
        result[key_type][str(layer)][str(k)]["bce"] = sum(values) / len(values)
    return result


def summarize_probe(result: dict[str, Any], level: str = "token") -> dict[str, Any]:
    """Best layer and its AUC per key type and K, plus the AUC at every layer pooled over K."""
    summary: dict[str, Any] = {}
    for key_type, by_layer in result.items():
        ks = sorted({k for layers in by_layer.values() for k in layers}, key=int)
        per_k = {}
        for k in ks:
            values = {layer: cells[k][level] for layer, cells in by_layer.items() if k in cells and cells[k][level] is not None}
            if values:
                best = max(values, key=values.get)
                per_k[k] = {"best_layer": int(best), "best_auc": values[best],
                            "by_layer": [values.get(str(layer)) for layer in range(len(by_layer))]}
        summary[key_type] = per_k
    return summary


# -- unit entries (GPU) ------------------------------------------------------------------
def run_oracle(backend, suite: dict[str, Any], unit: dict[str, Any]) -> dict[str, Any]:
    from ctx_to_lora.utils import get_layers

    layers = get_layers(backend.model.base_model)
    results = [oracle_rows(backend, layers, suite, source, unit["banks"][source],
                           layer_from_values=tuple(unit.get("layer_from", ORACLE_LAYER_FROM)),
                           ks=tuple(unit.get("ks", ORACLE_K)), ranks_per_document=unit.get("ranks_per_document", RANKS_PER_DOCUMENT),
                           position_modes=tuple(unit.get("position_modes", ())))
               for source in unit["sources"]]
    return {"sources": results, "suite": unit["suite"], "layer_from": list(unit.get("layer_from", ORACLE_LAYER_FROM)),
            "position_modes": list(unit.get("position_modes", ())),
            "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0}


def new_probe(backend, layers, unit: dict[str, Any]) -> MembershipProbe:
    model = backend.model.base_model
    torch.manual_seed(unit.get("seed", 217))
    return MembershipProbe(model.config.hidden_size, model.config.intermediate_size,
                           backend.model.hypernet.config.latent_size, len(layers), dim=unit.get("dim", PROBE_DIM),
                           key_types=tuple(unit.get("key_types", KEY_TYPES))).to(model.device)


def fitted_probe(backend, layers, payload: dict[str, Any], latents: dict[str, Any], selection: dict[str, Any],
                 unit: dict[str, Any]) -> tuple[MembershipProbe, list[dict[str, Any]], list[dict[str, Any]]]:
    """A probe fit on reader-training banks (held-out documents excluded) per the unit's recipe:
    (probe, training plans, fit log)."""
    from write_site_readers_reader_training import filtered_rows, held_out_hashes, pools

    model = backend.model.base_model
    scaling = model.peft_config["default"].lora_alpha
    held = held_out_hashes(selection)
    documents = {d["id"]: d for d in payload["documents"]}
    rows = [r for r in filtered_rows(payload, held)["train"] if r["document"]["id"] in latents]
    pool = pools(latents, documents, held)["train"]
    train_plans = probe_plans(rows, pool, unit.get("train_plans", 1500), unit.get("seed", 217),
                              tuple(unit.get("train_ks", PROBE_K)))
    probe = new_probe(backend, layers, unit)
    log = fit_probe(backend, layers, probe, train_plans, latents, scaling, unit.get("max_document_occurrences", 12),
                    query_mode=unit.get("query_mode", "response"), silent=bool(unit.get("silent", False)))
    return probe, train_plans, log


def probe_recipe(unit: dict[str, Any]) -> dict[str, Any]:
    return {k: unit[k] for k in ("seed", "dim", "key_types", "query_mode", "silent", "train_plans", "train_ks",
                                 "max_document_occurrences") if k in unit}


def save_probe(probe: MembershipProbe, path, recipe: dict[str, Any], extra: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": probe.state_dict(), "recipe": recipe, "shape": {"width": probe.width, "input_width": probe.input_width,
               "latent_width": probe.latent_width, "layers": probe.layers, "dim": probe.dim, "key_types": list(probe.key_types)},
               **(extra or {})}, path)


def load_probe(path, device) -> tuple[MembershipProbe, dict[str, Any]]:
    saved = torch.load(Path(path), map_location=device, weights_only=False)
    shape = saved["shape"]
    probe = MembershipProbe(shape["width"], shape["input_width"], shape["latent_width"], shape["layers"], dim=shape["dim"],
                            key_types=tuple(shape["key_types"]))
    probe.load_state_dict(saved["state"])
    return probe.to(device).eval(), saved


def run_probe(backend, payload: dict[str, Any], latents: dict[str, Any], selection: dict[str, Any],
              suite: dict[str, Any], banks: dict[str, list[dict[str, Any]]], unit: dict[str, Any]) -> dict[str, Any]:
    """Fit the probe on reader-training banks (held-out documents excluded), evaluate on the suite's banks."""
    from ctx_to_lora.utils import get_layers

    model = backend.model.base_model
    layers = get_layers(model)
    scaling = model.peft_config["default"].lora_alpha
    test_plans = suite_plans(suite, banks, tuple(unit.get("test_ks", ORACLE_K)))
    if unit.get("test_limit"):
        test_plans = test_plans[: unit["test_limit"]]
    needed = sorted({i for p in test_plans for i in p["bank_document_ids"]})
    records = compile_documents(backend, [suite["documents"][i]["text"] for i in needed])
    suite_latents = {i: r["latent"][0].cpu() for i, r in zip(needed, records, strict=True)}
    probe, train_plans, log = fitted_probe(backend, layers, payload, latents, selection, unit)
    occurrences = unit.get("max_document_occurrences", 12)
    query_mode = unit.get("query_mode", "response")
    silent = bool(unit.get("silent", False))
    if unit.get("save_path"):
        save_probe(probe, unit["save_path"], probe_recipe(unit), {"train_plans_sha256": hash_rank(*(p["query"]["id"] for p in train_plans))})
    result = evaluate_probe(backend, layers, probe, test_plans, suite_latents, scaling, occurrences, query_mode=query_mode,
                            silent=silent)
    return {"auc": result, "summary": {level: summarize_probe(result, level) for level in ("token", "question", "slot")},
            "query_mode": query_mode, "silent": silent,
            "fit_log_tail": log[-20:], "fit_bce_last_quarter": {t: sum(e[t] for e in log[-len(log) // 4:]) / max(1, len(log) // 4)
                                                                 for t in probe.learned_key_types} if log else {},
            "train_plans": len(train_plans), "test_plans": len(test_plans), "test_documents": len(needed),
            "train_plans_sha256": hash_rank(*(p["query"]["id"] for p in train_plans)),
            "key_types": list(probe.key_types), "dim": probe.dim, "layers": len(layers),
            "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0}
