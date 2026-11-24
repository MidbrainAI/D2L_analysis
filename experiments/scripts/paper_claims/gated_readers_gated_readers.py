"""GATED_READERS: WRITE_SITE_READERS's reader bodies with a per-rank gate output and a supervised document term.

WRITE_SITE_READERS's write-site readers emit an additive correction vector; their attention over slots is
never told which slots matter and (in the write-site reader selection) arms C and D converged to the same slot-agnostic
shrinkage. A gated reader keeps the body (arm B's single attention read or arm C's factored
deep read) and changes only what the read produces and what supervises it:

    logit_r  = q(h) . key_r / sqrt(d) + w . magnitude(m_r) + b + p     (per layer, per token; p = 4)
    ell_doc  = mean of logit_r over the 8 ranks of each document         [K], the membership logit
    ell_rank = logit_r - ell_doc(d(r))                                   [R], zero-mean per document
    g_r      = sigmoid(logit_r) / sigmoid(p)                             1 at the prior, 0 for -inf
    delta    = scaling * sum_r g_r c_r B_r   (= native + scaling * sum_r (g_r - 1) c_r B_r)

The prior ``p`` makes every document "applicable" at initialization: the membership
probability starts at sigmoid(4) = 0.98 and every gate at exactly 1, so the forward is exactly
native D2L. The document BCE then drives distractor logits negative (gates toward 0) and leaves
members near the prior (gates near 1, at most 1/sigmoid(4) = 1.018), so a confident
membership decision cannot amplify the source; an earlier softplus form tied the gate's
magnitude to the classifier's confidence. Nothing selects a document: every rank keeps its own
gate, several documents can stay open, and uniform shrinkage (normalization) is expressible.

Loss (``gated_loss``): WRITE_SITE_READERS's answer-token CE on the gated forward plus ``lam`` times the BCE of
``sigmoid(ell_doc)`` against "document d is the question's source", averaged over documents,
layers, and the response positions. The rank residual has no labels and learns only from the
CE. ``lam = 0`` is the gate-contract control.

Arm ``BS`` is the loss-only control: WRITE_SITE_READERS's arm B unchanged (additive output contract, softmax
attention over slots), with the document term applied to the attention mass its softmax
puts on each document (``logit(mass_d)`` plays the role of ``ell_doc``). It separates the gate
contract from the loss: if BS becomes immune, the contract was not needed.

Arms: ``GB`` (arm B body), ``GC`` (arm C body), ``GBL`` (arm B body whose gate keys also carry
the writer's per-layer pre-head latent of the slot: the D2L head maps one 512-wide latent to
one rank, so each rank has its own; ``uses_latents``). Checkpoints
load through ``load_gated_reader`` and WRITE_SITE_READERS's installer runs them unchanged (``correction``).
"""

from __future__ import annotations

import math
from typing import Any

import torch
from write_site_readers_readers import (
    FactoredDeepRankReader,
    WriteSiteShallowReader,
    effect_read,
    forward_with_reader,
    magnitude_features,
    make_reader,
    per_token_flops,
    signed_strength,
)
from einops import einsum
from torch import nn
from torch.nn import functional

GATED_ARMS = {"GB": ("B", False), "GC": ("C", False), "GBL": ("B", True), "BS": ("B", False)}
MASS_EPS = 1e-6
RANKS_PER_DOCUMENT = 8
GATE_WIDTH = 32
GATE_PRIOR = 4.0  # membership logit at initialization: sigmoid(4) = 0.98, every document applicable
RECIPE = "gated_readers-gated-ce-plus-document-bce-v1"


class GatedReader(nn.Module):
    """A write-site reader whose output is a gate per rank slot (see module docstring)."""

    execution_site = "write-site"

    def __init__(self, body: nn.Module, arm: str, *, use_latents: bool, latent_width: int | None = None,
                 gate_width: int = GATE_WIDTH, ranks_per_document: int = RANKS_PER_DOCUMENT) -> None:
        super().__init__()
        if arm not in GATED_ARMS:
            raise ValueError(f"unknown gated arm {arm}")
        self.body = body
        self.arm = arm
        self.gate_width = gate_width
        self.ranks_per_document = ranks_per_document
        self.uses_latents = use_latents
        self.latent_width = latent_width
        key_dim = body.bottleneck if isinstance(body, WriteSiteShallowReader) else body.hidden
        self.gate_key = nn.Linear(key_dim, gate_width, bias=False)
        self.gate_query = nn.Linear(body.hidden * body.reads, gate_width, bias=False)
        self.gate_magnitude = nn.Linear(2, gate_width, bias=False)
        self.gate_bias = nn.Parameter(torch.zeros(1))
        if use_latents:
            if latent_width is None:
                raise ValueError("latent keys need the latent width")
            self.latent_key = nn.Linear(latent_width, gate_width, bias=False)
        nn.init.zeros_(self.gate_query.weight)  # logits start at the prior: every gate exactly 1, native D2L
        self.doc_logits: list[torch.Tensor] = []
        self.last: dict[str, float] = {}

    # -- bank and read -----------------------------------------------------------------
    def encode_bank(self, a_rows, b_rows, layer, latents=None):
        bank = self.body.encode_bank(a_rows, b_rows, layer)
        keys = self.gate_key(bank["keys"] if "keys" in bank else bank["slots"])  # [batch, R, g]
        if self.uses_latents:
            if latents is None:
                raise ValueError("this arm keys on document latents")
            from layer_interventions_gates import per_rank_latents

            flat = per_rank_latents(latents, self.ranks_per_document, self.latent_width)  # [batch, R, latent]
            keys = keys + self.latent_key(functional.layer_norm(flat, (self.latent_width,)))
        bank["gate_keys"] = keys
        if layer == 0:
            self.doc_logits = []
        return bank

    def slot_logits(self, h: torch.Tensor, m: torch.Tensor, bank: dict[str, torch.Tensor]) -> torch.Tensor:
        keys = bank["gate_keys"].unsqueeze(1) + self.gate_magnitude(magnitude_features(m))  # [b, s, R, g]
        query = self.gate_query(h)
        return einsum(query, keys, "b s g, b s r g -> b s r") / math.sqrt(self.gate_width) + self.gate_bias

    def gates(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(gate [b, s, R], membership logits [b, s, K]) from slot logits (prior included in both)."""
        prior = torch.full_like(logits, GATE_PRIOR)
        shifted = logits + prior
        doc = shifted.reshape(*logits.shape[:-1], -1, self.ranks_per_document).mean(-1)
        gate = torch.sigmoid(shifted) / torch.sigmoid(prior)
        return gate, doc

    def correction(self, state, coefficients, b_rows, bank, layer, scaling, reference=None):
        if coefficients.shape[-1] % self.ranks_per_document:
            raise ValueError("the rank count is not a multiple of the ranks per document")
        m = signed_strength(coefficients, bank["b_norms"], scaling)
        h, _ = self.body.read(state, m, bank, layer)
        h = self.body.read_norm(h)
        gate, doc = self.gates(self.slot_logits(h, m, bank))
        self.doc_logits.append(doc)
        return effect_read(gate - 1.0, coefficients, b_rows, scaling)

    # -- bookkeeping -------------------------------------------------------------------
    def diagnostics(self) -> dict[str, float]:
        return dict(self.last)

    def payload_extras(self) -> dict[str, Any]:
        body = self.body
        extras: dict[str, Any] = {"gate_width": self.gate_width, "use_latents": self.uses_latents,
                                  "latent_width": self.latent_width, "ranks_per_document": self.ranks_per_document,
                                  "body_arm": GATED_ARMS[self.arm][0]}
        if isinstance(body, WriteSiteShallowReader):
            extras["transition_bottleneck"] = body.bottleneck
        else:
            extras.update({"hidden": body.hidden, "reads": body.reads, "bank_blocks": len(body.bank_blocks),
                           "read_blocks": len(body.cross_attention), "heads": body.heads})
        return extras


class AttentionSupervisedShallowReader(WriteSiteShallowReader):
    """Arm BS: WRITE_SITE_READERS's arm B with its attention mass per document exposed as a membership logit."""

    arm = "BS"
    uses_latents = False

    def __init__(self, width: int, layers: int, bottleneck: int = 32, ranks_per_document: int = RANKS_PER_DOCUMENT) -> None:
        super().__init__(width, layers, bottleneck)
        self.ranks_per_document = ranks_per_document
        self.doc_logits: list[torch.Tensor] = []
        self.last: dict[str, float] = {}

    def encode_bank(self, a_rows, b_rows, layer):
        if layer == 0:
            self.doc_logits = []
        return super().encode_bank(a_rows, b_rows, layer)

    def read(self, state, m, bank, layer):
        h, weights = super().read(state, m, bank, layer)  # weights [b, s, 1, R], a softmax over slots
        mass = weights[:, :, 0].reshape(*weights.shape[:2], -1, self.ranks_per_document).sum(-1)  # [b, s, K]
        mass = mass.clamp(MASS_EPS, 1.0 - MASS_EPS)
        self.doc_logits.append(torch.log(mass) - torch.log1p(-mass))
        return h, weights

    def diagnostics(self) -> dict[str, float]:
        return dict(self.last)

    def payload_extras(self) -> dict[str, Any]:
        return {"transition_bottleneck": self.bottleneck, "ranks_per_document": self.ranks_per_document,
                "body_arm": "B", "use_latents": False, "latent_width": None, "gate_width": None}


# -- construction, loss, persistence -------------------------------------------------------
def make_gated_reader(arm: str, model, latent_width: int, layers: int, *, bottleneck: int = 32, hidden: int = 128,
                      gate_width: int = GATE_WIDTH):
    if arm == "BS":
        return AttentionSupervisedShallowReader(model.config.hidden_size, layers, bottleneck)
    body_arm, use_latents = GATED_ARMS[arm]
    body = make_reader(body_arm, model, latent_width, layers, bottleneck=bottleneck, hidden=hidden)
    return GatedReader(body, arm, use_latents=use_latents, latent_width=latent_width, gate_width=gate_width)


def response_document_logits(doc_logits: list[torch.Tensor], prompt_lengths: list[int],
                             response_lengths: list[int]) -> torch.Tensor:
    """[layers, n_total, K]: every layer's document logits at the response positions of every row."""
    chunks = []
    for row, (p, r) in enumerate(zip(prompt_lengths, response_lengths, strict=True)):
        chunks.append(torch.stack([layer[row, p - 1:p - 1 + r] for layer in doc_logits]))  # [layers, r, K]
    return torch.cat(chunks, dim=1)


def membership_labels(plans: list[dict[str, Any]], response_lengths: list[int], device) -> torch.Tensor:
    """[n_total, K] float labels aligned with ``response_document_logits``."""
    rows = []
    for plan, length in zip(plans, response_lengths, strict=True):
        members = [i == plan["query"]["document_id"] for i in plan["bank_document_ids"]]
        if not any(members):
            raise ValueError(f"source of {plan['query']['id']} is not in its bank")
        rows.append(torch.tensor(members, dtype=torch.float32).expand(length, -1))
    return torch.cat(rows).to(device)


def gated_loss(lam: float):
    """WRITE_SITE_READERS's answer loss plus ``lam`` times the document-gate BCE; signature of ``answer_loss``."""

    def loss_fn(backend, layers, arm, reader, plans, latents, scaling):
        from write_site_readers_reader_training import make_answer_batch
        from official_d2l_k1_representation_training import gold_answer_loss

        targets = make_answer_batch(backend, plans)
        compiled = representation_batch(backend, plans, latents)
        logits = forward_with_reader(backend, layers, arm, reader, compiled, targets, scaling)
        ce = gold_answer_loss(logits, targets)
        doc = response_document_logits(reader.doc_logits, targets["student_prompt_lengths"], targets["response_lengths"])
        labels = membership_labels(plans, targets["response_lengths"], doc.device).expand_as(doc)
        bce = functional.binary_cross_entropy_with_logits(doc, labels)
        reader.last = {"ce": float(ce.detach()), "gate_bce": float(bce.detach())}
        return ce + lam * bce, sum(targets["response_lengths"])

    return loss_fn


def representation_batch(backend, plans, latents):
    from distractor_curves_reader_training import representation_batch as build

    return build(backend, plans, latents)


def load_gated_reader(path, backend, layers):
    model = backend.model.base_model
    saved = torch.load(path, map_location=model.device, weights_only=False)
    arm = saved["arm"]
    if arm not in GATED_ARMS:
        raise ValueError(f"{path} is not a gated reader checkpoint")
    if arm == "BS":
        reader = AttentionSupervisedShallowReader(model.config.hidden_size, len(layers), saved["transition_bottleneck"],
                                                  saved["ranks_per_document"])
        reader.load_state_dict(saved["transition"])
        return arm, reader.to(model.device).eval()
    body_arm = GATED_ARMS[arm][0]
    if body_arm == "B":
        body = WriteSiteShallowReader(model.config.hidden_size, len(layers), saved["transition_bottleneck"])
    else:
        body = FactoredDeepRankReader(model.config.hidden_size, model.config.intermediate_size, len(layers),
                                      hidden=saved["hidden"], reads=saved["reads"], bank_blocks=saved["bank_blocks"],
                                      read_blocks=saved["read_blocks"], heads=saved["heads"], aggregate=False)
    reader = GatedReader(body, arm, use_latents=saved["use_latents"], latent_width=saved["latent_width"],
                         gate_width=saved["gate_width"], ranks_per_document=saved["ranks_per_document"])
    reader.load_state_dict(saved["transition"])
    return arm, reader.to(model.device).eval()


def gated_per_token_flops(reader, slots: int, layers: int, width: int, input_width: int) -> dict[str, float]:
    """Body read cost (WRITE_SITE_READERS's estimate without its readout) plus the gate heads and the gated effect read."""
    if isinstance(reader, AttentionSupervisedShallowReader):
        return per_token_flops(reader, slots, layers, width, input_width)
    body = per_token_flops(reader.body, slots, layers, width, input_width)
    g, hidden = reader.gate_width, reader.body.hidden * reader.body.reads
    per_layer = 2 * hidden * g + 2 * 2 * g * slots + 2 * slots * g + 2 * slots * width
    return {"reader": body["reader"] + layers * per_layer, "per_bank": body["per_bank"] + layers * 2 * slots * g * reader.gate_key.in_features,
            "native_coefficients": body["native_coefficients"]}
