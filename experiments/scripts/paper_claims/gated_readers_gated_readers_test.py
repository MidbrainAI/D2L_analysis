"""CPU tests for the GATED_READERS gated readers (tiny widths, no model)."""

import math

import write_site_readers_readers as readers
import gated_readers_gated_readers as gated
import pytest
import torch
from einops import einsum

WIDTH, INPUT, LATENT, LAYERS, RPD = 16, 24, 12, 3, 4


def body(arm: str):
    if arm == "B":
        return readers.WriteSiteShallowReader(WIDTH, LAYERS, bottleneck=8)
    return readers.FactoredDeepRankReader(WIDTH, INPUT, LAYERS, hidden=16, reads=2, bank_blocks=1, read_blocks=1, heads=2)


def reader(arm: str):
    body_arm, use_latents = gated.GATED_ARMS[arm]
    return gated.GatedReader(body(body_arm), arm, use_latents=use_latents, latent_width=LATENT, gate_width=8,
                             ranks_per_document=RPD)


def bank(k: int, batch: int = 2, seq: int = 3, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(batch, k * RPD, INPUT, generator=g)
    b = torch.randn(batch, k * RPD, WIDTH, generator=g)
    x = torch.randn(batch, seq, INPUT, generator=g)
    state = torch.randn(batch, seq, WIDTH, generator=g)
    latents = torch.randn(batch, k, 1, LATENT, generator=g)
    return a, b, state, einsum(a, x, "b r i, b s i -> b s r"), latents


def encode(module, a, b, layer, latents):
    return module.encode_bank(a, b, layer, latents=latents) if module.uses_latents else module.encode_bank(a, b, layer)


@pytest.mark.parametrize("arm", ["GB", "GC", "GBL"])
def test_zero_init_is_exactly_native(arm):
    module = reader(arm)
    a, b, state, coefficients, latents = bank(k=3)
    data = encode(module, a, b, 0, latents[:, :, 0])
    out = module.correction(state, coefficients, b, data, 0, 2.0)
    assert torch.equal(out, torch.zeros_like(state))
    gate, doc = module.gates(module.slot_logits(module.body.read_norm(module.body.read(
        state, readers.signed_strength(coefficients, data["b_norms"], 2.0), data, 0)[0]), readers.signed_strength(
        coefficients, data["b_norms"], 2.0), data))
    assert torch.equal(gate, torch.ones_like(gate)) and doc.shape == (2, 3, 3)


def test_gates_are_one_at_the_prior_zero_for_distractors_and_capped_for_members():
    module = reader("GB")
    logits = torch.tensor([[[-50.0, 0.0, 50.0, 0.0]]])
    gate, doc = module.gates(logits)
    assert gate[0, 0, 1] == 1.0 and gate[0, 0, 0] < 1e-6 and 1.0 < gate[0, 0, 2] <= 1.0 / torch.sigmoid(torch.tensor(4.0)) + 1e-6
    assert doc.shape == (1, 1, 1) and torch.allclose(doc, logits.mean(-1, keepdim=True) + gated.GATE_PRIOR)


def test_document_logits_are_the_mean_over_each_documents_ranks_plus_the_prior():
    module = reader("GB")
    logits = torch.arange(2 * RPD, dtype=torch.float32).reshape(1, 1, -1)
    _, doc = module.gates(logits)
    assert torch.allclose(doc[0, 0], torch.tensor([1.5, 5.5]) + gated.GATE_PRIOR)


@pytest.mark.parametrize("arm", ["GB", "GC", "GBL"])
def test_oracle_gates_reproduce_source_only_and_latents_change_keys(arm):
    module = reader(arm)
    with torch.no_grad():
        for p in module.parameters():
            p.normal_()
    a, b, state, coefficients, latents = bank(k=2)
    data = encode(module, a, b, 1, latents[:, :, 0])
    out = module.correction(state, coefficients, b, data, 1, 1.0)
    assert out.shape == state.shape and len(module.doc_logits) == 1 and module.doc_logits[0].shape == (2, 3, 2)
    if module.uses_latents:
        other = encode(module, a, b, 1, torch.randn_like(latents[:, :, 0]))
        assert not torch.allclose(other["gate_keys"], data["gate_keys"])
        assert torch.allclose(other["gate_keys"][:, :RPD] - data["gate_keys"][:, :RPD],
                              (other["gate_keys"][:, :1] - data["gate_keys"][:, :1]).expand(-1, RPD, -1), atol=1e-5)


def test_response_document_logits_and_labels_align():
    doc_logits = [torch.arange(2 * 6 * 2, dtype=torch.float32).reshape(2, 6, 2) for _ in range(LAYERS)]
    gathered = gated.response_document_logits(doc_logits, prompt_lengths=[3, 2], response_lengths=[2, 3])
    assert gathered.shape == (LAYERS, 5, 2)
    assert torch.equal(gathered[0, :2], doc_logits[0][0, 2:4]) and torch.equal(gathered[0, 2:], doc_logits[0][1, 1:4])
    plans = [{"query": {"id": "q", "document_id": "s"}, "bank_document_ids": ["a", "s"]},
             {"query": {"id": "r", "document_id": "t"}, "bank_document_ids": ["t", "b"]}]
    labels = gated.membership_labels(plans, [2, 3], "cpu")
    assert labels.tolist() == [[0, 1], [0, 1], [1, 0], [1, 0], [1, 0]]
    with pytest.raises(ValueError):
        gated.membership_labels([{"query": {"id": "q", "document_id": "x"}, "bank_document_ids": ["a"]}], [1], "cpu")


@pytest.mark.parametrize("arm", ["GB", "GC", "GBL"])
def test_payload_round_trips_through_the_write_site_readers_payload_helper(arm):
    module = reader(arm)
    with torch.no_grad():
        for p in module.parameters():
            p.normal_()
    payload = readers.reader_payload(module, arm, selected_step=3, seed=1, recipe="t")
    assert payload["arm"] == arm and payload["use_latents"] == module.uses_latents and payload["gate_width"] == 8
    clone = reader(arm)
    clone.load_state_dict(payload["transition"])
    a, b, state, coefficients, latents = bank(k=2)
    for m in (module, clone):
        m.eval()
    out = module.correction(state, coefficients, b, encode(module, a, b, 0, latents[:, :, 0]), 0, 1.0)
    out2 = clone.correction(state, coefficients, b, encode(clone, a, b, 0, latents[:, :, 0]), 0, 1.0)
    assert torch.allclose(out, out2, atol=1e-6)


def test_gate_flops_exceed_the_body():
    module = reader("GB")
    body = readers.per_token_flops(module.body, 16, LAYERS, WIDTH, INPUT)
    total = gated.gated_per_token_flops(module, 16, LAYERS, WIDTH, INPUT)
    assert total["reader"] > body["reader"] and total["native_coefficients"] == body["native_coefficients"]
    assert math.isclose(1 / (1 + math.exp(-gated.GATE_PRIOR)), 0.982, abs_tol=1e-3)


def test_attention_supervised_reader_exposes_document_mass_and_keeps_arm_b_output():
    module = gated.AttentionSupervisedShallowReader(WIDTH, LAYERS, bottleneck=8, ranks_per_document=RPD)
    plain = readers.WriteSiteShallowReader(WIDTH, LAYERS, bottleneck=8)
    with torch.no_grad():
        for p, q in zip(module.parameters(), plain.parameters(), strict=True):
            p.normal_()
            q.copy_(p)
    a, b, state, coefficients, _ = bank(k=3)
    out = module.correction(state, coefficients, b, module.encode_bank(a, b, 0), 0, 1.0)
    expected = plain.correction(state, coefficients, b, plain.encode_bank(a, b, 0), 0, 1.0)
    assert torch.allclose(out, expected, atol=1e-6)
    assert len(module.doc_logits) == 1 and module.doc_logits[0].shape == (2, 3, 3)
    mass = torch.sigmoid(module.doc_logits[0])
    assert torch.allclose(mass.sum(-1), torch.ones(2, 3), atol=1e-4)
    payload = readers.reader_payload(module, "BS", selected_step=1, seed=1, recipe="t")
    assert payload["arm"] == "BS" and payload["transition_bottleneck"] == 8 and payload["use_latents"] is False
    assert gated.gated_per_token_flops(module, 16, LAYERS, WIDTH, INPUT)["reader"] > 0
