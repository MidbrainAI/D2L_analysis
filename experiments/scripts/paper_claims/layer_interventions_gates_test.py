"""CPU tests for the LAYER_INTERVENTIONS oracles and membership probe (tiny widths, no model)."""

import layer_interventions_gates as gates
import pytest
import torch
from einops import einsum

WIDTH, INPUT, LATENT, LAYERS, RPD = 16, 24, 12, 3, 4


def bank(k: int, batch: int = 2, seq: int = 5, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(batch, k * RPD, INPUT, generator=g)
    b = torch.randn(batch, k * RPD, WIDTH, generator=g)
    x = torch.randn(batch, seq, INPUT, generator=g)
    state = torch.randn(batch, seq, WIDTH, generator=g)
    latents = torch.randn(batch, k, 1, LATENT, generator=g)
    return a, b, state, einsum(a, x, "b r i, b s i -> b s r"), latents


def test_membership_and_rank_mask_follow_bank_order():
    members = gates.membership(["d2", "src", "d5"], "src")
    assert members == [False, True, False]
    assert gates.rank_mask(members, 3).tolist() == [0, 0, 0, 1, 1, 1, 0, 0, 0]
    with pytest.raises(ValueError):
        gates.membership(["d2", "d5"], "src")


def test_oracle_gates_keep_native_below_layer_from():
    g = gates.oracle_gates([True, False], layers=4, layer_from=2, ranks_per_document=2)
    assert g.shape == (4, 4)
    assert torch.equal(g[:2], torch.ones(2, 4)) and torch.equal(g[2:], torch.tensor([[1., 1., 0., 0.]] * 2))
    assert torch.equal(gates.oracle_gates([True, False], 4, 0, 2)[0], torch.tensor([1., 1., 0., 0.]))
    with pytest.raises(ValueError):
        gates.oracle_gates([True], 4, 5, 2)


def test_fixed_gate_reader_cancels_exactly_the_distractor_effects():
    a, b, state, coefficients, _ = bank(k=3)
    members = [False, True, False]
    reader = gates.FixedGateReader(gates.oracle_gates(members, LAYERS, layer_from=1, ranks_per_document=RPD))
    bank_data = reader.encode_bank(a, b, 0)
    native = einsum(coefficients, b, "b s r, b r d -> b s d") * 2.0
    assert torch.equal(reader.correction(state, coefficients, b, bank_data, 0, 2.0), torch.zeros_like(state))
    corrected = native + reader.correction(state, coefficients, b, bank_data, 1, 2.0)
    mask = gates.rank_mask(members, RPD)
    source_only = einsum(coefficients * mask, b, "b s r, b r d -> b s d") * 2.0
    assert torch.allclose(corrected, source_only, atol=1e-5)


def test_state_collector_records_every_layer_and_returns_zero():
    a, b, state, coefficients, latents = bank(k=2)
    collector = gates.StateCollector()
    for layer in range(LAYERS):
        data = collector.encode_bank(a, b, layer, latents=latents[:, :, layer % 1])
        out = collector.correction(state, coefficients, b, data, layer, 1.5)
        assert torch.equal(out, torch.zeros_like(state))
    assert sorted(collector.layers) == list(range(LAYERS))
    assert torch.equal(collector.layers[2]["m"], gates.signed_strength(coefficients, b.norm(dim=-1), 1.5))
    collector.encode_bank(a, b, 0)  # layer 0 starts a fresh forward
    assert collector.layers == {}


def test_document_logits_and_auc():
    slots = torch.tensor([[1., 1., 3., 3., 0., 0.]])
    assert torch.equal(gates.document_logits(slots, 2), torch.tensor([[1., 3., 0.]]))
    assert gates.auc(torch.tensor([0.9, 0.8, 0.1]), torch.tensor([True, True, False])) == 1.0
    assert gates.auc(torch.tensor([0.1, 0.9]), torch.tensor([True, False])) == 0.0
    assert gates.auc(torch.tensor([0.5, 0.5]), torch.tensor([True, False])) == 0.5
    assert gates.auc(torch.tensor([0.5]), torch.tensor([True])) is None


def test_per_rank_latents_accepts_one_per_document_or_one_per_rank():
    one = torch.randn(2, 3, 1, LATENT)
    out = gates.per_rank_latents(one, RPD, LATENT)
    assert out.shape == (2, 3 * RPD, LATENT) and torch.equal(out[0, :RPD], one[0, 0, 0].expand(RPD, -1))
    per_rank = torch.randn(2, 3, RPD, LATENT)
    assert torch.equal(gates.per_rank_latents(per_rank, RPD, LATENT)[1, RPD:2 * RPD], per_rank[1, 1])
    with pytest.raises(ValueError):
        gates.per_rank_latents(torch.randn(2, 3, RPD + 1, LATENT), RPD, LATENT)


def test_probe_logits_shapes_and_latent_keys_broadcast_per_document():
    probe = gates.MembershipProbe(WIDTH, INPUT, LATENT, LAYERS, dim=8)
    a, b, state, coefficients, latents = bank(k=2, batch=1, seq=3)
    m = gates.signed_strength(coefficients, b.norm(dim=-1), 1.0)
    for key_type in gates.KEY_TYPES:
        keys = probe.keys(key_type, 1, a[0:1], b[0:1], latents[:, :, 0], RPD)
        assert keys.shape == (1, 2 * RPD, 8)
        logits = probe.logits(key_type, 1, state, m, keys)
        assert logits.shape == (1, 3, 2 * RPD)
    latent_keys = probe.keys("latent", 0, a[0:1], b[0:1], latents[:, :, 0], RPD)
    assert torch.allclose(latent_keys[0, :RPD], latent_keys[0, :1].expand(RPD, -1))
    assert not torch.allclose(latent_keys[0, 0], latent_keys[0, RPD])


def test_probe_batch_gathers_response_positions_and_learns_a_separable_bank():
    """A bank whose source slots have a key direction the state can match is learned to AUC 1."""
    torch.manual_seed(0)
    probe = gates.MembershipProbe(WIDTH, INPUT, LATENT, layers=1, dim=8, key_types=("b_rows",))
    optimizer = torch.optim.Adam(probe.parameters(), lr=5e-2)
    direction = torch.randn(WIDTH)
    for _ in range(150):
        a, b, state, coefficients, _ = bank(k=3, batch=2, seq=6, seed=int(torch.randint(0, 10_000, (1,))))
        labels = torch.stack([gates.rank_mask([True, False, False], RPD), gates.rank_mask([False, False, True], RPD)])
        b = b + 3.0 * labels.unsqueeze(-1) * direction  # source slots share a direction
        state = state + 3.0 * direction  # the state points along it
        captures = {0: {"state": state, "m": gates.signed_strength(coefficients, b.norm(dim=-1), 1.0),
                        "a_rows": a, "b_rows": b, "latents": None}}
        positions = gates.response_positions([3, 2], [2, 4])
        logits = gates.probe_batch(probe, captures, positions, RPD)
        assert logits["b_rows"].shape == (1, 6, 3 * RPD)
        expanded = gates.expand_labels(labels, [2, 4])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits["b_rows"][0], expanded)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert gates.auc(logits["b_rows"][0].reshape(-1), expanded.reshape(-1) > 0.5) > 0.95


def test_probe_plans_are_k_balanced_and_sorted():
    rows = [{"document": {"id": f"d{i}"}, "query": {"id": f"q{i}", "document_id": f"d{i}"}} for i in range(40)]
    pool = [f"d{i}" for i in range(40)]
    plans = gates.probe_plans(rows, pool, 16, seed=1, ks=(2, 4))
    assert [p["k"] for p in plans] == [2] * 8 + [4] * 8
    assert all(p["query"]["document_id"] in p["bank_document_ids"] and len(set(p["bank_document_ids"])) == p["k"]
               for p in plans)


def test_suite_plans_cover_every_random_bank_at_the_chosen_k():
    suite = {"questions": {"s": [{"id": "q1", "document_id": "s"}, {"id": "q2", "document_id": "s"}]}}
    banks = {"s": [{"bank_id": "s:random:1", "condition": "random", "k": 1, "document_ids": ["s"]},
                   {"bank_id": "s:random:4", "condition": "random", "k": 4, "document_ids": ["a", "s", "b", "c"]},
                   {"bank_id": "s:same:4", "condition": "same_article", "k": 4, "document_ids": ["a", "s", "b", "c"]}]}
    plans = gates.suite_plans(suite, banks, (4,))
    assert len(plans) == 2 and all(p["bank_id"] == "s:random:4" for p in plans)


def test_summarize_probe_picks_the_best_layer():
    result = {"b_rows": {"0": {"4": {"token": 0.6, "question": 0.7, "slot": 0.5, "bce": 0.6}},
                         "1": {"4": {"token": 0.9, "question": 0.8, "slot": 0.7, "bce": 0.3}}}}
    summary = gates.summarize_probe(result, "token")
    assert summary["b_rows"]["4"] == {"best_layer": 1, "best_auc": 0.9, "by_layer": [0.6, 0.9]}


def test_auc_rank_based_matches_pairwise_definition_with_ties():
    import torch

    from layer_interventions_gates import auc

    torch.manual_seed(3)
    scores = torch.randint(0, 5, (200,)).float()  # many ties
    labels = torch.rand(200) < 0.3
    pos, neg = scores[labels], scores[~labels]
    pairwise = float(((pos[:, None] > neg[None, :]).float() + 0.5 * (pos[:, None] == neg[None, :]).float()).mean())
    assert abs(auc(scores, labels) - pairwise) < 1e-6  # the pairwise reference is float32
    assert auc(torch.tensor([0.1, 0.9]), torch.tensor([False, True])) == 1.0
    assert auc(torch.tensor([0.1, 0.9]), torch.tensor([True, True])) is None


def test_response_only_gate_reader_masks_decode_calls_and_optionally_the_last_prompt_position():
    import torch
    from layer_interventions_gates import position_mode_reader, rank_mask

    members = [False, True, False]
    reader = position_mode_reader(members, "decode", ranks_per_document=2)
    last = position_mode_reader(members, "decode_last", ranks_per_document=2)
    coefficients = torch.ones(1, 4, 6)
    b_rows = torch.eye(6, 8).unsqueeze(0)  # [batch, R, D]
    bank = {"b_norms": b_rows.norm(dim=-1)}
    prefill = reader.correction(None, coefficients, b_rows, bank, 0, 1.0)
    assert torch.equal(prefill, torch.zeros(1, 4, 8))  # prompt: native K everywhere
    prefill_last = last.correction(None, coefficients, b_rows, bank, 0, 1.0)
    assert torch.equal(prefill_last[0, :3], torch.zeros(3, 8))
    expected = ((rank_mask(members, 2) - 1.0) @ b_rows[0])  # distractor ranks subtracted at the last prompt position
    assert torch.allclose(prefill_last[0, 3], expected)
    decode = reader.correction(None, coefficients[:, :1], b_rows, bank, 5, 1.0)
    assert torch.allclose(decode[0, 0], expected)
    with __import__("pytest").raises(ValueError):
        position_mode_reader(members, "prefill", ranks_per_document=2)


def test_query_positions_and_pooled_gathering():
    import torch
    from layer_interventions_gates import gathered, query_positions

    positions, pooled = query_positions("response", [5, 3], [2, 4])
    assert positions == [(4, 2), (2, 4)] and not pooled
    positions, pooled = query_positions("last_prompt", [5, 3], [2, 4])
    assert positions == [(4, 1), (2, 1)] and not pooled
    positions, pooled = query_positions("prompt_mean", [5, 3], [2, 4])
    assert positions == [(0, 5), (0, 3)] and pooled
    captured = {"state": torch.arange(2 * 9 * 2, dtype=torch.float32).reshape(2, 9, 2), "m": torch.ones(2, 9, 3)}
    states, strengths, sizes = gathered(captured, positions, pooled)
    assert sizes == [1, 1] and states.shape == (2, 2) and strengths.shape == (2, 3)
    assert torch.allclose(states[0], captured["state"][0, :5].mean(0)) and torch.allclose(states[1], captured["state"][1, :3].mean(0))
    with __import__("pytest").raises(ValueError):
        query_positions("middle", [1], [1])


def test_silent_collector_cancels_the_native_sum_and_still_records_strengths():
    a, b, state, coefficients, latents = bank(k=2)
    collector = gates.StateCollector(silent=True)
    data = collector.encode_bank(a, b, 0, latents=latents[:, :, 0])
    out = collector.correction(state, coefficients, b, data, 0, 1.5)
    native = einsum(coefficients, b, "b s r, b r d -> b s d") * 1.5
    assert torch.allclose(out, -native, atol=1e-5)
    assert torch.equal(collector.layers[0]["m"], gates.signed_strength(coefficients, b.norm(dim=-1), 1.5))


def test_strength_key_is_parameter_free_and_pooled_over_the_query_positions():
    a, b, state, coefficients, latents = bank(k=2)
    probe = gates.MembershipProbe(WIDTH, INPUT, LATENT, LAYERS, dim=8, key_types=("b_rows", gates.STRENGTH_KEY))
    assert probe.learned_key_types == ("b_rows",) and gates.STRENGTH_KEY not in probe.query
    collector = gates.StateCollector()
    for layer in range(LAYERS):
        collector.correction(state, coefficients, b, collector.encode_bank(a, b, layer, latents=latents[:, :, 0]), layer, 1.5)
    positions = [(1, 2), (0, 3)]
    logits = gates.probe_batch(probe, collector.layers, positions, RPD)
    m = collector.layers[1]["m"]
    assert logits[gates.STRENGTH_KEY].shape == (LAYERS, 5, 2 * RPD)
    assert torch.allclose(logits[gates.STRENGTH_KEY][1][:2], m[0, 1:3].abs())
    pooled = gates.probe_batch(probe, collector.layers, positions, RPD, pooled=True)
    assert torch.allclose(pooled[gates.STRENGTH_KEY][1][1], m[1, 0:3].abs().mean(0))
    with pytest.raises(ValueError):
        gates.MembershipProbe(WIDTH, INPUT, LATENT, LAYERS, key_types=("b_rows", "nope"))
