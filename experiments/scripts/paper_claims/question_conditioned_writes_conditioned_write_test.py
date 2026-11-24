from types import SimpleNamespace

import torch

from question_conditioned_writes_conditioned_write import WriteEncoder, bank_slots, layer_factors, question_rows


def test_encoder_starts_as_identity_and_gets_gradients():
    torch.manual_seed(0)
    encoder = WriteEncoder(latent_width=16, hidden_width=12, layers=3, width=8, heads=2, blocks=2)
    slots = torch.randn(2, 6, 16)
    question = torch.randn(2, 5, 12)
    padding = torch.tensor([[False] * 5, [False, False, True, True, True]])
    out = encoder(slots, question, padding, layer=1)
    assert torch.equal(out, slots)
    out.sum().backward()
    assert encoder.out.weight.grad is not None and encoder.out.weight.grad.abs().sum() > 0


def test_encoder_ignores_padded_question_rows():
    torch.manual_seed(0)
    encoder = WriteEncoder(16, 12, 3, width=8, heads=2, blocks=1)
    torch.nn.init.normal_(encoder.out.weight)
    slots, question = torch.randn(1, 4, 16), torch.randn(1, 5, 12)
    padding = torch.tensor([[False, False, False, True, True]])
    changed = question.clone()
    changed[:, 3:] = 100.0
    assert torch.allclose(encoder(slots, question, padding, 0), encoder(slots, changed, padding, 0), atol=1e-6)


def test_question_rows_exclude_answer_predictor_and_response():
    states = torch.arange(2 * 7, dtype=torch.float32).view(2, 7, 1)
    rows, padding = question_rows(states, prompt_lengths=[5, 3])
    assert rows.shape == (2, 4, 1)
    assert padding.tolist() == [[False, False, False, False], [False, False, True, True]]


def fake_hypernet(layers=2, rank=2, d_latent=3, d_in=4, d_out=5, extra=1):
    torch.manual_seed(1)
    return SimpleNamespace(
        head=SimpleNamespace(weight=torch.randn(layers, d_latent, d_in + d_out + extra)),
        d_in={"down_proj": d_in}, d_out={"down_proj": d_out},
        scaler_A={"down_proj": torch.randn(1, layers, rank, 1)},
        scaler_B={"down_proj": torch.randn(1, layers, rank, 1)},
        bias_A={"down_proj": torch.randn(layers, rank, d_in)},
        bias_B={"down_proj": torch.randn(layers, rank, d_out)})


def reference_merged(hypernet, latents):
    """D2L's path: head per document (all layers), per-rank scalers, rank-concatenate documents, append bias."""
    k = latents.shape[0]
    flat = torch.einsum("klrd,ldo->klro", latents, hypernet.head.weight)
    d_in, d_out = hypernet.d_in["down_proj"], hypernet.d_out["down_proj"]
    a = flat[..., :d_in] * hypernet.scaler_A["down_proj"]
    b = flat[..., d_in:d_in + d_out] * hypernet.scaler_B["down_proj"]
    a = torch.cat([a[i] for i in range(k)] + [hypernet.bias_A["down_proj"]], dim=1)  # [layers, k*r + r, d_in]
    b = torch.cat([b[i] for i in range(k)] + [hypernet.bias_B["down_proj"]], dim=1)
    return a, b


def test_layer_factors_match_head_scalers_and_merge():
    hypernet = fake_hypernet()
    latents = torch.randn(3, 2, 2, 3)  # [K, layers, r, d_latent]
    expected_a, expected_b = reference_merged(hypernet, latents)
    bank = latents[None]  # [B=1, K, layers, r, d]
    for layer in range(2):
        a, b = layer_factors(hypernet, bank_slots(bank, layer), layer)
        assert torch.allclose(a[0], expected_a[layer], atol=1e-6)
        assert torch.allclose(b[0], expected_b[layer], atol=1e-6)


def test_bank_slots_keep_document_major_order():
    bank = torch.arange(1 * 2 * 3 * 2 * 1, dtype=torch.float32).view(1, 2, 3, 2, 1)
    slots = bank_slots(bank, layer=1)
    assert slots[0, :, 0].tolist() == [bank[0, 0, 1, 0, 0], bank[0, 0, 1, 1, 0], bank[0, 1, 1, 0, 0], bank[0, 1, 1, 1, 0]]


def test_partners_are_other_questions_on_the_same_source():
    from question_conditioned_writes_conditioned_write import partners, update_plans

    rows = [{"query": {"id": f"q{i}"}, "document": {"id": f"d{i % 3}"}} for i in range(9)]
    rows.append({"query": {"id": "solo"}, "document": {"id": "alone"}})
    paired = partners(rows, 212)
    for row, partner in zip(rows, paired):
        if row["query"]["id"] == "solo":
            assert partner is None
        else:
            source = next(r for r in rows if r["query"]["id"] == partner["id"])["document"]["id"]
            assert partner["id"] != row["query"]["id"] and source == row["document"]["id"]
    assert paired == partners(rows, 212)
    plans = update_plans(rows[:9], 2, 212, ["d0", "d1", "d2"], (1,), 4)
    assert all(p["partner"] is not None for step in plans for p in step)
