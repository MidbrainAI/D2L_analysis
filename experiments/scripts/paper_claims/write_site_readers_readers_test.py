"""CPU tests for the WRITE_SITE_READERS write-site reader arms (tiny widths, no model)."""

import math

import write_site_readers_readers as readers
import pytest
import torch
from einops import einsum

WIDTH, INPUT, LAYERS, RANK = 16, 24, 3, 8


def arm(name: str):
    if name == "B":
        return readers.WriteSiteShallowReader(WIDTH, LAYERS, bottleneck=8)
    return readers.FactoredDeepRankReader(WIDTH, INPUT, LAYERS, hidden=16, reads=2, bank_blocks=1, read_blocks=1,
                                          heads=2, aggregate=name == "D")


def inputs(k: int, batch: int = 2, seq: int = 3, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(batch, k * RANK, INPUT, generator=g)
    b = torch.randn(batch, k * RANK, WIDTH, generator=g)
    x = torch.randn(batch, seq, INPUT, generator=g)
    state = torch.randn(batch, seq, WIDTH, generator=g)
    coefficients = einsum(a, x, "b r i, b s i -> b s r")
    return a, b, x, state, coefficients


def native_delta(coefficients, b, scaling):
    return einsum(coefficients, b, "b s r, b r d -> b s d") * scaling


@pytest.mark.parametrize("name", ["B", "C", "D"])
def test_zero_init_gives_an_exactly_zero_correction(name):
    """Zero W_out/gamma/beta: the correction is exactly zero and tanh(0) keeps it so."""
    reader = arm(name)
    a, b, _, state, coefficients = inputs(k=4)
    bank = reader.encode_bank(a, b, layer=1)
    out = reader.correction(state, coefficients, b, bank, 1, scaling=2.0)
    assert out.shape == state.shape and torch.equal(out, torch.zeros_like(out))


@pytest.mark.parametrize("name", ["B", "C", "D"])
def test_aggregate_scale_can_express_normalization(name):
    """gamma = K**-0.5 - 1 on the native delta reproduces normalized native exactly."""
    reader = arm(name)
    k = 4
    a, b, _, state, coefficients = inputs(k=k)
    # Only the slot-count path feeds the read state, so h is one constant vector for every token.
    with torch.no_grad():
        for p in reader.parameters():
            p.zero_()
        reader.count_projection.weight.normal_()
    bank = reader.encode_bank(a, b, layer=0)
    h, _ = reader.read(state, readers.signed_strength(coefficients, bank["b_norms"], 2.0), bank, 0)
    with torch.no_grad():
        reader.read_norm.weight.fill_(1.0)  # the LayerNorm the heads see (zeroed above)
    h = reader.read_norm(h).detach()
    assert torch.allclose(h, h[:1, :1].expand_as(h), atol=1e-6) and abs(float(h[0, 0, 0])) > 1e-3
    with torch.no_grad():  # tanh(w h0 / sqrt(dim)) = K**-0.5 - 1
        reader.aggregate_scale.weight[0, 0] = math.atanh(k ** -0.5 - 1) * math.sqrt(h.shape[-1]) / float(h[0, 0, 0])
    out = reader.correction(state, coefficients, b, bank, 0, scaling=2.0)
    delta = native_delta(coefficients, b, 2.0)
    assert torch.allclose(delta + out, delta * k ** -0.5, atol=1e-5)


def test_effect_read_matches_materialized_effects():
    _, b, _, _, coefficients = inputs(k=2)
    weights = torch.rand(*coefficients.shape)
    effects = (coefficients.unsqueeze(-1) * b.unsqueeze(1)) * 1.5  # [b, s, r, d]
    expected = (weights.unsqueeze(-1) * effects).sum(2)
    assert torch.allclose(readers.effect_read(weights, coefficients, b, 1.5), expected, atol=1e-5)


def test_signed_strength_keeps_sign_and_magnitude():
    coefficients = torch.tensor([[[2.0, -0.5]]])
    norms = torch.tensor([[3.0, 4.0]])
    m = readers.signed_strength(coefficients, norms, 0.5)
    assert torch.allclose(m, torch.tensor([[[3.0, -1.0]]]))
    features = readers.magnitude_features(m)
    assert features.shape == (1, 1, 2, 2) and features[0, 0, 1, 1] == pytest.approx(-math.log1p(1.0))


@pytest.mark.parametrize("name", ["B", "C", "D"])
def test_attention_weights_are_a_distribution_over_slots(name):
    reader = arm(name)
    a, b, _, state, coefficients = inputs(k=2)
    bank = reader.encode_bank(a, b, layer=2)
    m = readers.signed_strength(coefficients, bank["b_norms"], 1.0)
    h, weights = reader.read(state, m, bank, 2)
    assert h.shape == (*state.shape[:2], reader.reads * reader.hidden)
    slots = 1 if name == "D" else coefficients.shape[-1]
    assert weights.shape == (*state.shape[:2], reader.reads, slots)
    assert torch.allclose(weights.sum(-1), torch.ones(*state.shape[:2], reader.reads), atol=1e-5)


def test_c_and_d_differ_only_in_pooling_and_payload_round_trips():
    for name in ("B", "C", "D"):
        reader = arm(name)
        with torch.no_grad():
            for p in reader.parameters():
                p.normal_()
        payload = readers.reader_payload(reader, name, selected_step=5, seed=1, recipe="test")
        clone = arm(name)
        clone.load_state_dict(payload["transition"])
        a, b, _, state, coefficients = inputs(k=2)
        for module in (reader, clone):
            module.eval()
        bank, bank2 = reader.encode_bank(a, b, 0), clone.encode_bank(a, b, 0)
        out = reader.correction(state, coefficients, b, bank, 0, 1.0)
        out2 = clone.correction(state, coefficients, b, bank2, 0, 1.0)
        assert torch.allclose(out, out2, atol=1e-6) and payload["arm"] == name and payload["execution_site"] == "write-site"
    c, d = arm("C"), arm("D")
    a, b, _, _, _ = inputs(k=2)
    assert c.encode_bank(a, b, 0)["slots"].shape[1] == 2 * RANK and d.encode_bank(a, b, 0)["slots"].shape[1] == 1


def test_flop_estimates_are_positive_and_scale_with_slots():
    for name in ("B", "C", "D"):
        reader = arm(name)
        small = readers.per_token_flops(reader, slots=8, layers=LAYERS, width=WIDTH, input_width=INPUT)
        large = readers.per_token_flops(reader, slots=64, layers=LAYERS, width=WIDTH, input_width=INPUT)
        assert small["reader"] > 0 and large["reader"] >= small["reader"]
        assert large["native_coefficients"] == 8 * small["native_coefficients"]
    assert readers.parameter_count(arm("B")) > 0


@pytest.mark.parametrize("name", ["B", "C", "D"])
def test_readout_scales_with_the_branch_rms(name):
    reader = arm(name)
    with torch.no_grad():
        for p in reader.parameters():
            p.normal_()
    a, b, _, state, coefficients = inputs(k=2)
    bank = reader.encode_bank(a, b, 0)
    reference = torch.randn(*state.shape)
    once = reader.correction(state, coefficients, b, bank, 0, 1.0, reference=reference)
    twice = reader.correction(state, coefficients, b, bank, 0, 1.0, reference=2 * reference)
    thrice = reader.correction(state, coefficients, b, bank, 0, 1.0, reference=3 * reference)
    # the W_out term is linear in the reference RMS; the gamma/beta terms do not depend on it
    assert not torch.allclose(once, twice)
    assert torch.allclose(thrice - twice, twice - once, atol=1e-4)


def test_mlp_input_norm_finds_the_pre_mlp_normalization_on_gemma_and_qwen():
    transformers = pytest.importorskip("transformers")
    gemma = transformers.Gemma2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, attn_implementation="eager")
    qwen = transformers.Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                    num_attention_heads=4, num_key_value_heads=2, head_dim=8, attn_implementation="eager")
    gemma_layer = transformers.Gemma2ForCausalLM(gemma).model.layers[0]
    qwen_layer = transformers.Qwen3ForCausalLM(qwen).model.layers[0]
    assert readers.mlp_input_norm(gemma_layer) is gemma_layer.pre_feedforward_layernorm
    assert readers.mlp_input_norm(qwen_layer) is qwen_layer.post_attention_layernorm
    with pytest.raises(ValueError):
        readers.mlp_input_norm(torch.nn.Linear(2, 2))
