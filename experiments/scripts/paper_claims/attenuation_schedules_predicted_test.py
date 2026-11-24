"""CPU tests for the predicted-gate readers (tiny widths; tiny Gemma-2 for the re-step loop)."""

import layer_interventions_gates as gates
import attenuation_schedules_predicted as predicted
import attenuation_schedules_schedules as schedules
import pytest
import torch
from einops import einsum

WIDTH, INPUT, LATENT, LAYERS, RPD = 16, 24, 12, 4, 4


def bank(k: int = 2, seq: int = 5, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(1, k * RPD, INPUT, generator=g)
    b = torch.randn(1, k * RPD, WIDTH, generator=g)
    x = torch.randn(1, seq, INPUT, generator=g)
    state = torch.randn(1, seq, WIDTH, generator=g)
    latents = torch.randn(1, k, 1, LATENT, generator=g)
    return a, b, state, einsum(a, x, "b r i, b s i -> b s r"), latents


def probe():
    torch.manual_seed(0)
    return gates.MembershipProbe(WIDTH, INPUT, LATENT, LAYERS, dim=8, key_types=("b_rows", gates.STRENGTH_KEY))


def test_gates_from_scores_modes_and_pools():
    scores = torch.tensor([[2.0, -1.0, 0.5, -3.0]])
    assert torch.allclose(predicted.gates_from_scores(scores, "b_rows", "soft", "rank", 2), torch.sigmoid(scores))
    assert predicted.gates_from_scores(scores, "b_rows", "hard", "rank", 2).tolist() == [[1.0, 0.0, 1.0, 0.0]]
    assert predicted.gates_from_scores(scores, "b_rows", "hard", "document", 2).tolist() == [[1.0, 1.0, 0.0, 0.0]]
    strength = torch.tensor([[4.0, 1.0, 2.0, 0.0]])
    assert predicted.gates_from_scores(strength, gates.STRENGTH_KEY, "soft", "rank", 2).tolist() == [[1.0, 0.25, 0.5, 0.0]]
    assert predicted.gates_from_scores(strength, gates.STRENGTH_KEY, "hard", "rank", 2).tolist() == [[1.0, 0.0, 1.0, 0.0]]
    with pytest.raises(ValueError):
        predicted.gates_from_scores(scores, "b_rows", "top1", "rank", 2)


def test_per_position_scores_last_versus_causal_mean():
    a, b, state, coefficients, latents = bank()
    m = gates.signed_strength(coefficients, b.norm(dim=-1), 1.5)
    captured = {"state": state, "m": m, "a_rows": a, "b_rows": b, "latents": latents[:, :, 0]}
    pr = probe()
    last = predicted.per_position_scores(pr, "b_rows", 1, captured, "last_prompt", RPD)
    mean = predicted.per_position_scores(pr, "b_rows", 1, captured, "prompt_mean", RPD)
    assert last.shape == (5, 2 * RPD) and torch.allclose(last[0], mean[0], atol=1e-5) and not torch.allclose(last[4], mean[4])
    strength = predicted.per_position_scores(pr, gates.STRENGTH_KEY, 1, captured, "prompt_mean", RPD)
    assert torch.allclose(strength[2], m[0, :3].abs().mean(0))
    with pytest.raises(ValueError):
        predicted.per_position_scores(pr, "b_rows", 1, captured, "response", RPD)


def run_prefill(reader, scaling=1.5):
    a, b, state, coefficients, latents = bank()
    reference = torch.randn(1, 5, WIDTH) + einsum(coefficients, b, "b s r, b r d -> b s d") * scaling
    outs = {}
    for layer in range(LAYERS):
        data = reader.encode_bank(a, b, layer, latents=latents[:, :, 0])
        outs[layer] = reader.correction(state, coefficients, b, data, layer, scaling, reference=reference)
    return a, b, state, coefficients, reference, outs


def test_schedule_a_reader_is_silent_in_prefill_and_gated_at_single_tokens():
    reader = predicted.PredictedGateReader(probe(), "b_rows", 1, "last_prompt", "soft", "rank", "A", ranks_per_document=RPD)
    a, b, state, coefficients, _reference, outs = run_prefill(reader)
    native = einsum(coefficients, b, "b s r, b r d -> b s d") * 1.5
    for layer in range(LAYERS):
        assert torch.allclose(outs[layer], -native, atol=1e-5)
    assert reader.gate.shape == (2 * RPD,) and reader.records == {}
    step = reader.correction(state[:, :1], coefficients[:, :1], b, reader.encode_bank(a, b, 2), 2, 1.5)
    expected = einsum(coefficients[:, :1] * (reader.gate - 1), b, "b s r, b r d -> b s d") * 1.5
    assert torch.allclose(step, expected, atol=1e-5)


def test_schedule_c_reader_records_clean_outputs_and_gated_writes():
    reader = predicted.PredictedGateReader(probe(), gates.STRENGTH_KEY, 0, "prompt_mean", "hard", "rank", "C", ranks_per_document=RPD)
    _a, b, _state, coefficients, reference, _outs = run_prefill(reader)
    native = einsum(coefficients, b, "b s r, b r d -> b s d") * 1.5
    writes = reader.writes(1.5)
    assert sorted(writes) == list(range(LAYERS))
    assert torch.allclose(writes[3]["clean"], reference - native, atol=1e-5)
    assert torch.allclose(writes[3]["write"], einsum(coefficients * reader.gate, b, "b s r, b r d -> b s d") * 1.5, atol=1e-5)
    assert set(reader.gate.tolist()) <= {0.0, 1.0}


def test_schedule_b_reader_gates_prefill_positions_from_layer_from():
    reader = predicted.PredictedGateReader(probe(), "b_rows", 1, "prompt_mean", "soft", "document", "B", layer_from=2, ranks_per_document=RPD)
    _a, b, _state, coefficients, _reference, outs = run_prefill(reader)
    native = einsum(coefficients, b, "b s r, b r d -> b s d") * 1.5
    assert torch.allclose(outs[0], -native, atol=1e-5) and torch.allclose(outs[1], -native, atol=1e-5)
    expected = einsum(coefficients * (reader.prompt_gates.unsqueeze(0) - 1), b, "b s r, b r d -> b s d") * 1.5
    assert torch.allclose(outs[2], expected, atol=1e-5) and torch.allclose(outs[3], expected, atol=1e-5)
    g = reader.prompt_gates.reshape(5, 2, RPD)
    assert torch.allclose(g[:, :, 0], g[:, :, 1])  # document pool broadcasts over ranks
    with pytest.raises(ValueError):
        predicted.PredictedGateReader(probe(), "b_rows", 3, "prompt_mean", "soft", "rank", "B", layer_from=2)


@pytest.fixture(scope="module")
def tiny_gemma():
    transformers = pytest.importorskip("transformers")
    config = transformers.Gemma2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                                       num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                                       max_position_embeddings=128, sliding_window=64, attn_implementation="eager")
    torch.manual_seed(0)
    return transformers.Gemma2ForCausalLM(config).eval()


def test_restep_after_full_prefill_matches_plain_decoding(tiny_gemma):
    from transformers.cache_utils import HybridCache

    prompt = torch.tensor([[2, 10, 11, 12, 13]])
    p, n = prompt.shape[1], 6
    make = lambda: HybridCache(tiny_gemma.config, max_batch_size=1, max_cache_len=p + n, dtype=tiny_gemma.dtype, device="cpu")  # noqa: E731
    stop = schedules.stop_ids(tiny_gemma)
    with torch.inference_mode():
        plain = schedules.greedy_decode(tiny_gemma, prompt, make(), stop, n)
        cache = make()
        tiny_gemma(prompt, attention_mask=torch.ones(1, p, dtype=torch.long), past_key_values=cache, use_cache=True,
                   cache_position=torch.arange(p))
        restep = schedules.greedy_decode(tiny_gemma, prompt[:, -1:], cache, stop, n, offset=p - 1)
    assert restep == plain and len(plain) > 0
