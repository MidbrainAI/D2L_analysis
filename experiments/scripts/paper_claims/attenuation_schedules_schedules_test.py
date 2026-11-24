"""CPU tests for the ATTENUATION_SCHEDULES schedules: reader weights, the replay's first-affected-layer exactness, and the
manual greedy loop against ``generate`` on a tiny Gemma-2."""

import attenuation_schedules_schedules as schedules
import pytest
import torch
from einops import einsum

R = 8  # two documents x 4 ranks


def bank(seq: int = 5, width: int = 16, inputs: int = 24, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(1, R, inputs, generator=g)
    b = torch.randn(1, R, width, generator=g)
    x = torch.randn(1, seq, inputs, generator=g)
    state = torch.randn(1, seq, width, generator=g)
    return a, b, state, einsum(a, x, "b r i, b s i -> b s r")


MASK = torch.tensor([0., 0., 0., 0., 1., 1., 1., 1.])


def test_schedule_reader_weights_follow_each_design():
    a, b, state, coefficients = bank()
    reader_a = schedules.ScheduleReader(MASK, "A")
    prefill = reader_a.weights(1, 5, layer=3, device="cpu")
    assert torch.equal(prefill[0, :4], torch.full((4, R), -1.0)) and torch.equal(prefill[0, 4], MASK - 1)
    assert torch.equal(reader_a.weights(1, 1, layer=0, device="cpu")[0, 0], MASK - 1)
    reader_b = schedules.ScheduleReader(MASK, "B", layer_from=2)
    assert torch.equal(reader_b.weights(1, 5, layer=1, device="cpu"), torch.full((1, 5, R), -1.0))
    assert torch.equal(reader_b.weights(1, 5, layer=2, device="cpu")[0, 2], MASK - 1)
    none = schedules.ScheduleReader(MASK, "none")
    assert torch.equal(none.weights(1, 5, layer=9, device="cpu"), torch.full((1, 5, R), -1.0))
    # the correction cancels exactly the native sum where silent and the distractors where masked
    native = einsum(coefficients, b, "b s r, b r d -> b s d") * 2.0
    corrected = native + reader_a.correction(state, coefficients, b, reader_a.encode_bank(a, b, 0), 0, 2.0)
    assert torch.allclose(corrected[0, :4], torch.zeros(4, 16), atol=1e-5)
    assert torch.allclose(corrected[0, 4], (einsum(coefficients * MASK, b, "b s r, b r d -> b s d") * 2.0)[0, 4], atol=1e-5)
    with pytest.raises(ValueError):
        schedules.ScheduleReader(MASK, "C")


def test_replay_reader_records_clean_output_and_masked_write_in_prefill_only():
    a, b, state, coefficients = bank()
    reader = schedules.ReplayReader(MASK)
    native = einsum(coefficients, b, "b s r, b r d -> b s d") * 2.0
    reference = torch.randn(1, 5, 16) + native
    out = reader.correction(state, coefficients, b, reader.encode_bank(a, b, 0), 0, 2.0, reference=reference)
    assert torch.allclose(reference + out, reference - native, atol=1e-5)
    assert torch.allclose(reader.records[0]["clean"], reference - native, atol=1e-5)
    assert torch.allclose(reader.records[0]["write"], einsum(coefficients * MASK, b, "b s r, b r d -> b s d") * 2.0, atol=1e-5)
    decode = reader.correction(state[:, :1], coefficients[:, :1], b, reader.encode_bank(a, b, 1), 1, 2.0, reference=reference[:, :1])
    assert 1 not in reader.records and torch.allclose(decode, (einsum(coefficients[:, :1] * (MASK - 1), b, "b s r, b r d -> b s d") * 2.0))
    with pytest.raises(ValueError):
        reader.correction(state, coefficients, b, reader.encode_bank(a, b, 2), 2, 2.0)


@pytest.fixture(scope="module")
def tiny_gemma():
    transformers = pytest.importorskip("transformers")
    config = transformers.Gemma2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                                       num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                                       max_position_embeddings=128, sliding_window=64, attn_implementation="eager")
    torch.manual_seed(0)
    return transformers.Gemma2ForCausalLM(config).eval()


def fresh_cache(model, length: int):
    from transformers.cache_utils import HybridCache

    return HybridCache(model.config, max_batch_size=1, max_cache_len=length, dtype=model.dtype, device=model.device)


def test_greedy_decode_matches_generate(tiny_gemma):
    prompt = torch.tensor([[2, 10, 11, 12, 13]])
    with torch.inference_mode():
        reference = tiny_gemma.generate(prompt, attention_mask=torch.ones_like(prompt), do_sample=False, max_new_tokens=6,
                                        cache_implementation="hybrid", pad_token_id=0)
        ours = schedules.greedy_decode(tiny_gemma, prompt, fresh_cache(tiny_gemma, prompt.shape[1] + 6),
                                       schedules.stop_ids(tiny_gemma), 6)
    expected = reference[0, prompt.shape[1]:].tolist()
    stop = schedules.stop_ids(tiny_gemma)
    expected = expected[: next((i for i, t in enumerate(expected) if t in stop), len(expected))]
    assert ours == expected and len(ours) > 0


def clean_prefill(model, prompt):
    """Prefill ``prompt[:, :-1]`` capturing per layer the residual entering the layer and the down_proj output."""
    layers = model.model.layers
    residuals, outputs, handles = {}, {}, []
    for index, layer in enumerate(layers):
        handles.append(layer.input_layernorm.register_forward_pre_hook(
            lambda _m, inputs, index=index: residuals.__setitem__(index, inputs[0].detach())))
        handles.append(layer.mlp.down_proj.register_forward_hook(
            lambda _m, _i, output, index=index: outputs.__setitem__(index, output.detach())))
    cache = fresh_cache(model, prompt.shape[1] + 4)
    with torch.inference_mode():
        model(prompt[:, :-1], attention_mask=torch.ones(1, prompt.shape[1] - 1, dtype=torch.long), past_key_values=cache,
              use_cache=True, cache_position=torch.arange(prompt.shape[1] - 1))
    for handle in handles:
        handle.remove()
    return cache, residuals, outputs


def test_replay_is_exact_at_the_first_affected_layer_and_identity_without_writes(tiny_gemma):
    model = tiny_gemma
    layers = model.model.layers
    prompt = torch.tensor([[2, 10, 11, 12, 13, 14]])
    positions = prompt.shape[1] - 1
    cache, residuals, outputs = clean_prefill(model, prompt)
    clean_keys = [k.clone() for k in cache.key_cache]
    cos, sin = model.model.rotary_emb(residuals[0], torch.arange(positions).unsqueeze(0))
    zero = {i: {"clean": outputs[i].float(), "write": torch.zeros_like(outputs[i]).float()} for i in range(len(layers))}
    with torch.inference_mode():
        delta = schedules.replay_prompt_cache(layers, cache, residuals, zero, cos, sin, positions)
    assert torch.equal(delta, torch.zeros_like(delta))
    for before, after in zip(clean_keys, cache.key_cache, strict=True):
        assert torch.allclose(before, after, atol=1e-6)
    # a write at layer 0 only: layer 1's recomputed keys/values must equal an exact forward with that write
    write = 0.5 * torch.randn(1, positions, model.config.hidden_size, generator=torch.Generator().manual_seed(1))
    records = {i: {"clean": outputs[i].float(), "write": (write if i == 0 else torch.zeros_like(write)).float()}
               for i in range(len(layers))}
    with torch.inference_mode():
        schedules.replay_prompt_cache(layers, cache, residuals, records, cos, sin, positions)
    handle = layers[0].mlp.down_proj.register_forward_hook(lambda _m, _i, output: output + write.to(output.dtype))
    exact = fresh_cache(model, prompt.shape[1] + 4)
    with torch.inference_mode():
        model(prompt[:, :-1], attention_mask=torch.ones(1, positions, dtype=torch.long), past_key_values=exact,
              use_cache=True, cache_position=torch.arange(positions))
    handle.remove()
    assert torch.allclose(cache.key_cache[1][:, :, :positions], exact.key_cache[1][:, :, :positions], atol=1e-5)
    assert torch.allclose(cache.value_cache[1][:, :, :positions], exact.value_cache[1][:, :, :positions], atol=1e-5)
    assert torch.allclose(cache.key_cache[0], clean_keys[0], atol=1e-6)  # layer 0 reads the embeddings
    # layer 2 is first-order: changed, and not exactly the re-evaluated forward
    assert not torch.allclose(cache.key_cache[2][:, :, :positions], clean_keys[2][:, :, :positions], atol=1e-5)
    assert not torch.allclose(cache.key_cache[2][:, :, :positions], exact.key_cache[2][:, :, :positions], atol=1e-6)


def test_schedule_mask_and_response_names():
    assert torch.equal(schedules.schedule_mask([False, True], "A", 4), torch.tensor([0., 0., 0., 0., 1., 1., 1., 1.]))
    assert torch.equal(schedules.schedule_mask([False, True], "none", 4), torch.zeros(8))
    assert torch.equal(schedules.schedule_mask([False, True], "C-none", 4), torch.zeros(8))
    assert schedules.SCHEDULES == ("none", "A", "B", "C") and schedules.DESIGN_B_LAYER_FROM == 8
