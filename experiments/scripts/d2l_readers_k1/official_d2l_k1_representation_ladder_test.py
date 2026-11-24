from __future__ import annotations

from pathlib import Path

import pytest
import torch

from official_d2l_k1_representation_evaluation import (
    query_ids_from_artifact,
    zero_document_representation,
)
from official_d2l_k1_representation_ladder import (
    ActiveMemoryTransition,
    CompiledRepresentationBatch,
    NonlinearRankOperator,
    QueryConditionedLatentEncoder,
    capture_pre_head_latents,
    combine_captured_latents,
    combine_projected_sources,
    factor_rank_effects,
    installed_representation_transition,
    last_prompt_layer_states,
    make_representation_transition,
)
from official_d2l_k1_representation_training import (
    _pad_compact_targets,
    gold_answer_loss,
    make_response_replays,
    reader_payload,
)
from post_block_transitions import factor_delta


def test_active_memory_transition_is_exact_identity_for_both_memory_layouts() -> None:
    transition = ActiveMemoryTransition(width=5, memory_width=3, layers=2, bottleneck=4)
    state = torch.randn(2, 6, 5)
    bias = torch.randn_like(state)
    static = torch.randn(2, 8, 3)
    tokenwise = torch.randn(2, 6, 8, 3)
    assert torch.equal(transition(state, static, bias, 1), state)
    assert torch.equal(transition(state, tokenwise, bias, 1), state)


def test_active_memory_transition_uses_memory_after_output_is_enabled() -> None:
    transition = ActiveMemoryTransition(width=5, memory_width=3, layers=2, bottleneck=4)
    with torch.no_grad():
        transition.output_projection.weight.fill_(0.1)
    state = torch.randn(2, 6, 5)
    bias = torch.randn_like(state)
    memory = torch.randn(2, 8, 3)
    actual = transition(state, memory, bias, 0)
    assert not torch.equal(actual, state)
    assert not torch.equal(actual, transition(state, torch.zeros_like(memory), bias, 0))


def test_active_memory_transition_masks_padded_static_memory() -> None:
    transition = ActiveMemoryTransition(width=5, memory_width=3, layers=1, bottleneck=4)
    with torch.no_grad():
        transition.output_projection.weight.fill_(0.1)
    state = torch.randn(2, 6, 5)
    bias = torch.randn_like(state)
    memory = torch.randn(2, 4, 3)
    mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
    changed = memory.clone()
    changed[~mask] = 1000
    torch.testing.assert_close(
        transition(state, memory, bias, 0, mask),
        transition(state, changed, bias, 0, mask),
    )


def test_query_conditioned_latent_encoder_is_exact_identity_at_initialization() -> None:
    encoder = QueryConditionedLatentEncoder(query_width=7, latent_width=5, layers=3, bottleneck=4)
    latents = torch.randn(2, 3, 8, 5)
    query_states = torch.randn(2, 3, 7)
    assert torch.equal(encoder(latents, query_states), latents)


def test_query_conditioned_latent_encoder_uses_document_and_request_after_enabled() -> None:
    encoder = QueryConditionedLatentEncoder(query_width=7, latent_width=5, layers=3, bottleneck=4)
    with torch.no_grad():
        encoder.output_projection.weight.fill_(0.1)
    latents = torch.randn(2, 3, 8, 5)
    query_states = torch.randn(2, 3, 7)
    actual = encoder(latents, query_states)
    assert not torch.equal(actual, latents)
    assert not torch.equal(actual, encoder(latents, torch.zeros_like(query_states)))
    assert not torch.equal(actual, encoder(torch.zeros_like(latents), query_states))


def test_last_prompt_layer_states_handles_left_and_right_padding_and_detaches() -> None:
    hidden_states = tuple(
        torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3).requires_grad_() + layer * 100
        for layer in range(3)
    )
    attention_mask = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1]])
    actual = last_prompt_layer_states(hidden_states, attention_mask, layers=2)
    expected = torch.stack(
        (
            torch.stack((hidden_states[1][0, 1], hidden_states[1][1, 3])),
            torch.stack((hidden_states[2][0, 1], hidden_states[2][1, 3])),
        ),
        dim=1,
    )
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad


def test_query_conditioned_latent_encoder_validates_shapes() -> None:
    encoder = QueryConditionedLatentEncoder(query_width=7, latent_width=5, layers=3, bottleneck=4)
    with pytest.raises(ValueError, match="latents"):
        encoder(torch.randn(2, 8, 5), torch.randn(2, 3, 7))
    with pytest.raises(ValueError, match="query states"):
        encoder(torch.randn(2, 3, 8, 5), torch.randn(2, 7))
    with pytest.raises(ValueError, match="embeddings"):
        last_prompt_layer_states((torch.randn(2, 4, 7),), torch.ones(2, 4), layers=2)


def test_representation_factory_changes_only_memory_input_width() -> None:
    summed = make_representation_transition("summed-effect", 7, 11, 3, 5)
    ranks = make_representation_transition("rank-effect", 7, 11, 3, 5)
    latent = make_representation_transition("pre-head-latent", 7, 11, 3, 5)
    projected = make_representation_transition("projected-source", 7, 11, 3, 5)
    assert summed.memory_width == ranks.memory_width == 7
    assert latent.memory_width == 11
    assert projected.memory_width == 11
    assert summed.bottleneck == ranks.bottleneck == latent.bottleneck == 5
    with pytest.raises(ValueError, match="unknown representation"):
        make_representation_transition("source-text", 7, 11, 3, 5)


def test_rank_effects_sum_to_standard_factor_delta() -> None:
    inputs = torch.randn(2, 5, 7)
    a = torch.randn(2, 3, 7)
    b = torch.randn(2, 3, 11)
    per_rank = factor_rank_effects(inputs, a, b, 2.5)
    torch.testing.assert_close(per_rank.sum(dim=2), factor_delta(inputs, a, b, 2.5))


def test_nonlinear_rank_operator_is_jointly_permutation_invariant() -> None:
    operator = NonlinearRankOperator(output_width=7, layers=3, bottleneck=8, heads=2, feedforward_width=16)
    coefficients = torch.randn(2, 5, 4)
    b_rows = torch.randn(2, 4, 7)
    permutation = torch.tensor([2, 0, 3, 1])
    expected = operator(coefficients, b_rows, 1)
    actual = operator(coefficients[:, :, permutation], b_rows[:, permutation], 1)
    torch.testing.assert_close(actual, expected)


def test_nonlinear_rank_operator_depends_on_pairing() -> None:
    operator = NonlinearRankOperator(output_width=7, layers=3, bottleneck=8, heads=2, feedforward_width=16)
    coefficients = torch.randn(2, 5, 4)
    b_rows = torch.randn(2, 4, 7)
    expected = operator(coefficients, b_rows, 1)
    mispaired = operator(coefficients[:, :, torch.tensor([2, 0, 3, 1])], b_rows, 1)
    assert not torch.equal(mispaired, expected)


def test_pre_head_capture_is_observational() -> None:
    head = torch.nn.Linear(3, 2, bias=False)
    inputs = torch.randn(2, 4, 1, 8, 3)
    expected = head(inputs)
    with capture_pre_head_latents(type("Hypernet", (), {"head": head})()) as captured:
        actual = head(inputs)
    torch.testing.assert_close(actual, expected)
    assert len(captured) == 1
    assert torch.equal(captured[0], inputs)
    inputs.zero_()
    assert torch.count_nonzero(captured[0]) > 0


def test_captured_latents_restore_microbatch_order_and_enforce_one_chunk() -> None:
    first = torch.randn(2, 3, 1, 8, 5)
    second = torch.randn(1, 3, 1, 8, 5)
    actual = combine_captured_latents([first, second], [1, 1, 1])
    torch.testing.assert_close(actual, torch.cat((first, second), dim=0)[:, :, 0])
    with pytest.raises(ValueError, match="one compiler chunk"):
        combine_captured_latents([first], [2])


def test_projected_sources_restore_layers_and_padding_masks() -> None:
    first = [torch.randn(2, 4, 5) for _ in range(3)]
    second = [torch.randn(1, 2, 5) for _ in range(3)]
    projected, mask = combine_projected_sources(
        [first, second],
        [torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]), torch.tensor([[1, 1]])],
        layers=3,
    )
    assert projected.shape == (3, 3, 3, 5)
    assert mask.tolist() == [[True, True, True], [True, True, False], [True, True, False]]
    torch.testing.assert_close(projected[0, :, :3], torch.stack(first, dim=1)[0, :, :3])


class _MLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down_proj = torch.nn.Linear(4, 3, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.down_proj(value)


class _Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _MLP()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value[..., :3] + self.mlp(value)


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = _Layer()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layer(value)


def _compiled_batch() -> CompiledRepresentationBatch:
    generated = {
        "down_proj": {
            "A": torch.randn(2, 1, 2, 4),
            "B": torch.randn(2, 1, 2, 3),
        }
    }
    bias = {
        "down_proj": {
            "A": torch.randn(2, 1, 2, 4),
            "B": torch.randn(2, 1, 2, 3),
        }
    }
    factors = {
        "down_proj": {
            "A": torch.cat((generated["down_proj"]["A"], bias["down_proj"]["A"]), dim=2),
            "B": torch.cat((generated["down_proj"]["B"], bias["down_proj"]["B"]), dim=2),
        }
    }
    return CompiledRepresentationBatch(
        factors,
        generated,
        bias,
        torch.randn(2, 1, 2, 5),
        torch.randn(2, 1, 4, 5),
        torch.tensor([[True, True, True, False], [True, True, False, False]]),
    )


@pytest.mark.parametrize(
    "representation",
    ["summed-effect", "rank-effect", "pre-head-latent", "projected-source"],
)
def test_installed_zero_reader_exactly_matches_native_factors(representation: str) -> None:
    model = _Model()
    compiled = _compiled_batch()
    transition = make_representation_transition(representation, 3, 5, 1, 2)
    value = torch.randn(2, 4, 4)
    base = model(value)
    expected = base + factor_delta(
        value,
        compiled.factors["down_proj"]["A"][:, 0],
        compiled.factors["down_proj"]["B"][:, 0],
        1.0,
    )
    with installed_representation_transition(
        model,
        [model.layer],
        compiled,
        transition,
        representation,
        1.0,
    ):
        actual = model(value)
    torch.testing.assert_close(actual, expected)


def test_latent_only_control_removes_generated_native_factors() -> None:
    model = _Model()
    compiled = _compiled_batch()
    transition = make_representation_transition("pre-head-latent", 3, 5, 1, 2)
    value = torch.randn(2, 4, 4)
    base = model(value)
    expected = base + factor_delta(
        value,
        compiled.bias_factors["down_proj"]["A"][:, 0],
        compiled.bias_factors["down_proj"]["B"][:, 0],
        1.0,
    )
    with installed_representation_transition(
        model,
        [model.layer],
        compiled,
        transition,
        "pre-head-latent",
        1.0,
        apply_document_factors=False,
    ):
        actual = model(value)
    torch.testing.assert_close(actual, expected)


def test_gold_answer_loss_uses_per_example_prompt_offsets() -> None:
    logits = torch.full((2, 6, 4), -10.0)
    logits[0, 1, 2] = 10.0
    logits[0, 2, 3] = 10.0
    logits[1, 3, 1] = 10.0
    targets = {
        "student_prompt_lengths": [2, 4],
        "response_lengths": [2, 1],
        "hard_targets": torch.tensor([[2, 3], [1, -100]]),
    }
    assert float(gold_answer_loss(logits, targets)) < 1e-6


def test_response_replays_share_detached_generated_prefixes() -> None:
    responses = [torch.tensor([7, 8, 9]), torch.tensor([6])]
    student = make_response_replays(
        [torch.tensor([1, 2]), torch.tensor([3])],
        responses,
    )
    teacher = make_response_replays(
        [torch.tensor([4, 5, 2]), torch.tensor([4, 3])],
        responses,
    )
    assert student[0].tolist() == [1, 2, 7, 8]
    assert teacher[0].tolist() == [4, 5, 2, 7, 8]
    assert student[1].tolist() == [3]
    assert teacher[1].tolist() == [4, 3]
    with pytest.raises(ValueError, match="nonempty and aligned"):
        make_response_replays([torch.tensor([1])], [])
    with pytest.raises(ValueError, match="nonempty one-dimensional"):
        make_response_replays([torch.tensor([1])], [torch.tensor([])])


def test_compact_targets_pad_response_axis_before_concatenation() -> None:
    first = {
        "indices": torch.tensor([[[1, 2], [2, 3]]], dtype=torch.int32),
        "log_probs": torch.randn(1, 2, 2),
        "tail_log_prob": torch.randn(1, 2),
    }
    second = {
        "indices": torch.tensor([[[3, 4]]], dtype=torch.int32),
        "log_probs": torch.randn(1, 1, 2),
        "tail_log_prob": torch.randn(1, 1),
    }
    combined = _pad_compact_targets([first, second], 2)
    assert combined["indices"].shape == (2, 2, 2)
    assert combined["log_probs"].shape == (2, 2, 2)
    assert combined["tail_log_prob"].shape == (2, 2)
    assert torch.count_nonzero(combined["indices"][1, 1]) == 0


def test_reader_payload_persists_representation_contract() -> None:
    transition = make_representation_transition("pre-head-latent", 7, 11, 3, 5)
    payload = reader_payload(transition, "pre-head-latent", 500)
    assert payload["representation"] == "pre-head-latent"
    assert payload["memory_width"] == 11
    assert payload["execution_site"] == "post-block"
    assert payload["selected_step"] == 500


def test_zero_document_representation_retains_only_shared_bias() -> None:
    compiled = _compiled_batch()
    zero = zero_document_representation(compiled)
    assert torch.equal(zero.factors["down_proj"]["A"], compiled.bias_factors["down_proj"]["A"])
    assert torch.count_nonzero(zero.generated_factors["down_proj"]["A"]) == 0
    assert torch.count_nonzero(zero.pre_head_latents) == 0


def test_query_ids_from_artifact_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "rows.json"
    path.write_text('{"rows": [{"query_id": "q1"}, {"query_id": "q2"}]}')
    assert query_ids_from_artifact(path) == ["q1", "q2"]
    path.write_text('{"rows": [{"query_id": "q1"}, {"query_id": "q1"}]}')
    with pytest.raises(ValueError, match="duplicates"):
        query_ids_from_artifact(path)
