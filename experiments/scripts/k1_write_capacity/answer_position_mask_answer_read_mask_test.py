import torch

from answer_position_mask_answer_read_mask import (
    ALL,
    CONDITIONS,
    LAYERS,
    block,
    blocked_rows,
    causal_additive_mask,
    document_mask,
    row_sets,
)


def test_row_sets_partition_teacher_sequence():
    span = {"start": 3, "end": 10}
    rows = row_sets(span, teacher_prompt_len=15, history_len=2)
    assert rows["document"] == list(range(3, 10))
    assert rows["question"] == [10, 11, 12, 13]
    assert rows["answer"] == [14, 15, 16]  # last prompt token plus two generated tokens


def test_blocked_rows_follow_condition_layers():
    rows = {"document": [1], "question": [5, 6], "answer": [7]}
    assert blocked_rows(CONDITIONS["teacher"], rows) == {}
    answer_all = blocked_rows(CONDITIONS["answer_read_all"], rows)
    assert set(answer_all) == set(range(LAYERS)) and all(v == [5, 6] for v in answer_all.values())
    late = blocked_rows(CONDITIONS["answer_read_from_18"], rows)
    assert late[0] == [5, 6, 7] and late[18] == [5, 6]
    assert blocked_rows(CONDITIONS["no_read"], rows)[25] == [5, 6, 7]
    assert blocked_rows(CONDITIONS["question_read_only"], rows)[3] == [7]


def test_block_sets_min_on_float_and_false_on_bool():
    mask = causal_additive_mask(5, torch.float32, "cpu")
    out = block(mask, [3, 4], [1, 2])
    low = torch.finfo(torch.float32).min
    assert out[0, 0, 3, 1] == low and out[0, 0, 4, 2] == low and out[0, 0, 3, 0] == 0
    assert mask[0, 0, 3, 1] == 0  # original untouched
    keep = torch.ones(1, 1, 5, 5, dtype=torch.bool).tril()
    assert not block(keep, [4], [0])[0, 0, 4, 0]


class TinyAttention(torch.nn.Module):
    def forward(self, hidden_states=None, attention_mask=None):
        scores = hidden_states @ hidden_states.transpose(-1, -2) + attention_mask[0, 0]
        return scores.softmax(-1) @ hidden_states


class TinyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = TinyAttention()


def test_document_mask_hook_reaches_attention_kwargs_and_is_removed():
    torch.manual_seed(0)
    layers = [TinyLayer(), TinyLayer()]
    hidden = torch.randn(1, 6, 4)
    mask = causal_additive_mask(6, torch.float32, "cpu")
    plain = layers[0].self_attn(hidden_states=hidden, attention_mask=mask)
    with document_mask(layers, {0: [4, 5]}, [1, 2]):
        hooked0 = layers[0].self_attn(hidden_states=hidden, attention_mask=mask)
        hooked1 = layers[1].self_attn(hidden_states=hidden, attention_mask=mask)
    after = layers[0].self_attn(hidden_states=hidden, attention_mask=mask)
    assert torch.equal(hooked1, plain) and torch.equal(after, plain)
    assert torch.equal(hooked0[:, :4], plain[:, :4]) and not torch.equal(hooked0[:, 4:], plain[:, 4:])


def test_conditions_cover_preregistered_arms():
    assert set(CONDITIONS) >= {"teacher", "answer_read_all", "answer_read_from_9", "answer_read_from_18",
                               "answer_read_until_18", "question_read_only", "no_read",
                               "answer_read_all_native", "no_read_native"}
    assert CONDITIONS["teacher"]["question_layers"] == ALL
