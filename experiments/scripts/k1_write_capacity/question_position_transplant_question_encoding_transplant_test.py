
import pytest
import torch

from question_position_transplant_question_encoding_transplant import CONDITIONS, LAYERS, position_sets, transplant_hooks


def test_position_sets_map_prompt_and_history_with_constant_offset():
    span = {"start": 3, "end": 6, "prefix": 3, "suffix": 8}
    sets = position_sets(span, teacher_prompt_len=14, student_prompt_len=11, history_len=2)
    assert sets["offset"] == 3
    assert sets["question"] == ([0, 1, 2, 6, 7, 8, 9, 10, 11, 12], list(range(10)))
    assert sets["history"] == ([14, 15], [11, 12])
    assert sets["question_history"][1] == [*range(10), 11, 12]
    assert sets["all"] == ([0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], list(range(13)))


class Layer(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, hidden):
        return (hidden * self.scale,)


def test_transplant_hooks_overwrite_only_requested_positions_and_boundaries_and_clean_up():
    layers = torch.nn.ModuleList([Layer(2.0), Layer(3.0), Layer(5.0)])
    source = [None, torch.full((1, 6, 2), 100.0), torch.full((1, 6, 2), 200.0), torch.full((1, 6, 2), 300.0)]
    x = torch.ones(1, 4, 2)

    def run(h):
        for layer in layers:
            h = layer(h)[0]
        return h

    with transplant_hooks(layers, boundaries=[2], student_idx=[0, 3], source_hidden=source, source_idx=[1, 5]):
        out = run(x)
    # boundary 2 = output of layer index 1: positions 0 and 3 become 200, then layer 2 scales by 5
    assert torch.equal(out[0, 0], torch.tensor([1000.0, 1000.0])) and torch.equal(out[0, 3], torch.tensor([1000.0, 1000.0]))
    assert torch.equal(out[0, 1], torch.tensor([30.0, 30.0]))
    assert all(not layer._forward_hooks for layer in layers)
    with pytest.raises(RuntimeError), transplant_hooks(layers, [1], [0], source, [0]):
        raise RuntimeError("interrupted")
    assert all(not layer._forward_hooks for layer in layers)


def test_condition_table_is_well_formed():
    assert CONDITIONS["native"]["boundaries"] == []
    assert CONDITIONS["q_from_0"]["boundaries"] == list(range(1, LAYERS))
    assert CONDITIONS["q_until_18"]["boundaries"] == list(range(1, 18)) and CONDITIONS["q_from_18"]["boundaries"][0] == 18
    assert CONDITIONS["teacher_all_positions"]["positions"] == "all"
    assert CONDITIONS["teacher_all_positions"]["boundaries"][-1] == LAYERS
