from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from official_d2l_compiler_finetuning_compiler_evaluation import (
    checkpoint_step,
    paired_comparison,
    reorder_plans,
    selection_key,
    wrong_document_plans,
)
from official_d2l_compiler_finetuning_compiler_training import (
    compile_trainable_batch,
    document_chunks,
    gradient_audit,
    split_digest,
    split_microbatches,
    trainability_audit,
)
from official_d2l_joint_finetuning_joint_training import (
    checkpoint_payload,
    compile_joint_batch,
    joint_gradient_audit,
)


class _Hypernetwork(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.bias_a = torch.nn.Parameter(torch.ones(1, 1, 1))
        self.bias_b = torch.nn.Parameter(torch.ones(1, 1, 1))
        self.config = SimpleNamespace(use_bias=True, latent_size=3)

    def get_head_bias(self) -> dict:
        return {"down_proj": {"A": self.bias_a, "B": self.bias_b}}


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_model = torch.nn.Linear(2, 2)
        self.ctx_encoder = torch.nn.Linear(2, 2)
        self.hypernet = _Hypernetwork()
        self.device = torch.device("cpu")

    def generate_weights(self, ids, mask, positions):
        assert positions is None and ids.shape == mask.shape
        batch = ids.shape[0]
        value = self.hypernet.weight * torch.ones(batch, 1, 1, 1)
        return {"down_proj": {"A": value, "B": value + 1}}, None


class _Backend:
    def __init__(self, *, chunks: int = 1) -> None:
        self.model = _Model()
        self.model_id = "fake"
        self.config = {"model": {"context_chunk_len": 32}}
        self.tokenizer = SimpleNamespace(pad_token_id=0)
        self.tokenize_context = lambda sample, tokenizer: {"ctx_ids": [[1, 2, 3]]}
        self.split_context = lambda sample, **kwargs: {
            "ctx_ids": [[1, 2, 3] for _ in range(chunks)]
        }

    @staticmethod
    def merge(generated, n_chunks, *, lora_bias, bias_scaler):
        assert n_chunks.tolist() == [1, 1]
        assert bias_scaler == 1.0
        return {
            "down_proj": {
                key: torch.cat(
                    (
                        generated["down_proj"][key],
                        lora_bias["down_proj"][key].unsqueeze(0).expand(2, -1, -1, -1),
                    ),
                    dim=2,
                )
                for key in ("A", "B")
            }
        }


def _plans() -> list[dict]:
    return [
        {
            "k": 1,
            "bank_document_ids": [f"d{index}"],
            "query": {"id": f"q{index}"},
        }
        for index in range(2)
    ]


def _documents() -> dict[str, dict]:
    return {
        "d0": {"text": "first", "content_hash": "hash-0"},
        "d1": {"text": "second", "content_hash": "hash-1"},
    }


def test_trainability_audit_accepts_only_hypernetwork_parameters() -> None:
    model = _Model()
    model.requires_grad_(False)
    model.hypernet.requires_grad_(True)
    audit = trainability_audit(model)
    assert audit["base_model"]["trainable_parameters"] == 0
    assert audit["context_encoder"]["trainable_parameters"] == 0
    assert audit["hypernetwork"]["trainable_parameters"] > 0
    model.base_model.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen target"):
        trainability_audit(model)


def test_document_chunks_enforces_k1_and_single_chunk() -> None:
    chunks, accounting = document_chunks(_Backend(), _plans(), _documents())
    assert len(chunks) == 2
    assert accounting == {"document_occurrences": 2, "context_chunks": 2, "source_tokens": 6}
    with pytest.raises(ValueError, match="exactly one compiler chunk"):
        document_chunks(_Backend(chunks=2), _plans(), _documents())
    invalid = [{**_plans()[0], "k": 2}]
    with pytest.raises(ValueError, match="K=1"):
        document_chunks(_Backend(), invalid, _documents())


def test_compiled_native_factors_retain_hypernetwork_gradients() -> None:
    backend = _Backend()
    backend.model.requires_grad_(False)
    backend.model.hypernet.requires_grad_(True)
    factors, accounting = compile_trainable_batch(
        backend,
        _plans(),
        _documents(),
        compiler_chunk_batch_size=1,
    )
    assert factors["down_proj"]["A"].requires_grad
    assert accounting["generated_factor_bytes"] > 0
    sum(tensor.sum() for part in factors.values() for tensor in part.values()).backward()
    assert backend.model.hypernet.weight.grad is not None
    assert torch.count_nonzero(backend.model.hypernet.weight.grad)
    assert gradient_audit(backend.model)["nonzero"]


def test_gradient_audit_rejects_frozen_model_gradients() -> None:
    model = _Model()
    model.requires_grad_(False)
    model.hypernet.requires_grad_(True)
    model.hypernet.weight.grad = torch.ones_like(model.hypernet.weight)
    model.base_model.weight.grad = torch.ones_like(model.base_model.weight)
    with pytest.raises(RuntimeError, match="target model"):
        gradient_audit(model)


def test_split_digest_covers_query_document_and_content_identity() -> None:
    original = split_digest(_plans(), _documents())
    changed = _documents()
    changed["d1"] = {**changed["d1"], "content_hash": "different"}
    assert original != split_digest(_plans(), changed)


def test_microbatch_split_preserves_order_and_effective_batch() -> None:
    plans = [{"query": {"id": str(index)}} for index in range(7)]
    parts = split_microbatches(plans, 3)
    assert [len(part) for part in parts] == [3, 3, 1]
    assert [plan["query"]["id"] for part in parts for plan in part] == [str(index) for index in range(7)]
    with pytest.raises(ValueError, match="positive"):
        split_microbatches(plans, 0)


def test_wrong_document_plans_never_reuse_the_source() -> None:
    plans = _plans()
    wrong = wrong_document_plans(plans)
    assert [plan["bank_document_ids"] for plan in wrong] == [["d1"], ["d0"]]
    with pytest.raises(ValueError, match="multiple documents"):
        wrong_document_plans([plans[0], {**plans[1], "bank_document_ids": ["d0"]}])


def test_checkpoint_selection_prioritizes_fresh_task_metrics() -> None:
    def summary(f1, exact, teacher):
        return {
            "summary": {
                "macro_over_k": {
                    "qa_f1": f1,
                    "exact_match": exact,
                    "teacher_response_similarity_f1": teacher,
                }
            }
        }

    assert selection_key(summary(0.8, 0.5, 0.9)) > selection_key(summary(0.79, 1.0, 1.0))
    assert selection_key(summary(0.8, 0.6, 0.7)) > selection_key(summary(0.8, 0.5, 1.0))
    assert checkpoint_step(Path("step-000500.pt")) == 500


def test_fresh_evaluation_restores_persisted_plan_order() -> None:
    plans = _plans()
    identity = [
        ["q1", 1, ["d1"]],
        ["q0", 1, ["d0"]],
    ]
    assert [plan["query"]["id"] for plan in reorder_plans(plans, identity)] == ["q1", "q0"]
    with pytest.raises(ValueError, match="training contract"):
        reorder_plans(plans, identity[:1])


def test_paired_comparison_reports_question_and_document_intervals() -> None:
    rows = [
        {
            "document_id": f"d{index // 2}",
            "scores": {
                "candidate": {"qa_f1": 0.5 + index / 10, "exact_match": 1.0, "rouge_l": 0.4},
                "reference": {"qa_f1": 0.5, "exact_match": 0.0, "rouge_l": 0.5},
            },
        }
        for index in range(4)
    ]
    result = paired_comparison(rows, "candidate", "reference", samples=100)
    assert result["questions"] == 4 and result["documents"] == 2
    assert result["deltas"]["qa_f1"]["mean"] == pytest.approx(0.15)
    assert len(result["deltas"]["qa_f1"]["document_cluster_bootstrap_95"]) == 2


def test_joint_compilation_preserves_generated_and_merged_factor_gradients() -> None:
    backend = _Backend()
    backend.model.requires_grad_(False)
    backend.model.hypernet.requires_grad_(True)
    compiled, _ = compile_joint_batch(
        backend,
        _plans(),
        _documents(),
        compiler_chunk_batch_size=1,
    )
    assert compiled.factors["down_proj"]["A"].requires_grad
    assert compiled.generated_factors["down_proj"]["A"].requires_grad
    assert compiled.bias_factors["down_proj"]["A"].shape[0] == 2
    sum(
        tensor.sum()
        for value in (compiled.factors, compiled.generated_factors)
        for part in value.values()
        for tensor in part.values()
    ).backward()
    assert torch.count_nonzero(backend.model.hypernet.weight.grad)


def test_joint_gradient_audit_requires_writer_and_reader_gradients() -> None:
    model = _Model()
    model.requires_grad_(False)
    model.hypernet.requires_grad_(True)
    reader = torch.nn.Linear(2, 2)
    model.hypernet.weight.grad = torch.ones_like(model.hypernet.weight)
    reader.weight.grad = torch.ones_like(reader.weight)
    audit = joint_gradient_audit(model, reader)
    assert audit["hypernetwork_gradient_tensors"] == 1
    assert audit["reader_gradient_tensors"] == 1
    reader.weight.grad = None
    with pytest.raises(RuntimeError, match="reader gradients"):
        joint_gradient_audit(model, reader)


def test_joint_checkpoint_contains_both_trainable_states() -> None:
    backend = _Backend()
    reader = torch.nn.Linear(2, 2)
    reader.bottleneck = 2
    reader.memory_width = 2
    reader.execution_site = "post-block"
    payload = checkpoint_payload(backend, reader, 500, {"version": 1}, {"train": []})
    assert payload["step"] == 500
    assert payload["representation"] == "rank-effect"
    assert set(payload) >= {"hypernet", "transition", "recipe", "identity"}
