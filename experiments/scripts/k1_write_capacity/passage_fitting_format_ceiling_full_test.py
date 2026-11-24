from __future__ import annotations

import math

import pytest
from passage_fitting_format_ceiling_full import (
    DEFAULTS,
    build_plan,
    cosine_lr,
    elicited,
    fit_unit,
    is_passage_span,
    is_question,
    max_overlap,
    micro_batches,
    oracle_choice,
    position_weights,
    select_final,
    select_recipe,
    sentences_needing_spans,
    span_sentence,
)
from experiment_plan import validate_plan


def _suite(final: int = 6, dev: int = 3) -> dict:
    return {"suite_sha256": "abc",
            "suite": [{"document_id": f"f{i}", "role": "a2_final", "questions": []} for i in range(final)]
            + [{"document_id": "x", "role": "a3_only", "questions": []}],
            "development": [{"document_id": f"d{i}", "role": "a2_dev", "questions": []} for i in range(dev)]
            + [{"document_id": f"p{i}", "role": "dev_pool", "questions": []} for i in range(6)]}


def test_span_helpers():
    passage = "The Eiffel Tower opened in 1889. It is 330 metres tall."
    assert is_passage_span("1889", passage) and is_passage_span("the eiffel tower", passage)
    assert not is_passage_span("18", passage) and not is_passage_span("", passage)
    assert not is_passage_span("Paris", passage)
    sentences = ["The Eiffel Tower opened in 1889.", "It is 330 metres tall."]
    assert span_sentence("330 metres", sentences) == 1 and span_sentence("Paris", sentences) is None
    rows = [{"response_text": t} for t in ("1889", "Eiffel Tower", "1889", "330 metres")]
    assert sentences_needing_spans(rows, sentences, [0, 1]) == [0, 1]
    rows.append({"response_text": "opened"})
    assert sentences_needing_spans(rows, sentences, [0, 1]) == [1]
    assert elicited(["1889"], rows) and not elicited(["Gustave Eiffel"], rows)
    assert max_overlap("When did it open?", ["When did the tower open?"]) > 0.5
    assert max_overlap("Who built it?", []) == 0.0


def test_full_batch_weights_reproduce_position_mean():
    lengths = [[30], [10], [5]]
    weights = position_weights(lengths)
    means = [2.0, 4.0, 1.0]
    full = (2.0 * 30 + 4.0 * 10 + 1.0 * 5) / 45
    assert math.isclose(sum(m * w for m, w in zip(means, weights, strict=True)), full)
    assert micro_batches(5, 2) == [[0, 1], [2, 3], [4]]


def test_cosine_truncation_keeps_trajectory():
    assert cosine_lr(1e-4, 1, 400) == pytest.approx(1e-4)
    assert cosine_lr(1e-4, 201, 400) == pytest.approx(5e-5)
    assert cosine_lr(1e-4, 400, 400) < 1e-8 * 1e-4 * 1e6
    assert [cosine_lr(1e-4, s, 400) for s in range(1, 101)] == [cosine_lr(1e-4, s, 400) for s in range(1, 101)]


def _dev_record(rank, lr, step_metrics, coverage="oracle"):
    evaluations = {"0": {"fit": {"agreement_f1": 0.0, "kl": 1.0}}}
    evaluations.update({str(s): {"fit": {"agreement_f1": a, "kl": k}} for s, (a, k) in step_metrics.items()})
    return {"unit": {"args": {"rank": rank, "lr": lr, "coverage": coverage}}, "result": {"evaluations": evaluations}}


def test_selection_picks_smallest_passing_step_then_lr_and_flags_fallback():
    records = [
        _dev_record(8, 1e-4, {50: (0.96, 0.01), 100: (0.99, 0.005)}),
        _dev_record(8, 3e-5, {50: (0.97, 0.01), 100: (0.98, 0.004)}),
        _dev_record(8, 1e-5, {50: (0.90, 0.05), 100: (0.96, 0.01)}),
        _dev_record(64, 1e-4, {50: (0.80, 0.05), 100: (0.90, 0.03)}),
        _dev_record(64, 1e-5, {50: (0.70, 0.05), 100: (0.85, 0.03)}),
        _dev_record(8, 1e-4, {50: (0.10, 0.9)}, coverage="blind"),  # ignored: tuning uses oracle coverage
    ]
    selection = select_recipe(records, [8, 64])
    assert (selection["8"]["step"], selection["8"]["lr"], selection["8"]["gate_met"]) == (50, 3e-5, True)
    assert (selection["64"]["step"], selection["64"]["lr"], selection["64"]["gate_met"]) == (100, 1e-4, False)
    with pytest.raises(ValueError, match="rank 16"):
        select_recipe(records, [16])


def test_plan_and_expansion():
    config = {**DEFAULTS, "ranks": [8, 16, 32, 64]}
    plan = build_plan(_suite(), "s.json", "m.json", "e.json", config=config, dev_documents=2, final_documents=4)
    validate_plan(plan)
    kinds = [u["kind"] for u in plan["units"]]
    assert kinds.count("retention") == 1 and kinds.count("prep") == 6  # 2 dev + 4 final
    assert kinds.count("fit") == 2 * 4 * 3 and all(u["args"]["coverage"] == "oracle" for u in plan["units"]
                                                   if u["kind"] == "fit")
    records = {}
    for unit in plan["units"]:
        if unit["kind"] == "fit":
            metrics = {50: (0.96, 0.01), 100: (0.99, 0.005), 200: (0.99, 0.004), 400: (0.99, 0.004)}
            if unit["args"]["rank"] == 64:
                metrics[50] = (0.5, 0.5)
            records[unit["id"]] = {"unit": unit, "result": {"evaluations": {str(k): {"fit": {"agreement_f1": a, "kl": b}}
                                                                            for k, (a, b) in metrics.items()}}}
    final = select_final(plan, {"prep": {}, "dev": records})
    assert {u["args"]["stop_step"] for u in final} == {200}  # largest selected step 100, plus one checkpoint
    assert all(u["args"]["checkpoints"] == [5, 10, 25, 50, 100, 200] for u in final)
    per_doc = [u for u in final if u["args"]["document_id"] == "f0"]
    assert sorted((u["args"]["rank"], u["args"]["coverage"]) for u in per_doc) == [
        (8, "blind"), (8, "oracle"), (8, "oracle_relaxed"), (16, "oracle"), (32, "blind"), (32, "oracle"),
        (32, "oracle_relaxed"), (64, "oracle")]
    wrong = [u for u in final if u["args"]["wrongdoc"]]
    assert {(u["args"]["rank"], u["args"]["coverage"]) for u in wrong} == {(8, "oracle")}
    assert next(u for u in wrong if u["args"]["document_id"] == "f3")["args"]["wrongdoc"] == "f0"
    assert {u["priority"] for u in final if u["args"]["rank"] in (8, 32)} == {2}
    assert len({u["id"] for u in final}) == len(final)
    assert fit_unit("d", 8, "oracle", 1e-4, stage="dev", stop_step=100, checkpoints=[50, 100, 200])["args"][
        "checkpoints"] == [50, 100]


def test_oracle_choice_prefers_covering_rows():
    rows = [{"response_text": t} for t in ("Sanchez", "Change Nothing by Sanchez", "Change Nothing", "Home")]
    chosen = oracle_choice(rows, ["Change Nothing"])
    assert [r["response_text"] for r in chosen] == ["Change Nothing", "Change Nothing by Sanchez"]
    assert oracle_choice([{"response_text": "Home"}], ["Change Nothing"]) == []


def test_is_question_rejects_answers_and_statements():
    assert is_question("Who closed the government mint?", "Emperor Gaozu")
    assert not is_question("Emperor Gaozu", "Emperor Gaozu")
    assert not is_question("Did Emperor Gaozu close the mint?", "Emperor Gaozu")
    assert not is_question("The Han dynasty inherited the coin type from the Qin.")
    assert not is_question("What?\nAnswer: x?")


def test_smoke_plans_never_touch_a2_documents():
    plan = build_plan(_suite(), "s.json", "m.json", "e.json", config={**DEFAULTS}, dev_documents=2,
                      final_documents=3, smoke=True)
    assert plan["config"]["dev_documents"] == ["p0", "p1"] and plan["config"]["final_documents"] == ["p2", "p3", "p4"]
    assert not any(u["args"].get("document_id", "").startswith(("f", "d")) for u in plan["units"])


def test_extended_parity_is_structural():
    import torch
    from passage_fitting_format_ceiling_full import extended_parity

    doc_a, doc_b = torch.randn(2, 8, 6), torch.randn(2, 8, 4)
    bias_a, bias_b = torch.randn(2, 8, 6), torch.randn(2, 8, 4)
    reference = {"down_proj": {"A": torch.cat([doc_a, bias_a], 1)[None], "B": torch.cat([doc_b, bias_b], 1)[None]}}

    class Fake:
        def __init__(self, new_b):
            self.a = torch.cat([doc_a, torch.randn(2, 4, 6), bias_a], 1)
            self.b = torch.cat([doc_b, new_b, bias_b], 1)

        def down_state(self):
            return {"down_proj": {"A": self.a[None], "B": self.b[None]}}

    assert extended_parity(Fake(torch.zeros(2, 4, 4)), reference) == {
        "strict": False, "shared_rows_exact": True, "appended_rows": 4}
    assert not extended_parity(Fake(torch.full((2, 4, 4), 1e-3)), reference)["shared_rows_exact"]
