"""CPU tests for the write-site reader report: selection arithmetic and the confirmation merge."""

import json

import pytest
from distractor_curves_core import K_VALUES
from write_site_readers_report import (
    accounting,
    confirmation,
    merge_rows,
    select,
    selection_report,
)


def arm_c_parameters() -> int:
    """Arm C's parameter count at the recipe shape, so the fixture follows the module."""
    from write_site_readers_readers import parameter_count
    from write_site_readers_report import reader_module

    return parameter_count(reader_module("C"))


def scores(**values):
    return {name: {"f1": value, "em": value} for name, value in values.items()}


def row(source: str, k: int, arm_scores: dict, condition: str = "random", query: str | None = None) -> dict:
    return {"query_id": query or f"{source}q", "source_id": source, "bank_id": f"{source}:{condition}:{k}",
            "condition": condition, "k": k, "responses": {arm: "x" for arm in arm_scores}, "scores": scores(**arm_scores)}


def unit_entry(unit: dict, rows: list[dict], per_question: list[dict] | None = None) -> dict:
    return {"unit": unit, "result": {"sources": [{"source_id": "s", "rows": rows, "per_question": per_question or []}]}}


FLOPS = {"A": 500.0, "B": 30.0, "C": 400.0, "D": 190.0}


def selection_rows(reader_f1: dict[str, dict[int, float]], base: dict[int, dict[str, float]],
                   sources=("d1", "d2", "d3")) -> list[dict]:
    """Rows over three sources with fixed base arms and per-K reader F1 (same on every source)."""
    rows = []
    for source in sources:
        for k in K_VALUES:
            values = dict(base[k])
            values.update({name: f1[k] for name, f1 in reader_f1.items()})
            rows.append(row(source, k, values))
    return rows


BASE = {1: {"native": 0.5, "icl": 0.8}, **{k: {"native": 0.3, "normalized_b0.5": 0.4, "icl": 0.8} for k in (2, 4, 8, 16)}}


def test_selection_switches_reference_at_k1_and_ranks_by_macro_difference():
    readers = {
        # A: strong at K=1 only. Against native at K=1: +0.3; against normalized at 4, 8: 0, 0 -> 0.1
        "A_best": {1: 0.8, 2: 0.4, 4: 0.4, 8: 0.4, 16: 0.4}, "A_last": {k: 0.1 for k in K_VALUES},
        # C: +0.1 at K=1 (vs native 0.5), +0.2 at K=4 and K=8 (vs normalized 0.4) -> 0.1667
        "C_best": {1: 0.6, 2: 0.6, 4: 0.6, 8: 0.6, 16: 0.6}, "C_last": {k: 0.59 for k in K_VALUES},
    }
    choice = select(selection_rows(readers, BASE), FLOPS)
    assert choice["arms"]["A"]["checkpoint"] == "best" and choice["arms"]["C"]["checkpoint"] == "best"
    assert choice["arms"]["A"]["score"] == pytest.approx(0.1)
    assert choice["arms"]["C"]["score"] == pytest.approx((0.1 + 0.2 + 0.2) / 3)
    assert choice["arms"]["C"]["per_k"]["1"]["reference"] == "native"
    assert choice["arms"]["C"]["per_k"]["4"]["reference"] == "normalized_b0.5"
    assert choice["ranking"] == ["C", "A"] and choice["selected"] == {"arm": "C", "checkpoint": "best",
                                                                     "reader": "C_best", "score": choice["arms"]["C"]["score"]}
    assert not choice["arms"]["B"]["trained"] and not choice["arms"]["D"]["trained"]


def test_selection_ties_go_to_the_cheaper_arm_and_checkpoint_ties_to_best():
    same = {1: 0.6, 2: 0.5, 4: 0.5, 8: 0.5, 16: 0.5}
    readers = {"C_best": same, "C_last": same, "B_best": same, "B_last": {k: v - 0.01 for k, v in same.items()},
               "D_best": same, "D_last": {k: v + 0.5 for k, v in same.items()}}  # D: last wins by macro F1
    choice = select(selection_rows(readers, BASE), FLOPS)
    assert choice["arms"]["C"]["checkpoint"] == "best"  # tie -> best
    assert choice["arms"]["D"]["checkpoint"] == "last" and choice["arms"]["D"]["reader"] == "D_last"
    # B and C have equal scores; B is cheaper. D's last checkpoint scores higher than both.
    assert choice["arms"]["B"]["score"] == pytest.approx(choice["arms"]["C"]["score"])
    assert choice["ranking"] == ["D", "B", "C"]


def test_checkpoint_rule_ignores_rows_lacking_the_arm_and_is_macro_over_k():
    rows = [row("d1", 1, {"X_best": 0.9, "X_last": 0.2, "native": 0.1})] \
        + [row("d1", 4, {"X_best": 0.1, "X_last": 0.9, "normalized_b0.5": 0.1}, query=f"q{i}") for i in range(5)] \
        + [row("d1", 8, {"X_best": 0.5, "X_last": 0.5, "normalized_b0.5": 0.1})]
    choice = select(rows, {"X": 1.0}, arms=("X",))
    assert choice["arms"]["X"]["checkpoint"] == "last"  # (0.2 + 0.9 + 0.5) / 3 > (0.9 + 0.1 + 0.5) / 3


def test_accounting_matches_the_distractor_curves_checkpoint_shape():
    account = accounting("A", trained_parameters=443200)
    assert account["parameters"] == 443200
    assert set(account["per_token_flops"]) == {"1", "8", "16"}
    assert account["per_token_flops"]["8"]["reader"] > account["per_token_flops"]["1"]["reader"]
    assert 0 < account["per_token_flops"]["1"]["reader_over_backbone"] < 1
    with pytest.raises(ValueError):
        accounting("B", trained_parameters=1)


def test_selection_report_reads_state_and_tags_suites(tmp_path):
    (tmp_path / "units").mkdir()
    readers = {"C_best": {k: 0.7 for k in K_VALUES}, "C_last": {k: 0.6 for k in K_VALUES}}
    for index, suite in enumerate(("selection", "distractor_curves-dev")):
        sources = tuple(f"{suite}-d{i}" for i in range(3))  # real suites hold disjoint documents
        (tmp_path / "units" / f"selection-base-{suite}-000.json").write_text(json.dumps(unit_entry(
            {"id": f"b{index}", "suite": suite, "readers": {}},
            selection_rows({}, BASE, sources))))
        (tmp_path / "units" / f"selection-reader-{suite}-000.json").write_text(json.dumps(unit_entry(
            {"id": f"r{index}", "suite": suite, "readers": {"C_best": "p", "C_last": "p"}},
            selection_rows(readers, {k: {} for k in K_VALUES}, sources))))
    (tmp_path / "units" / "train-C-seed216.json").write_text(json.dumps(
        {"unit": {"id": "train-C-seed216"}, "result": {"arm": "C", "accounting": {"reader_parameters": arm_c_parameters()}}}))
    report = selection_report(tmp_path)
    assert report["selected"]["arm"] == "C" and report["selected"]["checkpoint"] == "best"
    assert set(report["table"]) == {"selection", "distractor_curves-dev", "pooled"}
    # rows merged across base and reader units: the same (question, bank) carries both arms
    assert report["table"]["pooled"]["random:4"]["f1"]["C_best"] == pytest.approx(0.7)
    assert report["table"]["pooled"]["random:4"]["f1"]["normalized_b0.5"] == pytest.approx(0.4)
    assert report["table"]["pooled"]["random:4"]["questions"] == 6
    assert report["table"]["selection"]["random:4"]["questions"] == 3 and report["rows"] == {"distractor_curves-dev": 15, "selection": 15}
    assert report["accounting"]["C"]["parameters"] == arm_c_parameters() and report["units"] == 4


def test_confirmation_merges_recorded_base_rows_with_new_reader_rows(tmp_path):
    distractor_curves, confirmation_state = tmp_path / "distractor_curves", tmp_path / "confirmation"
    (distractor_curves / "units").mkdir(parents=True)
    (confirmation_state / "units").mkdir(parents=True)
    base_rows, reader_rows, per_q_base = [], [], []
    for source in ("d1", "d2", "d3"):
        for k in K_VALUES:
            base_rows.append(row(source, k, {"native": 0.4, "normalized_b0.5": 0.5, "icl": 0.9}))
            reader_rows.append(row(source, k, {"reader_s216": 0.7, "reader_s1216": 0.8, "reader_s2216": 0.9}))
        per_q_base.append({"query_id": f"{source}q", "responses": {"base": "x"}, "scores": scores(base=0.1)})
    (distractor_curves / "units" / "gemma2-test-base-000.json").write_text(json.dumps(unit_entry(
        {"id": "gemma2-test-base-000", "readers": {}}, base_rows, per_q_base)))
    (distractor_curves / "units" / "gemma2-test-reader-000.json").write_text("not read")
    readers = {f"reader_s{s}": "p" for s in (216, 1216, 2216)}
    (confirmation_state / "units" / "confirm-reader-distractor_curves-test-000.json").write_text(json.dumps(unit_entry(
        {"id": "confirm-reader-distractor_curves-test-000", "suite": "distractor_curves-test", "readers": readers}, reader_rows)))
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"documents": [{"document_id": d, "article": a} for d, a in
                                               (("d1", "A"), ("d2", "A"), ("d3", "B"))]}))
    report = confirmation(confirmation_state, distractor_curves, suite, None)
    assert report["seeds"] == ["reader_s1216", "reader_s216", "reader_s2216"]
    assert report["table"]["random:4"]["f1"]["reader_mean"] == pytest.approx(0.8)
    assert report["table"]["random:4"]["f1"]["icl"] == 0.9 and set(report["table"]) == {f"random:{k}" for k in K_VALUES}
    assert report["preregistered_reader_rule"]["holds"]
    assert report["contrasts"]["random:8:reader_mean-normalized_b0.5"]["difference"] == pytest.approx(0.3)
    assert report["contrasts"]["random:8:reader_mean-normalized_b0.5"]["article_cluster"]["clusters"] == 2
    assert report["base_units"] == 1 and report["reader_units"] == 1 and report["rows"] == 3 * len(K_VALUES)


def test_merge_rows_keeps_suite_and_per_question():
    rows, per_question, units = merge_rows([
        unit_entry({"id": "a", "suite": "selection"}, [row("d", 1, {"native": 0.1})], [{"query_id": "dq", "responses": {"base": "x"}, "scores": scores(base=0.1)}]),
        unit_entry({"id": "b", "suite": "selection"}, [row("d", 1, {"C_best": 0.5})]),
    ])
    assert len(rows) == 1 and rows[0]["suite"] == "selection" and set(rows[0]["scores"]) == {"native", "C_best"}
    assert per_question[0]["responses"] == {"base": "x"} and len(units) == 2
