"""CPU tests for the LAYER_INTERVENTIONS/GATED_READERS report over synthetic unit results."""

import json

import layer_interventions_report as report


def unit_file(state, name, unit, sources):
    (state / "units").mkdir(parents=True, exist_ok=True)
    (state / "units" / f"{name}.json").write_text(json.dumps({"unit": unit, "result": {"sources": sources}}))


def row(query, source, k, responses):
    return {"query_id": query, "source_id": source, "bank_id": f"{source}:random:{k}", "condition": "random", "k": k,
            "responses": responses, "scores": {name: {"f1": 1.0 if text == "gold" else 0.0, "em": 0.0}
                                               for name, text in responses.items()}}


def write_states(tmp_path):
    base, state = tmp_path / "base", tmp_path / "state"
    sources = []
    for s in ("s1", "s2", "s3"):
        rows = [row(f"{s}q", s, 1, {"icl": "gold", "native": "gold", "normalized_b0.5": "gold"})]
        for k in (4, 16):
            rows.append(row(f"{s}q", s, k, {"icl": "gold", "native": "x", "normalized_b0.5": "x" if k == 16 else "gold"}))
        sources.append({"source_id": s, "per_question": [], "rows": rows})
    unit_file(base, "selection-base-selection-000", {"suite": "selection", "kind": "eval"}, sources)
    oracle_sources, gated_sources = [], []
    for s in ("s1", "s2", "s3"):
        oracle_sources.append({"source_id": s, "per_question": [], "rows": [
            row(f"{s}q", s, k, {"oracle_from0": "gold", "oracle_from13": "gold" if k == 4 else "x"}) for k in (4, 16)]})
        gated_sources.append({"source_id": s, "per_question": [], "rows": [
            row(f"{s}q", s, k, {"GB-lam1-seed218_best": "gold" if (k < 16 or s != "s3") else "x"}) for k in (1, 4, 16)]})
    unit_file(state, "oracle-selection-000", {"suite": "selection", "kind": "oracle"}, oracle_sources)
    unit_file(state, "gated-reader-selection-000", {"suite": "selection", "kind": "eval"}, gated_sources)
    unit_file(state, "probe-selection", {"kind": "probe"}, [])
    (state / "units" / "probe-selection.json").write_text(json.dumps({"unit": {"kind": "probe"}, "result": {
        "summary": {level: {"b_rows": {"4": {"best_layer": 2, "best_auc": 0.9, "by_layer": [0.5, 0.7, 0.9]}}}
                    for level in ("token", "question", "slot")},
        "fit_bce_last_quarter": {"b_rows": 0.4}, "train_plans": 4, "test_plans": 6, "key_types": ["b_rows"]}}))
    (state / "units" / "train-GB-lam1-seed218.json").write_text(json.dumps({"unit": {"id": "train-GB-lam1-seed218"}, "result": {
        "steps": 2, "best_step": 2, "validation_history": [{"step": 0, "answer_ce": 2.0, "fresh_f1_per_k": {"1": 0.9}}],
        "loss_tail": [{"step": 1, "ce": 1.5, "gate_bce": 0.6}], "accounting": {"reader_parameters": 10}}}))
    return base, state


def test_oracle_report_merges_base_rows_and_checks_from0(tmp_path):
    base, state = write_states(tmp_path)
    out = report.oracle_report(state, base)
    assert out["arms"] == ["oracle_from0", "oracle_from13"]
    assert out["f1"]["random:4"]["f1"]["oracle_from13"] == 1.0 and out["f1"]["random:16"]["f1"]["oracle_from13"] == 0.0
    assert out["f1"]["random:16"]["f1"]["native"] == 0.0 and out["f1"]["random:16"]["f1"]["native_k1"] == 1.0
    assert out["from0_equals_native_k1"] == {"4": {"rows": 3, "mismatching": 0}, "16": {"rows": 3, "mismatching": 0}}
    assert out["contrasts"]["16"]["oracle_from13"]["vs_native_k1"]["difference"] == -1.0
    text = report.oracle_table(out)
    assert "from 13" in text and "| 16 |" in text


def test_probe_report_and_table(tmp_path):
    _, state = write_states(tmp_path)
    out = report.probe_report(state)
    assert out["key_types"] == ["b_rows"]
    assert "| b_rows | 4 | 2 | 0.900 | 0.50 0.70 0.90 |" in report.probe_table(out)


def test_gated_report_immunity_and_training(tmp_path):
    base, state = write_states(tmp_path)
    out = report.gated_report(state, base)
    arm = "GB-lam1-seed218_best"
    assert out["arms"] == [arm]
    assert out["f1"]["random:1"]["f1"][arm] == 1.0 and abs(out["f1"]["random:16"]["f1"][arm] - 2 / 3) < 1e-9
    imm = out["immunity"][arm]
    assert imm["k_high"] == 16 and abs(imm["k_high_minus_k1"]["difference"] + 1 / 3) < 1e-9 and abs(imm["ratio"] - 2 / 3) < 1e-9
    rows = [row(f"{s}q", s, k, {arm: "gold" if k == 1 else "x"}) for s in ("s1", "s2", "s3") for k in (1, 4)]
    lower = report.immunity(rows, arm, k_high=4)
    assert lower["k_high"] == 4 and lower["k_high_f1"] == 0.0 and abs(lower["k_high_minus_k1"]["difference"] + 1.0) < 1e-9
    assert report.immunity(rows, arm)["k_high"] == 4  # default: the largest evaluated K
    assert out["contrasts"][arm]["16"]["reference"] == "normalized_b0.5"
    assert out["training"]["train-GB-lam1-seed218"]["gate_bce_tail"] == [0.6]
    text = report.gated_table(out)
    assert "| 16 |" in text and arm in text and "(K=16)" in text


def test_schedule_report_uses_the_schedule_prefix(tmp_path):
    base, state = write_states(tmp_path)
    sources = [{"source_id": s, "per_question": [], "rows": [
        row(f"{s}q", s, k, {"schedule_none": "x", "schedule_A": "gold", "schedule_B8": "gold" if s != "s3" else "x",
                            "schedule_C": "gold"}) for k in (1, 4)]} for s in ("s1", "s2", "s3")]
    unit_file(state, "schedule-selection-000", {"suite": "selection", "kind": "schedule"}, sources)
    out = report.schedule_report(state, base)
    assert out["arms"] == ["schedule_A", "schedule_B8", "schedule_C", "schedule_none"] and out["prefix"] == "schedule_"
    assert out["f1"]["random:4"]["f1"]["schedule_A"] == 1.0 and out["f1"]["random:4"]["f1"]["schedule_none"] == 0.0
    assert out["contrasts"]["4"]["schedule_none"]["vs_native_k1"]["difference"] == -1.0
    assert out["from0_equals_native_k1"] == {"1": {"rows": 0, "mismatching": 0}, "4": {"rows": 0, "mismatching": 0}}
    text = report.oracle_table(out)
    assert "| B8 |" in text and "| none |" in text and "| 4 |" in text


def test_generality_report_reads_each_cell_from_its_own_base_pattern(tmp_path):
    base, state = write_states(tmp_path)
    # Gemma secondary cell: base rows live under a different pattern and suite name
    sec = tmp_path / "sec"
    rows = [row("t1q", "t1", 1, {"icl": "gold", "native": "gold"}), row("t1q", "t1", 8, {"icl": "gold", "native": "x", "normalized_b0.5": "x"})]
    unit_file(sec, "secondary-base-secondary-000", {"suite": "secondary", "kind": "eval"}, [{"source_id": "t1", "per_question": [], "rows": rows}])
    unit_file(state, "oracle-secondary-gemma2-000", {"suite": "secondary", "kind": "oracle"},
              [{"source_id": "t1", "per_question": [], "rows": [row("t1q", "t1", 8, {"oracle_from0": "gold", "oracle_from8": "x"})]}])
    unit_file(state, "schedule-secondary-gemma2-L6-000", {"suite": "secondary", "kind": "schedule"},
              [{"source_id": "t1", "per_question": [], "rows": [row("t1q", "t1", 8, {"schedule_B6": "gold"})]}])
    # Qwen cell: base rows in the LAYER_REPLICATION state itself
    unit_file(state, "selection-base-qwen4-000", {"suite": "selection", "kind": "eval"},
              [{"source_id": "s1", "per_question": [], "rows": [row("s1q", "s1", 1, {"icl": "gold", "native": "gold"}),
                                                                 row("s1q", "s1", 8, {"icl": "gold", "native": "x", "normalized_b0.5": "x"})]}])
    unit_file(state, "oracle-selection-qwen4-000", {"suite": "selection", "kind": "oracle"},
              [{"source_id": "s1", "per_question": [], "rows": [row("s1q", "s1", 8, {"oracle_from0": "gold", "oracle_from11": "gold"})]}])
    cells = (("gemma2", "secondary", sec, "secondary-base-*.json"), ("qwen4", "selection", None, "selection-base-qwen4-*.json"),
             ("gemma2", "selection", base, "selection-base-*.json"))
    out = report.generality_report(state, cells)
    g = out["cells"]["secondary:gemma2"]
    assert g["oracle"]["f1"]["random:8"]["f1"]["oracle_from0"] == 1.0 and g["oracle"]["f1"]["random:8"]["f1"]["native_k1"] == 1.0
    assert g["oracle"]["from0_equals_native_k1"]["8"] == {"rows": 1, "mismatching": 0}
    assert g["schedule"]["arms"] == ["schedule_B6"] and g["schedule"]["f1"]["random:8"]["f1"]["schedule_B6"] == 1.0
    q = out["cells"]["selection:qwen4"]
    assert q["oracle"]["arms"] == ["oracle_from0", "oracle_from11"] and q["oracle"]["f1"]["random:8"]["f1"]["icl"] == 1.0
    assert "selection:gemma2" in out["missing"]  # no LAYER_REPLICATION units for that cell in this fixture
    assert report.generality_tables(out)
