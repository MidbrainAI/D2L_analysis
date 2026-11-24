"""CPU test for the ATTENUATION_SCHEDULES gate-calibration analysis on synthetic rows."""

import json

import attenuation_schedules_gate_calibration as cal


def test_analyze_reports_ranking_and_alternative_gates():
    # two documents x 2 ranks; source is document 1 in the first row, document 0 in the second
    rows = [{"arm": "a", "k": 2, "logits": [-3.0, -3.0, -1.0, -1.0], "source": 1},
            {"arm": "a", "k": 2, "logits": [-1.0, -1.0, -2.0, -2.0], "source": 0},
            {"arm": "a", "k": 1, "logits": [-1.0, -1.0], "source": 0}]
    out = cal.analyze(rows, ranks_per_document=2)
    e = out["a"]["2"]
    assert e["rows"] == 2 and e["source_is_top_document"] == 1.0
    assert abs(e["source_logit_mean"] + 1.0) < 1e-9 and abs(e["distractor_logit_mean"] + 2.5) < 1e-9
    assert e["gates"]["doc exp(mean-max)"]["source"] == 1.0 and e["gates"]["doc exp(mean-max)"]["distractor"] < 0.4
    assert e["gates"]["sigmoid(x)"]["source"] < 0.3  # the run's gate: nearly silent
    assert out["a"]["1"]["distractor_logit_mean"] is None and out["a"]["1"]["source_is_top_document"] == 1.0


def test_load_rows_uses_bank_membership(tmp_path):
    state = tmp_path / "state"
    (state / "units").mkdir(parents=True)
    unit = {"sources": ["s"], "banks": {"s": [{"bank_id": "b", "document_ids": ["d", "s"], "k": 2}]}}
    row = {"source_id": "s", "bank_id": "b", "k": 2, "gates": {"x": {"scores_last": [0.0, 0.0, 1.0, 1.0]}}}
    (state / "units" / "predicted-selection-000.json").write_text(json.dumps({"unit": unit, "result": {"sources": [{"rows": [row]}]}}))
    rows = cal.load_rows(state, ("x",))
    assert rows == [{"arm": "x", "k": 2, "logits": [0.0, 0.0, 1.0, 1.0], "source": 1}]
