import json

from distractor_curves_report import bootstrap, contains, report


def scores(**values):
    return {name: {"f1": value, "em": value} for name, value in values.items()}


def test_bootstrap_interval_brackets_a_constant_difference():
    estimate, low, high = bootstrap({"a": [0.2, 0.2], "b": [0.2]})
    assert estimate == low == high == 0.2


def test_report_reads_units_and_applies_the_preregistered_reader_rule(tmp_path):
    (tmp_path / "units").mkdir()
    (tmp_path / "expansions").mkdir()
    selection = {"beta": {"beta": 0.5}, "checkpoints": {}}
    unit = {"readers": {"reader_s1": "p", "reader_s2": "p"}, "selection": selection}
    (tmp_path / "expansions" / "gemma2-test-base.json").write_text(
        json.dumps({"units": [{"selection": {"beta": selection["beta"]}}]}))
    (tmp_path / "expansions" / "gemma2-test-reader.json").write_text(
        json.dumps({"units": [{"selection": {"checkpoints": {}}}]}))
    rows = []
    for source in ("d1", "d2", "d3"):
        for k in (1, 2, 4, 8, 16):
            rows.append({"query_id": f"{source}q", "source_id": source, "bank_id": f"{source}:random:{k}",
                         "condition": "random", "k": k,
                         "responses": {"native": "x"},
                         "scores": scores(native=0.4, **{"normalized_b0.5": 0.5},
                                          reader_s1=0.7, reader_s2=0.8, icl=0.9)})
    per_question = [{"query_id": f"{s}q", "responses": {"base": "x"}, "scores": scores(base=0.1)}
                    for s in ("d1", "d2", "d3")]
    result = {"sources": [{"source_id": "d1", "rows": rows, "per_question": per_question}]}
    (tmp_path / "units" / "gemma2-test-reader-000.json").write_text(json.dumps({"unit": unit, "result": result}))
    output = report(tmp_path, "gemma2", {"d1": "A", "d2": "A", "d3": "B"})
    assert output["preregistered_reader_rule"]["holds"]
    assert output["table"]["random:4"]["f1"]["reader_mean"] == 0.75
    assert set(output["table"]) == {f"random:{k}" for k in (1, 2, 4, 8, 16)}


def test_containment_accepts_sentence_answers():
    assert contains("The first destinations were Switzerland and the Netherlands.", ["the Netherlands"]) == 1.0
    assert contains("Netherlandish art", ["Netherlands"]) == 0.0
