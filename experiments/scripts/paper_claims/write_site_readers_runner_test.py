"""CPU tests for the WRITE_SITE_READERS runner: dispatch per unit kind with fakes, and the latents gate."""

import json

import write_site_readers_runner as runner
import pytest
from experiment_plan import GateError


class FakeBackend:
    closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def context(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPERIMENT_STATE_DIR", str(tmp_path / "state"))
    plan = {"suites": {"selection": {"path": "inputs/write_site_readers/selection-v1.json"},
                       "distractor_curves_suite": {"path": "inputs/distractor_curves/suite/x/suite.json"}}}
    ctx = runner.setup(plan, "w0")
    ctx["cache"]["manifest:gemma2"] = {"documents": [{"id": "d0", "text": "t0"}, {"id": "d1", "text": "t1"}]}
    monkeypatch.setattr(runner, "backend_for", lambda context, model: FakeBackend())
    return ctx


def test_regression_delegates_to_distractor_curves(context, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "run_regression", lambda ctx, backend, unit: calls.append(unit) or {"agreement": 1})
    unit = {"kind": "regression", "model": "gemma2", "id": "r"}
    assert runner.run(context, unit) == {"agreement": 1} and calls == [unit]


def test_latents_unit_writes_the_shard_under_side(context, monkeypatch):
    seen = {}

    def fake_shard(backend, documents, ids, output):
        seen.update({"ids": ids, "output": output, "documents": documents})
        return {"documents": len(ids)}

    monkeypatch.setattr("distractor_curves_reader_training.build_latent_shard", fake_shard)
    result = runner.run(context, {"kind": "latents", "model": "gemma2", "shard": 3, "documents": ["d1"]})
    assert result == {"documents": 1} and seen["ids"] == ["d1"] and seen["documents"]["d1"]["text"] == "t1"
    assert seen["output"] == runner.side_dir() / "latents" / "gemma2" / "003.pt"


def test_train_gates_on_missing_latents(context, monkeypatch):
    monkeypatch.setattr("write_site_readers_reader_training.train_reader", lambda *a, **k: pytest.fail("ran without latents"))
    train = {"kind": "train", "model": "gemma2", "arm": "C", "seed": 216, "steps": 2, "max_document_occurrences": 12}
    with pytest.raises(GateError):
        runner.run(context, train)
    (runner.side_dir() / "latents" / "gemma2").mkdir(parents=True)  # empty directory is still a gate failure
    with pytest.raises(GateError):
        runner.run(context, train)


def with_latents(monkeypatch):
    directory = runner.side_dir() / "latents" / "gemma2"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "000.pt").write_bytes(b"shard")
    monkeypatch.setattr("distractor_curves_reader_training.load_latents", lambda path: {"d1": "latent"})


def test_train_single_process_uses_the_unit_contract(context, monkeypatch, tmp_path):
    with_latents(monkeypatch)
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"held_out_document_ids": []}))
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    seen = {}

    def fake_train(backend, payload, latents, selection, output_dir, **kwargs):
        seen.update({"selection": selection, "output_dir": output_dir, **kwargs})
        return {"checkpoints": {"best": str(output_dir / "best.pt")}}

    monkeypatch.setattr("write_site_readers_reader_training.train_reader", fake_train)
    unit = {"kind": "train", "model": "gemma2", "arm": "B", "seed": 1216, "steps": 4362, "max_document_occurrences": 12,
            "validation_interval": 500, "validation_limit": None, "fresh_per_k": 40, "selection_suite": "selection.json"}
    result = runner.run(context, unit)
    assert result["checkpoints"]["best"].endswith("readers/gemma2/B-seed1216/best.pt")
    assert seen["selection"] == {"held_out_document_ids": []}
    assert seen["output_dir"] == runner.side_dir() / "readers" / "gemma2" / "B-seed1216"
    assert (seen["arm"], seen["seed"], seen["steps"], seen["max_document_occurrences"]) == ("B", 1216, 4362, 12)
    assert (seen["validation_interval"], seen["validation_limit"], seen["fresh_per_k"]) == (500, None, 40)


def test_eval_resolves_readers_and_uses_the_unit_suite(context, monkeypatch):
    class FakeArms:
        def __init__(self, backend):
            self.layers = ["layer"]

    loaded = []
    monkeypatch.setattr("write_site_readers_core.WRITE_SITE_READERSArms", FakeArms)
    monkeypatch.setattr("write_site_readers_core.load_any_reader", lambda path, backend, layers: loaded.append(path) or ("write_site_readers", path))
    seen = []

    def fake_evaluate(arms, suite, source, banks, **kwargs):
        seen.append({"source": source, "banks": banks, "suite": suite, **kwargs})
        return {"source_id": source, "rows": []}

    monkeypatch.setattr("distractor_curves_core.evaluate_source", fake_evaluate)
    context["cache"]["suite:selection"] = {"documents": {}, "questions": {}, "name": "selection"}
    unit = {"kind": "eval", "model": "gemma2", "suite": "selection", "sources": ["s1", "s2"],
            "banks": {"s1": [{"k": 1}], "s2": [{"k": 2}]},
            "readers": {"C_best": "side:readers/gemma2/C-seed216/best.pt", "readout_ladderc": "artifacts/x.pt"},
            "betas": [0.5], "include_icl": True, "include_native": True}
    result = runner.run(context, unit)
    assert [s["source_id"] for s in result["sources"]] == ["s1", "s2"] and result["suite"] == "selection"
    assert loaded == [runner.side_dir() / "readers/gemma2/C-seed216/best.pt", runner.ROOT / "artifacts/x.pt"]
    assert seen[0]["suite"]["name"] == "selection" and seen[1]["banks"] == [{"k": 2}]
    assert set(seen[0]["readers"]) == {"C_best", "readout_ladderc"} and seen[0]["betas"] == [0.5]
    assert seen[0]["include_icl"] and seen[0]["include_native"]
    with pytest.raises(ValueError):
        runner.suite_for(context, "nonsense")


def test_unknown_kind_is_an_error(context):
    with pytest.raises(ValueError):
        runner.run(context, {"kind": "timing", "model": "gemma2"})
