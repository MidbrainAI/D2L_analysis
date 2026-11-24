"""CPU tests for the LAYER_INTERVENTIONS runner: dispatch per unit kind with fakes."""

import layer_interventions_runner as runner
import pytest
from experiment_plan import GateError


class FakeBackend:
    def close(self):
        pass


@pytest.fixture
def context(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPERIMENT_STATE_DIR", str(tmp_path / "state"))
    ctx = runner.setup({"suites": {"selection": {"path": "inputs/write_site_readers/selection-v1.json"}}}, "w0")
    ctx["cache"]["manifest:gemma2"] = {"documents": []}
    ctx["cache"]["suite:selection"] = {"documents": {}, "questions": {}}
    monkeypatch.setattr(runner, "backend_for", lambda context, model: FakeBackend())
    return ctx


def test_oracle_unit_passes_the_suite_and_unit(context, monkeypatch):
    import layer_interventions_gates

    seen = {}
    monkeypatch.setattr(layer_interventions_gates, "run_oracle", lambda backend, suite, unit: seen.update(suite=suite, unit=unit) or {"ok": 1})
    unit = {"kind": "oracle", "model": "gemma2", "suite": "selection", "id": "o"}
    assert runner.run(context, unit) == {"ok": 1} and seen["unit"] is unit and seen["suite"] == context["cache"]["suite:selection"]


def test_probe_and_train_units_need_the_latents(context):
    with pytest.raises(GateError):
        runner.run(context, {"kind": "probe", "model": "gemma2", "suite": "selection", "banks": {}, "id": "p"})
    with pytest.raises(GateError):
        runner.run(context, {"kind": "train", "model": "gemma2", "arm": "GB", "lam": 1.0, "seed": 218, "steps": 1,
                             "max_document_occurrences": 12, "id": "t"})


def test_train_unit_uses_the_gated_factory_and_loss(context, monkeypatch, tmp_path):
    import write_site_readers_reader_training
    import gated_readers_gated_readers

    latents = tmp_path / "state" / "side" / "latents" / "gemma2"
    latents.mkdir(parents=True)
    (latents / "000.pt").write_bytes(b"")
    monkeypatch.setattr("distractor_curves_reader_training.load_latents", lambda directory: {"d": None})
    monkeypatch.setattr(runner, "selection_for", lambda unit: {"held_out_document_ids": []})
    seen = {}

    def fake_train(backend, payload, latents, selection, output_dir, **kwargs):
        seen.update(kwargs, output_dir=output_dir)
        return {"done": True}

    monkeypatch.setattr(write_site_readers_reader_training, "train_reader", fake_train)
    unit = {"kind": "train", "model": "gemma2", "arm": "GBL", "lam": 0.5, "seed": 218, "steps": 3,
            "max_document_occurrences": 12, "id": "t"}
    assert runner.run(context, unit) == {"done": True}
    assert seen["reader_factory"] is gated_readers_gated_readers.make_gated_reader
    assert seen["known_arms"] == tuple(gated_readers_gated_readers.GATED_ARMS) and seen["recipe"].endswith("-lam0.5")
    assert seen["output_dir"].name == "GBL-lam0.5-seed218" and callable(seen["loss_fn"])


def test_unknown_kind_is_rejected(context):
    with pytest.raises(ValueError):
        runner.run(context, {"kind": "mystery", "model": "gemma2", "id": "m"})


def test_schedule_unit_passes_the_suite_and_unit(context, monkeypatch):
    import attenuation_schedules_schedules

    seen = {}
    monkeypatch.setattr(attenuation_schedules_schedules, "run_schedule", lambda backend, suite, unit: seen.update(suite=suite, unit=unit) or {"ok": 2})
    unit = {"kind": "schedule", "model": "gemma2", "suite": "selection", "id": "s"}
    assert runner.run(context, unit) == {"ok": 2} and seen["unit"] is unit and seen["suite"] == context["cache"]["suite:selection"]


def test_predicted_units_dispatch_and_probe_fit_needs_latents(context, monkeypatch):
    import attenuation_schedules_predicted

    seen = {}
    monkeypatch.setattr(attenuation_schedules_predicted, "run_predicted", lambda backend, suite, unit, side: seen.update(unit=unit, side=side) or {"ok": 3})
    unit = {"kind": "predicted", "model": "gemma2", "suite": "selection", "id": "p"}
    assert runner.run(context, unit) == {"ok": 3} and seen["unit"] is unit and seen["side"].name == "side"
    with pytest.raises(GateError):
        runner.run(context, {"kind": "probe-fit", "model": "gemma2", "scorer": "x", "id": "f"})
