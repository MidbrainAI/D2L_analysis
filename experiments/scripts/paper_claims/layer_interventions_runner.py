"""Single-unit runner for the LAYER_INTERVENTIONS/GATED_READERS experiment run (plan from ``layer_interventions_plan.py``).

Unit kinds (everything not listed is ``write_site_readers_runner``'s):

* ``oracle``: ``layer_interventions_gates.run_oracle``: layer-prefix oracle gates on a few selection-suite
  sources over their persisted random banks (DISTRACTOR_CURVES row shape; native/ICL/normalized rows come
  from the WRITE_SITE_READERS selection run and are merged by ``layer_interventions_report``).
* ``probe``: ``layer_interventions_gates.run_probe``: fit the membership probe on reader-training banks from
  the cached latents (``GateError`` without them, as in WRITE_SITE_READERS) and score it on the suite's banks.
* ``schedule``: ``attenuation_schedules_schedules.run_schedule`` (ATTENUATION_SCHEDULES): the attenuation-schedule ceilings with the
  true membership mask on a few selection-suite sources (DISTRACTOR_CURVES row shape, responses ``schedule_*``).
* ``train``: one GATED_READERS gated arm (``GB``, ``GC``, ``GBL``) with its ``lam``; checkpoints under
  ``side/readers/<model>/<arm>-lam<lam>-seed<seed>/``; data-parallel through
  the separately invoked ``gated_readers_ddp_train`` entry point.
* ``eval``: ``write_site_readers_runner.run_eval`` (gated checkpoints load through ``write_site_readers_core.load_any_reader``).
* ``probe-fit`` / ``predicted``: ``attenuation_schedules_predicted``: fit and save a clean-prefill probe under
  ``side/probes/<model>/<scorer>.pt``; run the schedules with gates predicted by it (responses ``predicted_*``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from write_site_readers_runner import (  # noqa: F401  (setup/teardown are the runner interface)
    ROOT,
    backend_for,
    manifest_for,
    release_cuda_memory,
    require_latents,
    run_eval,
    run_regression,
    selection_for,
    setup,
    side_dir,
    suite_for,
    teardown,
)


def run(context: dict[str, Any], unit: dict[str, Any]) -> dict[str, Any]:
    try:
        return run_unit(context, unit)
    finally:
        release_cuda_memory()


def run_unit(context: dict[str, Any], unit: dict[str, Any]) -> dict[str, Any]:
    kind = unit["kind"]
    backend = backend_for(context, unit["model"])
    if kind == "regression":
        return run_regression(context, backend, unit)
    if kind == "oracle":
        return run_oracle_unit(context, backend, unit)
    if kind == "probe":
        return run_probe_unit(context, backend, unit)
    if kind == "schedule":
        return run_schedule_unit(context, backend, unit)
    if kind == "probe-fit":
        return run_probe_fit_unit(context, backend, unit)
    if kind == "predicted":
        return run_predicted_unit(context, backend, unit)
    if kind == "train":
        return run_train(context, backend, unit)
    if kind == "eval":
        return run_eval(context, backend, unit)
    raise ValueError(f"unknown unit kind {kind}")


def run_oracle_unit(context: dict[str, Any], backend, unit: dict[str, Any]) -> dict[str, Any]:
    from layer_interventions_gates import run_oracle

    return run_oracle(backend, suite_for(context, unit["suite"]), unit)


def run_schedule_unit(context: dict[str, Any], backend, unit: dict[str, Any]) -> dict[str, Any]:
    from attenuation_schedules_schedules import run_schedule

    return run_schedule(backend, suite_for(context, unit["suite"]), unit)


def run_probe_fit_unit(context: dict[str, Any], backend, unit: dict[str, Any]) -> dict[str, Any]:
    from distractor_curves_reader_training import load_latents
    from attenuation_schedules_predicted import probe_path, run_probe_fit

    directory = require_latents(unit["model"])
    return run_probe_fit(backend, manifest_for(context, unit["model"]), load_latents(directory), selection_for(unit), unit,
                         probe_path(side_dir(), unit["model"], unit["scorer"]))


def run_predicted_unit(context: dict[str, Any], backend, unit: dict[str, Any]) -> dict[str, Any]:
    from attenuation_schedules_predicted import run_predicted

    return run_predicted(backend, suite_for(context, unit["suite"]), unit, side_dir())


def run_probe_unit(context: dict[str, Any], backend, unit: dict[str, Any]) -> dict[str, Any]:
    from distractor_curves_reader_training import load_latents
    from layer_interventions_gates import run_probe

    directory = require_latents(unit["model"])
    return run_probe(backend, manifest_for(context, unit["model"]), load_latents(directory), selection_for(unit),
                     suite_for(context, unit["suite"]), unit["banks"], unit)


def reader_name(unit: dict[str, Any]) -> str:
    return f"{unit['arm']}-lam{float(unit['lam']):g}-seed{unit['seed']}"  # must equal layer_interventions_plan.train_name


def reader_dir(unit: dict[str, Any]) -> Path:
    return side_dir() / "readers" / unit["model"] / reader_name(unit)


def train_gated(backend, payload: dict[str, Any], latents: dict[str, Any], selection: dict[str, Any],
                output_dir: Path, unit: dict[str, Any]) -> dict[str, Any]:
    """WRITE_SITE_READERS's training loop with the gated factory, the CE + lam * document-BCE loss, and GATED_READERS's recipe."""
    from write_site_readers_reader_training import train_reader
    from gated_readers_gated_readers import GATED_ARMS, RECIPE, gated_loss, make_gated_reader

    return train_reader(backend, payload, latents, selection, output_dir, arm=unit["arm"], seed=unit["seed"],
                        steps=unit["steps"], max_document_occurrences=unit["max_document_occurrences"],
                        validation_interval=unit.get("validation_interval", 500),
                        validation_limit=unit.get("validation_limit"), fresh_per_k=unit.get("fresh_per_k", 40),
                        reader_factory=make_gated_reader, loss_fn=gated_loss(float(unit["lam"])),
                        known_arms=tuple(GATED_ARMS), recipe=f"{RECIPE}-lam{unit['lam']}")


def run_train(context: dict[str, Any], backend, unit: dict[str, Any]) -> dict[str, Any]:
    from distractor_curves_reader_training import load_latents

    directory = require_latents(unit["model"])
    return train_gated(backend, manifest_for(context, unit["model"]), load_latents(directory), selection_for(unit),
                       reader_dir(unit), unit)
