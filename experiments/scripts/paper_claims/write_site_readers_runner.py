"""Runner for the write-site reader plans (``write_site_readers_plan.py``; Table 12).

Unit kinds:

* ``regression``: a 12-row runtime check that reproduces recorded single-document reader responses.
* ``latents``: compile one shard of reader-training documents (smoke plans only; the full selection
  plan reuses the distractor-curve run's Gemma shards under ``side/latents/gemma2/``).
* ``train``: one arm and seed of ``write_site_readers_reader_training.train_reader``. Distributed training
  uses the separate ``write_site_readers_ddp_train`` entry point.
* ``eval``: every arm of a few source documents over their persisted banks. The unit's ``suite`` selects
  the frozen suite (``selection``, ``secondary``, ``distractor_curves-dev``, or ``distractor_curves-test``).

The confirmation plan evaluates seed 216's checkpoint from the selection run (its ``requires`` entry).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from distractor_curves_runner import (  # noqa: F401  (setup/teardown are the runner interface)
    ROOT,
    SUITE_MANIFEST,
    backend_for,
    cached,
    reader_manifest,
    release_cuda_memory,
    resolve_reader,
    run_regression,
    setup,
    side_dir,
    teardown,
)

DISTRACTOR_CURVES_SUITES = ("distractor_curves-dev", "distractor_curves-test")


def latents_dir(model: str) -> Path:
    return side_dir() / "latents" / model


def latent_shards(model: str) -> list[Path]:
    return sorted(latents_dir(model).glob("*.pt"))


def require_latents(model: str) -> Path:
    """The cached latents directory, or a ``GateError`` when it holds no shard."""
    from experiment_plan import GateError

    directory = latents_dir(model)
    if not latent_shards(model):
        raise GateError(f"no latent shards under {directory}: copy DISTRACTOR_CURVES's cached latents into the state "
                        "directory before launch")
    return directory


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
    if kind == "latents":
        return run_latents(context, backend, unit)
    if kind == "train":
        return run_train(context, backend, unit)
    if kind == "eval":
        return run_eval(context, backend, unit)
    raise ValueError(f"unknown unit kind {kind}")


def manifest_for(context: dict[str, Any], model: str) -> dict[str, Any]:
    return cached(context, f"manifest:{model}", lambda: reader_manifest(model))


def run_latents(context: dict[str, Any], backend, unit: dict[str, Any]) -> dict[str, Any]:
    from distractor_curves_reader_training import build_latent_shard

    payload = manifest_for(context, unit["model"])
    documents = {d["id"]: d for d in payload["documents"]}
    return build_latent_shard(backend, documents, unit["documents"],
                              latents_dir(unit["model"]) / f"{unit['shard']:03d}.pt")


def selection_for(unit: dict[str, Any]) -> dict[str, Any]:
    from write_site_readers_reader_training import SELECTION_SUITE

    return json.loads((ROOT / unit.get("selection_suite", str(SELECTION_SUITE))).read_text())


def reader_dir(unit: dict[str, Any]) -> Path:
    return side_dir() / "readers" / unit["model"] / f"{unit['arm']}-seed{unit['seed']}"


def run_train(context: dict[str, Any], backend, unit: dict[str, Any]) -> dict[str, Any]:
    """Single-process training (one GPU, or the CPU smoke)."""
    from distractor_curves_reader_training import load_latents
    from write_site_readers_reader_training import train_reader

    directory = require_latents(unit["model"])
    payload = manifest_for(context, unit["model"])
    return train_reader(backend, payload, load_latents(directory), selection_for(unit), reader_dir(unit),
                        arm=unit["arm"], seed=unit["seed"], steps=unit["steps"],
                        max_document_occurrences=unit["max_document_occurrences"],
                        validation_interval=unit.get("validation_interval", 500),
                        validation_limit=unit.get("validation_limit"), fresh_per_k=unit.get("fresh_per_k", 40))


def frozen_manifest(context: dict[str, Any]) -> dict[str, Any]:
    from qa_protocol_protocol import load_frozen_manifest

    return cached(context, "manifest:qa_protocol", lambda: load_frozen_manifest(ROOT / SUITE_MANIFEST))


def suite_for(context: dict[str, Any], name: str) -> dict[str, Any]:
    """The frozen suite a unit evaluates, loaded once per worker and keyed by the unit's ``suite``."""
    def load() -> dict[str, Any]:
        suites = context["plan"]["suites"]
        if name in DISTRACTOR_CURVES_SUITES:
            from distractor_curves_core import load_suite

            return load_suite(ROOT / suites["distractor_curves_suite"]["path"], frozen_manifest(context))
        from write_site_readers_core import load_write_site_readers_suite

        return load_write_site_readers_suite(ROOT / suites[name]["path"], frozen_manifest(context))

    if name not in DISTRACTOR_CURVES_SUITES and name not in ("selection", "secondary"):
        raise ValueError(f"unknown suite {name}")
    return cached(context, f"suite:{name}", load)


def run_eval(context: dict[str, Any], backend, unit: dict[str, Any]) -> dict[str, Any]:
    import torch
    from distractor_curves_core import evaluate_source
    from write_site_readers_core import WRITE_SITE_READERSArms, load_any_reader

    arms = WRITE_SITE_READERSArms(backend)
    readers = {name: load_any_reader(resolve_reader(path), backend, arms.layers) for name, path in unit["readers"].items()}
    data = suite_for(context, unit["suite"])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    results = [evaluate_source(arms, data, source, unit["banks"][source], betas=unit["betas"], readers=readers, include_icl=unit["include_icl"],
                               include_native=unit["include_native"])
               for source in unit["sources"]]
    return {"sources": results, "suite": unit["suite"],
            "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0}
