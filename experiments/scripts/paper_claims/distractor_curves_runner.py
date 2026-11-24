"""Single-unit runner for the bundled SERVING_COST/DISTRACTOR_CURVES/QWEN_REPLICATION experiment run.

Unit kinds:

* ``latents``: compile one shard of reader-training documents to pre-head latents.
* ``train``: train one seed of the mixed-K rank-effect reader from the cached latents.
  Distributed training uses the separate ``distractor_curves_ddp_train`` entry point.
* ``eval``: every arm of a few source documents over their persisted banks.
* ``timing``: SERVING_COST's measured serving benchmark over both models, run last and alone.
* ``regression``: reproduce recorded READOUT_LADDER responses on the first exact-400 rows.

One backend is resident per worker; a unit for another model swaps it. Side outputs
(latents, reader checkpoints) live under ``$EXPERIMENT_STATE_DIR/side``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = {"gemma2": "gemma2_squad_matched_distill_scale60k_seed141", "qwen4": "qwen4_qwen_replication"}
SUITE_MANIFEST = Path("inputs/qa_protocol/manifest/pinned-v1/manifest.json")


def side_dir() -> Path:
    return Path(os.environ["EXPERIMENT_STATE_DIR"]) / "side"


def setup(plan: dict[str, Any], worker: str) -> dict[str, Any]:
    return {"plan": plan, "worker": worker, "model": None, "backend": None, "cache": {}}


def teardown(context: dict[str, Any]) -> None:
    if context["backend"] is not None:
        context["backend"].close()


def backend_for(context: dict[str, Any], model: str):
    if context["model"] != model:
        if context["backend"] is not None:
            # Fully release the previous model: an earlier run leaked ~18 GB per switch
            # through reference cycles, which OOMed workers sharing a GPU with reader training.
            context["backend"].close()
            context["backend"] = None
            context["model"] = None
            context["cache"].clear()
            release_cuda_memory()
        from d2l_scale import load_config
        from d2l_scale_backend import OfficialBackend

        context["backend"] = OfficialBackend(load_config(ROOT, CONFIGS[model]), ROOT)
        context["model"] = model
    return context["backend"]


def release_cuda_memory() -> None:
    """Collect cycles and return cached blocks, so the co-located worker on this GPU can use them."""
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cached(context: dict[str, Any], key: str, loader):
    if key not in context["cache"]:
        context["cache"][key] = loader()
    return context["cache"][key]


def reader_manifest(model: str) -> dict[str, Any]:
    from d2l_scale import load_config, load_manifest, paths

    return load_manifest(paths(ROOT, load_config(ROOT, CONFIGS[model]))["manifest"])


def suite(context: dict[str, Any]) -> dict[str, Any]:
    from qa_protocol_protocol import load_frozen_manifest
    from distractor_curves_core import load_suite

    return cached(context, "suite", lambda: load_suite(ROOT / context["plan"]["suite"],
                                                       load_frozen_manifest(ROOT / SUITE_MANIFEST)))


def run(context: dict[str, Any], unit: dict[str, Any]) -> dict[str, Any]:
    try:
        return run_unit(context, unit)
    finally:
        release_cuda_memory()


def run_unit(context: dict[str, Any], unit: dict[str, Any]) -> dict[str, Any]:
    kind = unit["kind"]
    if kind == "timing":
        return run_timing(context, unit)
    backend = backend_for(context, unit["model"])
    if kind == "latents":
        from distractor_curves_reader_training import build_latent_shard

        payload = cached(context, f"manifest:{unit['model']}", lambda: reader_manifest(unit["model"]))
        documents = {d["id"]: d for d in payload["documents"]}
        return build_latent_shard(backend, documents, unit["documents"],
                                  side_dir() / "latents" / unit["model"] / f"{unit['shard']:03d}.pt")
    if kind == "train":
        from distractor_curves_reader_training import load_latents, train_reader

        payload = cached(context, f"manifest:{unit['model']}", lambda: reader_manifest(unit["model"]))
        latents = load_latents(side_dir() / "latents" / unit["model"])
        return train_reader(backend, payload, latents, side_dir() / "readers" / unit["model"] / f"seed{unit['seed']}",
                            seed=unit["seed"], steps=unit["steps"],
                            max_document_occurrences=unit["max_document_occurrences"],
                            validation_interval=unit.get("validation_interval", 500),
                            validation_limit=unit.get("validation_limit"))
    if kind == "eval":
        return run_eval(context, backend, unit)
    if kind == "regression":
        return run_regression(context, backend, unit)
    raise ValueError(f"unknown unit kind {kind}")


def run_timing(context: dict[str, Any], unit: dict[str, Any]) -> dict[str, Any]:
    """SERVING_COST benchmark for every model in turn on this worker's otherwise idle GPU."""
    from serving_cost import benchmark

    data = suite(context)
    texts = [data["documents"][i]["text"] for i in sorted(data["documents"])
             if data["documents"][i]["role"] == "test_distractor"]
    return {model: benchmark(backend_for(context, model), texts, lengths=unit["lengths"], documents=unit["documents"],
                             batch_sizes=unit["batch_sizes"], output_tokens=unit["output_tokens"],
                             repeats=unit["repeats"]) for model in unit["models"]}


def resolve_reader(path: str) -> Path:
    """Saved readers are staged under the workspace; trained ones live in the side directory."""
    return side_dir() / path[len("side:"):] if path.startswith("side:") else ROOT / path


def run_eval(context: dict[str, Any], backend, unit: dict[str, Any]) -> dict[str, Any]:
    import torch
    from distractor_curves_core import Arms, evaluate_source, load_reader

    arms = Arms(backend)
    readers = {name: load_reader(resolve_reader(path), backend, arms.layers) for name, path in unit["readers"].items()}
    data = suite(context)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    results = [evaluate_source(arms, data, source, unit["banks"][source], betas=unit["betas"], readers=readers, include_icl=unit["include_icl"],
                               include_native=unit["include_native"]) for source in unit["sources"]]
    return {"sources": results,
            "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0}


def run_regression(context: dict[str, Any], backend, unit: dict[str, Any]) -> dict[str, Any]:
    """Reproduce READOUT_LADDER's recorded teacher/base/native/rank responses through the DISTRACTOR_CURVES arms."""
    from distractor_curves_core import Arms, bank_state, compile_documents, load_reader
    from experiment_plan import GateError

    recorded = json.loads((ROOT / unit["artifact"]).read_text())["rows"][: unit["rows"]]
    payload = cached(context, f"manifest:{unit['model']}", lambda: reader_manifest(unit["model"]))
    queries = {q["id"]: q for q in payload["queries"]}
    documents = {d["id"]: d for d in payload["documents"]}
    arms = Arms(backend)
    kind, reader = load_reader(resolve_reader(unit["reader"]), backend, arms.layers)
    matches = {"teacher": 0, "base": 0, "native_d2l": 0, "rank": 0}
    rows = []
    for row in recorded:
        query = {**queries[row["query_id"]], "answers": tuple(queries[row["query_id"]]["answers"])}
        records = compile_documents(backend, [documents[row["document_id"]]["text"]])
        now = {"teacher": arms.plain(query, context=documents[row["document_id"]]["text"]),
               "base": arms.plain(query),
               "native_d2l": arms.plain(query, state=bank_state(backend, records)),
               "rank": arms.reader(kind, reader, query, records)}
        for name, text in now.items():
            matches[name] += text == row["responses"][name]
        rows.append({"query_id": row["query_id"], "responses": now,
                     "recorded": {name: row["responses"][name] for name in now}})
    agreement = {name: count / len(recorded) for name, count in matches.items()}
    if min(agreement.values()) < unit["min_agreement"]:
        raise GateError(f"READOUT_LADDER regression agreement below {unit['min_agreement']}: {agreement}")
    return {"agreement": agreement, "rows": rows}
