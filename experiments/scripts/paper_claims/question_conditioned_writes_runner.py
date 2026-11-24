"""Single-unit runner for the question-conditioned writes (Table 9).

Unit kinds:

* ``gates``: the preregistered correctness gates on one development bank (fails without writing a result).
* ``latents``: compile a shard of training passages to cached D2L latents (SINGLE_DOCUMENT_WRITES).
* ``teacher``: cache the source-visible teacher's greedy response for a shard of training questions.
* ``train``: train the write encoder, in process, or with the separate distributed entry point.
* ``eval``: native, normalized, and encoder arms over DISTRACTOR_CURVES suite sources and banks.

Gemma-2-2B only (DISTRACTOR_CURVES's config). QUESTION_CONDITIONED_WRITES trains on the scale60k rows with DISTRACTOR_CURVES v5's cached latents, mounted
read-only. SINGLE_DOCUMENT_WRITES (``config["rows"]`` set) trains on frozen self-study SQuAD rows over QA_PROTOCOL passages and compiles
their latents in the run. Latents, teacher responses, and encoder checkpoints live under
``$EXPERIMENT_STATE_DIR/side``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MODEL = "gemma2"
SUITE_MANIFEST = Path("inputs/qa_protocol/manifest/pinned-v1/manifest.json")


def side_dir() -> Path:
    return Path(os.environ["EXPERIMENT_STATE_DIR"]) / "side"


def setup(plan: dict[str, Any], worker: str) -> dict[str, Any]:
    return {"plan": plan, "worker": worker, "backend": None, "cache": {}}


def teardown(context: dict[str, Any]) -> None:
    release(context)


def release(context: dict[str, Any]) -> None:
    import gc

    import torch

    if context["backend"] is not None:
        context["backend"].close()
        context["backend"] = None
    context["cache"].clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def backend(context: dict[str, Any]):
    if context["backend"] is None:
        from distractor_curves_runner import CONFIGS

        from d2l_scale import load_config
        from d2l_scale_backend import OfficialBackend

        context["backend"] = OfficialBackend(load_config(ROOT, CONFIGS[MODEL]), ROOT)
    return context["backend"]


def cached(context: dict[str, Any], key: str, loader):
    if key not in context["cache"]:
        context["cache"][key] = loader()
    return context["cache"][key]


def payload(context: dict[str, Any]) -> dict[str, Any]:
    from distractor_curves_runner import reader_manifest

    return cached(context, "payload", lambda: reader_manifest(MODEL))


def data(context: dict[str, Any]) -> dict[str, Any]:
    """Training documents and rows: SINGLE_DOCUMENT_WRITES's frozen rows file over QA_PROTOCOL passages, else QUESTION_CONDITIONED_WRITES's scale60k rows."""
    def load():
        config = context["plan"]["config"]
        if not config.get("rows"):
            from distractor_curves_reader_training import training_rows

            source = payload(context)
            return {"documents": {d["id"]: d for d in source["documents"]}, "rows": training_rows(source)}
        frozen = json.loads((ROOT / config["rows"]).read_text())
        needed = {r["document"]["id"] for split in ("train", "validation") for r in frozen[split]}
        manifest = json.loads((ROOT / SUITE_MANIFEST).read_text())
        return {"documents": {d["id"]: d for d in manifest["documents"] if d["id"] in needed},
                "rows": {split: frozen[split] for split in ("train", "validation")}}
    return cached(context, "data", load)


def suite(context: dict[str, Any]) -> dict[str, Any]:
    from qa_protocol_protocol import load_frozen_manifest
    from distractor_curves_core import load_suite

    return cached(context, "suite", lambda: load_suite(ROOT / context["plan"]["config"]["suite"],
                                                       load_frozen_manifest(ROOT / SUITE_MANIFEST)))


def latents(context: dict[str, Any]) -> dict[str, Any]:
    from distractor_curves_reader_training import load_latents

    configured = context["plan"]["config"].get("latents")
    return cached(context, "latents", lambda: load_latents(ROOT / configured if configured else side_dir() / "latents"))


def responses(context: dict[str, Any]) -> dict[str, Any]:
    import torch

    def load():
        table = {}
        teacher = context["plan"]["config"].get("teacher_dir")
        directory = ROOT / teacher if teacher else side_dir() / "teacher"
        for path in sorted(directory.glob("*.pt")):
            table.update(torch.load(path, map_location="cpu", weights_only=False))
        return table
    return cached(context, "responses", load)


def run(context: dict[str, Any], unit: dict[str, Any]) -> dict[str, Any]:
    try:
        return run_unit(context, unit)
    finally:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_unit(context: dict[str, Any], unit: dict[str, Any]) -> dict[str, Any]:
    kind = unit["kind"]
    if kind == "train":
        return run_train(context, unit)
    if kind == "gates":
        return run_gates(context, unit)
    if kind == "latents":
        return run_latents(context, unit)
    if kind == "teacher":
        return run_teacher(context, unit)
    if kind == "eval":
        return run_eval(context, unit)
    raise ValueError(f"unknown unit kind {kind}")


def run_gates(context: dict[str, Any], unit: dict[str, Any]) -> dict[str, Any]:
    from distractor_curves_core import compile_documents
    from question_conditioned_writes_conditioned_write import WriteEncoder, gates

    from experiment_plan import GateError

    data = suite(context)
    model = backend(context)
    ids = unit["document_ids"]
    records = compile_documents(model, [data["documents"][i]["text"] for i in ids])
    table = {i: r["latent"][0] for i, r in zip(ids, records, strict=True)}
    first = records[0]["latent"][0]
    encoder = WriteEncoder(first.shape[-1], model.model.base_model.config.hidden_size, first.shape[0]).to(
        model.model.device)
    try:
        return gates(model, encoder, table, ids, data["questions"][unit["source"]][0])
    except RuntimeError as error:
        raise GateError(str(error)) from error


def run_latents(context: dict[str, Any], unit: dict[str, Any]) -> dict[str, Any]:
    from distractor_curves_reader_training import build_latent_shard

    return build_latent_shard(backend(context), data(context)["documents"], unit["document_ids"],
                              side_dir() / "latents" / f"{unit['shard']:03d}.pt")


def run_teacher(context: dict[str, Any], unit: dict[str, Any]) -> dict[str, Any]:
    import torch
    from question_conditioned_writes_conditioned_write import teacher_responses

    training = data(context)
    wanted = set(unit["query_ids"])
    rows = [r for split in training["rows"].values() for r in split if r["query"]["id"] in wanted]
    out = teacher_responses(backend(context), rows, training["documents"])
    path = side_dir() / "teacher" / f"{unit['shard']:03d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, path.with_name(f".{path.name}.tmp"))
    path.with_name(f".{path.name}.tmp").replace(path)
    return {"questions": len(out), "mean_response_tokens": sum(v.numel() for v in out.values()) / max(1, len(out))}


def run_train(context: dict[str, Any], unit: dict[str, Any]) -> dict[str, Any]:
    from question_conditioned_writes_conditioned_write import train

    return train(backend(context), data(context), latents(context), responses(context),
                 side_dir() / "encoders" / f"seed{unit['seed']}", **train_options(unit))


def train_options(unit: dict[str, Any]) -> dict[str, Any]:
    from question_conditioned_writes_conditioned_write import EXAMPLES_PER_UPDATE, LEARNING_RATE

    return {"seed": unit["seed"], "steps": unit["steps"], "validation_interval": unit["validation_interval"],
            "validation_limit": unit.get("validation_limit"), "k_values": tuple(unit.get("k_values", (1, 2, 4, 8, 16))),
            "batch_size": unit.get("batch_size", EXAMPLES_PER_UPDATE),
            "learning_rate": unit.get("learning_rate", LEARNING_RATE),
            "contrastive_weight": unit.get("contrastive_weight", 0.0), "margin": unit.get("margin", 0.1)}


def run_eval(context: dict[str, Any], unit: dict[str, Any]) -> dict[str, Any]:
    import torch
    from question_conditioned_writes_conditioned_write import evaluate_source, load_encoder

    model = backend(context)
    data = suite(context)
    encoders = {name: load_encoder(side_dir() / path, model.model.device) for name, path in unit["encoders"].items()}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    results = [evaluate_source(model, data, source, unit["banks"][source], encoders, beta=unit.get("beta"),
                               budget=unit.get("budget", 32),
                               question_limit=unit.get("question_limit")) for source in unit["sources"]]
    return {"sources": results,
            "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0}
