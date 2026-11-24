"""Runner for the answer-position document-read mask (Table 3, right panel).

Each ``answer_position_mask`` unit scores every mask condition for one shard of the 402-question
diagnostic set with Gemma-2-2B and the released D2L checkpoint. Fresh baseline responses must agree with
the recorded activation-transplant reference before a unit produces a result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONFIG = "gemma2_squad_matched_distill_scale60k_seed141"
MANIFEST = Path("inputs/qa_protocol/manifest/pinned-v1/manifest.json")


def setup(plan: dict[str, Any], worker: str) -> dict[str, Any]:
    return {"plan": plan, "worker": worker, "backend": None, "cache": {}}


def teardown(context: dict[str, Any]) -> None:
    release(context)


def release(context: dict[str, Any]) -> None:
    if context["backend"] is not None:
        context["backend"].close()
        context["backend"] = None
    release_cuda_memory()


def release_cuda_memory() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def backend(context: dict[str, Any]):
    if context["backend"] is None:
        from d2l_scale import load_config
        from d2l_scale_backend import OfficialBackend

        context["backend"] = OfficialBackend(load_config(ROOT, CONFIG), ROOT)
    return context["backend"]


def cached(context: dict[str, Any], key: str, loader):
    if key not in context["cache"]:
        context["cache"][key] = loader()
    return context["cache"][key]


def manifest(context: dict[str, Any]) -> dict[str, Any]:
    from qa_protocol_protocol import load_frozen_manifest

    return cached(context, "manifest", lambda: load_frozen_manifest(ROOT / MANIFEST))


def documents(context: dict[str, Any]) -> dict[str, Any]:
    from qa_protocol_runner_common import d2l_documents

    return cached(context, "documents", lambda: d2l_documents(manifest(context)))


def plans_by_id(context: dict[str, Any], split: str) -> dict[str, dict]:
    from qa_protocol_runner_common import d2l_plans

    return cached(context, f"plans:{split}",
                  lambda: {p["query"]["id"]: p for p in d2l_plans(manifest(context), split)})


def inputs_dir(context: dict[str, Any]) -> Path:
    return ROOT / context["plan"]["config"]["inputs"]


def run(context: dict[str, Any], unit: dict[str, Any]) -> dict[str, Any]:
    try:
        return run_unit(context, unit)
    finally:
        release_cuda_memory()


def run_unit(context: dict[str, Any], unit: dict[str, Any]) -> dict[str, Any]:
    if unit["kind"] != "answer_position_mask":
        raise ValueError(f"unknown unit kind {unit['kind']}")
    return run_answer_position_mask(context, unit)


def run_answer_position_mask(context: dict[str, Any], unit: dict[str, Any]) -> dict[str, Any]:
    from answer_position_mask_answer_read_mask import MAX_NEW_TOKENS, PARITY_MIN_AGREEMENT, run_shard

    from experiment_plan import GateError

    by_id = plans_by_id(context, unit["split"])
    plans = [by_id[q] for q in unit["query_ids"]]
    reference = None
    if unit["suite"] == "exact402":
        confirmation = cached(context, "confirmation", lambda: json.loads(
            (inputs_dir(context) / "activation_transplant" / "reference.json").read_text()))
        reference = {row["query_id"]: row for row in confirmation["rows"]}
    limits = context["plan"]["config"].get("limits") or {}
    result = run_shard(backend(context), plans, documents(context), reference,
                       budget=limits.get("max_new_tokens", MAX_NEW_TOKENS))
    agreement = result["gates"]["reference_agreement"]
    low = {arm: value for arm, value in agreement.items() if value is not None and value < PARITY_MIN_AGREEMENT}
    if low:
        raise GateError(f"fresh responses disagree with the recorded transplant reference: {low}")
    return {"suite": unit["suite"], **result}
