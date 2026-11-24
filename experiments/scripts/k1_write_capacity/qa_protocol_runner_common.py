"""Shared no-download interfaces for the four QA_PROTOCOL execution stages."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from qa_protocol_protocol import SPLITS, digest, load_frozen_manifest, stage_contract, validate_resume


def rows(manifest: dict[str, Any], split: str) -> list[dict[str, Any]]:
    if split not in SPLITS:
        raise ValueError(f"unknown QA_PROTOCOL split: {split}")
    selected = [row for row in manifest["rows"] if row["split"] == split]
    return sorted(selected, key=lambda row: (row["family"], row["document_hash"], row["query_id"]))


def d2l_documents(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {document["id"]: {**document, "contenthash": document["content_hash"]} for document in manifest["documents"]}


def d2l_plans(manifest: dict[str, Any], split: str) -> list[dict[str, Any]]:
    documents = d2l_documents(manifest)
    plans = []
    for row in rows(manifest, split):
        query = {
            "id": row["query_id"], "document_id": row["document_id"], "input": row["question"],
            "answers": tuple(row["answers"]), "split": split, "dataset": row["family"],
            "relevant_document_ids": (),
        }
        plans.append({
            "document": documents[row["document_id"]], "query": query, "k": 1,
            "bank_document_ids": [row["document_id"]], "source_position": 0,
        })
    return plans


def family_batches(plans: list[dict[str, Any]], backend, batch_size: int, max_padded_tokens: int) -> list[list[dict[str, Any]]]:
    from official_d2l_online_batch_distillation import make_plan_batches, render_prompt

    teacher_tokens = {
        plan["query"]["id"]: render_prompt(
            backend, plan["query"]["input"], plan["query"]["dataset"], plan["document"]["text"]
        ).numel()
        for plan in plans
    }
    student_tokens = {
        plan["query"]["id"]: render_prompt(backend, plan["query"]["input"], plan["query"]["dataset"]).numel()
        for plan in plans
    }
    return make_plan_batches(plans, batch_size, batch_size, teacher_tokens, student_tokens, max_padded_tokens)


def identity(manifest: dict[str, Any], splits: tuple[str, ...]) -> dict[str, Any]:
    selected = [row for split in splits for row in rows(manifest, split)]
    return {
        "splits": list(splits),
        "rows": [
            [row["query_id"], row["document_id"], row["document_hash"], row["answer_hash"], row["family"], row["split"]]
            for row in selected
        ],
        "digest": digest(selected),
    }


def common_values(manifest: dict[str, Any], splits: tuple[str, ...], **values: Any) -> dict[str, Any]:
    return {
        "row_identity": identity(manifest, splits),
        "family_order": ["drop", "ropes", "squad"],
        "family_loss_weighting": "equal_aggregate_family_weight_response_token_weighted",
        "effective_batch_size": 12,
        "max_padded_student_tokens": 2048,
        "reader_learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "gradient_clip": 1.0,
        "representation": "rank-effect",
        "reader_bottleneck": 32,
        "reader_parameters": 443200,
        "existing_primitives": {
            "representation": "official_d2l_k1_representation_ladder",
            "batching": "official_d2l_online_batch_distillation",
            "gold_training": "official_d2l_k1_representation_training",
            "checkpoint_resume": "official_d2l_joint_finetuning_joint_training",
            "fresh_generation": "official_d2l_k1_representation_evaluation",
        },
        **values,
    }


def write_plan(output: Path, contract: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    if output.exists() and not dry_run:
        saved = json.loads(output.read_text(encoding="utf-8"))
        validate_resume(saved, contract)
        return saved
    result = {"contract": contract, "status": "planned" if dry_run else "ready_for_model_execution"}
    if not dry_run:
        _atomic_json(output, result)
    return result


def load_contract(manifest_path: Path, stage: str, splits: tuple[str, ...], **values: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = load_frozen_manifest(manifest_path)
    if not all(rows(manifest, split) for split in splits):
        raise ValueError(f"QA_PROTOCOL manifest has no rows for {splits}")
    contract = stage_contract(manifest, stage, **common_values(manifest, splits, **values))
    return manifest, contract


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    temporary.replace(path)
