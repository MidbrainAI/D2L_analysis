"""Evaluate fresh-selected COMPILER_FINETUNING on EXACT_400's exact held-out 400 questions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import transformers

sys.path.insert(0, str(Path(__file__).parent))
from d2l_scale import load_config, load_manifest, paths
from d2l_scale_core import Query, score_response
from official_d2l_compiler_finetuning_compiler_evaluation import file_digest, paired_comparison
from official_d2l_k1_representation_evaluation import query_ids_from_artifact
from official_d2l_k1_representation_ladder import compile_k1_representation_batch
from reader_banks import build_dataset_local_plans
from reader_banks import split_rows
from official_d2l_online_reader_evaluation import summarize_evaluation
from post_block_transitions import checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="gemma2_squad_matched_distill_scale60k_seed141")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fresh-validation", type=Path, required=True)
    parser.add_argument("--comparison-artifact", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--compiler-chunk-batch-size", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.max_new_tokens, args.compiler_chunk_batch_size) < 1:
        raise ValueError("token and compiler limits must be positive")
    if args.output.exists() != args.resume:
        expected = "existing" if args.resume else "absent"
        raise FileExistsError(f"held-out output must be {expected}: {args.output}")

    selection = json.loads(args.fresh_validation.read_text())
    if selection["status"] != "completed":
        raise ValueError("fresh-validation selection is incomplete")
    selected = selection["selection"]
    selected_path = Path(selected["checkpoint"]["path"])
    if file_digest(selected_path) != selected["checkpoint"]["sha256"]:
        raise ValueError("selected checkpoint hash differs from fresh validation")
    comparison = json.loads(args.comparison_artifact.read_text())
    if comparison["status"] != "completed" or comparison["contracts"]["split"] != "test":
        raise ValueError("comparison artifact is not the completed exact held-out evaluation")
    query_ids = query_ids_from_artifact(args.comparison_artifact)

    root = Path(__file__).parents[2]
    cfg = load_config(root, args.config)
    payload = load_manifest(paths(root, cfg)["manifest"])
    documents = {document["id"]: document for document in payload["documents"]}
    by_query = {row["query"]["id"]: row for row in split_rows(payload)["test"]}
    missing = [query_id for query_id in query_ids if query_id not in by_query]
    if missing:
        raise ValueError(f"exact held-out artifact contains {len(missing)} unknown query IDs")
    rows = [by_query[query_id] for query_id in query_ids]
    _, plans = build_dataset_local_plans(rows, documents, split="test", limit=None, fixed_k=1)
    comparison_by_query = {row["query_id"]: row for row in comparison["rows"]}
    if [plan["query"]["id"] for plan in plans] != query_ids:
        raise ValueError("held-out plan order differs from the exact EXACT_400 artifact")
    if any(
        comparison_by_query[plan["query"]["id"]]["document_id"] != plan["bank_document_ids"][0]
        for plan in plans
    ):
        raise ValueError("held-out source documents differ from the exact EXACT_400 artifact")

    contracts = {
        "config": args.config,
        "fresh_validation": str(args.fresh_validation),
        "fresh_validation_sha256": file_digest(args.fresh_validation),
        "comparison_artifact": str(args.comparison_artifact),
        "comparison_artifact_sha256": file_digest(args.comparison_artifact),
        "selected_step": selected["step"],
        "selected_checkpoint": str(selected_path),
        "selected_checkpoint_sha256": selected["checkpoint"]["sha256"],
        "query_ids": query_ids,
        "max_new_tokens": args.max_new_tokens,
        "compiler_chunk_batch_size": args.compiler_chunk_batch_size,
        "split": "test",
        "suite": "EXACT_400-exact-heldout400",
    }
    report = {
        "schema_version": 1,
        "experiment": "COMPILER_FINETUNING-answer-aligned-document-only-d2l-compiler-heldout",
        "status": "running",
        "contracts": contracts,
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        },
        "rows": [],
    }
    if args.resume:
        saved = json.loads(args.output.read_text())
        if saved["contracts"] != contracts:
            raise ValueError("held-out resume contract differs")
        report = saved
    completed = {row["query_id"] for row in report["rows"]}

    from d2l_scale_backend import OfficialBackend

    backend = OfficialBackend(cfg, root)
    try:
        state = torch.load(selected_path, map_location=backend.model.device, weights_only=False)
        if state["step"] != selected["step"]:
            raise ValueError("selected checkpoint step differs")
        backend.model.hypernet.load_state_dict(state["hypernet"])
        with torch.inference_mode():
            bias_state = {
                module: {key: value.unsqueeze(0) for key, value in part.items()}
                for module, part in backend.model.hypernet.get_head_bias().items()
            }
        started = time.perf_counter()
        for index, plan in enumerate(plans):
            query = Query(**plan["query"])
            if query.id in completed:
                continue
            compiled, compilation = compile_k1_representation_batch(
                backend,
                [plan],
                documents,
                args.compiler_chunk_batch_size,
            )
            wrong_plan = {**plan, "bank_document_ids": [plan["replacement_document_id"]]}
            wrong, wrong_compilation = compile_k1_representation_batch(
                backend,
                [wrong_plan],
                documents,
                args.compiler_chunk_batch_size,
            )
            baseline = comparison_by_query[query.id]["responses"]
            responses = {
                "teacher": baseline["teacher"],
                "base": baseline["base"],
                "native_d2l": baseline["native_d2l"],
                "readout_ladder_rank": baseline["rank"],
                "compiler_finetuning_compiler": backend.generate(
                    query.input,
                    state=compiled.factors,
                    dataset=query.dataset,
                    max_new_tokens=args.max_new_tokens,
                )["text"],
                "compiler_finetuning_no_effect": backend.generate(
                    query.input,
                    state=bias_state,
                    dataset=query.dataset,
                    max_new_tokens=args.max_new_tokens,
                )["text"],
                "compiler_finetuning_wrong_document": backend.generate(
                    query.input,
                    state=wrong.factors,
                    dataset=query.dataset,
                    max_new_tokens=args.max_new_tokens,
                )["text"],
            }
            teacher = responses["teacher"]
            report["rows"].append(
                {
                    "query_id": query.id,
                    "document_id": plan["bank_document_ids"][0],
                    "replacement_document_id": plan["replacement_document_id"],
                    "dataset": query.dataset,
                    "k": 1,
                    "responses": responses,
                    "scores": {
                        condition: score_response(response, query.answers)
                        for condition, response in responses.items()
                    },
                    "teacher_similarity": {
                        condition: score_response(response, [teacher])
                        for condition, response in responses.items()
                    },
                    "accounting": {
                        "correct_document": compilation,
                        "wrong_document": wrong_compilation,
                    },
                }
            )
            if (index + 1) % 20 == 0:
                checkpoint(args.output, report)
        conditions = tuple(report["rows"][0]["responses"])
        report["summaries"] = summarize_evaluation(report["rows"], conditions)
        report["paired_comparisons"] = {
            "compiler_vs_native": paired_comparison(
                report["rows"], "compiler_finetuning_compiler", "native_d2l"
            ),
            "compiler_vs_readout_ladder_rank": paired_comparison(
                report["rows"], "compiler_finetuning_compiler", "readout_ladder_rank"
            ),
        }
        report["accounting"] = {
            "wall_seconds": time.perf_counter() - started,
            "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        }
        report["status"] = "completed"
        checkpoint(args.output, report)
        print(
            json.dumps(
                {
                    condition: values["summary"]["macro_over_k"]
                    for condition, values in report["summaries"].items()
                },
                indent=2,
            )
        )
    finally:
        backend.close()


if __name__ == "__main__":
    main()
