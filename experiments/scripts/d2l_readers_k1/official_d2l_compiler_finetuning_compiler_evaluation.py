"""Select COMPILER_FINETUNING compiler checkpoints by fresh document-disjoint validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch
import transformers

sys.path.insert(0, str(Path(__file__).parent))
from d2l_scale import load_config, load_manifest, paths
from d2l_scale_core import Query, score_response
from official_d2l_k1_representation_ladder import compile_k1_representation_batch
from reader_banks import build_dataset_local_plans
from reader_banks import split_rows
from official_d2l_online_batch_distillation import plan_identity
from official_d2l_online_reader_evaluation import summarize_evaluation
from post_block_transitions import checkpoint


def file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.stem.removeprefix("step-"))
    except ValueError as error:
        raise ValueError(f"invalid checkpoint filename: {path}") from error


def wrong_document_plans(plans: list[dict]) -> list[dict]:
    if len(plans) < 2:
        raise ValueError("wrong-document controls require at least two plans")
    result = []
    for index, plan in enumerate(plans):
        document_id = plan["bank_document_ids"][0]
        replacement = next(
            (
                candidate
                for offset in range(1, len(plans))
                if (candidate := plans[(index + offset) % len(plans)])["bank_document_ids"][0]
                != document_id
            ),
            None,
        )
        if replacement is None:
            raise ValueError("wrong-document controls require multiple documents")
        result.append({**plan, "bank_document_ids": replacement["bank_document_ids"]})
    return result


def reorder_plans(plans: list[dict], expected_identity: list) -> list[dict]:
    keyed = {
        (plan["query"]["id"], plan["k"], tuple(plan["bank_document_ids"])): plan
        for plan in plans
    }
    expected = [(query_id, k, tuple(document_ids)) for query_id, k, document_ids in expected_identity]
    if len(keyed) != len(plans) or set(keyed) != set(expected):
        raise ValueError("fresh-validation plans differ from the training contract")
    return [keyed[key] for key in expected]


def selection_key(summary: dict) -> tuple[float, float, float]:
    values = summary["summary"]["macro_over_k"]
    return (
        values["qa_f1"],
        values["exact_match"],
        values["teacher_response_similarity_f1"],
    )


def paired_comparison(
    rows: list[dict],
    candidate: str,
    reference: str,
    *,
    samples: int = 2000,
    seed: int = 172,
) -> dict:
    if not rows or samples < 1:
        raise ValueError("paired comparison requires rows and bootstrap samples")
    metrics = ("qa_f1", "exact_match", "rouge_l")
    values = {
        metric: [
            row["scores"][candidate][metric] - row["scores"][reference][metric]
            for row in rows
        ]
        for metric in metrics
    }
    documents: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        documents.setdefault(row["document_id"], []).append(index)
    document_ids = sorted(documents)
    generator = random.Random(seed)
    question_draws = {metric: [] for metric in metrics}
    document_draws = {metric: [] for metric in metrics}
    for _ in range(samples):
        question_indices = [generator.randrange(len(rows)) for _ in rows]
        selected_documents = [generator.choice(document_ids) for _ in document_ids]
        document_indices = [index for document in selected_documents for index in documents[document]]
        for metric in metrics:
            question_draws[metric].append(
                sum(values[metric][index] for index in question_indices) / len(question_indices)
            )
            document_draws[metric].append(
                sum(values[metric][index] for index in document_indices) / len(document_indices)
            )

    def interval(draws: list[float]) -> list[float]:
        ordered = sorted(draws)
        return [ordered[int(0.025 * samples)], ordered[min(int(0.975 * samples), samples - 1)]]

    return {
        "candidate": candidate,
        "reference": reference,
        "questions": len(rows),
        "documents": len(document_ids),
        "bootstrap_samples": samples,
        "deltas": {
            metric: {
                "mean": sum(values[metric]) / len(values[metric]),
                "question_bootstrap_95": interval(question_draws[metric]),
                "document_cluster_bootstrap_95": interval(document_draws[metric]),
            }
            for metric in metrics
        },
    }


def evaluation_row(plan: dict, responses: dict[str, str]) -> dict:
    query = Query(**plan["query"])
    teacher = responses["teacher"]
    return {
        "query_id": query.id,
        "document_id": plan["bank_document_ids"][0],
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
    }


def generate_compiled(backend, plan, documents, compiler_chunk_batch_size, max_new_tokens) -> tuple[str, dict]:
    compiled, accounting = compile_k1_representation_batch(
        backend,
        [plan],
        documents,
        compiler_chunk_batch_size,
    )
    query = Query(**plan["query"])
    response = backend.generate(
        query.input,
        state=compiled.factors,
        dataset=query.dataset,
        max_new_tokens=max_new_tokens,
    )["text"]
    return response, accounting


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="gemma2_squad_matched_distill_scale60k_seed141")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--max-source-queries", type=int, default=400)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--compiler-chunk-batch-size", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.max_source_queries, args.max_new_tokens, args.compiler_chunk_batch_size) < 1:
        raise ValueError("query, token, and compiler limits must be positive")
    if args.output.exists() != args.resume:
        expected = "existing" if args.resume else "absent"
        raise FileExistsError(f"evaluation output must be {expected}: {args.output}")
    training = json.loads(args.training_report.read_text())
    if training["experiment"] != "COMPILER_FINETUNING-answer-aligned-document-only-d2l-compiler":
        raise ValueError("training report is not an COMPILER_FINETUNING compiler run")
    checkpoint_paths = sorted(args.checkpoint_dir.glob("step-*.pt"), key=checkpoint_step)
    if not checkpoint_paths or checkpoint_step(checkpoint_paths[0]) != 0:
        raise ValueError("COMPILER_FINETUNING fresh selection requires checkpoint zero")

    root = Path(__file__).parents[2]
    cfg = load_config(root, args.config)
    payload = load_manifest(paths(root, cfg)["manifest"])
    documents = {document["id"]: document for document in payload["documents"]}
    rows = split_rows(payload)
    _, plans = build_dataset_local_plans(
        rows["validation"],
        documents,
        split="validation",
        limit=args.max_source_queries,
        fixed_k=1,
    )
    plans = reorder_plans(plans, training["identity"]["validation"])
    identity = plan_identity(plans)
    checkpoints = [
        {
            "step": checkpoint_step(path),
            "path": str(path),
            "sha256": file_digest(path),
        }
        for path in checkpoint_paths
    ]
    contracts = json.loads(json.dumps({
        "config": args.config,
        "training_report": str(args.training_report),
        "training_report_sha256": file_digest(args.training_report),
        "max_source_queries": args.max_source_queries,
        "max_new_tokens": args.max_new_tokens,
        "compiler_chunk_batch_size": args.compiler_chunk_batch_size,
        "selection_order": ["qa_f1", "exact_match", "teacher_response_similarity_f1"],
        "split": "validation",
        "plans": identity,
        "checkpoints": checkpoints,
        "exact_400_test_opened": False,
    }))
    report = {
        "schema_version": 1,
        "experiment": "COMPILER_FINETUNING-answer-aligned-document-only-d2l-compiler-fresh-validation",
        "status": "running",
        "contracts": contracts,
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        },
        "baseline_rows": [],
        "checkpoint_rows": {},
    }
    if args.resume:
        saved = json.loads(args.output.read_text())
        if saved["contracts"] != contracts:
            raise ValueError("fresh-validation resume contract differs")
        report = saved

    from d2l_scale_backend import OfficialBackend

    backend = OfficialBackend(cfg, root)
    try:
        started = time.perf_counter()
        completed_baselines = {row["query_id"] for row in report["baseline_rows"]}
        for index, plan in enumerate(plans):
            query = Query(**plan["query"])
            if query.id in completed_baselines:
                continue
            document = documents[plan["bank_document_ids"][0]]["text"]
            native, accounting = generate_compiled(
                backend,
                plan,
                documents,
                args.compiler_chunk_batch_size,
                args.max_new_tokens,
            )
            responses = {
                "teacher": backend.generate(
                    query.input,
                    context=document,
                    dataset=query.dataset,
                    max_new_tokens=args.max_new_tokens,
                )["text"],
                "base": backend.generate(
                    query.input,
                    dataset=query.dataset,
                    max_new_tokens=args.max_new_tokens,
                )["text"],
                "native_d2l": native,
            }
            report["baseline_rows"].append(
                {**evaluation_row(plan, responses), "accounting": accounting}
            )
            if (index + 1) % 20 == 0:
                checkpoint(args.output, report)
        checkpoint(args.output, report)
        teacher_by_query = {
            row["query_id"]: row["responses"]["teacher"] for row in report["baseline_rows"]
        }
        native_by_query = {
            row["query_id"]: row["responses"]["native_d2l"] for row in report["baseline_rows"]
        }

        for checkpoint_info in checkpoints:
            step_key = str(checkpoint_info["step"])
            rows_for_step = report["checkpoint_rows"].setdefault(step_key, [])
            state = torch.load(checkpoint_info["path"], map_location=backend.model.device, weights_only=False)
            checkpoint_identity = json.loads(json.dumps(state["identity"]))
            if checkpoint_identity != training["identity"]:
                raise ValueError(f"checkpoint {step_key} identity differs")
            if any(training["contracts"].get(key) != value for key, value in state["recipe"].items()):
                raise ValueError(f"checkpoint {step_key} recipe differs")
            if checkpoint_info["step"] == 0:
                current = backend.model.hypernet.state_dict()
                if any(not torch.equal(current[name], value) for name, value in state["hypernet"].items()):
                    raise RuntimeError("checkpoint zero hypernetwork differs from released D2L")
                rows_for_step = [
                    evaluation_row(
                        plan,
                        {
                            "teacher": teacher_by_query[plan["query"]["id"]],
                            "compiler": native_by_query[plan["query"]["id"]],
                        },
                    )
                    for plan in plans
                ]
                report["checkpoint_rows"][step_key] = rows_for_step
                checkpoint(args.output, report)
                continue
            completed = {row["query_id"] for row in rows_for_step}
            backend.model.hypernet.load_state_dict(state["hypernet"])
            for index, plan in enumerate(plans):
                query = Query(**plan["query"])
                if query.id in completed:
                    continue
                response, accounting = generate_compiled(
                    backend,
                    plan,
                    documents,
                    args.compiler_chunk_batch_size,
                    args.max_new_tokens,
                )
                responses = {"teacher": teacher_by_query[query.id], "compiler": response}
                rows_for_step.append({**evaluation_row(plan, responses), "accounting": accounting})
                if (index + 1) % 20 == 0:
                    checkpoint(args.output, report)
            checkpoint(args.output, report)

        report["baseline_summaries"] = summarize_evaluation(
            report["baseline_rows"], ("teacher", "base", "native_d2l")
        )
        report["checkpoint_summaries"] = {
            step: summarize_evaluation(rows_for_step, ("compiler",))["compiler"]
            for step, rows_for_step in report["checkpoint_rows"].items()
        }
        selected_step = max(
            report["checkpoint_summaries"],
            key=lambda step: selection_key(report["checkpoint_summaries"][step]),
        )
        selected_checkpoint = next(item for item in checkpoints if str(item["step"]) == selected_step)
        selected_state = torch.load(
            selected_checkpoint["path"], map_location=backend.model.device, weights_only=False
        )
        backend.model.hypernet.load_state_dict(selected_state["hypernet"])
        wrong_plans = wrong_document_plans(plans)
        control_rows = []
        with torch.inference_mode():
            bias_state = {
                module: {key: value.unsqueeze(0) for key, value in part.items()}
                for module, part in backend.model.hypernet.get_head_bias().items()
            }
        selected_by_query = {
            row["query_id"]: row["responses"]["compiler"]
            for row in report["checkpoint_rows"][selected_step]
        }
        for plan, wrong_plan in zip(plans, wrong_plans, strict=True):
            query = Query(**plan["query"])
            wrong_response, wrong_accounting = generate_compiled(
                backend,
                wrong_plan,
                documents,
                args.compiler_chunk_batch_size,
                args.max_new_tokens,
            )
            responses = {
                "teacher": teacher_by_query[query.id],
                "selected_compiler": selected_by_query[query.id],
                "no_effect": backend.generate(
                    query.input,
                    state=bias_state,
                    dataset=query.dataset,
                    max_new_tokens=args.max_new_tokens,
                )["text"],
                "wrong_document": wrong_response,
            }
            control_rows.append(
                {**evaluation_row(plan, responses), "wrong_document_accounting": wrong_accounting}
            )
            if len(control_rows) % 20 == 0:
                report["control_rows"] = control_rows
                checkpoint(args.output, report)
        report["control_rows"] = control_rows
        report["control_summaries"] = summarize_evaluation(
            control_rows,
            ("teacher", "selected_compiler", "no_effect", "wrong_document"),
        )
        report["selection"] = {
            "step": int(selected_step),
            "checkpoint": selected_checkpoint,
            "metrics": report["checkpoint_summaries"][selected_step],
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
                    "selected_step": int(selected_step),
                    "selection": selection_key(report["checkpoint_summaries"][selected_step]),
                },
                indent=2,
            )
        )
    finally:
        backend.close()


if __name__ == "__main__":
    main()
