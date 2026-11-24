"""Select JOINT_FINETUNING joint writer-reader checkpoints by fresh validation."""

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
from official_d2l_compiler_finetuning_compiler_evaluation import (
    checkpoint_step,
    file_digest,
    paired_comparison,
    reorder_plans,
    selection_key,
)
from official_d2l_k1_representation_evaluation import (
    generate_reader_condition,
    zero_document_representation,
)
from official_d2l_k1_representation_ladder import (
    compile_k1_representation_batch,
    make_representation_transition,
)
from reader_banks import build_dataset_local_plans
from reader_banks import split_rows
from official_d2l_online_batch_distillation import plan_identity
from official_d2l_online_reader_evaluation import summarize_evaluation
from post_block_transitions import checkpoint


def row_for(plan: dict, responses: dict[str, str]) -> dict:
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


def load_transition(backend, layers, state: dict):
    transition = make_representation_transition(
        "rank-effect",
        backend.model.base_model.config.hidden_size,
        backend.model.hypernet.config.latent_size,
        len(layers),
        state["transition_bottleneck"],
    ).to(backend.model.device)
    if state["representation"] != "rank-effect" or state["execution_site"] != "post-block":
        raise ValueError("JOINT_FINETUNING checkpoint reader contract differs")
    transition.load_state_dict(state["transition"])
    transition.eval()
    return transition


def generate_joint(backend, layers, transition, compiled, query, scaling, max_new_tokens) -> str:
    return generate_reader_condition(
        backend,
        layers,
        compiled,
        transition,
        "rank-effect",
        scaling,
        query.input,
        max_new_tokens,
        query.dataset,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="gemma2_squad_matched_distill_scale60k_seed141")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--compiler_finetuning-validation", type=Path, required=True)
    parser.add_argument("--readout_ladder-validation", type=Path, required=True)
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
    if training["experiment"] != "JOINT_FINETUNING-joint-d2l-hypernetwork-rank-effect-reader":
        raise ValueError("training report is not JOINT_FINETUNING")
    checkpoint_paths = sorted(args.checkpoint_dir.glob("step-*.pt"), key=checkpoint_step)
    if not checkpoint_paths or checkpoint_step(checkpoint_paths[0]) != 0:
        raise ValueError("JOINT_FINETUNING fresh selection requires checkpoint zero")
    compiler_finetuning = json.loads(args.compiler_finetuning_validation.read_text())
    readout_ladder = json.loads(args.readout_ladder_validation.read_text())
    if compiler_finetuning["status"] != "completed" or readout_ladder["status"] != "completed":
        raise ValueError("comparison validation artifacts must be completed")

    root = Path(__file__).parents[2]
    cfg = load_config(root, args.config)
    payload = load_manifest(paths(root, cfg)["manifest"])
    documents = {document["id"]: document for document in payload["documents"]}
    _, plans = build_dataset_local_plans(
        split_rows(payload)["validation"],
        documents,
        split="validation",
        limit=args.max_source_queries,
        fixed_k=1,
    )
    validation_identity = compiler_finetuning["contracts"]["plans"]
    plans = reorder_plans(plans, validation_identity)
    diagnostic_identity = training["identity"]["validation"]
    diagnostic_query_ids = {item[0] for item in diagnostic_identity}
    selection_query_ids = {item[0] for item in validation_identity}
    identity = plan_identity(plans)
    baseline_by_query = {row["query_id"]: row for row in compiler_finetuning["baseline_rows"]}
    readout_ladder_by_query = {row["query_id"]: row for row in readout_ladder["rows"]}
    planned_query_ids = {plan["query"]["id"] for plan in plans}
    if not planned_query_ids <= set(baseline_by_query) or not planned_query_ids <= set(readout_ladder_by_query):
        raise ValueError("comparison rows differ from JOINT_FINETUNING validation plans")
    checkpoints = [
        {"step": checkpoint_step(path), "path": str(path), "sha256": file_digest(path)}
        for path in checkpoint_paths
    ]
    contracts = json.loads(
        json.dumps(
            {
                "config": args.config,
                "training_report": str(args.training_report),
                "training_report_sha256": file_digest(args.training_report),
                "compiler_finetuning_validation": str(args.compiler_finetuning_validation),
                "compiler_finetuning_validation_sha256": file_digest(args.compiler_finetuning_validation),
                "readout_ladder_validation": str(args.readout_ladder_validation),
                "readout_ladder_validation_sha256": file_digest(args.readout_ladder_validation),
                "max_source_queries": args.max_source_queries,
                "max_new_tokens": args.max_new_tokens,
                "compiler_chunk_batch_size": args.compiler_chunk_batch_size,
                "plans": identity,
                "training_ce_plan_count": len(diagnostic_identity),
                "training_ce_plan_overlap": len(diagnostic_query_ids & selection_query_ids),
                "checkpoints": checkpoints,
                "selection_order": ["qa_f1", "exact_match", "teacher_response_similarity_f1"],
                "exact_400_test_opened": False,
            }
        )
    )
    report = {
        "schema_version": 1,
        "experiment": "JOINT_FINETUNING-joint-rank-effect-fresh-validation",
        "status": "running",
        "contracts": contracts,
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        },
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
        from ctx_to_lora.utils import get_layers

        layers = get_layers(backend.model.base_model)
        scaling = backend.model.base_model.peft_config["default"].lora_alpha
        teacher_by_query = {
            query_id: row["responses"]["teacher"] for query_id, row in baseline_by_query.items()
        }
        native_by_query = {
            query_id: row["responses"]["native_d2l"] for query_id, row in baseline_by_query.items()
        }
        rank_by_query = {
            query_id: row["responses"]["rank_s004365"] for query_id, row in readout_ladder_by_query.items()
        }
        started = time.perf_counter()
        for checkpoint_info in checkpoints:
            key = str(checkpoint_info["step"])
            state = torch.load(checkpoint_info["path"], map_location=backend.model.device, weights_only=False)
            if json.loads(json.dumps(state["identity"])) != training["identity"]:
                raise ValueError(f"checkpoint {key} identity differs")
            if any(training["contracts"].get(name) != value for name, value in state["recipe"].items()):
                raise ValueError(f"checkpoint {key} recipe differs")
            if checkpoint_info["step"] == 0:
                current = backend.model.hypernet.state_dict()
                if any(not torch.equal(current[name], value) for name, value in state["hypernet"].items()):
                    raise RuntimeError("JOINT_FINETUNING checkpoint-zero writer differs from released D2L")
                transition = load_transition(backend, layers, state)
                if torch.count_nonzero(transition.output_projection.weight):
                    raise RuntimeError("JOINT_FINETUNING checkpoint-zero reader is not identity initialized")
                report["checkpoint_rows"][key] = [
                    row_for(
                        plan,
                        {
                            "teacher": teacher_by_query[plan["query"]["id"]],
                            "joint": native_by_query[plan["query"]["id"]],
                        },
                    )
                    for plan in plans
                ]
                checkpoint(args.output, report)
                continue
            backend.model.hypernet.load_state_dict(state["hypernet"])
            transition = load_transition(backend, layers, state)
            rows_for_step = report["checkpoint_rows"].setdefault(key, [])
            completed = {row["query_id"] for row in rows_for_step}
            for index, plan in enumerate(plans):
                query = Query(**plan["query"])
                if query.id in completed:
                    continue
                compiled, accounting = compile_k1_representation_batch(
                    backend,
                    [plan],
                    documents,
                    args.compiler_chunk_batch_size,
                )
                response = generate_joint(
                    backend,
                    layers,
                    transition,
                    compiled,
                    query,
                    scaling,
                    args.max_new_tokens,
                )
                rows_for_step.append(
                    {
                        **row_for(plan, {"teacher": teacher_by_query[query.id], "joint": response}),
                        "accounting": accounting,
                    }
                )
                if (index + 1) % 20 == 0:
                    checkpoint(args.output, report)
            checkpoint(args.output, report)
        report["checkpoint_summaries"] = {
            step: summarize_evaluation(rows, ("joint",))["joint"]
            for step, rows in report["checkpoint_rows"].items()
        }
        selected_step = max(
            report["checkpoint_summaries"],
            key=lambda step: selection_key(report["checkpoint_summaries"][step]),
        )
        selected_info = next(item for item in checkpoints if str(item["step"]) == selected_step)
        selected_state = torch.load(selected_info["path"], map_location=backend.model.device, weights_only=False)
        backend.model.hypernet.load_state_dict(selected_state["hypernet"])
        transition = load_transition(backend, layers, selected_state)
        selected_by_query = {
            row["query_id"]: row["responses"]["joint"]
            for row in report["checkpoint_rows"][selected_step]
        }
        controls = []
        for index, plan in enumerate(plans):
            query = Query(**plan["query"])
            compiled, _ = compile_k1_representation_batch(
                backend, [plan], documents, args.compiler_chunk_batch_size
            )
            wrong_plan = {**plan, "bank_document_ids": [plan["replacement_document_id"]]}
            wrong, _ = compile_k1_representation_batch(
                backend, [wrong_plan], documents, args.compiler_chunk_batch_size
            )
            zero = zero_document_representation(compiled)
            responses = {
                "teacher": teacher_by_query[query.id],
                "native_d2l": native_by_query[query.id],
                "readout_ladder_rank": rank_by_query[query.id],
                "joint": selected_by_query[query.id],
                "no_effect": generate_joint(
                    backend, layers, transition, zero, query, scaling, args.max_new_tokens
                ),
                "wrong_document": generate_joint(
                    backend, layers, transition, wrong, query, scaling, args.max_new_tokens
                ),
            }
            controls.append(row_for(plan, responses))
            if (index + 1) % 20 == 0:
                report["control_rows"] = controls
                checkpoint(args.output, report)
        report["control_rows"] = controls
        conditions = tuple(controls[0]["responses"])
        report["control_summaries"] = summarize_evaluation(controls, conditions)
        report["paired_comparisons"] = {
            "joint_vs_native": paired_comparison(controls, "joint", "native_d2l"),
            "joint_vs_readout_ladder_rank": paired_comparison(controls, "joint", "readout_ladder_rank"),
        }
        report["selection"] = {
            "step": int(selected_step),
            "checkpoint": selected_info,
            "metrics": report["checkpoint_summaries"][selected_step],
        }
        report["accounting"] = {
            "wall_seconds": time.perf_counter() - started,
            "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        }
        report["status"] = "completed"
        checkpoint(args.output, report)
        print(json.dumps({"selected_step": int(selected_step), "summaries": {
            name: values["summary"]["macro_over_k"]
            for name, values in report["control_summaries"].items()
        }}, indent=2))
    finally:
        backend.close()


if __name__ == "__main__":
    main()
