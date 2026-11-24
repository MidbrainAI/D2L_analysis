"""Cache-free fresh-generation evaluation for online-distilled D2L readers."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from d2l_scale import load_config, load_manifest, paths
from d2l_scale_core import Query, score_response
from post_block_transitions import installed_direct_transition
from reader_banks import build_dataset_local_plans
from reader_banks import split_rows, summarize_rows
from official_d2l_online_batch_distillation import (
    bank_context,
    compile_bank_batch,
)
from official_d2l_online_batch_distillation import (
    render_prompt as render_online_prompt,
)
from post_block_transitions import checkpoint, generate_condition, make_state_transition


def parse_reader_states(specifications: list[str]) -> dict[str, Path]:
    result = {}
    for specification in specifications:
        name, separator, path = specification.partition("=")
        if not separator or not name or not path:
            raise ValueError("reader states must use NAME=PATH")
        if name in ("teacher", "base", "native_sum") or name in result:
            raise ValueError(f"invalid or duplicate reader name: {name}")
        result[name] = Path(path)
    return result


def generate_reader_condition(
    backend,
    layers,
    state,
    reader,
    scaling: float,
    question: str,
    max_new_tokens: int,
    dataset: str,
) -> str:
    if reader.execution_site != "write-site":
        return generate_condition(
            backend,
            layers,
            state,
            reader,
            scaling,
            question,
            max_new_tokens,
            dataset,
        )
    prompt = render_online_prompt(backend, question, dataset).unsqueeze(0).to(backend.model.device)
    backend.model.reset()
    with installed_direct_transition(
        backend.model.base_model,
        layers,
        state,
        reader,
        scaling,
    ), torch.inference_mode():
        generated = backend.model.base_model.generate(
            prompt,
            attention_mask=torch.ones_like(prompt),
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=backend.tokenizer.pad_token_id,
        )
    return backend.tokenizer.decode(generated[0, prompt.shape[1] :], skip_special_tokens=True).strip()


def summarize_evaluation(rows: list[dict], conditions: tuple[str, ...]) -> dict:
    def summarize_with_similarity(condition_rows: list[dict]) -> dict:
        summary = summarize_rows(condition_rows)
        by_k = defaultdict(list)
        for row in condition_rows:
            by_k[str(row["k"])].append(row["teacher_similarity"]["qa_f1"])
        for k, values in by_k.items():
            summary["per_k"][k]["teacher_response_similarity_f1"] = sum(values) / len(values)
        summary["macro_over_k"]["teacher_response_similarity_f1"] = sum(
            values["teacher_response_similarity_f1"] for values in summary["per_k"].values()
        ) / len(summary["per_k"])
        return summary

    result = {}
    for condition in conditions:
        condition_rows = [
            {
                "query_id": row["query_id"],
                "dataset": row["dataset"],
                "k": row["k"],
                "response": row["responses"][condition],
                "score": row["scores"][condition],
                "teacher_similarity": row["teacher_similarity"][condition],
            }
            for row in rows
        ]
        datasets = sorted({row["dataset"] for row in condition_rows})
        result[condition] = {
            "summary": summarize_with_similarity(condition_rows),
            "by_dataset": {
                dataset: summarize_with_similarity([row for row in condition_rows if row["dataset"] == dataset])
                for dataset in datasets
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--squad-reader-state", type=Path)
    parser.add_argument("--mixed-reader-state", type=Path)
    parser.add_argument("--reader-state", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--max-test-source-queries", type=int, default=20)
    parser.add_argument("--query-ids-from", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--compiler-chunk-batch-size", type=int, default=8)
    parser.add_argument("--k", type=int)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--include-reader-controls", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.max_test_source_queries, args.max_new_tokens, args.compiler_chunk_batch_size) < 1:
        raise ValueError("query, token, and compiler limits must be positive")
    if args.k is not None and (args.k < 1 or args.k > 10):
        raise ValueError("K must be between 1 and 10")
    reader_paths = parse_reader_states(args.reader_state)
    if args.squad_reader_state is not None:
        reader_paths["squad_reader"] = args.squad_reader_state
    if args.mixed_reader_state is not None:
        reader_paths["mixed_reader"] = args.mixed_reader_state
    if not reader_paths:
        raise ValueError("at least one reader state is required")
    if args.include_reader_controls:
        generated_names = {
            condition
            for name in reader_paths
            for condition in (f"{name}_no_effect", f"{name}_wrong_document")
        }
        if generated_names & reader_paths.keys():
            raise ValueError("reader names collide with generated control names")
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    if args.resume and not args.output.exists():
        raise FileNotFoundError(args.output)

    root = Path(__file__).parents[2]
    cfg = load_config(root, args.config)
    payload = load_manifest(paths(root, cfg)["manifest"])
    documents = {document["id"]: document for document in payload["documents"]}
    rows_by_split = split_rows(payload)
    test_rows = rows_by_split[args.split]
    query_ids = None
    if args.query_ids_from is not None:
        source = json.loads(args.query_ids_from.read_text())
        query_ids = [row["query_id"] for row in source["rows"]]
        by_query_id = {row["query"]["id"]: row for row in test_rows}
        missing = [query_id for query_id in query_ids if query_id not in by_query_id]
        if missing:
            raise ValueError(f"query-ID source contains {len(missing)} IDs outside the {args.split} split")
        test_rows = [by_query_id[query_id] for query_id in query_ids]
    _, plans = build_dataset_local_plans(
        test_rows,
        documents,
        split=args.split,
        limit=None if query_ids is not None else args.max_test_source_queries,
        nested=args.k is None,
        fixed_k=args.k,
    )
    contracts = {
        "config": args.config,
        "split": args.split,
        "max_test_source_queries_per_dataset": args.max_test_source_queries,
        "query_ids": query_ids,
        "fixed_k": args.k,
        "reader_states": {name: str(path) for name, path in reader_paths.items()},
        "include_reader_controls": args.include_reader_controls,
        "max_new_tokens": args.max_new_tokens,
        "compiler_chunk_batch_size": args.compiler_chunk_batch_size,
        "plans": [[plan["query"]["id"], plan["k"], plan["bank_document_ids"]] for plan in plans],
        "persistent_lora_cache": False,
        "persistent_teacher_traces": False,
    }
    report = {
        "schema_version": 1,
        "experiment": "official-d2l-online-reader-evaluation",
        "status": "running",
        "contracts": contracts,
        "rows": [],
    }
    if args.resume:
        saved = json.loads(args.output.read_text())
        if saved["contracts"] != contracts:
            raise ValueError("evaluation resume contract differs")
        report = saved
    completed = {(row["query_id"], row["k"]) for row in report["rows"]}

    from d2l_scale_backend import OfficialBackend

    backend = OfficialBackend(cfg, root)
    try:
        from ctx_to_lora.utils import get_layers

        model = backend.model.base_model
        layers = get_layers(model)
        scaling = model.peft_config["default"].lora_alpha
        readers = {}
        for name, path in reader_paths.items():
            saved = torch.load(path, map_location=model.device, weights_only=False)
            reader = make_state_transition(
                saved.get("transition_type", "shared"),
                model.config.hidden_size,
                len(layers),
                saved.get("transition_bottleneck", 32),
            ).to(model.device)
            reader.load_state_dict(saved["transition"])
            readers[name] = reader
        for reader in readers.values():
            reader.eval()
        started = time.perf_counter()
        for plan in plans:
            query = Query(**plan["query"])
            key = (query.id, plan["k"])
            if key in completed:
                continue
            teacher = backend.generate(
                query.input,
                context=bank_context(plan, documents),
                dataset=query.dataset,
                max_new_tokens=args.max_new_tokens,
            )["text"]
            state, compilation = compile_bank_batch(
                backend,
                [plan],
                documents,
                args.compiler_chunk_batch_size,
            )
            responses = {
                "teacher": teacher,
                "base": backend.generate(
                    query.input,
                    dataset=query.dataset,
                    max_new_tokens=args.max_new_tokens,
                )["text"],
                "native_sum": backend.generate(
                    query.input,
                    state=state,
                    dataset=query.dataset,
                    max_new_tokens=args.max_new_tokens,
                )["text"],
            }
            responses.update(
                {
                    name: generate_reader_condition(
                        backend,
                        layers,
                        state,
                        reader,
                        scaling,
                        query.input,
                        args.max_new_tokens,
                        query.dataset,
                    )
                    for name, reader in readers.items()
                }
            )
            if args.include_reader_controls:
                wrong_plan = {**plan, "bank_document_ids": [plan["replacement_document_id"]]}
                wrong_state, _ = compile_bank_batch(
                    backend,
                    [wrong_plan],
                    documents,
                    args.compiler_chunk_batch_size,
                )
                for name, reader in readers.items():
                    responses[f"{name}_no_effect"] = generate_reader_condition(
                        backend,
                        layers,
                        None,
                        reader,
                        scaling,
                        query.input,
                        args.max_new_tokens,
                        query.dataset,
                    )
                    responses[f"{name}_wrong_document"] = generate_reader_condition(
                        backend,
                        layers,
                        wrong_state,
                        reader,
                        scaling,
                        query.input,
                        args.max_new_tokens,
                        query.dataset,
                    )
            report["rows"].append(
                {
                    "query_id": query.id,
                    "dataset": query.dataset,
                    "k": plan["k"],
                    "bank_document_ids": plan["bank_document_ids"],
                    "responses": responses,
                    "scores": {
                        name: score_response(response, query.answers) for name, response in responses.items()
                    },
                    "teacher_similarity": {
                        name: score_response(response, [teacher]) for name, response in responses.items()
                    },
                    "compilation": compilation,
                }
            )
            report["completed"] = len(report["rows"])
            checkpoint(args.output, report)
        controls = (
            tuple(
                condition
                for name in readers
                for condition in (f"{name}_no_effect", f"{name}_wrong_document")
            )
            if args.include_reader_controls
            else ()
        )
        conditions = ("teacher", "base", "native_sum", *readers, *controls)
        report["evaluation"] = summarize_evaluation(report["rows"], conditions)
        report["status"] = "completed"
        report["accounting"] = {
            "plans": len(report["rows"]),
            "wall_seconds_this_process": time.perf_counter() - started,
            "persistent_lora_bytes": 0,
            "persistent_trace_bytes": 0,
        }
        checkpoint(args.output, report)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
