"""Fresh-generation evaluation and controls for READOUT_LADDER representation readers."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from d2l_scale import load_config, load_manifest, paths
from d2l_scale_core import Query, score_response
from official_d2l_k1_representation_ladder import (
    CompiledRepresentationBatch,
    compile_k1_representation_batch,
    installed_nonlinear_rank_operator,
    installed_representation_transition,
    make_representation_transition,
)
from reader_banks import build_dataset_local_plans
from reader_banks import split_rows
from official_d2l_online_batch_distillation import bank_context, render_prompt
from official_d2l_online_reader_evaluation import parse_reader_states, summarize_evaluation
from post_block_transitions import checkpoint


def generate_reader_condition(
    backend,
    layers,
    compiled,
    reader,
    representation: str,
    scaling: float,
    question: str,
    max_new_tokens: int,
    dataset: str,
    *,
    apply_document_factors: bool = True,
) -> str:
    prompt = render_prompt(backend, question, dataset).unsqueeze(0).to(backend.model.device)
    backend.model.reset()
    with (
        installed_representation_transition(
            backend.model.base_model,
            layers,
            compiled,
            reader,
            representation,
            scaling,
            apply_document_factors=apply_document_factors,
            official_backend=backend,
        ),
        torch.inference_mode(),
    ):
        generated = backend.model.base_model.generate(
            prompt,
            attention_mask=torch.ones_like(prompt),
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=backend.tokenizer.pad_token_id,
        )
    return backend.tokenizer.decode(
        generated[0, prompt.shape[1] :],
        skip_special_tokens=True,
    ).strip()


def generate_nonlinear_rank_condition(
    backend,
    layers,
    compiled,
    operator,
    scaling: float,
    question: str,
    max_new_tokens: int,
    dataset: str,
) -> str:
    prompt = render_prompt(backend, question, dataset).unsqueeze(0).to(backend.model.device)
    backend.model.reset()
    with (
        installed_nonlinear_rank_operator(
            backend.model.base_model,
            layers,
            compiled,
            operator,
            scaling,
            official_backend=backend,
        ),
        torch.inference_mode(),
    ):
        generated = backend.model.base_model.generate(
            prompt,
            attention_mask=torch.ones_like(prompt),
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=backend.tokenizer.pad_token_id,
        )
    return backend.tokenizer.decode(
        generated[0, prompt.shape[1] :],
        skip_special_tokens=True,
    ).strip()


def zero_document_representation(compiled: CompiledRepresentationBatch) -> CompiledRepresentationBatch:
    generated = {
        module: {key: torch.zeros_like(value) for key, value in part.items()}
        for module, part in compiled.generated_factors.items()
    }
    factors = {
        module: {key: value.clone() for key, value in part.items()}
        for module, part in compiled.bias_factors.items()
    }
    return CompiledRepresentationBatch(
        factors=factors,
        generated_factors=generated,
        bias_factors=compiled.bias_factors,
        pre_head_latents=torch.zeros_like(compiled.pre_head_latents),
        projected_source=(
            torch.zeros_like(compiled.projected_source)
            if compiled.projected_source is not None
            else None
        ),
        projected_source_mask=compiled.projected_source_mask,
    )


def query_ids_from_artifact(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    rows = payload.get("rows")
    if rows is None:
        raise ValueError("query-ID artifact has no rows")
    query_ids = [row["query_id"] for row in rows]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("query-ID artifact contains duplicates")
    return query_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="gemma2_squad_matched_distill_scale60k_seed141")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reader-state", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--max-source-queries", type=int, default=400)
    parser.add_argument("--query-ids-from", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--compiler-chunk-batch-size", type=int, default=8)
    parser.add_argument("--include-controls", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.max_source_queries, args.max_new_tokens, args.compiler_chunk_batch_size) < 1:
        raise ValueError("query, output-token, and compiler limits must be positive")
    reader_paths = parse_reader_states(args.reader_state)
    if not reader_paths:
        raise ValueError("at least one reader state is required")
    if args.output.exists() != args.resume:
        expected = "existing" if args.resume else "absent"
        raise FileExistsError(f"evaluation output must be {expected}: {args.output}")

    root = Path(__file__).parents[2]
    cfg = load_config(root, args.config)
    payload = load_manifest(paths(root, cfg)["manifest"])
    documents = {document["id"]: document for document in payload["documents"]}
    rows = split_rows(payload)[args.split]
    query_ids = query_ids_from_artifact(args.query_ids_from) if args.query_ids_from else None
    if query_ids is not None:
        by_query = {row["query"]["id"]: row for row in rows}
        missing = [query_id for query_id in query_ids if query_id not in by_query]
        if missing:
            raise ValueError(f"query-ID artifact contains {len(missing)} IDs outside {args.split}")
        rows = [by_query[query_id] for query_id in query_ids]
    _, plans = build_dataset_local_plans(
        rows,
        documents,
        split=args.split,
        limit=None if query_ids is not None else args.max_source_queries,
        fixed_k=1,
    )

    contracts = {
        "config": args.config,
        "split": args.split,
        "query_ids": query_ids,
        "reader_states": {name: str(path) for name, path in reader_paths.items()},
        "include_controls": args.include_controls,
        "max_new_tokens": args.max_new_tokens,
        "compiler_chunk_batch_size": args.compiler_chunk_batch_size,
        "plans": [[plan["query"]["id"], plan["bank_document_ids"]] for plan in plans],
    }
    report = {
        "schema_version": 1,
        "experiment": "READOUT_LADDER-k1-compiled-representation-evaluation",
        "status": "running",
        "contracts": contracts,
        "rows": [],
    }
    if args.resume:
        saved = json.loads(args.output.read_text())
        if saved["contracts"] != contracts:
            raise ValueError("evaluation resume contract differs")
        report = saved
    completed = {row["query_id"] for row in report["rows"]}

    from d2l_scale_backend import OfficialBackend

    backend = OfficialBackend(cfg, root)
    try:
        from ctx_to_lora.utils import get_layers

        model = backend.model.base_model
        layers = get_layers(model)
        scaling = model.peft_config["default"].lora_alpha
        latent_width = backend.model.hypernet.config.latent_size
        readers = {}
        metadata = {}
        for name, path in reader_paths.items():
            saved = torch.load(path, map_location=model.device, weights_only=False)
            representation = saved["representation"]
            reader = make_representation_transition(
                representation,
                model.config.hidden_size,
                latent_width,
                len(layers),
                saved["transition_bottleneck"],
            ).to(model.device)
            if reader.memory_width != saved["memory_width"] or saved["execution_site"] != "post-block":
                raise ValueError(f"reader contract mismatch: {path}")
            reader.load_state_dict(saved["transition"])
            reader.eval()
            readers[name] = reader
            metadata[name] = {
                "representation": representation,
                "selected_step": saved["selected_step"],
            }
        report["reader_metadata"] = metadata
        started = time.perf_counter()
        for plan in plans:
            query = Query(**plan["query"])
            if query.id in completed:
                continue
            compiled, compilation = compile_k1_representation_batch(
                backend,
                [plan],
                documents,
                args.compiler_chunk_batch_size,
                capture_projected=any(
                    value["representation"] == "projected-source" for value in metadata.values()
                ),
            )
            teacher = backend.generate(
                query.input,
                context=bank_context(plan, documents),
                dataset=query.dataset,
                max_new_tokens=args.max_new_tokens,
            )["text"]
            responses = {
                "teacher": teacher,
                "base": backend.generate(
                    query.input,
                    dataset=query.dataset,
                    max_new_tokens=args.max_new_tokens,
                )["text"],
                "native_d2l": backend.generate(
                    query.input,
                    state=compiled.factors,
                    dataset=query.dataset,
                    max_new_tokens=args.max_new_tokens,
                )["text"],
            }
            zero = None
            wrong = None
            if args.include_controls:
                zero = zero_document_representation(compiled)
                wrong_plan = {**plan, "bank_document_ids": [plan["replacement_document_id"]]}
                wrong, _ = compile_k1_representation_batch(
                    backend,
                    [wrong_plan],
                    documents,
                    args.compiler_chunk_batch_size,
                    capture_projected=any(
                        value["representation"] == "projected-source" for value in metadata.values()
                    ),
                )
            for name, reader in readers.items():
                representation = metadata[name]["representation"]
                responses[name] = generate_reader_condition(
                    backend,
                    layers,
                    compiled,
                    reader,
                    representation,
                    scaling,
                    query.input,
                    args.max_new_tokens,
                    query.dataset,
                )
                if args.include_controls:
                    responses[f"{name}_no_effect"] = generate_reader_condition(
                        backend,
                        layers,
                        zero,
                        reader,
                        representation,
                        scaling,
                        query.input,
                        args.max_new_tokens,
                        query.dataset,
                    )
                    responses[f"{name}_wrong_document"] = generate_reader_condition(
                        backend,
                        layers,
                        wrong,
                        reader,
                        representation,
                        scaling,
                        query.input,
                        args.max_new_tokens,
                        query.dataset,
                    )
                    if representation == "pre-head-latent":
                        responses[f"{name}_latent_only"] = generate_reader_condition(
                            backend,
                            layers,
                            compiled,
                            reader,
                            representation,
                            scaling,
                            query.input,
                            args.max_new_tokens,
                            query.dataset,
                            apply_document_factors=False,
                        )
            report["rows"].append(
                {
                    "query_id": query.id,
                    "document_id": plan["document"]["id"],
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
                    "compilation": compilation,
                }
            )
            checkpoint(args.output, report)
        conditions = tuple(report["rows"][0]["responses"])
        report["summaries"] = summarize_evaluation(report["rows"], conditions)
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
