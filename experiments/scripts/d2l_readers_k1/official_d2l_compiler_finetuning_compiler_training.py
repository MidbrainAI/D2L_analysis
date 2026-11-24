"""Train COMPILER_FINETUNING's released document-only D2L compiler with gold-answer CE."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import transformers
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from d2l_scale import load_config, load_manifest, paths
from d2l_scale_core import Query
from official_d2l_k1_representation_ladder import compile_k1_representation_batch
from official_d2l_k1_representation_training import gold_answer_loss, make_gold_batch
from reader_banks import build_dataset_local_plans
from reader_banks import split_rows
from official_d2l_online_batch_distillation import (
    atomic_torch_save,
    concatenate_generated_parts,
    make_plan_batches,
    pad_sequences,
    plan_identity,
    render_prompt,
)
from post_block_transitions import tensor_digest

SEED = 168
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
DEFAULT_BATCH_SIZE = 12
DEFAULT_PADDED_STUDENT_TOKEN_BUDGET = 2048
DEFAULT_VALIDATION_INTERVAL = 500
DEFAULT_CHECKPOINT_INTERVAL = 100


def trainability_audit(model) -> dict:
    groups = {
        "base_model": model.base_model,
        "context_encoder": model.ctx_encoder,
        "hypernetwork": model.hypernet,
    }
    audit = {}
    for name, module in groups.items():
        parameters = list(module.named_parameters())
        audit[name] = {
            "parameters": sum(parameter.numel() for _, parameter in parameters),
            "parameter_bytes": sum(
                parameter.numel() * parameter.element_size() for _, parameter in parameters
            ),
            "trainable_parameters": sum(
                parameter.numel() for _, parameter in parameters if parameter.requires_grad
            ),
            "trainable_names": [parameter_name for parameter_name, parameter in parameters if parameter.requires_grad],
        }
    if audit["base_model"]["trainable_parameters"] or audit["context_encoder"]["trainable_parameters"]:
        raise ValueError("COMPILER_FINETUNING requires frozen target and context-encoder models")
    if not audit["hypernetwork"]["trainable_parameters"]:
        raise ValueError("COMPILER_FINETUNING requires a trainable D2L hypernetwork")
    return audit


def document_chunks(backend, plans: list[dict], documents: dict[str, dict]) -> tuple[list[torch.Tensor], dict]:
    if not plans or any(plan["k"] != 1 or len(plan["bank_document_ids"]) != 1 for plan in plans):
        raise ValueError("COMPILER_FINETUNING accepts only nonempty K=1 batches")
    chunks = []
    source_tokens = 0
    for plan in plans:
        document = documents[plan["bank_document_ids"][0]]["text"]
        ids = backend.tokenize_context({"context": [document]}, backend.tokenizer)["ctx_ids"][0]
        split = backend.split_context(
            {"ctx_ids": ids},
            model_name_or_path=backend.model_id,
            num_chunk_probs=None,
            max_chunk_len=backend.config["model"]["context_chunk_len"],
            min_chunk_len=0,
            max_num_split=None,
            is_train=False,
        )["ctx_ids"]
        if len(split) != 1:
            raise ValueError("COMPILER_FINETUNING requires exactly one compiler chunk per document")
        chunks.append(torch.tensor(split[0], dtype=torch.long))
        source_tokens += len(ids)
    return chunks, {
        "document_occurrences": len(plans),
        "context_chunks": len(chunks),
        "source_tokens": source_tokens,
    }


def compile_trainable_batch(
    backend,
    plans: list[dict],
    documents: dict[str, dict],
    compiler_chunk_batch_size: int,
    *,
    require_grad: bool = True,
) -> tuple[dict, dict]:
    chunks, accounting = document_chunks(backend, plans, documents)
    generated_parts = []
    for start in range(0, len(chunks), compiler_chunk_batch_size):
        ctx_ids, ctx_mask = pad_sequences(
            chunks[start : start + compiler_chunk_batch_size],
            backend.tokenizer.pad_token_id,
            device=backend.model.device,
        )
        generated, layernorms = backend.model.generate_weights(ctx_ids, ctx_mask, None)
        if layernorms:
            raise ValueError("layernorm adaptation is outside the COMPILER_FINETUNING protocol")
        generated_parts.append(generated)
    generated = concatenate_generated_parts(generated_parts)
    n_chunks = torch.ones(len(plans), device=backend.model.device, dtype=torch.int32)
    head_bias = (
        backend.model.hypernet.get_head_bias()
        if backend.model.hypernet.config.use_bias
        else None
    )
    factors = backend.merge(generated, n_chunks, lora_bias=head_bias, bias_scaler=1.0)
    if require_grad and any(not tensor.requires_grad for part in factors.values() for tensor in part.values()):
        raise RuntimeError("compiled COMPILER_FINETUNING factors are detached from the hypernetwork")
    accounting.update(
        {
            "generated_factor_bytes": sum(
                tensor.numel() * tensor.element_size()
                for part in generated.values()
                for tensor in part.values()
            ),
            "loaded_factor_bytes": sum(
                tensor.numel() * tensor.element_size()
                for part in factors.values()
                for tensor in part.values()
            ),
        }
    )
    return factors, accounting


def forward_with_factors(backend, factors: dict, targets: dict) -> torch.Tensor:
    backend.model.reset()
    backend.model.patch_lora_forward()
    try:
        backend.install(
            backend.model.base_model,
            backend.model.hypernet.layer_indices,
            factors,
            torch.ones(factors["down_proj"]["A"].shape[0], device=backend.model.device, dtype=torch.int32),
        )
        outputs = backend.model.base_model(
            targets["student_ids"],
            attention_mask=targets["student_mask"],
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        return outputs.logits
    finally:
        backend.model.reset()


def gradient_audit(model) -> dict:
    hyper_gradients = [
        parameter.grad
        for parameter in model.hypernet.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not hyper_gradients:
        raise RuntimeError("gold-answer CE produced no hypernetwork gradients")
    finite = all(torch.isfinite(gradient).all() for gradient in hyper_gradients)
    nonzero = any(torch.count_nonzero(gradient) for gradient in hyper_gradients)
    if not finite or not nonzero:
        raise RuntimeError("gold-answer CE produced invalid hypernetwork gradients")
    if any(parameter.grad is not None for parameter in model.base_model.parameters()):
        raise RuntimeError("frozen target model received gradients")
    if any(parameter.grad is not None for parameter in model.ctx_encoder.parameters()):
        raise RuntimeError("frozen context encoder received gradients")
    return {
        "finite": finite,
        "nonzero": bool(nonzero),
        "gradient_tensors": len(hyper_gradients),
    }


def update_zero_parity(
    backend,
    plan: dict,
    documents: dict[str, dict],
    compiler_chunk_batch_size: int,
    max_new_tokens: int = 32,
) -> dict:
    with torch.inference_mode():
        expected, expected_accounting = compile_k1_representation_batch(
            backend,
            [plan],
            documents,
            compiler_chunk_batch_size,
        )
        actual, actual_accounting = compile_trainable_batch(
            backend,
            [plan],
            documents,
            compiler_chunk_batch_size,
            require_grad=False,
        )
        expected_digest = tensor_digest(expected.factors)
        actual_digest = tensor_digest(actual)
        if expected_digest != actual_digest:
            raise RuntimeError("COMPILER_FINETUNING update-zero factors differ from released native D2L")
        targets = make_gold_batch(backend, [plan])
        expected_logits = forward_with_factors(backend, expected.factors, targets)
        actual_logits = forward_with_factors(backend, actual, targets)
        if not torch.equal(expected_logits, actual_logits):
            raise RuntimeError("COMPILER_FINETUNING update-zero logits differ from released native D2L")
    query = Query(**plan["query"])
    expected_response = backend.generate(
        query.input,
        state=expected.factors,
        dataset=query.dataset,
        max_new_tokens=max_new_tokens,
    )["text"]
    actual_response = backend.generate(
        query.input,
        state=actual,
        dataset=query.dataset,
        max_new_tokens=max_new_tokens,
    )["text"]
    if expected_response != actual_response:
        raise RuntimeError("COMPILER_FINETUNING update-zero response differs from released native D2L")
    return {
        "factor_digest": expected_digest,
        "factor_exact": True,
        "logits_exact": True,
        "response_exact": True,
        "response": expected_response,
        "native_accounting": expected_accounting,
        "trainable_path_accounting": actual_accounting,
    }


def split_microbatches(plans: list[dict], microbatch_size: int) -> list[list[dict]]:
    if microbatch_size < 1:
        raise ValueError("microbatch size must be positive")
    return [plans[start : start + microbatch_size] for start in range(0, len(plans), microbatch_size)]


def train_batch(
    backend,
    optimizer,
    plans,
    documents,
    compiler_chunk_batch_size: int,
    microbatch_size: int,
) -> dict:
    started = time.perf_counter()
    microbatches = split_microbatches(plans, microbatch_size)
    targets = [make_gold_batch(backend, microbatch) for microbatch in microbatches]
    total_response_tokens = sum(
        sum(target["response_lengths"])
        for target in targets
    )
    optimizer.zero_grad(set_to_none=True)
    weighted_loss = 0.0
    compile_seconds = 0.0
    accounting = {
        "document_occurrences": 0,
        "context_chunks": 0,
        "source_tokens": 0,
        "generated_factor_bytes": 0,
        "loaded_factor_bytes": 0,
    }
    for microbatch, target in zip(microbatches, targets, strict=True):
        compile_started = time.perf_counter()
        factors, current = compile_trainable_batch(
            backend,
            microbatch,
            documents,
            compiler_chunk_batch_size,
        )
        compile_seconds += time.perf_counter() - compile_started
        logits = forward_with_factors(backend, factors, target)
        microbatch_tokens = sum(target["response_lengths"])
        loss = gold_answer_loss(logits, target)
        scaled_loss = loss * (microbatch_tokens / total_response_tokens)
        scaled_loss.backward()
        weighted_loss += float(loss.detach()) * microbatch_tokens
        for key in accounting:
            accounting[key] += current[key]
    gradients = gradient_audit(backend.model)
    gradient_norm = nn.utils.clip_grad_norm_(backend.model.hypernet.parameters(), GRADIENT_CLIP)
    optimizer.step()
    return {
        "loss": weighted_loss / total_response_tokens,
        "gradient_norm": float(gradient_norm),
        "compile_seconds": compile_seconds,
        "wall_seconds": time.perf_counter() - started,
        "student_replay_tokens": sum(int(target["student_mask"].sum()) for target in targets),
        "response_tokens": total_response_tokens,
        "microbatches": len(microbatches),
        "gradient_audit": gradients,
        **accounting,
    }


def validate_batches(backend, batches, documents, compiler_chunk_batch_size: int) -> dict:
    weighted_loss = 0.0
    response_tokens = 0
    source_tokens = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for plans in batches:
            factors, accounting = compile_trainable_batch(
                backend,
                plans,
                documents,
                compiler_chunk_batch_size,
                require_grad=False,
            )
            targets = make_gold_batch(backend, plans)
            logits = forward_with_factors(backend, factors, targets)
            tokens = sum(targets["response_lengths"])
            weighted_loss += float(gold_answer_loss(logits, targets)) * tokens
            response_tokens += tokens
            source_tokens += accounting["source_tokens"]
    return {
        "gold_answer_ce": weighted_loss / response_tokens,
        "examples": sum(map(len, batches)),
        "source_tokens": source_tokens,
        "response_tokens": response_tokens,
        "wall_seconds": time.perf_counter() - started,
    }


def cpu_state(module) -> dict:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def split_digest(plans: list[dict], documents: dict[str, dict]) -> str:
    rows = [
        {
            "query_id": plan["query"]["id"],
            "document_id": plan["bank_document_ids"][0],
            "document_hash": documents[plan["bank_document_ids"][0]].get("content_hash"),
        }
        for plan in plans
    ]
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="gemma2_squad_matched_distill_scale60k_seed141")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--max-source-queries", type=int, default=60000)
    parser.add_argument("--max-validation-source-queries", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--max-padded-student-tokens-per-batch",
        type=int,
        default=DEFAULT_PADDED_STUDENT_TOKEN_BUDGET,
    )
    parser.add_argument("--compiler-chunk-batch-size", type=int, default=8)
    parser.add_argument("--microbatch-size", type=int, default=6)
    parser.add_argument("--resume-microbatch-size", type=int)
    parser.add_argument("--validation-interval", type=int, default=DEFAULT_VALIDATION_INTERVAL)
    parser.add_argument("--checkpoint-interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    numeric = (
        args.steps,
        args.max_source_queries,
        args.max_validation_source_queries,
        args.batch_size,
        args.max_padded_student_tokens_per_batch,
        args.compiler_chunk_batch_size,
        args.microbatch_size,
        args.validation_interval,
        args.checkpoint_interval,
    )
    if min(numeric) < 1:
        raise ValueError("all count, size, and interval arguments must be positive")
    if args.resume_microbatch_size is not None and (
        args.resume_microbatch_size < 1 or not args.resume
    ):
        raise ValueError("resume microbatch size must be positive and requires --resume")
    resume_path = args.output.with_suffix(".resume.pt")
    state_output = args.output.with_suffix(".states.pt")
    if args.output.exists() or state_output.exists():
        raise FileExistsError(args.output if args.output.exists() else state_output)
    if args.resume != resume_path.exists():
        expected = "existing" if args.resume else "absent"
        raise FileNotFoundError(f"resume state must be {expected}: {resume_path}")

    root = Path(__file__).parents[2]
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    cfg = load_config(root, args.config)
    payload = load_manifest(paths(root, cfg)["manifest"])
    documents = {document["id"]: document for document in payload["documents"]}
    rows = split_rows(payload)
    _, plans = build_dataset_local_plans(
        rows["train"], documents, split="train", limit=args.max_source_queries, fixed_k=1
    )
    _, validation_plans = build_dataset_local_plans(
        rows["validation"],
        documents,
        split="validation",
        limit=args.max_validation_source_queries,
        fixed_k=1,
    )
    train_documents = {plan["bank_document_ids"][0] for plan in plans}
    validation_documents = {plan["bank_document_ids"][0] for plan in validation_plans}
    if train_documents & validation_documents:
        raise ValueError("COMPILER_FINETUNING requires document-disjoint train and validation plans")

    from d2l_scale_backend import OfficialBackend

    backend = OfficialBackend(cfg, root, train_compiler=True)
    try:
        teacher_tokens = {
            plan["query"]["id"]: render_prompt(
                backend,
                plan["query"]["input"],
                plan["query"]["dataset"],
                documents[plan["bank_document_ids"][0]]["text"],
            ).numel()
            for plan in plans + validation_plans
        }
        student_tokens = {
            plan["query"]["id"]: render_prompt(
                backend, plan["query"]["input"], plan["query"]["dataset"]
            ).numel()
            for plan in plans + validation_plans
        }
        batches = make_plan_batches(
            plans,
            args.batch_size,
            args.batch_size,
            teacher_tokens,
            student_tokens,
            args.max_padded_student_tokens_per_batch,
        )
        validation_batches = make_plan_batches(
            validation_plans,
            args.batch_size,
            args.batch_size,
            teacher_tokens,
            student_tokens,
            args.max_padded_student_tokens_per_batch,
        )
        if len(batches) < args.steps:
            raise ValueError(f"need {args.steps} batches, found {len(batches)}")
        selected_batches = batches[: args.steps]
        selected_plans = [plan for batch in selected_batches for plan in batch]
        identity = {
            "train": plan_identity(selected_plans),
            "validation": plan_identity([plan for batch in validation_batches for plan in batch]),
            "train_digest": split_digest(selected_plans, documents),
            "validation_digest": split_digest(validation_plans, documents),
        }
        recipe = {
            "version": 1,
            "config": args.config,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "max_padded_student_tokens_per_batch": args.max_padded_student_tokens_per_batch,
            "compiler_chunk_batch_size": args.compiler_chunk_batch_size,
            "microbatch_size": args.microbatch_size,
            "validation_interval": args.validation_interval,
            "checkpoint_interval": args.checkpoint_interval,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRADIENT_CLIP,
            "seed": SEED,
            "compiler_input": "document_only",
            "objective": "first_nonempty_gold_answer_plus_eos_ce",
        }
        optimizer = torch.optim.AdamW(
            backend.model.hypernet.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        audit = trainability_audit(backend.model)
        correctness = update_zero_parity(
            backend,
            selected_batches[0][0],
            documents,
            args.compiler_chunk_batch_size,
        )
        updates = []
        validation_history = []
        completed_steps = 0
        best_validation = float("inf")
        best_hypernet = None
        consumed_seconds = 0.0
        prior_peak_vram = 0
        if args.resume:
            saved = torch.load(resume_path, map_location=backend.model.device, weights_only=False)
            if saved["identity"] != identity or saved["recipe"] != recipe:
                raise ValueError("training resume contract differs")
            backend.model.hypernet.load_state_dict(saved["hypernet"])
            optimizer.load_state_dict(saved["optimizer"])
            updates = saved["updates"]
            validation_history = saved["validation_history"]
            completed_steps = saved["completed_steps"]
            best_validation = saved["best_validation"]
            best_hypernet = saved["best_hypernet"]
            consumed_seconds = saved["consumed_seconds"]
            prior_peak_vram = saved["peak_workload_vram_bytes"]
            torch.set_rng_state(saved["torch_rng_state"].cpu())
            if torch.cuda.is_available() and saved.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng_state"]])
        if args.checkpoint_dir is not None:
            args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if not validation_history:
            validation = {
                "step": 0,
                **validate_batches(
                    backend,
                    validation_batches,
                    documents,
                    args.compiler_chunk_batch_size,
                ),
            }
            validation_history.append(validation)
            best_validation = validation["gold_answer_ce"]
            best_hypernet = cpu_state(backend.model.hypernet)
            if args.checkpoint_dir is not None:
                atomic_torch_save(
                    {"hypernet": best_hypernet, "step": 0, "recipe": recipe, "identity": identity},
                    args.checkpoint_dir / "step-000000.pt",
                )
        segment_started = time.perf_counter()
        for step in range(completed_steps, args.steps):
            execution_microbatch_size = args.resume_microbatch_size or args.microbatch_size
            updates.append(
                {
                    "step": step + 1,
                    **train_batch(
                        backend,
                        optimizer,
                        selected_batches[step],
                        documents,
                        args.compiler_chunk_batch_size,
                        execution_microbatch_size,
                    ),
                }
            )
            completed_steps = step + 1
            should_validate = completed_steps % args.validation_interval == 0 or completed_steps == args.steps
            if should_validate:
                validation = {
                    "step": completed_steps,
                    **validate_batches(
                        backend,
                        validation_batches,
                        documents,
                        args.compiler_chunk_batch_size,
                    ),
                }
                validation_history.append(validation)
                if validation["gold_answer_ce"] < best_validation:
                    best_validation = validation["gold_answer_ce"]
                    best_hypernet = cpu_state(backend.model.hypernet)
                if args.checkpoint_dir is not None:
                    atomic_torch_save(
                        {
                            "hypernet": cpu_state(backend.model.hypernet),
                            "step": completed_steps,
                            "recipe": recipe,
                            "identity": identity,
                        },
                        args.checkpoint_dir / f"step-{completed_steps:06d}.pt",
                    )
            if completed_steps % args.checkpoint_interval == 0 or should_validate:
                consumed_seconds += time.perf_counter() - segment_started
                atomic_torch_save(
                    {
                        "identity": identity,
                        "recipe": recipe,
                        "hypernet": backend.model.hypernet.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "updates": updates,
                        "validation_history": validation_history,
                        "completed_steps": completed_steps,
                        "best_validation": best_validation,
                        "best_hypernet": best_hypernet,
                        "consumed_seconds": consumed_seconds,
                        "peak_workload_vram_bytes": max(
                            prior_peak_vram,
                            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                        ),
                        "torch_rng_state": torch.get_rng_state(),
                        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    },
                    resume_path,
                )
                segment_started = time.perf_counter()
        if best_hypernet is None:
            raise RuntimeError("training completed without validation")
        selected_step = min(validation_history, key=lambda row: row["gold_answer_ce"])["step"]
        atomic_torch_save(
            {
                "hypernet": best_hypernet,
                "step": selected_step,
                "recipe": recipe,
                "identity": identity,
                "selection": "provisional_gold_ce_pending_fresh_generation",
            },
            state_output,
        )
        report = {
            "schema_version": 1,
            "experiment": "COMPILER_FINETUNING-answer-aligned-document-only-d2l-compiler",
            "status": "completed_training_pending_fresh_selection",
            "contracts": {
                **recipe,
                "train_examples": len(selected_plans),
                "validation_examples": sum(map(len, validation_batches)),
                "single_chunk_required": True,
                "native_rank": 8,
                "native_target_modules": ["down_proj"],
                "final_exact_400_test_opened": False,
                "resume_microbatch_size": args.resume_microbatch_size,
            },
            "identity": identity,
            "trainability": audit,
            "correctness": correctness,
            "updates": updates,
            "validation": {
                "history": validation_history,
                "provisional_selected_step": selected_step,
                "provisional_selected_gold_answer_ce": best_validation,
            },
            "accounting": {
                "hypernetwork_backward_passes": completed_steps,
                "wall_seconds": consumed_seconds,
                "peak_workload_vram_bytes": max(
                    prior_peak_vram,
                    torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                ),
                "optimizer_state_bytes": sum(
                    value.numel() * value.element_size()
                    for state in optimizer.state.values()
                    for value in state.values()
                    if isinstance(value, torch.Tensor)
                ),
            },
            "runtime": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "transformers": transformers.__version__,
                "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
            },
            "backend_provenance": backend.provenance,
            "state_output": str(state_output),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        resume_path.unlink()
        print(json.dumps({"status": report["status"], "selected_step": selected_step}, indent=2))
    finally:
        backend.close()


if __name__ == "__main__":
    main()
