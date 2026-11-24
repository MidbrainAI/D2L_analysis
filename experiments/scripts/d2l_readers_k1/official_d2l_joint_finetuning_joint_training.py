"""Train JOINT_FINETUNING's D2L hypernetwork and READOUT_LADDER rank-effect reader jointly."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import transformers
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from d2l_scale import load_config, load_manifest, paths
from official_d2l_compiler_finetuning_compiler_training import (
    cpu_state,
    document_chunks,
    split_digest,
    split_microbatches,
    trainability_audit,
)
from official_d2l_k1_representation_evaluation import generate_reader_condition
from official_d2l_k1_representation_ladder import (
    CompiledRepresentationBatch,
    compile_k1_representation_batch,
    installed_representation_transition,
    make_representation_transition,
)
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
WRITER_LEARNING_RATE = 1e-5
READER_LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0


def compile_joint_batch(
    backend,
    plans: list[dict],
    documents: dict[str, dict],
    compiler_chunk_batch_size: int,
    *,
    require_grad: bool = True,
) -> tuple[CompiledRepresentationBatch, dict]:
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
            raise ValueError("layernorm adaptation is outside the JOINT_FINETUNING protocol")
        generated_parts.append(generated)
    generated = concatenate_generated_parts(generated_parts)
    head_bias = (
        backend.model.hypernet.get_head_bias()
        if backend.model.hypernet.config.use_bias
        else None
    )
    factors = backend.merge(
        generated,
        torch.ones(len(plans), device=backend.model.device, dtype=torch.int32),
        lora_bias=head_bias,
        bias_scaler=1.0,
    )
    bias_factors = {
        module: {
            key: value.unsqueeze(0).expand(len(plans), -1, -1, -1)
            for key, value in part.items()
        }
        for module, part in head_bias.items()
    }
    if require_grad and any(
        not tensor.requires_grad
        for value in (generated, factors, bias_factors)
        for part in value.values()
        for tensor in part.values()
    ):
        raise RuntimeError("JOINT_FINETUNING factors are detached from the trainable hypernetwork")
    sample = generated["down_proj"]["A"]
    compiled = CompiledRepresentationBatch(
        factors=factors,
        generated_factors=generated,
        bias_factors=bias_factors,
        pre_head_latents=sample.new_zeros(
            sample.shape[0],
            sample.shape[1],
            sample.shape[2],
            backend.model.hypernet.config.latent_size,
        ),
    )
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
    return compiled, accounting


def forward_joint_batch(backend, layers, transition, compiled, targets, scaling) -> torch.Tensor:
    backend.model.reset()
    with installed_representation_transition(
        backend.model.base_model,
        layers,
        compiled,
        transition,
        "rank-effect",
        scaling,
        official_backend=backend,
    ):
        outputs = backend.model.base_model(
            targets["student_ids"],
            attention_mask=targets["student_mask"],
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
    return outputs.logits


def module_gradient_norm(parameters) -> float:
    gradients = [parameter.grad.float().norm() for parameter in parameters if parameter.grad is not None]
    return float(torch.stack(gradients).norm()) if gradients else 0.0


def joint_gradient_audit(model, transition) -> dict:
    hyper_gradients = [
        parameter.grad
        for parameter in model.hypernet.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    reader_gradients = [
        parameter.grad
        for parameter in transition.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    for name, gradients in (("hypernetwork", hyper_gradients), ("reader", reader_gradients)):
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError(f"JOINT_FINETUNING produced missing or nonfinite {name} gradients")
        if not any(torch.count_nonzero(gradient) for gradient in gradients):
            raise RuntimeError(f"JOINT_FINETUNING produced zero {name} gradients")
    if any(parameter.grad is not None for parameter in model.base_model.parameters()):
        raise RuntimeError("frozen target model received gradients")
    if any(parameter.grad is not None for parameter in model.ctx_encoder.parameters()):
        raise RuntimeError("frozen context encoder received gradients")
    return {
        "hypernetwork_gradient_tensors": len(hyper_gradients),
        "reader_gradient_tensors": len(reader_gradients),
        "hypernetwork_gradient_norm": module_gradient_norm(model.hypernet.parameters()),
        "reader_gradient_norm": module_gradient_norm(transition.parameters()),
        "finite": True,
        "nonzero": True,
    }


def joint_trainability_audit(model, transition) -> dict:
    audit = trainability_audit(model)
    reader = list(transition.parameters())
    if not reader or any(not parameter.requires_grad for parameter in reader):
        raise ValueError("JOINT_FINETUNING requires a fully trainable READOUT_LADDER reader")
    audit["reader"] = {
        "parameters": sum(parameter.numel() for parameter in reader),
        "parameter_bytes": sum(parameter.numel() * parameter.element_size() for parameter in reader),
        "trainable_parameters": sum(parameter.numel() for parameter in reader),
    }
    return audit


def train_effective_batch(
    backend,
    layers,
    transition,
    optimizer,
    plans,
    documents,
    scaling,
    compiler_chunk_batch_size,
    microbatch_size,
) -> dict:
    started = time.perf_counter()
    microbatches = split_microbatches(plans, microbatch_size)
    targets = [make_gold_batch(backend, microbatch) for microbatch in microbatches]
    total_response_tokens = sum(sum(target["response_lengths"]) for target in targets)
    optimizer.zero_grad(set_to_none=True)
    transition.train()
    weighted_loss = 0.0
    accounting = {
        "document_occurrences": 0,
        "context_chunks": 0,
        "source_tokens": 0,
        "generated_factor_bytes": 0,
        "loaded_factor_bytes": 0,
    }
    for microbatch, target in zip(microbatches, targets, strict=True):
        compiled, current = compile_joint_batch(
            backend,
            microbatch,
            documents,
            compiler_chunk_batch_size,
        )
        logits = forward_joint_batch(
            backend,
            layers,
            transition,
            compiled,
            target,
            scaling,
        )
        tokens = sum(target["response_lengths"])
        loss = gold_answer_loss(logits, target)
        (loss * (tokens / total_response_tokens)).backward()
        weighted_loss += float(loss.detach()) * tokens
        for key in accounting:
            accounting[key] += current[key]
    gradients = joint_gradient_audit(backend.model, transition)
    joint_parameters = [*backend.model.hypernet.parameters(), *transition.parameters()]
    gradient_norm = nn.utils.clip_grad_norm_(joint_parameters, GRADIENT_CLIP)
    optimizer.step()
    return {
        "loss": weighted_loss / total_response_tokens,
        "gradient_norm": float(gradient_norm),
        "microbatches": len(microbatches),
        "student_replay_tokens": sum(int(target["student_mask"].sum()) for target in targets),
        "response_tokens": total_response_tokens,
        "wall_seconds": time.perf_counter() - started,
        "gradient_audit": gradients,
        **accounting,
    }


def validate_batches(
    backend,
    layers,
    transition,
    batches,
    documents,
    scaling,
    compiler_chunk_batch_size,
) -> dict:
    transition.eval()
    weighted_loss = 0.0
    response_tokens = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for plans in batches:
            compiled, _ = compile_joint_batch(
                backend,
                plans,
                documents,
                compiler_chunk_batch_size,
                require_grad=False,
            )
            targets = make_gold_batch(backend, plans)
            logits = forward_joint_batch(
                backend,
                layers,
                transition,
                compiled,
                targets,
                scaling,
            )
            tokens = sum(targets["response_lengths"])
            weighted_loss += float(gold_answer_loss(logits, targets)) * tokens
            response_tokens += tokens
    return {
        "gold_answer_ce": weighted_loss / response_tokens,
        "examples": sum(map(len, batches)),
        "response_tokens": response_tokens,
        "wall_seconds": time.perf_counter() - started,
    }


def joint_update_zero_parity(
    backend,
    layers,
    transition,
    plan,
    documents,
    scaling,
    compiler_chunk_batch_size,
    max_new_tokens=32,
) -> dict:
    transition.eval()
    with torch.inference_mode():
        expected, _ = compile_k1_representation_batch(
            backend,
            [plan],
            documents,
            compiler_chunk_batch_size,
        )
        actual, _ = compile_joint_batch(
            backend,
            [plan],
            documents,
            compiler_chunk_batch_size,
            require_grad=False,
        )
        if tensor_digest(expected.factors) != tensor_digest(actual.factors):
            raise RuntimeError("JOINT_FINETUNING update-zero factors differ from native D2L")
        targets = make_gold_batch(backend, [plan])
        backend.model.reset()
        backend.model.patch_lora_forward()
        try:
            backend.install(
                backend.model.base_model,
                backend.model.hypernet.layer_indices,
                expected.factors,
                torch.ones(1, device=backend.model.device, dtype=torch.int32),
            )
            native_logits = backend.model.base_model(
                targets["student_ids"],
                attention_mask=targets["student_mask"],
                use_cache=False,
                return_dict=True,
            ).logits
        finally:
            backend.model.reset()
        joint_logits = forward_joint_batch(
            backend,
            layers,
            transition,
            actual,
            targets,
            scaling,
        )
        if not torch.equal(native_logits, joint_logits):
            raise RuntimeError("JOINT_FINETUNING update-zero logits differ from native D2L")
    query = plan["query"]
    native_response = backend.generate(
        query["input"],
        state=expected.factors,
        dataset=query["dataset"],
        max_new_tokens=max_new_tokens,
    )["text"]
    joint_response = generate_reader_condition(
        backend,
        layers,
        actual,
        transition,
        "rank-effect",
        scaling,
        query["input"],
        max_new_tokens,
        query["dataset"],
    )
    if native_response != joint_response:
        raise RuntimeError("JOINT_FINETUNING update-zero response differs from native D2L")
    return {
        "factor_exact": True,
        "logits_exact": True,
        "response_exact": True,
        "reader_zero_exact": not torch.count_nonzero(transition.output_projection.weight),
        "response": native_response,
    }


def checkpoint_payload(backend, transition, step, recipe, identity) -> dict:
    return {
        "hypernet": cpu_state(backend.model.hypernet),
        "transition": cpu_state(transition),
        "step": step,
        "recipe": recipe,
        "identity": identity,
        "representation": "rank-effect",
        "transition_bottleneck": transition.bottleneck,
        "memory_width": transition.memory_width,
        "execution_site": transition.execution_site,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="gemma2_squad_matched_distill_scale60k_seed141")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--max-source-queries", type=int, default=60000)
    parser.add_argument("--max-validation-source-queries", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--microbatch-size", type=int, default=2)
    parser.add_argument("--resume-microbatch-size", type=int)
    parser.add_argument("--max-padded-student-tokens-per-batch", type=int, default=2048)
    parser.add_argument("--compiler-chunk-batch-size", type=int, default=8)
    parser.add_argument("--validation-interval", type=int, default=500)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    numeric = (
        args.steps,
        args.max_source_queries,
        args.max_validation_source_queries,
        args.batch_size,
        args.microbatch_size,
        args.max_padded_student_tokens_per_batch,
        args.compiler_chunk_batch_size,
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
    if {plan["bank_document_ids"][0] for plan in plans} & {
        plan["bank_document_ids"][0] for plan in validation_plans
    }:
        raise ValueError("JOINT_FINETUNING requires document-disjoint train and validation plans")

    from d2l_scale_backend import OfficialBackend

    backend = OfficialBackend(cfg, root, train_compiler=True)
    try:
        from ctx_to_lora.utils import get_layers

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
        model = backend.model.base_model
        layers = get_layers(model)
        scaling = model.peft_config["default"].lora_alpha
        transition = make_representation_transition(
            "rank-effect",
            model.config.hidden_size,
            backend.model.hypernet.config.latent_size,
            len(layers),
            32,
        ).to(model.device)
        optimizer = torch.optim.AdamW(
            [
                {"params": backend.model.hypernet.parameters(), "lr": WRITER_LEARNING_RATE},
                {"params": transition.parameters(), "lr": READER_LEARNING_RATE},
            ],
            weight_decay=WEIGHT_DECAY,
        )
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
            "microbatch_size": args.microbatch_size,
            "max_padded_student_tokens_per_batch": args.max_padded_student_tokens_per_batch,
            "compiler_chunk_batch_size": args.compiler_chunk_batch_size,
            "validation_interval": args.validation_interval,
            "checkpoint_interval": args.checkpoint_interval,
            "writer_learning_rate": WRITER_LEARNING_RATE,
            "reader_learning_rate": READER_LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRADIENT_CLIP,
            "seed": SEED,
            "compiler_input": "document_only",
            "representation": "rank-effect",
            "reader_bottleneck": 32,
            "execution_site": "post-block",
            "objective": "first_nonempty_gold_answer_plus_eos_ce",
        }
        audit = joint_trainability_audit(backend.model, transition)
        correctness = joint_update_zero_parity(
            backend,
            layers,
            transition,
            selected_batches[0][0],
            documents,
            scaling,
            args.compiler_chunk_batch_size,
        )
        updates = []
        validation_history = []
        completed_steps = 0
        consumed_seconds = 0.0
        prior_peak_vram = 0
        if args.resume:
            saved = torch.load(resume_path, map_location=model.device, weights_only=False)
            if saved["identity"] != identity or saved["recipe"] != recipe:
                raise ValueError("training resume contract differs")
            backend.model.hypernet.load_state_dict(saved["hypernet"])
            transition.load_state_dict(saved["transition"])
            optimizer.load_state_dict(saved["optimizer"])
            updates = saved["updates"]
            validation_history = saved["validation_history"]
            completed_steps = saved["completed_steps"]
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
            validation_history.append(
                {
                    "step": 0,
                    **validate_batches(
                        backend,
                        layers,
                        transition,
                        validation_batches,
                        documents,
                        scaling,
                        args.compiler_chunk_batch_size,
                    ),
                }
            )
            if args.checkpoint_dir is not None:
                atomic_torch_save(
                    checkpoint_payload(backend, transition, 0, recipe, identity),
                    args.checkpoint_dir / "step-000000.pt",
                )
        segment_started = time.perf_counter()
        for step in range(completed_steps, args.steps):
            execution_microbatch_size = args.resume_microbatch_size or args.microbatch_size
            updates.append(
                {
                    "step": step + 1,
                    **train_effective_batch(
                        backend,
                        layers,
                        transition,
                        optimizer,
                        selected_batches[step],
                        documents,
                        scaling,
                        args.compiler_chunk_batch_size,
                        execution_microbatch_size,
                    ),
                }
            )
            completed_steps = step + 1
            should_validate = completed_steps % args.validation_interval == 0 or completed_steps == args.steps
            if should_validate:
                validation_history.append(
                    {
                        "step": completed_steps,
                        **validate_batches(
                            backend,
                            layers,
                            transition,
                            validation_batches,
                            documents,
                            scaling,
                            args.compiler_chunk_batch_size,
                        ),
                    }
                )
                if args.checkpoint_dir is not None:
                    atomic_torch_save(
                        checkpoint_payload(
                            backend,
                            transition,
                            completed_steps,
                            recipe,
                            identity,
                        ),
                        args.checkpoint_dir / f"step-{completed_steps:06d}.pt",
                    )
            if completed_steps % args.checkpoint_interval == 0 or should_validate:
                consumed_seconds += time.perf_counter() - segment_started
                atomic_torch_save(
                    {
                        "identity": identity,
                        "recipe": recipe,
                        "hypernet": backend.model.hypernet.state_dict(),
                        "transition": transition.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "updates": updates,
                        "validation_history": validation_history,
                        "completed_steps": completed_steps,
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
        atomic_torch_save(
            checkpoint_payload(backend, transition, completed_steps, recipe, identity),
            state_output,
        )
        report = {
            "schema_version": 1,
            "experiment": "JOINT_FINETUNING-joint-d2l-hypernetwork-rank-effect-reader",
            "status": "completed_training_pending_fresh_selection",
            "contracts": {
                **recipe,
                "resume_microbatch_size": args.resume_microbatch_size,
                "train_examples": len(selected_plans),
                "validation_examples": sum(map(len, validation_batches)),
                "single_chunk_required": True,
                "native_rank": 8,
                "native_target_modules": ["down_proj"],
                "final_exact_400_test_opened": False,
            },
            "identity": identity,
            "trainability": audit,
            "correctness": correctness,
            "updates": updates,
            "validation": {
                "history": validation_history,
                "selection_status": "pending_fresh_generation",
            },
            "accounting": {
                "joint_backward_passes": completed_steps,
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
        print(json.dumps({"status": report["status"], "steps": completed_steps}, indent=2))
    finally:
        backend.close()


if __name__ == "__main__":
    main()
