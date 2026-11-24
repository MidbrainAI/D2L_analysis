"""Train READOUT_LADDER/STUDENT_PREFIX's matched post-block representation readers."""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional

sys.path.insert(0, str(Path(__file__).parent))
from d2l_scale import load_config, load_manifest, paths
from d2l_scale_core import Query
from official_d2l_k1_representation_ladder import (
    REPRESENTATIONS,
    compile_k1_representation_batch,
    compiler_capture_parity_check,
    installed_nonlinear_rank_operator,
    installed_representation_transition,
    make_representation_transition,
)
from reader_banks import (
    LEARNING_RATE,
    build_dataset_local_plans,
    compact_topk,
    sparse_coarsened_forward_kl,
)
from reader_banks import split_rows
from official_d2l_online_batch_distillation import (
    atomic_torch_save,
    bank_context,
    gather_response_logits,
    make_plan_batches,
    pad_sequences,
    plan_identity,
    render_prompt,
    response_position_mask,
    response_token_lengths,
)

DEFAULT_BATCH_SIZE = 12
DEFAULT_PADDED_STUDENT_TOKEN_BUDGET = 2048
DEFAULT_VALIDATION_INTERVAL = 500
DEFAULT_CHECKPOINT_INTERVAL = 100
SEED = 168
OBJECTIVES = ("gold-answer-ce", "student-prefix-forward-kl")


def make_gold_batch(backend, plans: list[dict]) -> dict:
    replays = []
    labels = []
    prompt_lengths = []
    response_lengths = []
    for plan in plans:
        query = Query(**plan["query"])
        answer = next((value.strip() for value in query.answers if value.strip()), None)
        if answer is None:
            raise ValueError(f"query {query.id} has no nonempty gold answer")
        answer_ids = backend.tokenizer(answer, add_special_tokens=False)["input_ids"]
        eos = backend.tokenizer.eos_token_id
        if eos is not None:
            answer_ids = [*answer_ids, eos]
        if not answer_ids:
            raise ValueError(f"query {query.id} produced no gold-answer tokens")
        prompt = render_prompt(backend, query.input, query.dataset)
        answer_tensor = torch.tensor(answer_ids, dtype=prompt.dtype)
        replays.append(torch.cat((prompt, answer_tensor[:-1])))
        labels.append(answer_tensor)
        prompt_lengths.append(prompt.numel())
        response_lengths.append(answer_tensor.numel())
    student_ids, student_mask = pad_sequences(
        replays,
        backend.tokenizer.pad_token_id,
        device=backend.model.device,
    )
    width = max(response_lengths)
    hard_targets = torch.full(
        (len(labels), width),
        -100,
        dtype=labels[0].dtype,
        device=backend.model.device,
    )
    for index, target in enumerate(labels):
        hard_targets[index, : target.numel()] = target.to(backend.model.device)
    return {
        "student_ids": student_ids,
        "student_mask": student_mask,
        "student_prompt_lengths": prompt_lengths,
        "response_lengths": response_lengths,
        "hard_targets": hard_targets,
    }


def gold_answer_loss(logits: torch.Tensor, targets: dict) -> torch.Tensor:
    response_logits = gather_response_logits(
        logits,
        targets["student_prompt_lengths"],
        targets["response_lengths"],
    )
    return functional.cross_entropy(
        response_logits.float().reshape(-1, response_logits.shape[-1]),
        targets["hard_targets"].reshape(-1),
        ignore_index=-100,
    )


def make_response_replays(
    prompts: list[torch.Tensor],
    responses: list[torch.Tensor],
) -> list[torch.Tensor]:
    if len(prompts) != len(responses) or not prompts:
        raise ValueError("response replay prompts and responses must be nonempty and aligned")
    if any(response.ndim != 1 or response.numel() < 1 for response in responses):
        raise ValueError("response replays require nonempty one-dimensional responses")
    return [
        torch.cat((prompt, response[:-1]))
        for prompt, response in zip(prompts, responses, strict=True)
    ]


def _pad_compact_targets(parts: list[dict], response_width: int) -> dict:
    padded = []
    for compact in parts:
        pad = response_width - compact["indices"].shape[1]
        if pad < 0:
            raise ValueError("compact target exceeds response width")
        if pad:
            compact = {
                "indices": functional.pad(compact["indices"], (0, 0, 0, pad)),
                "log_probs": functional.pad(compact["log_probs"], (0, 0, 0, pad)),
                "tail_log_prob": functional.pad(compact["tail_log_prob"], (0, pad)),
            }
        padded.append(compact)
    return {key: torch.cat([part[key] for part in padded], dim=0) for key in padded[0]}


def make_student_prefix_batch(
    backend,
    layers,
    transition,
    compiled,
    representation: str,
    plans: list[dict],
    documents: dict[str, dict],
    scaling: float,
    max_new_tokens: int,
    teacher_replay_batch_size: int,
) -> dict:
    model = backend.model.base_model
    device = model.device
    student_prompts, teacher_prompts = [], []
    for plan in plans:
        query = Query(**plan["query"])
        student_prompts.append(render_prompt(backend, query.input, query.dataset))
        teacher_prompts.append(
            render_prompt(
                backend,
                query.input,
                query.dataset,
                bank_context(plan, documents),
            )
        )

    rollout_ids, rollout_mask = pad_sequences(
        student_prompts,
        backend.tokenizer.pad_token_id,
        device=device,
        left=True,
    )
    backend.model.reset()
    transition.eval()
    with (
        installed_representation_transition(
            model,
            layers,
            compiled,
            transition,
            representation,
            scaling,
            official_backend=backend,
        ),
        torch.inference_mode(),
    ):
        generated = model.generate(
            rollout_ids,
            attention_mask=rollout_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=backend.tokenizer.pad_token_id,
        )
    continuations = generated[:, rollout_ids.shape[1] :]
    eos = backend.tokenizer.eos_token_id
    eos_ids = tuple(eos) if isinstance(eos, (list, tuple)) else (eos,)
    response_lengths = response_token_lengths(
        continuations,
        eos_ids,
        backend.tokenizer.pad_token_id,
    )
    responses = [
        continuations[index, :length].detach().cpu()
        for index, length in enumerate(response_lengths)
    ]

    teacher_replays = make_response_replays(teacher_prompts, responses)
    response_width = max(response_lengths)
    teacher_top_parts = []
    teacher_replay_tokens = 0
    backend.model.reset()
    for start in range(0, len(plans), teacher_replay_batch_size):
        end = min(start + teacher_replay_batch_size, len(plans))
        replay_ids, replay_mask = pad_sequences(
            teacher_replays[start:end],
            backend.tokenizer.pad_token_id,
            device=device,
        )
        teacher_replay_tokens += int(replay_mask.sum())
        with torch.inference_mode():
            outputs = model(
                replay_ids,
                attention_mask=replay_mask,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
            logits = gather_response_logits(
                outputs.logits,
                [prompt.numel() for prompt in teacher_prompts[start:end]],
                response_lengths[start:end],
            )
            teacher_top_parts.append(compact_topk(logits))

    student_replays = make_response_replays(student_prompts, responses)
    student_ids, student_mask = pad_sequences(
        student_replays,
        backend.tokenizer.pad_token_id,
        device=device,
    )
    return {
        "teacher_top": _pad_compact_targets(teacher_top_parts, response_width),
        "responses": responses,
        "student_ids": student_ids,
        "student_mask": student_mask,
        "student_prompt_lengths": [prompt.numel() for prompt in student_prompts],
        "response_lengths": response_lengths,
        "response_mask": response_position_mask(response_lengths, device=device),
        "student_rollout_prompt_tokens": int(rollout_mask.sum()),
        "rollout_response_tokens": sum(response_lengths),
        "teacher_replay_tokens": teacher_replay_tokens,
        "student_replay_tokens": int(student_mask.sum()),
        "teacher_replay_forward_passes": len(teacher_top_parts),
    }
def forward_gold_batch(
    backend,
    layers,
    transition,
    compiled,
    representation: str,
    targets: dict,
    scaling: float,
    *,
    apply_document_factors: bool = True,
) -> torch.Tensor:
    backend.model.reset()
    with installed_representation_transition(
        backend.model.base_model,
        layers,
        compiled,
        transition,
        representation,
        scaling,
        apply_document_factors=apply_document_factors,
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


def forward_nonlinear_rank_batch(
    backend,
    layers,
    operator,
    compiled,
    targets: dict,
    scaling: float,
) -> torch.Tensor:
    backend.model.reset()
    compute_dtype = getattr(operator, "compute_dtype", None)
    precision_context = (
        torch.autocast(device_type="cuda", dtype=compute_dtype)
        if compute_dtype is not None and backend.model.base_model.device.type == "cuda"
        else contextlib.nullcontext()
    )
    with (
        installed_nonlinear_rank_operator(
            backend.model.base_model,
            layers,
            compiled,
            operator,
            scaling,
            official_backend=backend,
        ),
        precision_context,
    ):
        outputs = backend.model.base_model(
            targets["student_ids"],
            attention_mask=targets["student_mask"],
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
    return outputs.logits


def train_batch(
    backend,
    layers,
    transition,
    optimizer,
    plans,
    documents,
    representation,
    scaling,
    compiler_chunk_batch_size,
    objective="gold-answer-ce",
    max_new_tokens=32,
    teacher_replay_batch_size=4,
) -> dict:
    compile_started = time.perf_counter()
    compiled, accounting = compile_k1_representation_batch(
        backend,
        plans,
        documents,
        compiler_chunk_batch_size,
        capture_projected=representation == "projected-source",
    )
    compile_seconds = time.perf_counter() - compile_started
    target_started = time.perf_counter()
    if objective == "gold-answer-ce":
        targets = make_gold_batch(backend, plans)
    elif objective == "student-prefix-forward-kl":
        targets = make_student_prefix_batch(
            backend,
            layers,
            transition,
            compiled,
            representation,
            plans,
            documents,
            scaling,
            max_new_tokens,
            teacher_replay_batch_size,
        )
    else:
        raise ValueError(f"unknown objective: {objective}")
    target_seconds = time.perf_counter() - target_started
    student_started = time.perf_counter()
    transition.train()
    logits = forward_gold_batch(
        backend,
        layers,
        transition,
        compiled,
        representation,
        targets,
        scaling,
    )
    if objective == "gold-answer-ce":
        loss = gold_answer_loss(logits, targets)
    else:
        response_logits = gather_response_logits(
            logits,
            targets["student_prompt_lengths"],
            targets["response_lengths"],
        )
        loss = sparse_coarsened_forward_kl(
            targets["teacher_top"],
            response_logits,
            targets["response_mask"],
        )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = nn.utils.clip_grad_norm_(transition.parameters(), 1.0)
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "gradient_norm": float(gradient_norm),
        "compile_seconds": compile_seconds,
        "target_seconds": target_seconds,
        "student_seconds": time.perf_counter() - student_started,
        "student_replay_tokens": int(targets["student_mask"].sum()),
        "response_tokens": sum(targets["response_lengths"]),
        **(
            {
                "student_rollout_prompt_tokens": targets["student_rollout_prompt_tokens"],
                "rollout_response_tokens": targets["rollout_response_tokens"],
                "teacher_replay_tokens": targets["teacher_replay_tokens"],
                "student_rollout_forward_passes": 1,
                "teacher_replay_forward_passes": targets["teacher_replay_forward_passes"],
            }
            if objective == "student-prefix-forward-kl"
            else {}
        ),
        **accounting,
    }


def validate_batches(
    backend,
    layers,
    transition,
    batches,
    documents,
    representation,
    scaling,
    compiler_chunk_batch_size,
) -> dict:
    transition.eval()
    weighted_loss = 0.0
    response_tokens = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for plans in batches:
            targets = make_gold_batch(backend, plans)
            compiled, _ = compile_k1_representation_batch(
                backend,
                plans,
                documents,
                compiler_chunk_batch_size,
                capture_projected=representation == "projected-source",
            )
            logits = forward_gold_batch(
                backend,
                layers,
                transition,
                compiled,
                representation,
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


def reader_payload(transition, representation: str, selected_step: int) -> dict:
    return {
        "transition": transition.state_dict(),
        "transition_type": "active-memory",
        "representation": representation,
        "transition_bottleneck": transition.bottleneck,
        "memory_width": transition.memory_width,
        "execution_site": transition.execution_site,
        "selected_step": selected_step,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="gemma2_squad_matched_distill_scale60k_seed141")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--representation", choices=REPRESENTATIONS, required=True)
    parser.add_argument("--objective", choices=OBJECTIVES, default="gold-answer-ce")
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
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--teacher-replay-batch-size", type=int, default=4)
    parser.add_argument("--transition-bottleneck", type=int, default=32)
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
        args.max_new_tokens,
        args.teacher_replay_batch_size,
        args.transition_bottleneck,
        args.validation_interval,
        args.checkpoint_interval,
    )
    if min(numeric) < 1:
        raise ValueError("all count, size, and interval arguments must be positive")
    if args.objective == "student-prefix-forward-kl" and args.representation != "rank-effect":
        raise ValueError("student-prefix forward KL is preregistered only for rank-effect state")
    resume_path = args.output.with_suffix(".resume.pt")
    state_output = args.output.with_suffix(".states.pt")
    if args.output.exists() or state_output.exists():
        raise FileExistsError(args.output if args.output.exists() else state_output)
    if args.resume != resume_path.exists():
        expected = "existing" if args.resume else "absent"
        raise FileNotFoundError(f"resume state must be {expected}: {resume_path}")

    root = Path(__file__).parents[2]
    torch.manual_seed(SEED)
    cfg = load_config(root, args.config)
    payload = load_manifest(paths(root, cfg)["manifest"])
    documents = {document["id"]: document for document in payload["documents"]}
    rows = split_rows(payload)
    _, plans = build_dataset_local_plans(
        rows["train"],
        documents,
        split="train",
        limit=args.max_source_queries,
        fixed_k=1,
    )
    _, validation_plans = build_dataset_local_plans(
        rows["validation"],
        documents,
        split="validation",
        limit=args.max_validation_source_queries,
        fixed_k=1,
    )

    from d2l_scale_backend import OfficialBackend

    backend = OfficialBackend(cfg, root)
    try:
        from ctx_to_lora.utils import get_layers

        teacher_tokens = {
            plan["query"]["id"]: render_prompt(
                backend,
                Query(**plan["query"]).input,
                plan["query"]["dataset"],
                bank_context(plan, documents),
            ).numel()
            for plan in plans + validation_plans
        }
        student_tokens = {
            plan["query"]["id"]: render_prompt(
                backend,
                Query(**plan["query"]).input,
                plan["query"]["dataset"],
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
        model = backend.model.base_model
        layers = get_layers(model)
        scaling = model.peft_config["default"].lora_alpha
        latent_width = backend.model.hypernet.config.latent_size
        transition = make_representation_transition(
            args.representation,
            model.config.hidden_size,
            latent_width,
            len(layers),
            args.transition_bottleneck,
        ).to(model.device)
        optimizer = torch.optim.AdamW(transition.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        identity = {
            "train": plan_identity([plan for batch in selected_batches for plan in batch]),
            "validation": plan_identity([plan for batch in validation_batches for plan in batch]),
        }
        recipe = {
            "version": 1,
            "config": args.config,
            "representation": args.representation,
            "objective": args.objective,
            "batch_size": args.batch_size,
            "max_padded_student_tokens_per_batch": args.max_padded_student_tokens_per_batch,
            "compiler_chunk_batch_size": args.compiler_chunk_batch_size,
            "max_new_tokens": args.max_new_tokens,
            "teacher_replay_batch_size": args.teacher_replay_batch_size,
            "transition_bottleneck": args.transition_bottleneck,
            "validation_interval": args.validation_interval,
            "learning_rate": LEARNING_RATE,
            "weight_decay": 1e-4,
            "seed": SEED,
        }
        correctness = compiler_capture_parity_check(backend, selected_batches[0][0], documents)
        updates = []
        validation_history = []
        completed_steps = 0
        best_validation = float("inf")
        best_transition = None
        consumed_seconds = 0.0
        prior_peak_vram = 0
        if args.resume:
            saved = torch.load(resume_path, map_location=model.device, weights_only=False)
            if saved["identity"] != identity or saved["recipe"] != recipe:
                raise ValueError("training resume contract differs")
            transition.load_state_dict(saved["transition"])
            optimizer.load_state_dict(saved["optimizer"])
            updates = saved["updates"]
            validation_history = saved["validation_history"]
            completed_steps = saved["completed_steps"]
            best_validation = saved["best_validation"]
            best_transition = saved["best_transition"]
            consumed_seconds = saved["consumed_seconds"]
            prior_peak_vram = saved["peak_workload_vram_bytes"]
        if args.checkpoint_dir is not None:
            args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if not validation_history:
            validation = {
                "step": 0,
                **validate_batches(
                    backend,
                    layers,
                    transition,
                    validation_batches,
                    documents,
                    args.representation,
                    scaling,
                    args.compiler_chunk_batch_size,
                ),
            }
            validation_history.append(validation)
            best_validation = validation["gold_answer_ce"]
            best_transition = copy.deepcopy(
                {name: value.detach().cpu() for name, value in transition.state_dict().items()}
            )
            if args.checkpoint_dir is not None:
                atomic_torch_save(
                    reader_payload(transition, args.representation, 0),
                    args.checkpoint_dir / "step-000000.pt",
                )
        segment_started = time.perf_counter()
        for step in range(completed_steps, args.steps):
            updates.append(
                {
                    "step": step + 1,
                    **train_batch(
                        backend,
                        layers,
                        transition,
                        optimizer,
                        selected_batches[step],
                        documents,
                        args.representation,
                        scaling,
                        args.compiler_chunk_batch_size,
                        args.objective,
                        args.max_new_tokens,
                        args.teacher_replay_batch_size,
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
                        layers,
                        transition,
                        validation_batches,
                        documents,
                        args.representation,
                        scaling,
                        args.compiler_chunk_batch_size,
                    ),
                }
                validation_history.append(validation)
                if validation["gold_answer_ce"] < best_validation:
                    best_validation = validation["gold_answer_ce"]
                    best_transition = copy.deepcopy(
                        {name: value.detach().cpu() for name, value in transition.state_dict().items()}
                    )
                if args.checkpoint_dir is not None:
                    atomic_torch_save(
                        reader_payload(transition, args.representation, completed_steps),
                        args.checkpoint_dir / f"step-{completed_steps:06d}.pt",
                    )
            if completed_steps % args.checkpoint_interval == 0 or should_validate:
                consumed_seconds += time.perf_counter() - segment_started
                atomic_torch_save(
                    {
                        "identity": identity,
                        "recipe": recipe,
                        "transition": transition.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "updates": updates,
                        "validation_history": validation_history,
                        "completed_steps": completed_steps,
                        "best_validation": best_validation,
                        "best_transition": best_transition,
                        "consumed_seconds": consumed_seconds,
                        "peak_workload_vram_bytes": max(
                            prior_peak_vram,
                            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                        ),
                    },
                    resume_path,
                )
                segment_started = time.perf_counter()
        if best_transition is None:
            raise RuntimeError("training completed without validation")
        transition.load_state_dict(best_transition)
        selected_step = min(validation_history, key=lambda row: row["gold_answer_ce"])["step"]
        atomic_torch_save(reader_payload(transition, args.representation, selected_step), state_output)
        report = {
            "schema_version": 1,
            "experiment": (
                "STUDENT_PREFIX-student-prefix-on-policy-rank-effect"
                if args.objective == "student-prefix-forward-kl"
                else (
                    "PROJECTED_SOURCE-projected-pre-perceiver-source-oracle"
                    if args.representation == "projected-source"
                    else "READOUT_LADDER-k1-compiled-representation-ladder"
                )
            ),
            "status": "completed",
            "contracts": {
                **recipe,
                "train_examples": sum(map(len, selected_batches)),
                "validation_examples": sum(map(len, validation_batches)),
                "single_chunk_required": True,
                "native_path_frozen": True,
                "selection_status": "provisional_gold_ce_pending_fresh_generation",
            },
            "updates": updates,
            "correctness": correctness,
            "validation": {
                "history": validation_history,
                "provisional_selected_step": selected_step,
                "provisional_selected_gold_answer_ce": best_validation,
            },
            "accounting": {
                "reader_parameter_bytes": sum(
                    parameter.numel() * parameter.element_size() for parameter in transition.parameters()
                ),
                "reader_backbone_backward_passes": completed_steps,
                "student_rollout_forward_passes": (
                    completed_steps if args.objective == "student-prefix-forward-kl" else 0
                ),
                "teacher_replay_forward_passes": sum(
                    update.get("teacher_replay_forward_passes", 0) for update in updates
                ),
                "wall_seconds": consumed_seconds,
                "peak_workload_vram_bytes": max(
                    prior_peak_vram,
                    torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                ),
            },
            "state_output": str(state_output),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        resume_path.unlink()
        print(json.dumps({"status": "completed", "selected_step": selected_step}, indent=2))
    finally:
        backend.close()


if __name__ == "__main__":
    main()
