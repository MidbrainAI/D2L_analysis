"""Cache-free minibatch teacher-to-D2L reader distillation.

Each update consumes raw document text, generates temporary batch-specific D2L
LoRAs on GPU, runs the text teacher and PK-free student sequentially, and then
discards the generated state. Only the shared post-block reader is trainable.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from d2l_scale import load_config, load_manifest, paths
from d2l_scale_core import Query
from post_block_transitions import installed_direct_transition
from reader_banks import (
    LEARNING_RATE,
    build_dataset_local_plans,
    compact_topk,
    sparse_coarsened_forward_kl,
)
from reader_banks import split_rows
from post_block_transitions import installed_transition, make_state_transition

DEFAULT_BATCH_SIZE = 16
DEFAULT_DOCUMENT_OCCURRENCE_BUDGET = 80
DEFAULT_PADDED_STUDENT_TOKEN_BUDGET = 4096
DEFAULT_VALIDATION_INTERVAL = 500
DEFAULT_CHECKPOINT_INTERVAL = 100


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def transition_payload(transition, transition_type: str, bottleneck: int, selected_step: int) -> dict:
    return {
        "transition": transition.state_dict(),
        "transition_type": transition_type,
        "transition_bottleneck": bottleneck,
        "execution_site": transition.execution_site,
        "selected_step": selected_step,
    }


def plan_identity(plans: list[dict]) -> list[tuple[str, int, tuple[str, ...]]]:
    return [
        (plan["query"]["id"], plan["k"], tuple(plan["bank_document_ids"]))
        for plan in plans
    ]


def validate_resume(saved: dict, identity: dict, recipe: dict) -> None:
    if saved.get("identity") != identity:
        raise ValueError("training resume identity differs")
    compatible = [recipe]
    if recipe.get("transition_type") == "shared":
        version3 = {**recipe, "version": 3}
        version3.pop("transition_type")
        version3.pop("transition_bottleneck")
        version2 = {**version3, "version": 2}
        version2.pop("compiler_chunk_batch_size", None)
        compatible.extend((version3, version2))
    if saved.get("recipe") not in compatible:
        raise ValueError("training resume recipe differs")


def concatenate_generated_parts(parts: list[dict]) -> dict:
    if not parts:
        raise ValueError("expected at least one generated-factor part")
    return {
        module: {
            key: torch.cat([part[module][key] for part in parts], dim=0)
            for key in ("A", "B")
        }
        for module in parts[0]
    }


def pad_sequences(
    sequences: list[torch.Tensor],
    pad_token_id: int,
    *,
    device: torch.device,
    left: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not sequences or any(sequence.ndim != 1 or sequence.numel() == 0 for sequence in sequences):
        raise ValueError("expected nonempty one-dimensional token sequences")
    width = max(sequence.numel() for sequence in sequences)
    ids = torch.full((len(sequences), width), pad_token_id, dtype=sequences[0].dtype, device=device)
    mask = torch.zeros((len(sequences), width), dtype=torch.long, device=device)
    for index, sequence in enumerate(sequences):
        sequence = sequence.to(device)
        target = slice(width - sequence.numel(), width) if left else slice(0, sequence.numel())
        ids[index, target] = sequence
        mask[index, target] = 1
    return ids, mask


def response_token_lengths(
    continuations: torch.Tensor,
    eos_token_ids: tuple[int, ...],
    pad_token_id: int,
) -> list[int]:
    lengths = []
    for continuation in continuations:
        length = continuation.numel()
        for index, token in enumerate(continuation.tolist()):
            if token in eos_token_ids:
                length = index + 1
                break
            if token == pad_token_id:
                length = index
                break
        if length < 1:
            raise RuntimeError("teacher generated an empty response")
        lengths.append(length)
    return lengths


def response_position_mask(lengths: list[int], *, device: torch.device) -> torch.Tensor:
    width = max(lengths)
    return torch.arange(width, device=device).unsqueeze(0) < torch.tensor(lengths, device=device).unsqueeze(1)


def gather_response_logits(
    logits: torch.Tensor,
    prompt_lengths: list[int],
    response_lengths: list[int],
) -> torch.Tensor:
    if logits.shape[0] != len(prompt_lengths) or len(prompt_lengths) != len(response_lengths):
        raise ValueError("logit batch and response metadata differ")
    width = max(response_lengths)
    offsets = torch.arange(width, device=logits.device).unsqueeze(0)
    starts = torch.tensor(prompt_lengths, device=logits.device).unsqueeze(1) - 1
    positions = (starts + offsets).clamp_max(logits.shape[1] - 1)
    batches = torch.arange(logits.shape[0], device=logits.device).unsqueeze(1).expand_as(positions)
    return logits[batches, positions]


def render_prompt(backend, question: str, dataset: str, context: str = "") -> torch.Tensor:
    prompt = backend.templates[dataset].format(input=question)
    if context:
        prompt = context.strip() + "\n\n" + prompt
    return backend.tokenizer.apply_chat_template(
        [{"role": "system", "content": ""}, {"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        add_special_tokens=False,
    )[0].cpu()


def bank_context(plan: dict, documents: dict[str, dict]) -> str:
    return "\n\n".join(documents[document_id]["text"] for document_id in plan["bank_document_ids"])


def make_plan_batches(
    plans: list[dict],
    max_examples: int,
    max_document_occurrences: int,
    teacher_tokens: dict[str, int],
    student_tokens: dict[str, int] | None = None,
    max_padded_student_tokens: int = DEFAULT_PADDED_STUDENT_TOKEN_BUDGET,
) -> list[list[dict]]:
    """Create dataset/K-homogeneous, length-sorted batches under fixed budgets."""
    grouped: dict[tuple[str, int], list[dict]] = {}
    for plan in plans:
        key = (plan["query"].get("dataset", ""), plan["k"])
        grouped.setdefault(key, []).append(plan)
    chunks_by_group = {}
    for key, group in grouped.items():
        _, k = key
        group.sort(key=lambda plan: teacher_tokens[plan["query"]["id"]])
        size = min(max_examples, max_document_occurrences // k)
        if size < 1:
            raise ValueError(f"document-occurrence budget cannot fit K={k}")
        chunks, current = [], []
        for plan in group:
            candidate = [*current, plan]
            padded_student_tokens = 0
            if student_tokens is not None:
                padded_student_tokens = len(candidate) * max(
                    student_tokens[item["query"]["id"]] for item in candidate
                )
            if current and (len(candidate) > size or padded_student_tokens > max_padded_student_tokens):
                chunks.append(current)
                current = [plan]
            else:
                current = candidate
        if current:
            chunks.append(current)
        chunks_by_group[key] = chunks
    return [
        chunks_by_group[key][round_index]
        for round_index in range(max(map(len, chunks_by_group.values()), default=0))
        for key in sorted(chunks_by_group)
        if round_index < len(chunks_by_group[key])
    ]


def make_teacher_batch(
    backend,
    plans: list[dict],
    documents: dict[str, dict],
    max_new_tokens: int,
    teacher_generation_batch_size: int,
    teacher_replay_batch_size: int,
) -> dict:
    model = backend.model.base_model
    device = model.device
    teacher_prompts, student_prompts = [], []
    for plan in plans:
        query = Query(**plan["query"])
        teacher_prompts.append(render_prompt(backend, query.input, query.dataset, bank_context(plan, documents)))
        student_prompts.append(render_prompt(backend, query.input, query.dataset))
    backend.model.reset()
    eos = backend.tokenizer.eos_token_id
    eos_ids = tuple(eos) if isinstance(eos, (list, tuple)) else (eos,)
    responses, response_lengths = [], []
    teacher_prompt_tokens = 0
    for start in range(0, len(plans), teacher_generation_batch_size):
        prompt_part = teacher_prompts[start : start + teacher_generation_batch_size]
        teacher_ids, teacher_mask = pad_sequences(
            prompt_part,
            backend.tokenizer.pad_token_id,
            device=device,
            left=True,
        )
        teacher_prompt_tokens += int(teacher_mask.sum())
        with torch.inference_mode():
            generated = model.generate(
                teacher_ids,
                attention_mask=teacher_mask,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=backend.tokenizer.pad_token_id,
            )
        continuations = generated[:, teacher_ids.shape[1] :]
        part_lengths = response_token_lengths(continuations, eos_ids, backend.tokenizer.pad_token_id)
        response_lengths.extend(part_lengths)
        responses.extend(
            continuations[index, :length].cpu()
            for index, length in enumerate(part_lengths)
        )

    teacher_replay = [
        torch.cat((prompt, response[:-1]))
        for prompt, response in zip(teacher_prompts, responses, strict=True)
    ]
    teacher_top_parts = []
    response_width = max(response_lengths)
    for start in range(0, len(plans), teacher_replay_batch_size):
        end = min(start + teacher_replay_batch_size, len(plans))
        replay_ids, replay_mask = pad_sequences(
            teacher_replay[start:end],
            backend.tokenizer.pad_token_id,
            device=device,
        )
        with torch.inference_mode():
            teacher_output = model(
                replay_ids,
                attention_mask=replay_mask,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
            teacher_logits = gather_response_logits(
                teacher_output.logits,
                [prompt.numel() for prompt in teacher_prompts[start:end]],
                response_lengths[start:end],
            )
            compact = compact_topk(teacher_logits)
        pad = response_width - compact["indices"].shape[1]
        if pad:
            compact = {
                "indices": torch.cat(
                    (
                        compact["indices"],
                        torch.zeros(
                            compact["indices"].shape[0],
                            pad,
                            compact["indices"].shape[2],
                            dtype=compact["indices"].dtype,
                        ),
                    ),
                    dim=1,
                ),
                "log_probs": torch.cat(
                    (
                        compact["log_probs"],
                        torch.zeros(
                            compact["log_probs"].shape[0],
                            pad,
                            compact["log_probs"].shape[2],
                            dtype=compact["log_probs"].dtype,
                        ),
                    ),
                    dim=1,
                ),
                "tail_log_prob": torch.cat(
                    (
                        compact["tail_log_prob"],
                        torch.zeros(
                            compact["tail_log_prob"].shape[0],
                            pad,
                            dtype=compact["tail_log_prob"].dtype,
                        ),
                    ),
                    dim=1,
                ),
            }
        teacher_top_parts.append(compact)
    teacher_top = {
        key: torch.cat([part[key] for part in teacher_top_parts], dim=0)
        for key in teacher_top_parts[0]
    }
    student_replay = [
        torch.cat((prompt, response[:-1]))
        for prompt, response in zip(student_prompts, responses, strict=True)
    ]
    student_ids, student_mask = pad_sequences(
        student_replay,
        backend.tokenizer.pad_token_id,
        device=device,
    )
    return {
        "teacher_top": teacher_top,
        "responses": responses,
        "student_ids": student_ids,
        "student_mask": student_mask,
        "student_prompt_lengths": [prompt.numel() for prompt in student_prompts],
        "response_lengths": response_lengths,
        "response_mask": response_position_mask(response_lengths, device=device),
        "teacher_prompt_tokens": teacher_prompt_tokens,
        "student_replay_tokens": int(student_mask.sum()),
    }


def compile_bank_batch(
    backend,
    plans: list[dict],
    documents: dict[str, dict],
    compiler_chunk_batch_size: int,
) -> tuple[dict, dict]:
    chunks, chunks_per_example, source_tokens = [], [], 0
    for plan in plans:
        count = 0
        for document_id in plan["bank_document_ids"]:
            text = documents[document_id]["text"]
            ids = backend.tokenize_context({"context": [text]}, backend.tokenizer)["ctx_ids"][0]
            document_chunks = backend.split_context(
                {"ctx_ids": ids},
                model_name_or_path=backend.model_id,
                num_chunk_probs=None,
                max_chunk_len=backend.config["model"]["context_chunk_len"],
                min_chunk_len=0,
                max_num_split=None,
                is_train=False,
            )["ctx_ids"]
            chunks.extend(torch.tensor(chunk, dtype=torch.long) for chunk in document_chunks)
            count += len(document_chunks)
            source_tokens += len(ids)
        chunks_per_example.append(count)
    generated_parts = []
    with torch.inference_mode():
        for start in range(0, len(chunks), compiler_chunk_batch_size):
            ctx_ids, ctx_mask = pad_sequences(
                chunks[start : start + compiler_chunk_batch_size],
                backend.tokenizer.pad_token_id,
                device=backend.model.device,
            )
            generated, layernorms = backend.model.generate_weights(ctx_ids, ctx_mask, None)
            if layernorms:
                raise ValueError("layernorm adaptation is outside the frozen LoRA protocol")
            generated_parts.append(generated)
        generated = concatenate_generated_parts(generated_parts)
        head_bias = {
            module: {key: value.to(backend.model.device) for key, value in part.items()}
            for module, part in backend.head_bias.items()
        }
        n_chunks = torch.tensor(chunks_per_example, device=backend.model.device, dtype=torch.int32)
        state = backend.merge(generated, n_chunks, lora_bias=head_bias, bias_scaler=1.0)
    # Reader autograd consumes the factors as constants, but inference tensors
    # cannot be saved by downstream differentiable operations.
    state = {
        module: {key: value.detach().clone() for key, value in part.items()}
        for module, part in state.items()
    }
    return state, {
        "document_occurrences": sum(plan["k"] for plan in plans),
        "context_chunks": len(chunks),
        "source_tokens": source_tokens,
    }


def forward_student_batch(backend, layers, transition, state, targets, scaling) -> torch.Tensor:
    backend.model.reset()
    context = (
        installed_direct_transition(
            backend.model.base_model,
            layers,
            state,
            transition,
            scaling,
        )
        if transition.execution_site == "write-site"
        else installed_transition(
            backend.model.base_model,
            layers,
            state,
            transition,
            scaling,
            official_backend=backend,
        )
    )
    with context:
        output = backend.model.base_model(
            targets["student_ids"],
            attention_mask=targets["student_mask"],
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
    return gather_response_logits(
        output.logits,
        targets["student_prompt_lengths"],
        targets["response_lengths"],
    )


def batch_parity_check(
    backend,
    layers,
    transition,
    plans,
    documents,
    scaling,
    max_new_tokens,
    teacher_generation_batch_size,
    teacher_replay_batch_size,
    compiler_chunk_batch_size,
) -> dict:
    targets = make_teacher_batch(
        backend,
        plans,
        documents,
        max_new_tokens,
        teacher_generation_batch_size,
        teacher_replay_batch_size,
    )
    state, _ = compile_bank_batch(backend, plans, documents, compiler_chunk_batch_size)
    with torch.inference_mode():
        batched_logits = forward_student_batch(backend, layers, transition, state, targets, scaling)
    factor_max_abs = 0.0
    student_max_abs = 0.0
    student_abs_sum = 0.0
    student_logit_count = 0
    student_forward_kl_sum = 0.0
    student_response_positions = 0
    student_top1_disagreements = 0
    teacher_top1_disagreements = 0
    teacher_top1_logprob_max_abs = 0.0
    response_mismatches = 0
    for index, plan in enumerate(plans):
        independent_targets = make_teacher_batch(backend, [plan], documents, max_new_tokens, 1, 1)
        independent_state, _ = compile_bank_batch(backend, [plan], documents, compiler_chunk_batch_size)
        if not torch.equal(targets["responses"][index], independent_targets["responses"][0]):
            response_mismatches += 1
        length = targets["response_lengths"][index]
        teacher_compare_length = min(length, independent_targets["response_lengths"][0])
        batch_top = targets["teacher_top"]
        independent_top = independent_targets["teacher_top"]
        teacher_top1_disagreements += int(
            (
                batch_top["indices"][index, :teacher_compare_length, 0]
                != independent_top["indices"][0, :teacher_compare_length, 0]
            ).sum()
        )
        teacher_top1_logprob_max_abs = max(
            teacher_top1_logprob_max_abs,
            float(
                (
                    batch_top["log_probs"][index, :teacher_compare_length, 0].float()
                    - independent_top["log_probs"][0, :teacher_compare_length, 0].float()
                )
                .abs()
                .max()
            ),
        )
        for module, part in state.items():
            for key, value in part.items():
                expected = torch.zeros_like(value[index])
                single = independent_state[module][key][0]
                expected[:, : single.shape[1]] = single
                factor_max_abs = max(factor_max_abs, float((value[index] - expected).abs().max()))
        sequence_length = int(targets["student_mask"][index].sum())
        independent_from_batch = {
            "student_ids": targets["student_ids"][index : index + 1, :sequence_length],
            "student_mask": targets["student_mask"][index : index + 1, :sequence_length],
            "student_prompt_lengths": [targets["student_prompt_lengths"][index]],
            "response_lengths": [length],
        }
        with torch.inference_mode():
            independent_logits = forward_student_batch(
                backend,
                layers,
                transition,
                independent_state,
                independent_from_batch,
                scaling,
            )[0, :length]
        compared = batched_logits[index, :length]
        difference = (compared - independent_logits).detach().float().abs()
        student_max_abs = max(student_max_abs, float(difference.max()))
        student_abs_sum += float(difference.sum())
        student_logit_count += difference.numel()
        independent_log = independent_logits.detach().float().log_softmax(-1)
        compared_log = compared.detach().float().log_softmax(-1)
        student_forward_kl_sum += float(
            (independent_log.exp() * (independent_log - compared_log)).sum(-1).sum()
        )
        student_response_positions += length
        student_top1_disagreements += int((compared.argmax(-1) != independent_logits.argmax(-1)).sum())
    return {
        "examples": len(plans),
        "response_mismatches": response_mismatches,
        "teacher_top1_disagreements": teacher_top1_disagreements,
        "teacher_top1_logprob_max_abs": teacher_top1_logprob_max_abs,
        "factor_max_abs": factor_max_abs,
        "student_logit_max_abs": student_max_abs,
        "student_logit_mean_abs": student_abs_sum / student_logit_count,
        "student_forward_kl_per_response_token": student_forward_kl_sum / student_response_positions,
        "student_top1_disagreements": student_top1_disagreements,
    }


def train_batch(
    backend,
    layers,
    transition,
    optimizer,
    plans,
    documents,
    scaling,
    max_new_tokens,
    teacher_generation_batch_size,
    teacher_replay_batch_size,
    compiler_chunk_batch_size,
) -> dict:
    started = time.perf_counter()
    targets = make_teacher_batch(
        backend,
        plans,
        documents,
        max_new_tokens,
        teacher_generation_batch_size,
        teacher_replay_batch_size,
    )
    teacher_seconds = time.perf_counter() - started
    compile_started = time.perf_counter()
    state, compilation = compile_bank_batch(backend, plans, documents, compiler_chunk_batch_size)
    compile_seconds = time.perf_counter() - compile_started
    optimizer.zero_grad(set_to_none=True)
    transition.train()
    student_started = time.perf_counter()
    candidate_logits = forward_student_batch(backend, layers, transition, state, targets, scaling)
    loss = sparse_coarsened_forward_kl(targets["teacher_top"], candidate_logits, targets["response_mask"])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(transition.parameters(), 1.0)
    optimizer.step()
    student_seconds = time.perf_counter() - student_started
    return {
        "loss": float(loss.detach()),
        "teacher_seconds": teacher_seconds,
        "compile_seconds": compile_seconds,
        "student_seconds": student_seconds,
        "teacher_prompt_tokens": targets["teacher_prompt_tokens"],
        "student_replay_tokens": targets["student_replay_tokens"],
        **compilation,
    }


def validate_batches(
    backend,
    layers,
    transition,
    batches,
    documents,
    scaling,
    max_new_tokens,
    teacher_generation_batch_size,
    teacher_replay_batch_size,
    compiler_chunk_batch_size,
) -> dict:
    transition.eval()
    weighted_loss = 0.0
    response_tokens = 0
    examples = 0
    by_dataset: dict[str, dict[str, float | int]] = {}
    started = time.perf_counter()
    for batch in batches:
        targets = make_teacher_batch(
            backend,
            batch,
            documents,
            max_new_tokens,
            teacher_generation_batch_size,
            teacher_replay_batch_size,
        )
        state, _ = compile_bank_batch(backend, batch, documents, compiler_chunk_batch_size)
        with torch.inference_mode():
            logits = forward_student_batch(backend, layers, transition, state, targets, scaling)
            loss = sparse_coarsened_forward_kl(targets["teacher_top"], logits, targets["response_mask"])
        tokens = int(targets["response_mask"].sum())
        weighted_loss += float(loss) * tokens
        response_tokens += tokens
        examples += len(batch)
        datasets = {plan["query"]["dataset"] for plan in batch}
        if len(datasets) != 1:
            raise ValueError("validation batches must be dataset-homogeneous")
        dataset = datasets.pop()
        summary = by_dataset.setdefault(dataset, {"weighted_loss": 0.0, "response_tokens": 0, "examples": 0})
        summary["weighted_loss"] += float(loss) * tokens
        summary["response_tokens"] += tokens
        summary["examples"] += len(batch)
    dataset_metrics = {
        dataset: {
            "sparse_forward_kl": summary["weighted_loss"] / summary["response_tokens"],
            "examples": summary["examples"],
            "response_tokens": summary["response_tokens"],
        }
        for dataset, summary in sorted(by_dataset.items())
    }
    return {
        "sparse_forward_kl": weighted_loss / response_tokens,
        "selection_sparse_forward_kl": sum(
            metrics["sparse_forward_kl"] for metrics in dataset_metrics.values()
        )
        / len(dataset_metrics),
        "by_dataset": dataset_metrics,
        "examples": examples,
        "response_tokens": response_tokens,
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--max-document-occurrences-per-batch",
        type=int,
        default=DEFAULT_DOCUMENT_OCCURRENCE_BUDGET,
    )
    parser.add_argument(
        "--max-padded-student-tokens-per-batch",
        type=int,
        default=DEFAULT_PADDED_STUDENT_TOKEN_BUDGET,
    )
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--max-source-queries", type=int, default=20)
    parser.add_argument("--max-validation-source-queries", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--teacher-generation-batch-size", type=int, default=8)
    parser.add_argument("--teacher-replay-batch-size", type=int, default=4)
    parser.add_argument("--compiler-chunk-batch-size", type=int, default=8)
    parser.add_argument("--k", type=int)
    parser.add_argument(
        "--transition-type",
        choices=("shared", "per-layer", "write-site-residual"),
        default="shared",
    )
    parser.add_argument("--transition-bottleneck", type=int, default=32)
    parser.add_argument("--sort-by-teacher-tokens", action="store_true")
    parser.add_argument(
        "--bucket-by-k-and-length",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--parity-check", action="store_true")
    parser.add_argument("--validation-interval", type=int, default=DEFAULT_VALIDATION_INTERVAL)
    parser.add_argument("--checkpoint-interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-state", type=Path)
    parser.add_argument("--state-output", type=Path)
    parser.add_argument("--validation-checkpoint-dir", type=Path)
    args = parser.parse_args()
    if min(
        args.batch_size,
        args.steps,
        args.max_source_queries,
        args.max_validation_source_queries,
        args.max_new_tokens,
        args.teacher_generation_batch_size,
        args.teacher_replay_batch_size,
        args.compiler_chunk_batch_size,
        args.transition_bottleneck,
        args.max_document_occurrences_per_batch,
        args.max_padded_student_tokens_per_batch,
        args.validation_interval,
        args.checkpoint_interval,
    ) < 1:
        raise ValueError("batch size, steps, query limit, and token limit must be positive")
    if args.k is not None and (args.k < 1 or args.k > 10):
        raise ValueError("K must be between 1 and 10")
    if args.output.exists():
        raise FileExistsError(args.output)
    resume_state_path = args.resume_state or args.output.with_suffix(".resume.pt")
    state_output_path = args.state_output or args.output.with_suffix(".states.pt")
    if args.resume and not resume_state_path.exists():
        raise FileNotFoundError(resume_state_path)
    if not args.resume and resume_state_path.exists():
        raise FileExistsError(resume_state_path)
    if state_output_path.exists():
        raise FileExistsError(state_output_path)
    root = Path(__file__).parents[2]
    cfg = load_config(root, args.config)
    payload = load_manifest(paths(root, cfg)["manifest"])
    documents = {document["id"]: document for document in payload["documents"]}
    rows = split_rows(payload)
    _, plans = build_dataset_local_plans(
        rows["train"],
        documents,
        split="train",
        limit=args.max_source_queries,
        fixed_k=args.k,
    )
    _, validation_plans = build_dataset_local_plans(
        rows["validation"],
        documents,
        split="validation",
        limit=args.max_validation_source_queries,
        fixed_k=args.k,
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
        if args.bucket_by_k_and_length:
            batches = make_plan_batches(
                plans,
                args.batch_size,
                args.max_document_occurrences_per_batch,
                teacher_tokens,
                student_tokens,
                args.max_padded_student_tokens_per_batch,
            )
        else:
            if args.sort_by_teacher_tokens:
                plans.sort(key=lambda plan: teacher_tokens[plan["query"]["id"]])
            batches = [
                plans[start : start + args.batch_size]
                for start in range(0, len(plans), args.batch_size)
            ]
        validation_batches = make_plan_batches(
            validation_plans,
            args.batch_size,
            args.max_document_occurrences_per_batch,
            teacher_tokens,
            student_tokens,
            args.max_padded_student_tokens_per_batch,
        )
        if len(batches) < args.steps:
            raise ValueError(f"need {args.steps} batches, found {len(batches)}")
        model = backend.model.base_model
        layers = get_layers(model)
        scaling = model.peft_config["default"].lora_alpha
        transition = make_state_transition(
            args.transition_type,
            model.config.hidden_size,
            len(layers),
            args.transition_bottleneck,
        ).to(model.device)
        optimizer = torch.optim.AdamW(transition.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        if args.validation_checkpoint_dir is not None:
            args.validation_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        selected_batches = batches[: args.steps]
        identity = {
            "train": plan_identity([plan for batch in selected_batches for plan in batch]),
            "validation": plan_identity([plan for batch in validation_batches for plan in batch]),
        }
        recipe = {
            "version": 4,
            "config": args.config,
            "batch_size": args.batch_size,
            "max_document_occurrences_per_batch": args.max_document_occurrences_per_batch,
            "max_padded_student_tokens_per_batch": args.max_padded_student_tokens_per_batch,
            "max_new_tokens": args.max_new_tokens,
            "teacher_generation_batch_size": args.teacher_generation_batch_size,
            "teacher_replay_batch_size": args.teacher_replay_batch_size,
            "compiler_chunk_batch_size": args.compiler_chunk_batch_size,
            "fixed_k": args.k,
            "transition_type": args.transition_type,
            "transition_bottleneck": args.transition_bottleneck,
            "bucket_by_k_and_length": args.bucket_by_k_and_length,
            "dataset_homogeneous_batches": True,
            "sort_by_teacher_tokens": args.sort_by_teacher_tokens,
            "validation_interval": args.validation_interval,
            "learning_rate": LEARNING_RATE,
            "weight_decay": 1e-4,
        }
        updates, validation_history = [], []
        completed_steps = 0
        best_validation = float("inf")
        best_transition = None
        consumed_seconds = 0.0
        prior_peak_vram = 0
        saved_parity = None
        if args.resume:
            saved = torch.load(resume_state_path, map_location=model.device, weights_only=False)
            validate_resume(saved, identity, recipe)
            transition.load_state_dict(saved["transition"])
            optimizer.load_state_dict(saved["optimizer"])
            updates = saved["updates"]
            validation_history = saved["validation_history"]
            completed_steps = saved["completed_steps"]
            best_validation = saved["best_validation"]
            best_transition = saved["best_transition"]
            consumed_seconds = saved["consumed_seconds"]
            prior_peak_vram = saved.get("peak_workload_vram_bytes", 0)
            saved_parity = saved.get("parity")
        initial_allocated = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        parity = (
            batch_parity_check(
                backend,
                layers,
                transition,
                batches[0],
                documents,
                scaling,
                args.max_new_tokens,
                args.teacher_generation_batch_size,
                args.teacher_replay_batch_size,
                args.compiler_chunk_batch_size,
            )
            if args.parity_check and completed_steps == 0
            else saved_parity
        )
        started = time.perf_counter()
        segment_started = started
        if completed_steps == 0 and not validation_history:
            initial_validation = {
                "step": 0,
                **validate_batches(
                    backend,
                    layers,
                    transition,
                    validation_batches,
                    documents,
                    scaling,
                    args.max_new_tokens,
                    args.teacher_generation_batch_size,
                    args.teacher_replay_batch_size,
                    args.compiler_chunk_batch_size,
                ),
            }
            validation_history.append(initial_validation)
            best_validation = initial_validation["selection_sparse_forward_kl"]
            best_transition = copy.deepcopy(
                {name: value.detach().cpu() for name, value in transition.state_dict().items()}
            )
            if args.validation_checkpoint_dir is not None:
                atomic_torch_save(
                    transition_payload(transition, args.transition_type, args.transition_bottleneck, 0),
                    args.validation_checkpoint_dir / "step-000000.pt",
                )
        for step in range(completed_steps, args.steps):
            batch = batches[step]
            updates.append(
                {
                    "step": step + 1,
                    "k_values": [plan["k"] for plan in batch],
                    **train_batch(
                        backend,
                        layers,
                        transition,
                        optimizer,
                        batch,
                        documents,
                        scaling,
                        args.max_new_tokens,
                        args.teacher_generation_batch_size,
                        args.teacher_replay_batch_size,
                        args.compiler_chunk_batch_size,
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
                        scaling,
                        args.max_new_tokens,
                        args.teacher_generation_batch_size,
                        args.teacher_replay_batch_size,
                        args.compiler_chunk_batch_size,
                    ),
                }
                validation_history.append(validation)
                if validation["selection_sparse_forward_kl"] < best_validation:
                    best_validation = validation["selection_sparse_forward_kl"]
                    best_transition = copy.deepcopy(
                        {name: value.detach().cpu() for name, value in transition.state_dict().items()}
                    )
                if args.validation_checkpoint_dir is not None:
                    atomic_torch_save(
                        transition_payload(
                            transition,
                            args.transition_type,
                            args.transition_bottleneck,
                            completed_steps,
                        ),
                        args.validation_checkpoint_dir / f"step-{completed_steps:06d}.pt",
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
                        "parity": parity,
                    },
                    resume_state_path,
                )
                segment_started = time.perf_counter()
        consumed_seconds += time.perf_counter() - segment_started
        if best_transition is None:
            raise RuntimeError("training completed without validation")
        transition.load_state_dict(best_transition)
        selected_step = min(
            validation_history, key=lambda item: item["selection_sparse_forward_kl"]
        )["step"]
        atomic_torch_save(
            transition_payload(
                transition,
                args.transition_type,
                args.transition_bottleneck,
                selected_step,
            ),
            state_output_path,
        )
        report = {
            "schema_version": 3,
            "experiment": "official-d2l-online-batch-distillation",
            "status": "completed",
            "config": args.config,
            "contracts": {
                "persistent_lora_cache": False,
                "persistent_teacher_traces": False,
                "teacher_and_student_share_target_model": True,
                "compiler_and_target_resident_together": True,
                "temporary_loras_discarded_per_batch": True,
                "shared_bias_once_per_example": True,
                "batch_size": args.batch_size,
                "max_document_occurrences_per_batch": args.max_document_occurrences_per_batch,
                "max_padded_student_tokens_per_batch": args.max_padded_student_tokens_per_batch,
                "teacher_generation_batch_size": args.teacher_generation_batch_size,
                "teacher_replay_batch_size": args.teacher_replay_batch_size,
                "compiler_chunk_batch_size": args.compiler_chunk_batch_size,
                "fixed_k": args.k,
                "transition_type": args.transition_type,
                "transition_bottleneck": args.transition_bottleneck,
                "sorted_by_teacher_tokens": args.sort_by_teacher_tokens,
                "bucketed_by_k_and_length": args.bucket_by_k_and_length,
                "dataset_homogeneous_batches": True,
                "validation_interval": args.validation_interval,
                "checkpoint_interval": args.checkpoint_interval,
                "validation_checkpoint_dir": (
                    str(args.validation_checkpoint_dir)
                    if args.validation_checkpoint_dir is not None
                    else None
                ),
                "plan_counts_by_dataset": {
                    "train": {
                        dataset: sum(
                            plan["query"]["dataset"] == dataset
                            for batch in selected_batches
                            for plan in batch
                        )
                        for dataset in sorted(
                            {plan["query"]["dataset"] for batch in selected_batches for plan in batch}
                        )
                    },
                    "validation": {
                        dataset: sum(
                            plan["query"]["dataset"] == dataset
                            for batch in validation_batches
                            for plan in batch
                        )
                        for dataset in sorted(
                            {plan["query"]["dataset"] for batch in validation_batches for plan in batch}
                        )
                    },
                },
            },
            "updates": updates,
            "validation": {
                "history": validation_history,
                "selected_macro_dataset_sparse_forward_kl": best_validation,
            },
            "parity": parity,
            "resume_state": str(resume_state_path),
            "state_output": str(state_output_path),
            "accounting": {
                "steps": completed_steps,
                "backward_passes": completed_steps,
                "examples": sum(len(batch) for batch in selected_batches),
                "total_wall_seconds": consumed_seconds,
                "initial_allocated_vram_bytes": initial_allocated,
                "peak_workload_vram_bytes": max(
                    prior_peak_vram,
                    torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                ),
                "persistent_lora_bytes": 0,
                "persistent_trace_bytes": 0,
                "resume_state_bytes": resume_state_path.stat().st_size,
                "transition_bytes": state_output_path.stat().st_size,
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        backend.close()


if __name__ == "__main__":
    main()
