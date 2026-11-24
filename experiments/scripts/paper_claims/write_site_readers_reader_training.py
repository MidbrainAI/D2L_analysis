"""Train one write-site reader arm (A/B/C/D) with DISTRACTOR_CURVES's data contract and an answer-token target.

Differences from ``distractor_curves_reader_training`` (everything else is reused from it):

* **Target.** The gold label is the answer's tokens followed by the chat terminator
  (``<end_of_turn>`` on Gemma; the tokenizer's eos where it is the terminator, as on Qwen).
  DISTRACTOR_CURVES appended ``<eos>``, which is not Gemma's terminator: it cost about 12.9 nats per row
  against about 0.8 per answer token and let a reader lower the CE by learning a stop-token
  substitution. An answer-only target was tried first and
  removed the stop supervision altogether: the reader answered correctly and then rambled
  (11 words per response against 1.5 for native), so fresh F1 fell while CE improved.
* **Held-out documents.** The 180 SQuAD-train contexts the released writer never saw
  (``inputs/write_site_readers/selection-v1.json``) are removed from the fitting rows, the validation rows,
  and both distractor pools; they form the arm-selection suite instead. Documents are matched
  by content-hash prefix because the reader-training manifest uses full hashes as IDs.
* **Arms.** ``arm`` selects DISTRACTOR_CURVES's post-block reader (A) or a write-site reader from
  ``write_site_readers_readers`` (B, C, D). Every arm starts as an exact zero correction.
* **Selection.** Inline validation measures answer-token CE per K and fresh greedy-generation
  F1 on a fixed subset of validation questions (40 per K). ``best`` is the checkpoint with
  the highest macro-F1 over K (ties to the earlier step); ``last`` is kept too.

Data-parallel use is unchanged: each rank takes every world-size-th question, one question
per backward pass, and token-weighted gradients are all-reduced.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from distractor_curves_core import K_VALUES, score
from distractor_curves_reader_training import (
    EXAMPLES_PER_UPDATE,
    RESUME_INTERVAL,
    VALIDATION_INTERVAL,
    all_reduce_sum,
    atomic_save,
    microbatches,
    representation_batch,
    training_rows,
    update_plans,
    validation_plans,
    world,
)
from write_site_readers_readers import ARMS, forward_with_reader, generate_with_reader, make_reader, parameter_count, reader_payload

RECIPE = "write_site_readers-answer-terminator-ce-mixed-k-v2"
END_OF_TURN_TOKEN = "<end_of_turn>"
# One SQuAD-train question (squad:train:75237) is 25,651 characters of whitespace padding (839
# tokens; the 99th percentile is 27). At K=16 DISTRACTOR_CURVES's reader materializes tokens x 128 slots x
# hidden per layer for backward and runs a 40 GB A100 out of memory.
MAX_QUESTION_CHARS = 2000
FRESH_PER_K = 40  # fresh-generation validation questions per K (200 in all)
SELECTION_SUITE = Path("inputs/write_site_readers/selection-v1.json")
QA_PROTOCOL_PREFIX = "qa_protocol-doc-"


# -- data (CPU) -------------------------------------------------------------------
def held_out_hashes(selection: dict[str, Any]) -> set[str]:
    """24-hex content-hash prefixes of the held-out documents (the QA_PROTOCOL ID suffix)."""
    return {i[len(QA_PROTOCOL_PREFIX):] for i in selection["held_out_document_ids"]}


def is_held_out(document: dict[str, Any], held: set[str]) -> bool:
    return document["contenthash"][:24] in held


def filtered_rows(payload: dict[str, Any], held: set[str],
                  max_question_chars: int = MAX_QUESTION_CHARS) -> dict[str, list[dict[str, Any]]]:
    """DISTRACTOR_CURVES's rows without questions on held-out documents or questions longer than the cap."""
    documents = {d["id"]: d for d in payload["documents"]}
    rows = training_rows(payload)
    return {split: [r for r in values if not is_held_out(documents[r["document"]["id"]], held)
                    and len(r["query"]["input"]) <= max_question_chars]
            for split, values in rows.items()}


def pools(latents: dict[str, Any], documents: dict[str, dict[str, Any]], held: set[str]) -> dict[str, list[str]]:
    return {split: sorted(i for i in latents if documents[i]["split"] == split and not is_held_out(documents[i], held))
            for split in ("train", "validation")}


def planned_steps(train_rows: int, requested: int) -> int:
    """One pass at 12 questions per update over the rows that remain after the held-out exclusion.

    DISTRACTOR_CURVES's 4,362 updates consumed 52,344 of READOUT_LADDER's 52,354 fitting questions; removing the
    questions on the 180 held-out documents leaves fewer, so the pass is shorter by the same
    amount for every arm. The actual count is reported beside the requested one.
    """
    available = train_rows // EXAMPLES_PER_UPDATE
    if available < 1:
        raise ValueError("no training rows remain after the held-out exclusion")
    return min(requested, available)


def fresh_subset(plans: list[dict[str, Any]], per_k: int = FRESH_PER_K) -> list[dict[str, Any]]:
    """The first ``per_k`` validation plans of every K, in plan order (fixed before any output)."""
    chosen = []
    for k in K_VALUES:
        chosen += [p for p in plans if p["k"] == k][:per_k]
    return chosen


# -- targets -------------------------------------------------------------------------
def terminator_id(tokenizer) -> int:
    """The chat terminator the model emits after an answer: ``<end_of_turn>`` if present, else eos."""
    vocab = tokenizer.get_vocab()
    if END_OF_TURN_TOKEN in vocab:
        return int(vocab[END_OF_TURN_TOKEN])
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer has neither <end_of_turn> nor an eos token")
    return int(tokenizer.eos_token_id)


def make_answer_batch(backend, plans: list[dict[str, Any]]) -> dict[str, Any]:
    """DISTRACTOR_CURVES's gold batch with the chat terminator as the stop label instead of ``<eos>``."""
    import torch
    from d2l_scale_core import Query
    from official_d2l_online_batch_distillation import pad_sequences, render_prompt

    stop = terminator_id(backend.tokenizer)
    replays, labels, prompt_lengths, response_lengths = [], [], [], []
    for plan in plans:
        query = Query(**plan["query"])
        answer = next((value.strip() for value in query.answers if value.strip()), None)
        if answer is None:
            raise ValueError(f"query {query.id} has no nonempty gold answer")
        answer_ids = backend.tokenizer(answer, add_special_tokens=False)["input_ids"]
        if not answer_ids:
            raise ValueError(f"query {query.id} produced no gold-answer tokens")
        answer_ids = [*answer_ids, stop]
        prompt = render_prompt(backend, query.input, query.dataset)
        answer_tensor = torch.tensor(answer_ids, dtype=prompt.dtype)
        replays.append(torch.cat((prompt, answer_tensor[:-1])))
        labels.append(answer_tensor)
        prompt_lengths.append(prompt.numel())
        response_lengths.append(answer_tensor.numel())
    student_ids, student_mask = pad_sequences(replays, backend.tokenizer.pad_token_id, device=backend.model.device)
    hard_targets = torch.full((len(labels), max(response_lengths)), -100, dtype=labels[0].dtype,
                              device=backend.model.device)
    for index, target in enumerate(labels):
        hard_targets[index, : target.numel()] = target.to(backend.model.device)
    return {"student_ids": student_ids, "student_mask": student_mask, "student_prompt_lengths": prompt_lengths,
            "response_lengths": response_lengths, "hard_targets": hard_targets}


def answer_tokens(backend, plan: dict[str, Any]) -> int:
    """Labelled positions of one plan: the answer tokens plus the terminator."""
    answer = next(value.strip() for value in plan["query"]["answers"] if value.strip())
    return len(backend.tokenizer(answer, add_special_tokens=False)["input_ids"]) + 1


def answer_loss(backend, layers, arm: str, reader, plans, latents, scaling) -> tuple[Any, int]:
    from official_d2l_k1_representation_training import gold_answer_loss

    targets = make_answer_batch(backend, plans)
    compiled = representation_batch(backend, plans, latents)
    logits = forward_with_reader(backend, layers, arm, reader, compiled, targets, scaling)
    return gold_answer_loss(logits, targets), sum(targets["response_lengths"])


LossFn = Callable[[Any, Any, str, Any, list, dict, float], tuple[Any, int]]


# -- validation ----------------------------------------------------------------------
def validate(backend, layers, arm: str, reader, plans, fresh, latents, scaling,
             max_document_occurrences, loss_fn: LossFn = answer_loss) -> dict[str, Any]:
    """Target CE (answer tokens + terminator) per K on every plan, and fresh greedy F1 per K on the fixed subset.

    ``loss_fn`` is the training objective (GATED_READERS adds a gate term to the CE); the reported
    ``answer_ce`` is whatever it returns, so the field is comparable within one arm.
    """
    import torch

    reader.eval()
    rank, size = world()
    ce = {k: [0.0, 0.0] for k in K_VALUES}
    with torch.no_grad():
        for k in K_VALUES:
            group = [p for p in plans if p["k"] == k]
            if not group:
                continue
            batches = [[p] for p in group[rank::size]] if size > 1 else microbatches(group, max_document_occurrences)
            for batch in batches:
                loss, count = loss_fn(backend, layers, arm, reader, batch, latents, scaling)
                ce[k][0] += float(loss) * count
                ce[k][1] += count
    f1 = {k: [0.0, 0.0] for k in K_VALUES}
    for plan in fresh[rank::size]:
        compiled = representation_batch(backend, [plan], latents)
        text = generate_with_reader(backend, layers, arm, reader, compiled, plan["query"]["input"],
                                    plan["query"]["dataset"])
        f1[plan["k"]][0] += score(text, plan["query"]["answers"])["f1"]
        f1[plan["k"]][1] += 1
    flat = all_reduce_sum([v for k in K_VALUES for v in ce[k]] + [v for k in K_VALUES for v in f1[k]],
                          backend.model.device)
    n = 2 * len(K_VALUES)
    ce_k = {str(k): flat[2 * i] / flat[2 * i + 1] for i, k in enumerate(K_VALUES) if flat[2 * i + 1]}
    f1_k = {str(k): flat[n + 2 * i] / flat[n + 2 * i + 1] for i, k in enumerate(K_VALUES) if flat[n + 2 * i + 1]}
    tokens = sum(flat[2 * i + 1] for i in range(len(K_VALUES)))
    return {"answer_ce": sum(flat[2 * i] for i in range(len(K_VALUES))) / tokens if tokens else None,
            "answer_ce_per_k": ce_k, "fresh_f1_per_k": f1_k,
            "fresh_macro_f1": sum(f1_k.values()) / len(f1_k) if f1_k else None,
            "fresh_questions": int(sum(flat[n + 2 * i + 1] for i in range(len(K_VALUES))))}


def better(candidate: dict[str, Any], incumbent: dict[str, Any] | None) -> bool:
    """Higher fresh macro-F1 wins; ties keep the earlier checkpoint."""
    if incumbent is None:
        return True
    return (candidate["fresh_macro_f1"] or -1.0) > (incumbent["fresh_macro_f1"] or -1.0)


# -- training (GPU) ---------------------------------------------------------------
def train_reader(backend, payload: dict[str, Any], latents: dict[str, Any], selection: dict[str, Any],
                 output_dir: Path, *, arm: str, seed: int, steps: int, max_document_occurrences: int,
                 validation_interval: int = VALIDATION_INTERVAL, validation_limit: int | None = None,
                 fresh_per_k: int = FRESH_PER_K, hidden: int = 128, bottleneck: int = 32,
                 reader_factory: Callable[..., Any] | None = None, loss_fn: LossFn = answer_loss,
                 known_arms: tuple[str, ...] = ARMS, recipe: str = RECIPE) -> dict[str, Any]:
    """Resumable mixed-K training of one arm; writes best.pt and last.pt under ``output_dir``.

    ``reader_factory(arm, model, latent_width, layers, bottleneck=, hidden=)`` builds the reader
    (default ``make_reader``); ``loss_fn`` is the per-microbatch objective (default answer CE);
    ``known_arms`` the arms the factory accepts. GATED_READERS's gated readers use all three.
    """
    import torch
    from ctx_to_lora.utils import get_layers
    from reader_banks import LEARNING_RATE

    if arm not in known_arms:
        raise ValueError(f"unknown arm {arm}")
    factory = reader_factory or make_reader
    held = held_out_hashes(selection)
    documents = {d["id"]: d for d in payload["documents"]}
    rows = {split: [r for r in values if r["document"]["id"] in latents]
            for split, values in filtered_rows(payload, held).items()}
    pool = pools(latents, documents, held)
    requested_steps, steps = steps, planned_steps(len(rows["train"]), steps)
    updates = update_plans(rows["train"], steps, seed, pool["train"])
    held_out_plans = validation_plans(rows["validation"][:validation_limit], pool["validation"])
    fresh = fresh_subset(held_out_plans, fresh_per_k)
    model = backend.model.base_model
    layers = get_layers(model)
    scaling = model.peft_config["default"].lora_alpha
    torch.manual_seed(seed)
    reader = factory(arm, model, backend.model.hypernet.config.latent_size, len(layers), bottleneck=bottleneck,
                     hidden=hidden).to(model.device)
    optimizer = torch.optim.AdamW(reader.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    resume = output_dir / "resume.pt"
    state: dict[str, Any] = {"step": 0, "history": [], "losses": [], "best": None, "best_validation": None,
                             "seconds": 0.0, "microbatches": 0}
    if resume.exists():
        saved = torch.load(resume, map_location=model.device, weights_only=False)
        if saved["seed"] != seed or saved["steps"] != steps or saved["arm"] != arm:  # steps: the planned count
            raise ValueError("resume state belongs to a different run")
        reader.load_state_dict(saved["reader"])
        optimizer.load_state_dict(saved["optimizer"])
        state = saved["state"]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    rank, size = world()

    def checkpoint_validation(step: int) -> None:
        result = {"step": step, **validate(backend, layers, arm, reader, held_out_plans, fresh, latents, scaling,
                                           max_document_occurrences, loss_fn)}
        state["history"].append(result)
        if better(result, state["best_validation"]):
            state["best_validation"] = result
            state["best"] = reader_payload(reader, arm, step, seed, recipe)
        if rank == 0:
            print(f"[train {arm} seed={seed}] step {step} answer CE {result['answer_ce']:.4f} "
                  f"fresh macro F1 {result['fresh_macro_f1']:.4f} {json.dumps(result['fresh_f1_per_k'])}", flush=True)

    def save_resume() -> None:
        if rank == 0:
            atomic_save({"seed": seed, "steps": steps, "arm": arm, "reader": reader.state_dict(),
                         "optimizer": optimizer.state_dict(), "state": state}, resume)

    if state["step"] == 0 and not state["history"]:
        checkpoint_validation(0)
    started = time.perf_counter()
    for step in range(state["step"], steps):
        reader.train()
        plans = updates[step]
        optimizer.zero_grad(set_to_none=True)
        tokens, losses = 0, []
        batches = [[p] for p in plans[rank::size]] if size > 1 else microbatches(plans, max_document_occurrences)
        total_tokens = sum(answer_tokens(backend, p) for p in plans)
        for batch in batches:
            loss, count = loss_fn(backend, layers, arm, reader, batch, latents, scaling)
            (loss * (count / total_tokens)).backward()
            losses.append(float(loss.detach()) * count)
            tokens += count
        if size > 1:
            import torch.distributed as dist

            for parameter in reader.parameters():
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                dist.all_reduce(parameter.grad)
        loss_sum, tokens, passes = all_reduce_sum([sum(losses), float(tokens), float(len(batches))], model.device)
        gradient_norm = torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0)
        optimizer.step()
        state["microbatches"] += int(passes)
        state["losses"].append({"step": step + 1, "k": plans[0]["k"], "loss": loss_sum / tokens,
                                "gradient_norm": float(gradient_norm), "microbatches": int(passes),
                                **(reader.diagnostics() if hasattr(reader, "diagnostics") else {})})
        state["step"] = step + 1
        if state["step"] % validation_interval == 0 or state["step"] == steps:
            state["seconds"] += time.perf_counter() - started
            started = time.perf_counter()
            checkpoint_validation(state["step"])
            save_resume()
        elif state["step"] % RESUME_INTERVAL == 0:
            state["seconds"] += time.perf_counter() - started
            started = time.perf_counter()
            save_resume()
    if rank == 0:
        atomic_save(state["best"], output_dir / "best.pt")
        atomic_save(reader_payload(reader, arm, steps, seed, recipe), output_dir / "last.pt")
    return {
        "world_size": size, "arm": arm, "seed": seed, "steps": steps, "requested_steps": requested_steps,
        "recipe": recipe,
        "best_step": state["best"]["selected_step"], "best_validation": state["best_validation"],
        "validation_history": state["history"],
        "loss_by_k": {str(k): _mean([row["loss"] for row in state["losses"] if row["k"] == k]) for k in K_VALUES},
        "loss_tail": state["losses"][-50:],
        "checkpoints": {"best": str(output_dir / "best.pt"), "last": str(output_dir / "last.pt")},
        "data": {"held_out_documents": len(held), "train_rows": len(rows["train"]),
                 "validation_rows": len(rows["validation"]), "train_pool": len(pool["train"]),
                 "validation_pool": len(pool["validation"]), "fresh_questions": len(fresh),
                 "selection_suite_sha256": selection["suite_sha256"]},
        "accounting": {
            "reader_parameters": parameter_count(reader), "updates": steps,
            "backbone_backward_passes": state["microbatches"], "training_questions": steps * EXAMPLES_PER_UPDATE,
            "max_document_occurrences_per_microbatch": max_document_occurrences, "wall_seconds": state["seconds"],
            "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        },
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
