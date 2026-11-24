"""Train the mixed-K rank-effect post-block reader of the distractor curves from cached compiler latents.

Recipe (fixed before training): READOUT_LADDER's rank-effect post-block reader
(``ActiveMemoryTransition``, bottleneck 32) and gold-answer CE, trained one pass over
READOUT_LADDER's 52,354 SQuAD-train fitting questions, but on nested random banks. Each update
draws one K from {1, 2, 4, 8, 16} (a seeded balanced schedule), takes 12 questions, and
adds K-1 random distractor documents per question from the training split. Microbatches
hold at most ``max_document_occurrences`` document occurrences; gradients accumulate
over them, weighted by answer tokens.

The compiler is frozen, so every training document is compiled once into its normalized
pre-head latent (stage ``latents``). Factors are regenerated with the frozen head, which
replays the compiler up to kernel rounding; a parity gate checks this before training. Validation is
gold CE on 400 SQuAD-train validation-split questions at mixed K; the best and the last
checkpoint are kept, and the fresh-generation choice between them is made on the DISTRACTOR_CURVES
dev split by a later stage.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

from distractor_curves_core import K_VALUES, compile_documents, factors_from_latents, hash_rank, head_bias

EXAMPLES_PER_UPDATE = 12
VALIDATION_QUESTIONS = 400
VALIDATION_INTERVAL = 500
RESUME_INTERVAL = 100
REPLAY_TOLERANCE = 1e-4  # relative to the largest factor entry
TRAIN_QUESTION_LIMIT = 60000  # READOUT_LADDER's limit; yields its 52,354 fitting questions


# -- data (CPU) -------------------------------------------------------------------
def training_rows(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """READOUT_LADDER's fitting rows and the validation rows, SQuAD only."""
    from reader_banks import SEED, stable_source_rows
    from reader_banks import split_rows

    rows = split_rows(payload)
    train = stable_source_rows([r for r in rows["train"] if r["query"]["dataset"] == "squad"], TRAIN_QUESTION_LIMIT,
                               seed=SEED)
    validation = stable_source_rows([r for r in rows["validation"] if r["query"]["dataset"] == "squad"],
                                    VALIDATION_QUESTIONS, seed=SEED)
    return {"train": train, "validation": validation}


def cache_documents(payload: dict[str, Any]) -> list[str]:
    """Every train/validation SQuAD document the reader can see, in a fixed order."""
    return sorted(d["id"] for d in payload["documents"]
                  if d.get("dataset") == "squad" and d.get("split") in ("train", "validation"))


def k_schedule(steps: int, seed: int) -> list[int]:
    """Every consecutive block of len(K_VALUES) updates holds each K once, in a seeded order."""
    rng = random.Random(seed)
    schedule: list[int] = []
    while len(schedule) < steps:
        block = list(K_VALUES)
        rng.shuffle(block)
        schedule += block
    return schedule[:steps]


def bank_for(row: dict[str, Any], k: int, pool: list[str], seed: object) -> list[str]:
    source = row["document"]["id"]
    rng = random.Random(hash_rank(seed, row["query"]["id"], k))
    distractors = []
    while len(distractors) < k - 1:
        candidate = pool[rng.randrange(len(pool))]
        if candidate != source and candidate not in distractors:
            distractors.append(candidate)
    bank = list(distractors)
    bank.insert(rng.randrange(k), source)
    return bank


def update_plans(rows: list[dict[str, Any]], steps: int, seed: int, pool: list[str]) -> list[list[dict[str, Any]]]:
    if len(rows) < steps * EXAMPLES_PER_UPDATE:
        raise ValueError("not enough training questions for the requested updates")
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    updates = []
    for step, k in enumerate(k_schedule(steps, seed)):
        chosen = [rows[i] for i in order[step * EXAMPLES_PER_UPDATE:(step + 1) * EXAMPLES_PER_UPDATE]]
        updates.append([{"query": row["query"], "k": k, "bank_document_ids": bank_for(row, k, pool, seed)}
                        for row in chosen])
    return updates


def validation_plans(rows: list[dict[str, Any]], pool: list[str]) -> list[dict[str, Any]]:
    return [{"query": row["query"], "k": K_VALUES[index % len(K_VALUES)],
             "bank_document_ids": bank_for(row, K_VALUES[index % len(K_VALUES)], pool, "validation")}
            for index, row in enumerate(rows)]


def microbatches(plans: list[dict[str, Any]], max_document_occurrences: int) -> list[list[dict[str, Any]]]:
    size = max(1, max_document_occurrences // plans[0]["k"])
    return [plans[i:i + size] for i in range(0, len(plans), size)]


# -- latent cache (GPU) -----------------------------------------------------------
def build_latent_shard(backend, documents: dict[str, dict[str, Any]], ids: list[str], output: Path) -> dict[str, Any]:
    """Compile one shard of documents; gate on exact head replay for the shard's first document."""
    import torch
    from experiment_plan import GateError

    records = compile_documents(backend, [documents[i]["text"] for i in ids])
    with torch.no_grad():
        replay = factors_from_latents(backend, records[0]["latent"])
    # Replay runs the head at another batch shape, so kernels may differ in the last bits (HEAD_INVERSION: ~3e-7).
    exact = all(torch.equal(replay[m][key], records[0]["generated"][m][key]) for m in replay for key in ("A", "B"))
    relative_error = max(float((replay[m][key] - records[0]["generated"][m][key]).abs().max()
                               / records[0]["generated"][m][key].abs().max()) for m in replay for key in ("A", "B"))
    if relative_error > REPLAY_TOLERANCE:
        raise GateError(f"head replay from cached latents deviates: relative error {relative_error}")
    latents = torch.cat([r["latent"] for r in records]).cpu()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    torch.save({"ids": ids, "latents": latents, "source_tokens": [r["source_tokens"] for r in records]}, temporary)
    temporary.replace(output)
    return {"documents": len(ids), "latent_bytes": latents.numel() * latents.element_size(),
            "head_replay_exact": exact, "head_replay_relative_error": relative_error, "path": str(output)}


def load_latents(directory: Path) -> dict[str, Any]:
    import torch

    table: dict[str, Any] = {}
    for path in sorted(directory.glob("*.pt")):
        shard = torch.load(path, map_location="cpu", weights_only=False)
        for index, document_id in enumerate(shard["ids"]):
            table[document_id] = shard["latents"][index]
    return table


# -- training (GPU) ---------------------------------------------------------------
def representation_batch(backend, plans: list[dict[str, Any]], latents: dict[str, Any]):
    """Native state and per-rank factors for K-homogeneous plans, regenerated from latents."""
    import torch
    from official_d2l_k1_representation_ladder import CompiledRepresentationBatch

    k = plans[0]["k"]
    device = backend.model.device
    stacked = torch.stack([latents[i] for plan in plans for i in plan["bank_document_ids"]]).to(device)
    with torch.no_grad():
        generated = factors_from_latents(backend, stacked)
        bias = head_bias(backend)
        factors = backend.merge(generated, torch.full((len(plans),), k, device=device, dtype=torch.int32),
                                lora_bias=bias, bias_scaler=1.0)
    per_rank = {m: {key: value.view(len(plans), k, *value.shape[1:]).transpose(1, 2)
                    .reshape(len(plans), value.shape[1], k * value.shape[2], value.shape[3])
                    for key, value in part.items()} for m, part in generated.items()}
    expanded_bias = {m: {key: value.unsqueeze(0).expand(len(plans), -1, -1, -1) for key, value in part.items()}
                     for m, part in bias.items()}
    return CompiledRepresentationBatch(
        factors={m: {key: v.detach().clone() for key, v in part.items()} for m, part in factors.items()},
        generated_factors={m: {key: v.detach().clone() for key, v in part.items()} for m, part in per_rank.items()},
        bias_factors=expanded_bias,
        pre_head_latents=torch.zeros(len(plans), 1, device=device),
        document_latents=stacked.view(len(plans), k, *stacked.shape[1:]),
    )


def world() -> tuple[int, int]:
    """(rank, world size) under torch.distributed, else (0, 1)."""
    import torch.distributed as dist

    return (dist.get_rank(), dist.get_world_size()) if dist.is_available() and dist.is_initialized() else (0, 1)


def all_reduce_sum(values: list[float], device) -> list[float]:
    import torch
    import torch.distributed as dist

    if world()[1] == 1:
        return values
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(tensor)
    return tensor.tolist()


def gold_loss(backend, layers, reader, plans, latents, scaling) -> tuple[Any, int]:
    from official_d2l_k1_representation_training import forward_gold_batch, gold_answer_loss, make_gold_batch

    targets = make_gold_batch(backend, plans)
    compiled = representation_batch(backend, plans, latents)
    logits = forward_gold_batch(backend, layers, reader, compiled, "rank-effect", targets, scaling)
    return gold_answer_loss(logits, targets), sum(targets["response_lengths"])


def validate(backend, layers, reader, plans, latents, scaling, max_document_occurrences) -> dict[str, Any]:
    import torch

    reader.eval()
    rank, size = world()
    sums = {k: [0.0, 0.0] for k in K_VALUES}  # summed token loss, tokens
    with torch.no_grad():
        for k in K_VALUES:
            group = [p for p in plans if p["k"] == k]
            # Distributed: each rank scores every size-th question, one per forward pass.
            if size > 1:
                batches = [[p] for p in group[rank::size]]
            else:
                batches = microbatches(group, max_document_occurrences) if group else []
            for batch in batches:
                loss, count = gold_loss(backend, layers, reader, batch, latents, scaling)
                sums[k][0] += float(loss) * count
                sums[k][1] += count
    flat = all_reduce_sum([value for k in K_VALUES for value in sums[k]], backend.model.device)
    per_k = {k: (flat[2 * i], flat[2 * i + 1]) for i, k in enumerate(K_VALUES)}
    total, tokens = sum(v[0] for v in per_k.values()), sum(v[1] for v in per_k.values())
    return {"gold_answer_ce": total / tokens, "per_k": {str(k): v[0] / v[1] for k, v in per_k.items() if v[1]}}


def reader_payload(reader, selected_step: int, seed: int) -> dict[str, Any]:
    return {"transition": {k: v.detach().cpu() for k, v in reader.state_dict().items()},
            "transition_type": "active-memory", "representation": "rank-effect",
            "transition_bottleneck": reader.bottleneck, "memory_width": reader.memory_width,
            "execution_site": reader.execution_site, "selected_step": selected_step, "seed": seed,
            "recipe": "distractor_curves-mixed-k-rank-effect-gold-ce-v1"}


def atomic_save(value: Any, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def train_reader(backend, payload: dict[str, Any], latents: dict[str, Any], output_dir: Path, *, seed: int,
                 steps: int, max_document_occurrences: int, validation_interval: int = VALIDATION_INTERVAL,
                 validation_limit: int | None = None) -> dict[str, Any]:
    """Resumable mixed-K training; writes best.pt and last.pt under ``output_dir``."""
    import torch
    from ctx_to_lora.utils import get_layers
    from official_d2l_k1_representation_ladder import make_representation_transition
    from reader_banks import LEARNING_RATE

    # Every reader-training document is cached in the full plan; the smoke caches a prefix.
    rows = {split: [r for r in values if r["document"]["id"] in latents]
            for split, values in training_rows(payload).items()}
    documents = {d["id"]: d for d in payload["documents"]}
    train_pool = sorted(i for i in latents if documents[i]["split"] == "train")
    validation_pool = sorted(i for i in latents if documents[i]["split"] == "validation")
    updates = update_plans(rows["train"], steps, seed, train_pool)
    held_out = validation_plans(rows["validation"][:validation_limit], validation_pool)
    model = backend.model.base_model
    layers = get_layers(model)
    scaling = model.peft_config["default"].lora_alpha
    torch.manual_seed(seed)
    reader = make_representation_transition("rank-effect", model.config.hidden_size,
                                            backend.model.hypernet.config.latent_size, len(layers), 32).to(model.device)
    optimizer = torch.optim.AdamW(reader.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    resume = output_dir / "resume.pt"
    state = {"step": 0, "history": [], "losses": [], "best": None, "best_ce": float("inf"), "seconds": 0.0,
             "microbatches": 0}
    if resume.exists():
        saved = torch.load(resume, map_location=model.device, weights_only=False)
        if saved["seed"] != seed or saved["steps"] != steps:
            raise ValueError("resume state belongs to a different run")
        reader.load_state_dict(saved["reader"])
        optimizer.load_state_dict(saved["optimizer"])
        state = saved["state"]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    rank, size = world()

    def checkpoint_validation(step: int) -> None:
        result = {"step": step, **validate(backend, layers, reader, held_out, latents, scaling,
                                           max_document_occurrences)}
        state["history"].append(result)
        if result["gold_answer_ce"] < state["best_ce"]:
            state["best_ce"] = result["gold_answer_ce"]
            state["best"] = reader_payload(reader, step, seed)
        if rank == 0:
            print(f"[train seed={seed}] step {step} validation CE {result['gold_answer_ce']:.4f}", flush=True)

    def save_resume() -> None:
        if rank == 0:
            atomic_save({"seed": seed, "steps": steps, "reader": reader.state_dict(),
                         "optimizer": optimizer.state_dict(), "state": state}, resume)

    if state["step"] == 0 and not state["history"]:
        checkpoint_validation(0)
    started = time.perf_counter()
    for step in range(state["step"], steps):
        reader.train()
        plans = updates[step]
        optimizer.zero_grad(set_to_none=True)
        tokens = 0
        losses = []
        # Distributed: each rank takes every size-th question, one per backward pass; summing the
        # token-weighted gradients across ranks gives the same update as one process over all 12.
        batches = [[p] for p in plans[rank::size]] if size > 1 else microbatches(plans, max_document_occurrences)
        total_tokens = sum(len([*_answer_tokens(backend, p)]) for p in plans)
        for batch in batches:
            loss, count = gold_loss(backend, layers, reader, batch, latents, scaling)
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
                                "gradient_norm": float(gradient_norm), "microbatches": int(passes)})
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
        atomic_save(reader_payload(reader, steps, seed), output_dir / "last.pt")
    return {
        "world_size": size,
        "seed": seed,
        "steps": steps,
        "best_step": state["best"]["selected_step"],
        "best_validation_ce": state["best_ce"],
        "validation_history": state["history"],
        "loss_by_k": {str(k): _mean([row["loss"] for row in state["losses"] if row["k"] == k]) for k in K_VALUES},
        "loss_tail": state["losses"][-50:],
        "checkpoints": {"best": str(output_dir / "best.pt"), "last": str(output_dir / "last.pt")},
        "accounting": {
            "reader_parameters": sum(p.numel() for p in reader.parameters()),
            "updates": steps,
            "backbone_backward_passes": state["microbatches"],
            "training_questions": steps * EXAMPLES_PER_UPDATE,
            "max_document_occurrences_per_microbatch": max_document_occurrences,
            "wall_seconds": state["seconds"],
            "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        },
    }


def _answer_tokens(backend, plan: dict[str, Any]) -> list[int]:
    """Answer tokens exactly as make_gold_batch builds them (first nonempty answer plus EOS)."""
    answer = next(value.strip() for value in plan["query"]["answers"] if value.strip())
    ids = backend.tokenizer(answer, add_special_tokens=False)["input_ids"]
    return ids + ([backend.tokenizer.eos_token_id] if backend.tokenizer.eos_token_id is not None else [])


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
