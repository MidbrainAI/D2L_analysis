"""Bank construction, answer filtering, summaries, and the sparse forward-KL loss shared by the reader runners."""

from __future__ import annotations

import hashlib
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from post_block_transitions import SEED

K_VALUES = tuple(range(1, 11))


def split_rows(payload: dict) -> dict[str, list[dict]]:
    documents = {document["id"]: document for document in payload["documents"]}
    rows = defaultdict(list)
    for query in payload["queries"]:
        rows[query["split"]].append({"document": documents[query["document_id"]], "query": query})
    for split in rows:
        rows[split].sort(key=lambda row: (row["document"]["id"], row["query"]["id"]))

    by_document = defaultdict(list)
    for row in rows["train"]:
        by_document[row["document"]["id"]].append(row)
    fitting, seen_query = [], []
    for document_rows in by_document.values():
        if len(document_rows) >= 2:
            fitting.extend(document_rows[:-1])
            seen_query.append(document_rows[-1])
        else:
            fitting.extend(document_rows)
    return {
        "train": sorted(fitting, key=lambda row: row["query"]["id"]),
        "seen_query": sorted(seen_query, key=lambda row: row["query"]["id"]),
        "validation": rows["validation"],
        "test": rows["test"],
    }


def stable_seed(*parts: object) -> int:
    value = "\0".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def meaningful_answers(answers: list[str]) -> list[str]:
    return [" ".join(answer.casefold().split()) for answer in answers if len("".join(answer.split())) >= 3]


def contains_answer(text: str, answers: list[str]) -> bool:
    normalized = " ".join(text.casefold().split())
    if any(
        re.search(rf"(?<!\w){re.escape(answer)}(?!\w)", normalized)
        for answer in meaningful_answers(answers)
    ):
        return True
    # DROP frequently has one- or two-digit answers; unlike short words, these
    # can be filtered without making nearly every passage an invalid distractor.
    tokens = {token.strip(".,:;!?()[]{}\"'") for token in normalized.split()}
    return any(answer.strip().isdecimal() and answer.strip() in tokens for answer in answers)


def answer_clean_distractors(
    row: dict,
    document_pool: list[str],
    documents: dict[str, dict],
    required: int,
    *,
    seed: int,
    variant: str,
    label: str,
) -> list[str]:
    """Traverse a deterministic pool permutation and stop once enough clean documents exist."""
    source = row["document"]["id"]
    if source not in document_pool:
        raise ValueError("source document is outside its dataset-local bank pool")
    if required == 0:
        return []
    size = len(document_pool)
    if required > size - 1:
        return []
    rng = random.Random(stable_seed(seed, variant, row["query"]["id"], label))
    start = rng.randrange(size)
    step = rng.randrange(1, size)
    while math.gcd(step, size) != 1:
        step = step % (size - 1) + 1
    clean = []
    for offset in range(size):
        document_id = document_pool[(start + offset * step) % size]
        if document_id != source and not contains_answer(
            documents[document_id]["text"], row["query"]["answers"]
        ):
            clean.append(document_id)
            if len(clean) == required:
                break
    return clean


def expand_nested_banks(
    rows: list[dict],
    document_pool: list[str],
    documents: dict[str, dict],
    *,
    seed: int,
    variant: str,
    k_values: tuple[int, ...] = K_VALUES,
) -> list[dict]:
    """Expand every query across K using one answer-clean nested distractor bank."""
    if not k_values or min(k_values) < 1 or max(k_values) > len(document_pool):
        raise ValueError("bank sizes must fit the split-local document pool")
    plans = []
    for row in rows:
        source = row["document"]["id"]
        if source not in document_pool:
            raise ValueError("source document is outside its bank pool")
        needed = max(k_values)
        clean = answer_clean_distractors(
            row,
            document_pool,
            documents,
            needed,
            seed=seed,
            variant=variant,
            label="nested-bank",
        )
        if len(clean) < needed:
            continue
        distractors = clean[: max(k_values) - 1]
        replacement = clean[max(k_values) - 1]
        for k in k_values:
            bank = [source, *distractors[: k - 1]]
            random.Random(stable_seed(seed, variant, row["query"]["id"], k)).shuffle(bank)
            replaced_bank = [replacement if document_id == source else document_id for document_id in bank]
            plans.append(
                {
                    **row,
                    "k": k,
                    "bank_document_ids": bank,
                    "source_position": bank.index(source),
                    "replacement_document_id": replacement,
                    "replaced_bank_document_ids": replaced_bank,
                    "variant": variant,
                }
            )
    return plans


def summarize_rows(rows: list[dict]) -> dict:
    by_k = defaultdict(list)
    for row in rows:
        by_k[row["k"]].append(row)

    def summarize(group: list[dict]) -> dict:
        result = {
            "rows": len(group),
            "qa_f1": sum(row["score"]["qa_f1"] for row in group) / len(group),
            "exact_match": sum(row["score"]["exact_match"] for row in group) / len(group),
            "rouge_l": sum(row["score"]["rouge_l"] for row in group) / len(group),
        }
        similarities = [row["base_similarity"]["qa_f1"] for row in group if "base_similarity" in row]
        if similarities:
            result["base_response_f1"] = sum(similarities) / len(similarities)
        return result

    per_k = {str(k): summarize(group) for k, group in sorted(by_k.items())}
    metrics = ("qa_f1", "exact_match", "rouge_l")
    macro = {metric: sum(row[metric] for row in per_k.values()) / len(per_k) for metric in metrics}
    macro["rows"] = len(rows)
    if all("base_response_f1" in row for row in per_k.values()):
        macro["base_response_f1"] = sum(row["base_response_f1"] for row in per_k.values()) / len(per_k)
    rng = random.Random(stable_seed("bootstrap", *(row["query_id"] for row in rows)))
    bootstrap = []
    for _ in range(500):
        bootstrap.append(
            sum(sum(rng.choice(group)["score"]["qa_f1"] for _ in group) / len(group) for group in by_k.values())
            / len(by_k)
        )
    bootstrap.sort()
    macro["qa_f1_bootstrap_95"] = [bootstrap[12], bootstrap[487]]
    return {"macro_over_k": macro, "per_k": per_k}


TOP_K = 128
LEARNING_RATE = 1e-4


def assign_balanced_banks(
    rows: list[dict],
    document_pool: list[str],
    documents: dict[str, dict],
    *,
    seed: int = SEED,
    variant: str = "train",
    k_values: tuple[int, ...] = K_VALUES,
) -> list[dict]:
    """Assign one deterministic, balanced K and an answer-clean bank per row."""
    if not k_values or min(k_values) < 1 or max(k_values) > len(document_pool):
        raise ValueError("bank sizes must fit the split-local document pool")
    ordered = list(rows)
    random.Random(stable_seed(seed, variant, "k-order")).shuffle(ordered)
    plans = []
    required = max(k_values) - 1
    for row in ordered:
        source = row["document"]["id"]
        if source not in document_pool:
            raise ValueError("source document is outside its dataset-local bank pool")
        clean = answer_clean_distractors(
            row,
            document_pool,
            documents,
            required,
            seed=seed,
            variant=variant,
            label="bank",
        )
        if len(clean) < required:
            continue
        k = k_values[len(plans) % len(k_values)]
        bank = [source, *clean[: k - 1]]
        rng = random.Random(stable_seed(seed, variant, row["query"]["id"], "bank-order"))
        rng.shuffle(bank)
        plans.append({**row, "k": k, "bank_document_ids": bank, "source_position": bank.index(source), "variant": variant})
    return plans


def build_dataset_local_plans(
    rows: list[dict],
    documents: dict[str, dict],
    *,
    split: str,
    limit: int | None,
    nested: bool = False,
    fixed_k: int | None = None,
    seed: int = SEED,
) -> tuple[list[dict], list[dict]]:
    """Select and bank rows independently so datasets cannot leak into each other."""
    selected, plans_by_dataset = [], []
    datasets = sorted({row["query"]["dataset"] for row in rows})
    for dataset in datasets:
        dataset_rows = stable_source_rows(
            [row for row in rows if row["query"]["dataset"] == dataset],
            limit,
            seed=seed,
        )
        pool = sorted(
            document_id
            for document_id, document in documents.items()
            if document.get("split") == split and document.get("dataset") == dataset
        )
        selected.extend(dataset_rows)
        if fixed_k is not None:
            dataset_plans = expand_nested_banks(
                dataset_rows,
                pool,
                documents,
                seed=seed,
                variant=split,
                k_values=(fixed_k,),
            )
        elif nested:
            dataset_plans = expand_nested_banks(
                dataset_rows,
                pool,
                documents,
                seed=seed,
                variant=split,
                k_values=K_VALUES,
            )
        else:
            dataset_plans = assign_balanced_banks(
                dataset_rows,
                pool,
                documents,
                seed=seed,
                variant=split,
            )
        plans_by_dataset.append(dataset_plans)
    plans = [
        dataset_plans[index]
        for index in range(max(map(len, plans_by_dataset), default=0))
        for dataset_plans in plans_by_dataset
        if index < len(dataset_plans)
    ]
    return selected, plans


def compact_topk(logits: torch.Tensor, top_k: int = TOP_K) -> dict[str, torch.Tensor]:
    """Store a distribution as top probabilities and one exact tail bucket."""
    log_probs = logits.float().log_softmax(-1)
    k = min(top_k, log_probs.shape[-1])
    values, indices = log_probs.topk(k, dim=-1)
    stored_values = values.to(torch.bfloat16)
    top_mass = stored_values.float().exp().sum(-1)
    tail = (1.0 - top_mass).clamp_min(0.0)
    return {
        "indices": indices.to(torch.int32).cpu(),
        "log_probs": stored_values.cpu(),
        "tail_log_prob": tail.clamp_min(torch.finfo(torch.float32).tiny).log().float().cpu(),
    }


def sparse_coarsened_forward_kl(
    target: dict[str, torch.Tensor],
    candidate_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Forward KL between the teacher's top-K-plus-tail coarsening and candidate."""
    indices = target["indices"].to(device=candidate_logits.device, dtype=torch.long)
    teacher_log = target["log_probs"].to(candidate_logits.device).float()
    teacher_tail_log = target["tail_log_prob"].to(candidate_logits.device, dtype=torch.float32)
    if indices.ndim != candidate_logits.ndim or indices.shape[:-1] != candidate_logits.shape[:-1]:
        raise ValueError("teacher top-index dimensions do not match candidate logits")
    if teacher_log.shape != indices.shape or teacher_tail_log.shape != indices.shape[:-1]:
        raise ValueError("teacher compact distribution dimensions are invalid")
    candidate_log = candidate_logits.float().log_softmax(-1)
    selected = candidate_log.gather(-1, indices)
    top = (teacher_log.exp() * (teacher_log - selected)).sum(-1)
    top_mask = torch.zeros_like(candidate_log, dtype=torch.bool).scatter(-1, indices, True)
    candidate_tail = candidate_log.masked_fill(top_mask, float("-inf")).logsumexp(-1)
    teacher_tail = teacher_tail_log.exp()
    tail = teacher_tail * (teacher_tail_log - candidate_tail)
    values = top + tail
    if mask is None:
        return values.mean()
    if mask.shape != values.shape or not mask.any():
        raise ValueError("KL mask must match target positions and select at least one token")
    return values.masked_select(mask.to(values.device, dtype=torch.bool)).mean()


def stable_source_rows(rows: list[dict], limit: int | None, *, seed: int = SEED) -> list[dict]:
    ordered = sorted(
        rows,
        key=lambda row: (stable_seed(seed, "source-row", row["query"]["id"]), row["query"]["id"]),
    )
    return ordered if limit is None else ordered[:limit]
