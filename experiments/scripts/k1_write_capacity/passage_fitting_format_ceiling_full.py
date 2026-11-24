"""PASSAGE_FITTING/A2: per-passage format ceiling with complete, retention-safe supervision.

Runs as a single-unit runner (``run-unit.py``) over the frozen
PASSAGE_FIT_SUITE suite. Units:

* ``retention`` -- 100 natural questions from other (QA_PROTOCOL-train) passages that the
  PK-free base model answers exactly without context; never trained on.
* ``prep`` (per document) -- source-teacher/native/bias-only baselines on the
  natural questions, plus two labeled probe sets with top-128 teacher targets:
  *blind* (sentence-focused factual generation until every sentence has >= 3
  distinct teacher answer spans or the cap is reached) and *oracle* (blind plus
  span-focused questions for natural answers the blind set did not elicit; the
  natural question text is never shown; token F1 to any natural question <= 0.5).
  Only rows whose teacher answer is a passage span are kept.
* ``fit`` (per document, rank, coverage, lr) -- rank-r ``down_proj`` from the D2L
  init (r - 8 new rows: Kaiming A, zero B), full-batch forward KL every step
  (micro-batches accumulated, position-weighted), AdamW wd 0, clip 1.0, cosine
  decay over the schedule length, inline evaluation at every checkpoint.

Development fits feed the preregistered selection rule (``select_final``), which
expands the final units. ``plan`` (CPU) writes the frozen plan from the suite.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from d2l_scale_core import normalize, qa_f1, score_response
from qa_protocol_protocol import digest
from answer_attribution_write_salience_diagnostic import sentence_spans

EXPERIMENT = "PASSAGE_FITTING"
SEED = 195
FAMILY = "squad"
BASE_RANK = 8
FACTUAL_CATEGORIES = ("direct_factual_use", "paraphrase")
SPANS_PER_SENTENCE = 3
MAX_BLIND_QUESTIONS = 192
BLIND_ROUNDS = 6
ORACLE_PER_ANSWER = 2
ORACLE_SAMPLES = 8
MAX_NATURAL_OVERLAP = 0.5
RETENTION_QUESTIONS = 100
AGREEMENT_GATE = 0.95
KL_GATE = 0.02
DEFAULTS = {
    "ranks": [8, 16, 32, 64],
    "learning_rates": [1e-5, 3e-5, 1e-4],
    "checkpoints": [5, 10, 25, 50, 100, 200, 400],
    "schedule_steps": 400,
    "micro_batch": 16,
    "max_new_tokens": 32,
    # Primary natural-question scores use PASSAGE_FORMAT_CEILING's batch-1 cache-free greedy (string-comparable with the transplant reference);
    # fit-agreement and retention diagnostics use batched KV-cached greedy, identically for every unit. On GPU the
    # two modes differ in 3/72 bf16 responses on GPU, so they are not interchangeable.
    "eval_mode": "cache_free",
    "diagnostic_eval_mode": "cached",
    "blind_ranks": [8, 32],
    "wrongdoc_ranks": [8],
    "max_blind_questions": MAX_BLIND_QUESTIONS,
    "blind_rounds": BLIND_ROUNDS,
    "oracle_samples": ORACLE_SAMPLES,
    "samples_per_prompt": 4,
    # Separately labeled leak-tolerant arm: oracle candidates up to this natural-question
    # token F1 cover answers the strict 0.5 bound cannot reach, so near-complete coverage is testable.
    "relaxed_overlap": 0.8,
    "relaxed_ranks": [8, 32],
    "retention_questions": RETENTION_QUESTIONS,
}


# ----------------------------------------------------------------------------- pure helpers (CPU-tested)
def norm_text(value: str) -> str:
    return " ".join(normalize(value))


def is_passage_span(answer: str, passage: str) -> bool:
    key = norm_text(answer)
    return bool(key) and f" {key} " in f" {norm_text(passage)} "


def span_sentence(answer: str, sentences: list[str]) -> int | None:
    key = f" {norm_text(answer)} "
    return next((i for i, s in enumerate(sentences) if key in f" {norm_text(s)} "), None)


def sentence_texts(passage: str) -> list[str]:
    return [passage[start:end].strip() for start, end in sentence_spans(passage)]


def sentences_needing_spans(rows: list[dict], sentences: list[str], focus: list[int]) -> list[int]:
    """Focus sentences that still have fewer than SPANS_PER_SENTENCE distinct teacher answer spans."""
    spans: dict[int, set[str]] = {i: set() for i in focus}
    for row in rows:
        index = span_sentence(row["response_text"], sentences)
        if index in spans:
            spans[index].add(norm_text(row["response_text"]))
    return [i for i in focus if len(spans[i]) < SPANS_PER_SENTENCE]


def elicited(natural_answers: list[str], rows: list[dict]) -> bool:
    gold = {norm_text(a) for a in natural_answers if a}
    return any(norm_text(r["response_text"]) in gold for r in rows if r.get("response_text"))


def is_question(text: str, answer: str | None = None) -> bool:
    """A single-line question of >= 3 words ending in '?' that does not contain the answer it targets."""
    text = text.strip()
    if len(text.split()) < 3 or "\n" in text or not text.endswith("?"):
        return False
    return answer is None or f" {norm_text(answer)} " not in f" {norm_text(text)} "


def oracle_choice(rows: list[dict], answers: list[str]) -> list[dict]:
    """Up to ORACLE_PER_ANSWER span rows, preferring those whose teacher answer is the gold span, then F1 >= 0.5."""
    ranked = sorted(rows, key=lambda r: (not elicited(answers, [r]), -qa_f1(r["response_text"], answers)))
    return [r for r in ranked if qa_f1(r["response_text"], answers) >= 0.5][:ORACLE_PER_ANSWER]


def max_overlap(question: str, natural_questions: list[str]) -> float:
    return max((qa_f1(question, [n]) for n in natural_questions), default=0.0)


def cosine_lr(base: float, step: int, total: int) -> float:
    """LR for 1-based ``step`` of a ``total``-step cosine decay to zero; truncating a run keeps its trajectory."""
    return base * 0.5 * (1.0 + math.cos(math.pi * (step - 1) / total))


def micro_batches(count: int, size: int) -> list[list[int]]:
    return [list(range(start, min(start + size, count))) for start in range(0, count, size)]


def position_weights(lengths: list[list[int]]) -> list[float]:
    """Weight per micro-batch so accumulated position-mean losses equal the full-batch position mean."""
    totals = [sum(group) for group in lengths]
    whole = sum(totals)
    return [t / whole for t in totals]


def select_recipe(dev_results: list[dict], ranks: list[int]) -> dict[str, dict]:
    """Preregistered rule per rank over oracle-coverage dev fits: the smallest (step, lr) whose dev-mean fit
    agreement >= 0.95 and fit KL < 0.02; otherwise the highest mean agreement (ties: lower KL, smaller step, lr),
    flagged as not meeting the gate."""
    selection = {}
    for rank in ranks:
        cells: dict[tuple[int, float], list[dict]] = {}
        for record in dev_results:
            args = record["unit"]["args"]
            if args["rank"] != rank or args["coverage"] != "oracle":
                continue
            for step, evaluation in record["result"]["evaluations"].items():
                if int(step) > 0:
                    cells.setdefault((int(step), args["lr"]), []).append(evaluation)
        if not cells:
            raise ValueError(f"no development results for rank {rank}")
        summary = {key: {"agreement": float(np.mean([e["fit"]["agreement_f1"] for e in values])),
                         "kl": float(np.mean([e["fit"]["kl"] for e in values])), "documents": len(values)}
                   for key, values in cells.items()}
        documents = max(s["documents"] for s in summary.values())
        complete = {key: s for key, s in summary.items() if s["documents"] == documents}
        passing = sorted(key for key, s in complete.items() if s["agreement"] >= AGREEMENT_GATE and s["kl"] < KL_GATE)
        if passing:
            step, lr = passing[0]
            met = True
        else:
            step, lr = min(complete, key=lambda k: (-complete[k]["agreement"], complete[k]["kl"], k[0], k[1]))
            met = False
        selection[str(rank)] = {"lr": lr, "step": step, "gate_met": met, **complete[(step, lr)],
                                "grid": {f"{k[0]}@{k[1]:g}": v for k, v in sorted(complete.items())}}
    return selection


def fit_unit(document_id: str, rank: int, coverage: str, lr: float, *, stage: str, stop_step: int,
             checkpoints: list[int], wrongdoc: str | None = None, priority: int = 0) -> dict:
    unit_id = f"fit.{stage}.{document_id}.r{rank}.{coverage}.lr{lr:g}"
    return {"id": unit_id, "stage": stage, "kind": "fit", "priority": priority, "cost": stop_step / 100.0,
            "args": {"document_id": document_id, "rank": rank, "coverage": coverage, "lr": lr, "stop_step": stop_step,
                     "checkpoints": [c for c in checkpoints if c <= stop_step], "wrongdoc": wrongdoc}}


def build_plan(suite: dict, suite_path: str, manifest_path: str, exclusions_path: str, *, config: dict,
               dev_documents: int, final_documents: int, smoke: bool = False) -> dict:
    """Real plans use ``a2_dev``/``a2_final``. Smoke plans draw both sets from ``dev_pool`` so smokes never inspect
    the frozen A2 documents."""
    if smoke:
        pool = [e["document_id"] for e in suite["development"] if e["role"] == "dev_pool"]
        dev, final = pool[:dev_documents], pool[dev_documents:dev_documents + final_documents]
    else:
        dev = [e["document_id"] for e in suite["development"] if e["role"] == "a2_dev"][:dev_documents]
        final = [e["document_id"] for e in suite["suite"] if e["role"] == "a2_final"][:final_documents]
    units = [{"id": "retention", "stage": "prep", "kind": "retention", "priority": 10, "cost": 0.5, "args": {}}]
    units += [{"id": f"prep.{d}", "stage": "prep", "kind": "prep", "priority": 5 if d in dev else 0, "cost": 0.3,
               "args": {"document_id": d}} for d in dev + final]
    stop = config["schedule_steps"]
    units += [fit_unit(d, rank, "oracle", lr, stage="dev", stop_step=stop, checkpoints=config["checkpoints"])
              for d in dev for rank in config["ranks"] for lr in config["learning_rates"]]
    return {"schema": "unit-plan-v1", "experiment": EXPERIMENT,
            "stages": [{"name": "prep"}, {"name": "dev", "after": ["prep"]},
                       {"name": "final", "after": ["prep", "dev"], "expand": "passage_fitting_format_ceiling_full:select_final"}],
            "config": {**config, "suite_path": suite_path, "suite_sha256": suite["suite_sha256"],
                       "manifest_path": manifest_path, "exclusions_path": exclusions_path,
                       "dev_documents": dev, "final_documents": final, "smoke": smoke},
            "units": units}


def select_final(plan: dict, prior: dict) -> list[dict]:
    """Expand final units from the frozen selection rule applied to the development fits."""
    config = plan["config"]
    selection = select_recipe(list(prior["dev"].values()), config["ranks"])
    stop = max(s["step"] for s in selection.values())
    later = [c for c in config["checkpoints"] if c > stop]
    stop = later[0] if later else stop  # plus one checkpoint beyond the largest selected step
    final = config["final_documents"]
    units = []
    for index, document_id in enumerate(final):
        pair = final[(index + 1) % len(final)] if len(final) > 1 else None
        for rank in config["ranks"]:
            lr = selection[str(rank)]["lr"]
            priority = 2 if rank in (8, 32) else 1
            coverages = (["oracle"] + (["blind"] if rank in config["blind_ranks"] else [])
                         + (["oracle_relaxed"] if rank in config.get("relaxed_ranks", []) else []))
            for coverage in coverages:
                wrongdoc = pair if pair and coverage == "oracle" and rank in config["wrongdoc_ranks"] else None
                unit = fit_unit(document_id, rank, coverage, lr, stage="final", stop_step=stop,
                                checkpoints=config["checkpoints"], wrongdoc=wrongdoc, priority=priority)
                unit["selection"] = selection[str(rank)]
                units.append(unit)
    return units


# ----------------------------------------------------------------------------- model-side runner
def state_dir() -> Path:
    return Path(os.environ.get("EXPERIMENT_STATE_DIR", "artifacts/passage_fitting"))


def setup(plan: dict, worker: str) -> dict:
    import torch
    from d2l_scale import load_config
    from d2l_scale_backend import OfficialBackend
    from qa_protocol_protocol import load_frozen_manifest
    from qa_protocol_runner_common import d2l_documents, d2l_plans

    config = plan["config"]
    manifest = load_frozen_manifest(Path(config["manifest_path"]))
    suite = json.loads(Path(config["suite_path"]).read_text())
    if suite["suite_sha256"] != config["suite_sha256"]:
        raise ValueError("suite digest differs from the plan")
    entries = {e["document_id"]: e for e in suite["suite"] + suite["development"]}
    by_query = {p["query"]["id"]: p for split in ("validation", "train") for p in d2l_plans(manifest, split)}
    root = Path(__file__).resolve().parents[2]
    backend = OfficialBackend(load_config(root, "gemma2_qa_protocol"), root)
    torch.manual_seed(SEED)
    return {"plan": plan, "config": config, "manifest": manifest, "suite": suite, "entries": entries,
            "documents": d2l_documents(manifest), "by_query": by_query, "backend": backend,
            "scale": float(backend.model.peft_config.lora_alpha), "worker": worker}


def teardown(context: dict) -> None:
    context["backend"].close()


def run(context: dict, unit: dict) -> dict:
    handler = {"retention": run_retention, "prep": run_prep, "fit": run_fit_unit}[unit["kind"]]
    return handler(context, unit)


def natural_plans(context: dict, document_id: str) -> list[dict]:
    return [context["by_query"][q] for q in context["entries"][document_id]["questions"]]


def generate(context: dict, prompts: list, down_state, *, counter: dict, kind: str, diagnostic: bool = False) -> list[str]:
    """Greedy PK-free/source responses under an installed state; cached batched or PASSAGE_FORMAT_CEILING cache-free mode.
    ``diagnostic`` selects the diagnostic mode (fit agreement, retention) instead of the primary one."""
    import torch
    from passage_fitting_core import _greedy_batch_ids
    from passage_fitting_core import greedy, installed

    backend, scale, budget = context["backend"], context["scale"], context["config"]["max_new_tokens"]
    device = backend.model.device
    counter[f"{kind}_generations"] = counter.get(f"{kind}_generations", 0) + len(prompts)
    mode = context["config"]["diagnostic_eval_mode" if diagnostic else "eval_mode"]
    if mode == "cache_free":
        return [greedy(backend, p.to(device), down_state, None, scale, budget, counter, kind)[0] for p in prompts]
    texts = []
    batch = context["config"].get("generation_batch", 32)
    with installed(backend, down_state, None, scale), torch.inference_mode():
        for start in range(0, len(prompts), batch):
            for ids in _greedy_batch_ids(backend, prompts[start:start + batch], budget):
                texts.append(backend.tokenizer.decode(ids, skip_special_tokens=True).strip())
    return texts


def score_plans(context: dict, plans: list[dict], texts: list[str]) -> list[dict]:
    return [{"query_id": p["query"]["id"], "response": t, "score": score_response(t, p["query"]["answers"])}
            for p, t in zip(plans, texts, strict=True)]


def question_prompts(context: dict, plans: list[dict], document: dict | None = None) -> list:
    from official_d2l_online_batch_distillation import render_prompt

    text = document["text"] if document else ""
    return [render_prompt(context["backend"], p["query"]["input"], FAMILY, text) for p in plans]


def run_retention(context: dict, unit: dict) -> dict:
    """First RETENTION_QUESTIONS natural questions, in a fixed hash order over QA_PROTOCOL-train SQuAD passages outside the
    suite and the exclusion scan, that the PK-free base model answers with exact match."""
    exclusions = set(json.loads(Path(context["config"]["exclusions_path"]).read_text())["excluded_documents"])
    reserved = set(context["entries"])
    candidates = sorted(
        (p for p in context["by_query"].values()
         if p["query"]["split"] == "train" and p["query"]["dataset"] == FAMILY
         and p["query"]["document_id"] not in exclusions | reserved),
        key=lambda p: digest([SEED, "retention", p["query"]["id"]]))
    counter: dict = {}
    kept, examined = [], 0
    context["backend"].model.reset()
    for start in range(0, len(candidates), 64):
        chunk = candidates[start:start + 64]
        scored = score_plans(context, chunk, generate(context, question_prompts(context, chunk), None,
                                                       counter=counter, kind="base", diagnostic=True))
        examined += len(chunk)
        kept += [s for s in scored if s["score"]["exact_match"] == 1.0]
        if len(kept) >= context["config"]["retention_questions"]:
            break
    kept = kept[:context["config"]["retention_questions"]]
    if len(kept) < context["config"]["retention_questions"]:
        raise RuntimeError("not enough base-correct retention questions")
    return {"questions": [k["query_id"] for k in kept], "examined": examined, "base_responses": kept,
            "replay": counter}


def prompt_text(category: str, focus: str, *, answer: str | None = None) -> str:
    """Probe request. With ``answer`` (oracle), the request states the target span explicitly as the answer, not
    the question; phrasing chosen on a smoke run because the earlier wording made the QA-templated teacher
    output the span itself instead of a question."""
    from passage_fitting_core import _source_prompt

    text = _source_prompt(category) + f' Focus on this part of the source: "{focus}"'
    if answer is not None:
        text += (f' The correct answer to your question must be exactly "{answer}". Write only the question itself,'
                 ' ending with a question mark; do not write the answer.')
    return text


def generate_questions(context: dict, document: dict, requests: list[str], *, sampled: bool, seed: int) -> list[str]:
    from passage_fitting_core import _greedy_batch_ids
    from passage_fitting_core import BATCH, _sample_batch_ids, decode_questions
    from official_d2l_online_batch_distillation import render_prompt

    backend = context["backend"]
    rendered = [render_prompt(backend, r, FAMILY, document["text"]) for r in requests]
    out = []
    backend.model.reset()
    for start in range(0, len(rendered), BATCH):
        chunk = rendered[start:start + BATCH]
        out += (_sample_batch_ids(backend, chunk, context["config"]["max_new_tokens"], seed + start) if sampled
                else _greedy_batch_ids(backend, chunk, context["config"]["max_new_tokens"]))
    return decode_questions(backend, out)


def answer_rows(context: dict, document: dict, questions: list[str], counter: dict, origin: str) -> list[dict]:
    """Teacher answers and targets; keep only rows whose teacher answer is a passage span."""
    from passage_fitting_core import teacher_rows

    rows = [{"question": q, "origin": origin} for q in questions]
    if rows:
        teacher_rows(context["backend"], document["text"], rows, context["config"]["max_new_tokens"], counter)
    return [r for r in rows if is_passage_span(r["response_text"], document["text"])]


def run_prep(context: dict, unit: dict) -> dict:
    import torch
    from passage_fitting_core import bias_only_state, native_factors, seed_int

    document_id = unit["args"]["document_id"]
    document = context["documents"][document_id]
    plans = natural_plans(context, document_id)
    counter: dict = {}
    started = time.perf_counter()
    factors, _, bias, accounting = native_factors(context["backend"], plans[0], context["documents"])
    baseline = {
        "teacher": score_plans(context, plans, generate(context, question_prompts(context, plans, document), None,
                                                        counter=counter, kind="teacher")),
        "native": score_plans(context, plans, generate(context, question_prompts(context, plans), factors,
                                                       counter=counter, kind="native")),
        "bias_only": score_plans(context, plans, generate(context, question_prompts(context, plans),
                                                          bias_only_state(bias), counter=counter, kind="bias")),
    }
    sentences = sentence_texts(document["text"])
    focus = [i for i, s in enumerate(sentences) if len(s.split()) >= 3]
    blind: list[dict] = []
    seen: set[str] = set()
    rounds = 0
    cap = context["config"]["max_blind_questions"]
    for round_index in range(context["config"]["blind_rounds"] + 1):
        needed = focus if round_index == 0 else sentences_needing_spans(blind, sentences, focus)
        if not needed or len(blind) >= cap:
            break
        copies = 1 if round_index == 0 else context["config"]["samples_per_prompt"]
        requests = [prompt_text(c, sentences[i]) for i in needed for c in FACTUAL_CATEGORIES for _ in range(copies)]
        questions = generate_questions(context, document, requests, sampled=round_index > 0,
                                       seed=seed_int(SEED, document_id, round_index))
        fresh = []
        for question in questions:
            key = norm_text(question)
            if is_question(question) and key and key not in seen:
                seen.add(key)
                fresh.append(question)
        blind += answer_rows(context, document, fresh[:cap - len(blind)], counter, "blind")
        rounds = round_index
    natural_questions = [p["query"]["input"] for p in plans]
    unelicited = [p for p in plans if not elicited(list(p["query"]["answers"]), blind)]
    oracle_added: list[dict] = []
    relaxed_added: list[dict] = []
    oracle_attempts: dict[str, dict] = {}
    relaxed_bound = context["config"]["relaxed_overlap"]
    for plan in unelicited:
        answers = list(plan["query"]["answers"])
        answer = answers[0]
        index = span_sentence(answer, sentences)
        focus_text = sentences[index] if index is not None else document["text"]
        request = prompt_text("direct_factual_use", focus_text, answer=answer)
        candidates = generate_questions(context, document, [request], sampled=False, seed=0)
        for sample in range(context["config"]["oracle_samples"]):
            candidates += generate_questions(context, document, [request], sampled=True,
                                             seed=seed_int(SEED, document_id, plan["query"]["id"], sample))
        accepted, overlapping = [], 0
        for question in candidates:
            key = norm_text(question)
            if not (is_question(question, answer) and key and key not in seen):
                continue
            overlap = max_overlap(question, natural_questions)
            if overlap > relaxed_bound:
                overlapping += 1
                continue
            seen.add(key)
            accepted.append(question)
        rows = answer_rows(context, document, accepted, counter, "oracle")
        for row in rows:
            row["natural_overlap"] = max_overlap(row["question"], natural_questions)
        strict = oracle_choice([r for r in rows if r["natural_overlap"] <= MAX_NATURAL_OVERLAP], answers)
        relaxed = strict if elicited(answers, strict) else oracle_choice(rows, answers)
        for row in relaxed:
            if row not in strict:
                row["origin"] = "oracle_relaxed"
        oracle_added += strict
        relaxed_added += relaxed
        oracle_attempts[plan["query"]["id"]] = {
            "candidates": len(candidates), "rejected_overlap_relaxed": overlapping,
            "rejected_overlap_strict": overlapping + sum(r["natural_overlap"] > MAX_NATURAL_OVERLAP for r in rows),
            "accepted": len(accepted), "span_rows": len(rows), "kept": len(strict), "kept_relaxed": len(relaxed),
            "covered": elicited(answers, strict), "covered_relaxed": elicited(answers, relaxed)}
    for row in blind:
        row["natural_overlap"] = max_overlap(row["question"], natural_questions)
    probes = {"document_id": document_id, "blind": blind, "oracle": blind + oracle_added,
              "oracle_relaxed": blind + relaxed_added}
    path = state_dir() / "probes" / f"{document_id}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(probes, temporary)
    temporary.replace(path)

    def stratum(plan: dict) -> str:
        answers = list(plan["query"]["answers"])
        if plan not in unelicited:
            return "blind_elicited"
        if elicited(answers, oracle_added):
            return "oracle_elicited"
        return "relaxed_elicited" if elicited(answers, relaxed_added) else "unelicited"

    strata = {p["query"]["id"]: stratum(p) for p in plans}
    return {"document_id": document_id, "baseline": baseline, "compile_accounting": accounting,
            "blind_rows": len(blind), "oracle_rows": len(blind) + len(oracle_added),
            "oracle_relaxed_rows": len(blind) + len(relaxed_added), "blind_rounds": rounds,
            "sentences": len(focus), "sentences_short_of_spans": len(sentences_needing_spans(blind, sentences, focus)),
            "natural_strata": strata,
            "max_oracle_overlap": max((r["natural_overlap"] for r in oracle_added), default=0.0),
            "max_relaxed_overlap": max((r["natural_overlap"] for r in relaxed_added), default=0.0),
            "oracle_attempts": oracle_attempts,
            "probe_path": str(path.relative_to(state_dir())), "replay": counter,
            "wall_seconds": time.perf_counter() - started}


def fit_metrics(context: dict, trainable, rows: list[dict], micro: int) -> dict:
    """Teacher-forced fit KL (position mean) and greedy agreement F1 with the teacher's own answers."""
    import torch
    from passage_fitting_core import kl_loss, make_batch

    backend, scale = context["backend"], context["scale"]
    device = backend.model.device
    total, positions = 0.0, 0
    with torch.inference_mode():
        for group in micro_batches(len(rows), micro):
            batch = make_batch(rows, group, backend.tokenizer.pad_token_id, device)
            count = int(batch["position_mask"].sum())
            total += float(kl_loss(backend, trainable, batch, scale)) * count
            positions += count
    state = detached_state(trainable)
    texts = generate(context, [r["student_prompt_ids"] for r in rows], state, counter={}, kind="fit", diagnostic=True)
    agreement = [qa_f1(t, [r["response_text"]]) for t, r in zip(texts, rows, strict=True)]
    return {"kl": total / positions, "agreement_f1": float(np.mean(agreement)), "rows": len(rows),
            "agreement_by_row": [round(a, 4) for a in agreement]}


def extended_parity(trainable, reference: dict) -> dict:
    """Rank > 8 step-0 gate: document and bias rows bit-identical to native and every appended
    B row exactly zero, so the installed state equals native in exact arithmetic. Appended rows change the bf16 GEMM
    reduction order, which can flip a near-tie greedy decode (A2 halt: max |dlogit| 0.72 on a long answer), so the
    logit difference is recorded by the step-0 evaluation rather than gated here."""
    import torch

    state = trainable.down_state()["down_proj"]
    ref_a, ref_b = reference["down_proj"]["A"][0], reference["down_proj"]["B"][0]
    a, b = state["A"][0].detach(), state["B"][0].detach()
    shared = ref_a.shape[1] - BASE_RANK
    exact = (torch.equal(a[:, :shared], ref_a[:, :shared].to(a.dtype)) and torch.equal(b[:, :shared], ref_b[:, :shared].to(b.dtype))
             and torch.equal(a[:, -BASE_RANK:], ref_a[:, -BASE_RANK:].to(a.dtype))
             and torch.equal(b[:, -BASE_RANK:], ref_b[:, -BASE_RANK:].to(b.dtype))
             and int(torch.count_nonzero(b[:, shared:-BASE_RANK])) == 0)
    return {"strict": False, "shared_rows_exact": bool(exact), "appended_rows": int(a.shape[1] - ref_a.shape[1])}


def detached_state(trainable) -> dict:
    return {m: {k: v.detach() for k, v in p.items()} for m, p in trainable.down_state().items()}


def evaluate(context: dict, trainable, plans: list[dict], rows: list[dict], retention: list[dict], micro: int,
             counter: dict) -> dict:
    state = detached_state(trainable)
    natural = score_plans(context, plans, generate(context, question_prompts(context, plans), state,
                                                   counter=counter, kind="natural"))
    kept = score_plans(context, retention, generate(context, question_prompts(context, retention), state,
                                                    counter=counter, kind="retention", diagnostic=True))
    return {"natural": natural, "mean_f1": float(np.mean([n["score"]["qa_f1"] for n in natural])),
            "mean_exact": float(np.mean([n["score"]["exact_match"] for n in natural])),
            "retention_exact": float(np.mean([k["score"]["exact_match"] for k in kept])),
            "retention": kept, "fit": fit_metrics(context, trainable, rows, micro)}


def run_fit_unit(context: dict, unit: dict) -> dict:
    import torch
    from passage_fitting_core import (
        Trainable,
        gradient_isolation,
        kl_loss,
        make_batch,
        native_factors,
        seed_int,
        step_zero_parity,
    )
    from experiment_plan import GateError
    from official_d2l_online_batch_distillation import render_prompt

    args, config = unit["args"], context["config"]
    backend, scale = context["backend"], context["scale"]
    device = backend.model.device
    document_id, rank = args["document_id"], args["rank"]
    plans = natural_plans(context, document_id)
    probes = torch.load(state_dir() / "probes" / f"{document_id}.pt", weights_only=False)
    rows = probes[args["coverage"]]
    if not rows:
        raise GateError(f"no {args['coverage']} probe rows for {document_id}")
    retention_ids = json.loads((state_dir() / "units" / "retention.json").read_text())["result"]["questions"]
    retention = [context["by_query"][q] for q in retention_ids]
    reference, doc_rows, bias_rows, accounting = native_factors(backend, plans[0], context["documents"])
    arm = "A" if rank == BASE_RANK else "C"
    generator = torch.Generator().manual_seed(seed_int(SEED, document_id, rank))
    trainable = Trainable(arm, doc_rows, bias_rows, None, generator, device, extra_rows=rank - BASE_RANK)
    prompt = render_prompt(backend, plans[0]["query"]["input"], FAMILY).to(device)
    if rank == BASE_RANK:
        try:
            parity = step_zero_parity(backend, trainable, reference, prompt, scale, strict=True,
                                      budget=config["max_new_tokens"])
        except RuntimeError as error:
            raise GateError(str(error)) from error
    else:
        parity = extended_parity(trainable, reference)
        if not parity["shared_rows_exact"]:
            raise GateError(f"step-zero parity failed for rank {rank}: {parity}")
    optimizer = torch.optim.AdamW(trainable.parameters(), lr=args["lr"], weight_decay=0.0)
    groups = micro_batches(len(rows), config["micro_batch"])
    batches = [make_batch(rows, g, backend.tokenizer.pad_token_id, device) for g in groups]
    weights = position_weights([[int(b["position_mask"].sum())] for b in batches])
    counter: dict = {}
    evaluations = {"0": evaluate(context, trainable, plans, rows, retention, config["micro_batch"], counter)}
    losses, grad_norms, isolation = [], [], None
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for step in range(1, args["stop_step"] + 1):
        for group in optimizer.param_groups:
            group["lr"] = cosine_lr(args["lr"], step, config["schedule_steps"])
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for batch, weight in zip(batches, weights, strict=True):
            loss = kl_loss(backend, trainable, batch, scale) * weight
            loss.backward()
            total += float(loss.detach())
            counter["student_backward_passes"] = counter.get("student_backward_passes", 0) + 1
            counter["student_training_positions"] = counter.get("student_training_positions", 0) + int(batch["mask"].sum())
        if step == 1:
            try:
                isolation = gradient_isolation(backend, trainable)
            except RuntimeError as error:
                raise GateError(str(error)) from error
        grad_norms.append(float(torch.nn.utils.clip_grad_norm_(trainable.parameters(), 1.0)))
        optimizer.step()
        losses.append(total)
        if step in args["checkpoints"]:
            evaluations[str(step)] = evaluate(context, trainable, plans, rows, retention, config["micro_batch"], counter)
    training_seconds = time.perf_counter() - started
    wrongdoc = None
    if args.get("wrongdoc"):
        other = natural_plans(context, args["wrongdoc"])
        texts = generate(context, question_prompts(context, other), detached_state(trainable), counter=counter,
                         kind="wrongdoc")
        scored = score_plans(context, other, texts)
        wrongdoc = {"document_id": args["wrongdoc"], "natural": scored,
                    "mean_f1": float(np.mean([s["score"]["qa_f1"] for s in scored]))}
    return {"document_id": document_id, "rank": rank, "coverage": args["coverage"], "lr": args["lr"],
            "schedule_steps": config["schedule_steps"], "stop_step": args["stop_step"], "parity": parity,
            "gradient_isolation": isolation, "losses": losses, "grad_norms": grad_norms, "evaluations": evaluations,
            "wrongdoc": wrongdoc, "fit_rows": len(rows), "micro_batches": len(batches),
            "trainable_bytes": trainable.trainable_bytes(),
            "installed_down_rank": int(trainable.down_state()["down_proj"]["A"].shape[2]),
            "document_row_bytes_fp32": int(sum(p.numel() for p in trainable.parameters()) * 4),
            "compile_accounting": accounting, "replay": counter, "training_seconds": training_seconds,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None}


def cache_parity(plan: dict, output: Path) -> dict:
    """Cached batched vs PASSAGE_FORMAT_CEILING cache-free greedy on every plan document's natural questions, for native, bias-only,
    and base states. The cached mode is valid for evaluation only if responses are identical."""
    from passage_fitting_core import bias_only_state, native_factors

    context = setup(plan, "cache-parity")
    rows, started = [], time.perf_counter()
    try:
        for document_id in plan["config"]["dev_documents"] + plan["config"]["final_documents"]:
            plans = natural_plans(context, document_id)
            factors, _, bias, _ = native_factors(context["backend"], plans[0], context["documents"])
            for name, state in (("native", factors), ("bias_only", bias_only_state(bias)), ("base", None)):
                responses = {}
                for mode in ("cached", "cache_free"):
                    context["config"]["eval_mode"] = mode  # primary path, switched per mode
                    responses[mode] = generate(context, question_prompts(context, plans), state, counter={}, kind=mode)
                rows += [{"document_id": document_id, "state": name, "query_id": p["query"]["id"], "cached": c,
                          "cache_free": f, "equal": c == f}
                         for p, c, f in zip(plans, responses["cached"], responses["cache_free"], strict=True)]
    finally:
        teardown(context)
    result = {"experiment": EXPERIMENT, "check": "cached_vs_cache_free_greedy", "rows": rows,
              "equal": sum(r["equal"] for r in rows), "total": len(rows), "wall_seconds": time.perf_counter() - started}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


# ----------------------------------------------------------------------------- CLI
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("plan", help="write the frozen unit plan from the PASSAGE_FIT_SUITE suite")
    plan.add_argument("--suite", type=Path, required=True)
    plan.add_argument("--manifest", default="inputs/qa_protocol/manifest/pinned-v1/manifest.json")
    plan.add_argument("--exclusions", default="inputs/passage_fit_suite/exclusions-v1.json")
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--dev-documents", type=int, default=24)
    plan.add_argument("--final-documents", type=int, default=160)
    plan.add_argument("--ranks", type=int, nargs="+", default=DEFAULTS["ranks"])
    plan.add_argument("--learning-rates", type=float, nargs="+", default=DEFAULTS["learning_rates"])
    plan.add_argument("--checkpoints", type=int, nargs="+", default=DEFAULTS["checkpoints"])
    plan.add_argument("--schedule-steps", type=int, default=DEFAULTS["schedule_steps"])
    plan.add_argument("--micro-batch", type=int, default=DEFAULTS["micro_batch"])
    plan.add_argument("--eval-mode", choices=("cached", "cache_free"), default=DEFAULTS["eval_mode"])
    plan.add_argument("--diagnostic-eval-mode", choices=("cached", "cache_free"),
                      default=DEFAULTS["diagnostic_eval_mode"])
    plan.add_argument("--blind-ranks", type=int, nargs="+", default=DEFAULTS["blind_ranks"])
    plan.add_argument("--relaxed-ranks", type=int, nargs="*", default=DEFAULTS["relaxed_ranks"])
    plan.add_argument("--config-json", default="{}", help="extra config overrides (smokes only), e.g. caps")
    plan.add_argument("--smoke", action="store_true", help="draw documents from dev_pool, never the A2 sets")
    parity = sub.add_parser("cache-parity", help="compare cached and cache-free greedy evaluation")
    parity.add_argument("--plan", type=Path, required=True)
    parity.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "cache-parity":
        result = cache_parity(json.loads(args.plan.read_text()), args.output)
        print(json.dumps({k: result[k] for k in ("equal", "total", "wall_seconds")}))
        sys.exit(0 if result["equal"] == result["total"] else 1)
    suite = json.loads(args.suite.read_text())
    config = {**DEFAULTS, "ranks": args.ranks, "learning_rates": args.learning_rates, "checkpoints": args.checkpoints,
              "schedule_steps": args.schedule_steps, "micro_batch": args.micro_batch, "eval_mode": args.eval_mode,
              "diagnostic_eval_mode": args.diagnostic_eval_mode,
              "blind_ranks": [r for r in args.blind_ranks if r in args.ranks],
              "relaxed_ranks": [r for r in args.relaxed_ranks if r in args.ranks],
              "wrongdoc_ranks": [r for r in DEFAULTS["wrongdoc_ranks"] if r in args.ranks],
              **json.loads(args.config_json)}
    if max(config["checkpoints"]) > config["schedule_steps"]:
        raise ValueError("checkpoints must lie inside the schedule")
    result = build_plan(suite, str(args.suite), args.manifest, args.exclusions, config=config,
                        dev_documents=args.dev_documents, final_documents=args.final_documents, smoke=args.smoke)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "units": len(result["units"]),
                      "plan_sha256": digest(result)}, indent=2))


if __name__ == "__main__":
    main()
