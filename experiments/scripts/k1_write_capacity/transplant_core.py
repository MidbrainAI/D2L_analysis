"""Shared setup and paired analysis for the answer-position transplant and mask runners (Table 3)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from official_d2l_k1_representation_ladder import compile_k1_representation_batch
from official_d2l_online_batch_distillation import bank_context, render_prompt


def setup_case(backend, plan, documents):
    compiled, accounting = compile_k1_representation_batch(backend, [plan], documents, 8)
    q = plan["query"]
    student = render_prompt(backend, q["input"], q["dataset"]).to(backend.model.device)
    teacher = render_prompt(backend, q["input"], q["dataset"], bank_context(plan, documents)).to(backend.model.device)
    if not torch.equal(student[-8:], teacher[-8:]):
        raise ValueError("unaligned response anchor")
    return compiled, student, teacher, accounting


def analyze(report, reference="native", seed=184, resamples=4000):
    rows = report["rows"]
    families = sorted({r["dataset"] for r in rows})
    rng = np.random.default_rng(seed)
    conditions = sorted(rows[0]["scores"])
    output = {}
    for condition in conditions:
        by_family = {}
        bootstrap = []
        for family in families:
            group = [r for r in rows if r["dataset"] == family]
            docs = sorted({r["document_id"] for r in group})
            sums, counts = [], []
            for doc in docs:
                selected = [r for r in group if r["document_id"] == doc]
                sums.append(sum(r["scores"][condition]["qa_f1"] - r["scores"][reference]["qa_f1"] for r in selected))
                counts.append(len(selected))
            index = rng.integers(0, len(docs), (resamples, len(docs)))
            bootstrap.append(np.asarray(sums)[index].sum(1) / np.asarray(counts)[index].sum(1))
            gaps = [r for r in group if r["scores"]["teacher"]["qa_f1"] >= .5
                    and r["scores"][reference]["qa_f1"] < .5]
            by_family[family] = {
                "f1": sum(r["scores"][condition]["qa_f1"] for r in group) / len(group),
                "delta": sum(sums) / sum(counts),
                "teacher_correct_native_wrong": len(gaps),
                "rescues": sum(r["scores"][condition]["qa_f1"] >= .5 for r in gaps),
                "regressions": sum(r["scores"][reference]["qa_f1"] >= .5
                                   and r["scores"][condition]["qa_f1"] < .5 for r in group),
            }
        output[condition] = {"by_family": by_family,
                             "macro_f1": sum(v["f1"] for v in by_family.values()) / len(families),
                             "macro_delta": sum(v["delta"] for v in by_family.values()) / len(families),
                             "document_cluster_delta_ci95": np.quantile(np.mean(bootstrap, axis=0), [.025, .975]).tolist()}
    errors = {}
    if "activation_errors" in rows[0]:
        for family in families:
            selected = [r for r in rows if r["dataset"] == family]
            errors[family] = {}
            for key in rows[0]["activation_errors"]:
                denominator = sum(r["activation_errors"][key]["zero_delta_mse"] for r in selected)
                errors[family][key] = {name: sum(r["activation_errors"][key][name] for r in selected) / denominator
                                       for name in ("raw_prediction_mse", "selected_prediction_mse")}
    return {"conditions": output, "activation_error_over_zero_delta": errors,
            "bootstrap": {"seed": seed, "resamples": resamples, "unit": "document", "stratified_by_family": True},
            "limitations": "Repeated validation; confidence intervals are descriptive, with multiple contrasts."}


# ----------------------------------------------------------------------------- pure helpers
def document_span(teacher_prompt: torch.Tensor, student_prompt: torch.Tensor) -> dict:
    """Teacher positions not shared with the student prompt: the passage plus boundary tokens."""
    t, s = teacher_prompt.tolist(), student_prompt.tolist()
    prefix = 0
    while prefix < min(len(t), len(s)) and t[prefix] == s[prefix]:
        prefix += 1
    suffix = 0
    while suffix < min(len(t), len(s)) - prefix and t[-1 - suffix] == s[-1 - suffix]:
        suffix += 1
    start, end = prefix, len(t) - suffix
    if start >= end or suffix < 8:
        raise ValueError("teacher prompt does not embed the student prompt around a document span")
    return {"start": start, "end": end, "prefix": prefix, "suffix": suffix, "document_tokens": end - start,
            "student_unmatched": len(s) - prefix - suffix}


def rest_mapping(span: dict, teacher_length: int, student_length: int) -> tuple[list[int], list[int]]:
    """Teacher/student position pairs for shared (non-document) tokens: common prefix and the aligned tail."""
    prefix = span["prefix"]
    tail = teacher_length - span["end"]
    teacher = list(range(prefix)) + list(range(span["end"], teacher_length))
    student = list(range(prefix)) + list(range(student_length - tail, student_length))
    if len(teacher) != len(student) or min(student) < 0:
        raise ValueError("rest mapping is inconsistent")
    return teacher, student
