"""WRITE_SITE_READERS: freeze the selection, secondary, and confirmation inputs for the deep-rank reader program.

Three frozen files under ``inputs/write_site_readers/``:

* ``selection-v1.json``: the 180 SQuAD *train* contexts that the released D2L writer never
  saw. Upstream ``squad_compact`` groups HF SQuAD train by unique ``context`` in index
  order and trains on ``train[180:]``, so the first 180 contexts are the writer's own
  held-out paragraphs. They are reconstructed from the pinned QA_PROTOCOL SQuAD source rows and
  mapped to QA_PROTOCOL manifest documents. Every document is a source and a distractor
  candidate for the others (no article grouping is available). The file also carries
  the confirmation pointer to the DISTRACTOR_CURVES SQuAD-validation test suite and its bank plan.
* ``secondary-v1.json``: DROP and ROPES ``test`` documents (the HF validation splits,
  D2L's own test sets) minus D2L's selection paragraphs (any question with validation
  index below 900), each marked ``untouched`` from the reference scan, with questions
  capped per document in seed-hash order and DROP answer types attached.
* ``exclusions-v1.json``: the disclosure of prior references to both pools.

The ``scan`` stage walks ``artifacts/`` and ``inputs/`` once and records every source
that references a held-out SQuAD context (by content-hash prefix or 10-word passage
window) or a DROP/ROPES test document (by QA_PROTOCOL ID, passage window, or validation index).
Sources referencing nearly the whole pool are raw copies or whole-split sweeps and are
listed with counts; the rest mark their documents as touched. Nothing here loads a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from qa_protocol_protocol import _atomic_json, digest, load_frozen_manifest
from passage_fit_suite_suite_freeze import TEXT_SUFFIXES, matching_passages, passage_keys

EXPERIMENT = "WRITE_SITE_READERS"
SELECTION_SCHEMA = "write_site_readers-selection-v1"
SECONDARY_SCHEMA = "write_site_readers-secondary-v1"
EXCLUSIONS_SCHEMA = "write_site_readers-exclusions-v1"
HELD_OUT_CONTEXTS = 180  # upstream squad_compact: train[180:] of the context-grouped HF train split
D2L_SELECTION_QUESTIONS = 900  # upstream definitions.py: validation[:900] for DROP and ROPES
SQUAD_TRAIN_ID = re.compile(r"squad:(?:squad:)?train:(\d+)")
SECONDARY_VALIDATION_ID = re.compile(r"(drop|ropes):(?:drop:|ropes:)?validation:(\d+)")
SCAN_VALIDATION_INDEX = re.compile(rb"(drop|ropes):validation:(\d+)")
SECONDARY_FAMILIES = ("drop", "ropes")
QUESTION_CAPS = {"drop": 3, "ropes": 5}
DOC_PREFIX = "qa_protocol-doc-"
# Sources at or above this fraction of a pool are raw copies or whole-split sweeps, not targeted uses.
WHOLE_POOL_FRACTION = 0.9
SKIP_DIRECTORIES = {"third_party", "venv", ".venv", "__pycache__", "write_site_readers"}
DISTRACTOR_CURVES_SUITE = Path("inputs/distractor_curves/suite/14b15feae84cd72b/suite.json")
DISTRACTOR_CURVES_PLAN = Path("inputs/distractor_curves/plans/full-v2.json")
CONFIRMATION_NOTE = "DISTRACTOR_CURVES test: evaluated once before (DISTRACTOR_CURVES readers); never used for selection. Disclosed second look."


def document_id(content_hash: str) -> str:
    return DOC_PREFIX + content_hash[:24]


def hash_rank(seed: str, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def whole_pool_threshold(total: int) -> int:
    return math.ceil(WHOLE_POOL_FRACTION * total)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def train_index(query_id: str) -> int:
    match = SQUAD_TRAIN_ID.fullmatch(query_id)
    if match is None:
        raise ValueError(f"not a SQuAD train query ID: {query_id}")
    return int(match.group(1))


def validation_index(query_id: str) -> int:
    match = SECONDARY_VALIDATION_ID.fullmatch(query_id)
    if match is None:
        raise ValueError(f"not a DROP/ROPES validation query ID: {query_id}")
    return int(match.group(2))


def source_query_id(query_id: str) -> str:
    """Manifest IDs double the family prefix (``drop:drop:validation:3``); source rows do not."""
    return re.sub(r"^(drop|ropes|squad):\1:", r"\1:", query_id)


def nonempty_answers(row: dict[str, Any]) -> None:
    answers = row["answers"]
    values = answers.get("text", answers.get("spans", [])) if isinstance(answers, dict) else answers
    if not any(str(value).strip() for value in values):
        raise ValueError(f"empty answers for {row['query_id']}")


# --- selection suite: the writer's 180 held-out SQuAD train contexts -----------------------------


def held_out_contexts(squad_jsonl: Path, count: int = HELD_OUT_CONTEXTS) -> list[dict[str, Any]]:
    """First ``count`` distinct contexts of HF SQuAD train in index order, with every question index of each."""
    rows = []
    for row in read_jsonl(squad_jsonl):
        match = SQUAD_TRAIN_ID.fullmatch(row["id"])
        if match is not None:
            rows.append((int(match.group(1)), row))
    rows.sort(key=lambda item: item[0])
    held: dict[str, dict[str, Any]] = {}
    for index, row in rows:
        entry = held.get(row["context"])
        if entry is None:
            if len(held) == count:
                continue
            entry = held[row["context"]] = {"document_hash": row["document_id"], "context": row["context"],
                                            "hf_train_index": index, "question_indices": []}
        if entry["document_hash"] != row["document_id"]:
            raise ValueError(f"context at HF index {index} carries two document hashes")
        entry["question_indices"].append(index)
    if len(held) != count:
        raise ValueError(f"only {len(held)} distinct train contexts in {squad_jsonl}")
    return list(held.values())


def selection_documents(manifest: dict[str, Any], held: list[dict[str, Any]]) -> list[dict[str, Any]]:
    documents = {d["id"]: d for d in manifest["documents"]}
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest["rows"]:
        if row["family"] == "squad" and row["split"] != "test":
            rows[row["document_id"]].append(row)
    entries = []
    for item in held:
        identifier = document_id(item["document_hash"])
        document = documents.get(identifier)
        if document is None:
            raise ValueError(f"held-out context at HF index {item['hf_train_index']} is not in the manifest")
        if document["content_hash"] != item["document_hash"]:
            raise ValueError(f"content hash mismatch for {identifier}")
        questions = sorted(rows[identifier], key=lambda r: train_index(r["query_id"]))
        if not questions:
            raise ValueError(f"{identifier} has no manifest questions")
        if [train_index(q["query_id"]) for q in questions] != sorted(item["question_indices"]):
            raise ValueError(f"manifest questions of {identifier} differ from the source rows")
        for row in questions:
            nonempty_answers(row)
        entries.append({"document_id": identifier, "content_hash": document["content_hash"],
                        "role": "selection_source", "source_tokens": document["source_tokens"],
                        "hf_train_index": item["hf_train_index"], "manifest_split": document["split"],
                        "questions": [row["query_id"] for row in questions]})
    return entries


def confirmation_pointer(suite_path: Path, plan_path: Path) -> dict[str, Any]:
    if not suite_path.exists() or not plan_path.exists():
        raise FileNotFoundError(f"confirmation inputs missing: {suite_path}, {plan_path}")
    return {"suite": str(suite_path), "suite_sha256": json.loads(suite_path.read_text())["suite_sha256"],
            "banks_plan": str(plan_path), "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            "note": CONFIRMATION_NOTE}


def freeze_selection(manifest: dict[str, Any], held: list[dict[str, Any]], confirmation: dict[str, Any],
                     *, seed: str) -> dict[str, Any]:
    entries = selection_documents(manifest, held)
    body = {
        "schema": SELECTION_SCHEMA,
        "experiment": EXPERIMENT,
        "manifest_sha256": manifest["manifest_sha256"],
        "seed": seed,
        "eligibility": {"family": "squad", "split": "HF train, first 180 distinct contexts (upstream train[180:] excludes them)",
                        "held_out_contexts": HELD_OUT_CONTEXTS, "dedupe": "exact context string"},
        "distractor_pool": "selection_source",
        "note": "Every document is a source and a distractor candidate for the others; no article grouping is available.",
        "manifest_splits": dict(Counter(e["manifest_split"] for e in entries)),
        "held_out_document_ids": [e["document_id"] for e in entries],
        "held_out_query_ids": [q for e in entries for q in e["questions"]],
        "confirmation": confirmation,
        "documents": entries,
    }
    body["suite_sha256"] = digest(body)
    return body


# --- secondary suite: DROP and ROPES test documents ----------------------------------------------


def secondary_documents(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    documents = [d for d in manifest["documents"] if d["family"] in SECONDARY_FAMILIES and d["split"] == "test"]
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest["rows"]:
        if row["family"] in SECONDARY_FAMILIES and row["split"] == "test":
            rows[row["document_id"]].append(row)
    return documents, {key: sorted(value, key=lambda r: validation_index(r["query_id"])) for key, value in rows.items()}


def d2l_selection_documents(rows: dict[str, list[dict[str, Any]]]) -> set[str]:
    return {identifier for identifier, group in rows.items()
            if any(validation_index(row["query_id"]) < D2L_SELECTION_QUESTIONS for row in group)}


def touched_documents(scan: dict[str, Any], total: int) -> tuple[set[str], dict[str, int], dict[str, int]]:
    """Split scan sources into whole-pool copies and targeted uses; return the targeted documents."""
    threshold = whole_pool_threshold(total)
    sizes = {name: value["docs"] if isinstance(value, dict) else value for name, value in scan.items()}
    whole = {name: len(docs) for name, docs in sizes.items() if len(docs) >= threshold}
    targeted = {name: len(docs) for name, docs in sizes.items() if docs and name not in whole}
    touched = set().union(*(set(sizes[name]) for name in targeted)) if targeted else set()
    return touched, whole, targeted


def capped_questions(seed: str, query_ids: list[str], cap: int) -> list[str]:
    return sorted(query_ids, key=lambda q: hash_rank(seed, q))[:cap]


def drop_answer_types(drop_jsonl: Path, wanted: set[str]) -> dict[str, list[str]]:
    types = {}
    for row in read_jsonl(drop_jsonl):
        if row["id"] in wanted:
            types[row["id"]] = list(row["answers"]["types"])
    missing = wanted - set(types)
    if missing:
        raise ValueError(f"{len(missing)} DROP questions have no source row, e.g. {sorted(missing)[:3]}")
    return types


def answer_type_label(types: list[str]) -> str:
    return "+".join(sorted(set(types))) or "none"


def freeze_secondary(manifest: dict[str, Any], scan: dict[str, Any], answer_types: dict[str, list[str]] | None,
                     *, seed: str, caps: dict[str, int] = QUESTION_CAPS) -> dict[str, Any]:
    documents, rows = secondary_documents(manifest)
    selection = d2l_selection_documents(rows)
    touched, whole, targeted = touched_documents(scan, len(documents))
    entries = []
    for document in documents:
        if document["id"] in selection:
            continue
        group = rows.get(document["id"], [])
        if not group:
            raise ValueError(f"{document['id']} has no manifest questions")
        for row in group:
            nonempty_answers(row)
        family = document["family"]
        questions = capped_questions(seed, [row["query_id"] for row in group], caps[family])
        entry = {"document_id": document["id"], "content_hash": document["content_hash"], "family": family,
                 "role": f"{family}_source", "source_tokens": document["source_tokens"],
                 "untouched": document["id"] not in touched, "questions": questions,
                 "questions_uncapped": len(group)}
        if family == "drop":
            if answer_types is None:
                raise ValueError("DROP answer types are required")
            entry["answer_types"] = {q: answer_types[source_query_id(q)] for q in questions}
            entry["answer_types_uncapped"] = {row["query_id"]: answer_types[source_query_id(row["query_id"])] for row in group}
        entries.append(entry)
    families = {}
    for family in SECONDARY_FAMILIES:
        members = [e for e in entries if e["family"] == family]
        untouched = [e for e in members if e["untouched"]]
        summary = {"documents": len(members), "questions": sum(len(e["questions"]) for e in members),
                   "questions_uncapped": sum(e["questions_uncapped"] for e in members),
                   "untouched_documents": len(untouched),
                   "untouched_questions": sum(len(e["questions"]) for e in untouched),
                   "untouched_questions_uncapped": sum(e["questions_uncapped"] for e in untouched),
                   "d2l_selection_excluded": sum(d["family"] == family for d in documents if d["id"] in selection),
                   "question_cap": caps[family]}
        if family == "drop":
            summary["answer_type_counts"] = dict(Counter(answer_type_label(t) for e in members for t in e["answer_types"].values()))
            summary["answer_type_counts_uncapped"] = dict(Counter(
                answer_type_label(t) for e in members for t in e["answer_types_uncapped"].values()))
        families[family] = summary
    body = {
        "schema": SECONDARY_SCHEMA,
        "experiment": EXPERIMENT,
        "manifest_sha256": manifest["manifest_sha256"],
        "scan_sha256": digest(scan),
        "seed": seed,
        "eligibility": {"families": list(SECONDARY_FAMILIES), "split": "HF validation (QA_PROTOCOL test; D2L's own test sets)",
                        "d2l_selection_questions": D2L_SELECTION_QUESTIONS, "question_caps": dict(caps),
                        "question_order": "sha256(seed:query_id) ascending",
                        "whole_pool_threshold": whole_pool_threshold(len(documents))},
        "distractor_pool": "other included documents of the same family (role <family>_source)",
        "excluded": {"d2l_selection_documents": sorted(selection)},
        "scan_sources": {"whole_pool": whole, "targeted": targeted},
        "families": families,
        "documents": entries,
    }
    body["suite_sha256"] = digest(body)
    return body


def distractor_pool(suite: dict[str, Any], identifier: str) -> list[str]:
    """Same-family included documents other than ``identifier``."""
    family = next(e["family"] for e in suite["documents"] if e["document_id"] == identifier)
    return [e["document_id"] for e in suite["documents"] if e["family"] == family and e["document_id"] != identifier]


# --- exclusions / disclosure ---------------------------------------------------------------------


def freeze_exclusions(manifest: dict[str, Any], held: list[dict[str, Any]], scan180: dict[str, Any],
                      scan_dr: dict[str, Any]) -> dict[str, Any]:
    suffix_to_id = {item["document_hash"][:24]: document_id(item["document_hash"]) for item in held}
    touched180, whole180, targeted180 = touched_documents(scan180, len(held))
    documents, rows = secondary_documents(manifest)
    selection = d2l_selection_documents(rows)
    touched, whole, targeted = touched_documents(scan_dr, len(documents))
    known = {d["id"] for d in documents}
    body = {
        "schema": EXCLUSIONS_SCHEMA,
        "experiment": EXPERIMENT,
        "manifest_sha256": manifest["manifest_sha256"],
        "scan180_sha256": digest(scan180),
        "scan_dr_sha256": digest(scan_dr),
        "selection_prior_references": {
            "held_out_documents": len(held),
            "whole_pool_threshold": whole_pool_threshold(len(held)),
            "whole_pool_sources": whole180,
            "targeted_sources": targeted180,
            "referenced_documents": sorted(suffix_to_id[s] for s in touched180 if s in suffix_to_id),
            "note": "The 180 contexts are the writer's held-out paragraphs and stay in the selection suite; "
                    "targeted prior references are disclosed, not excluded.",
        },
        "drop_ropes": {
            "test_documents": len(documents),
            "whole_pool_threshold": whole_pool_threshold(len(documents)),
            "d2l_selection_documents": sorted(selection),
            "touched_documents": sorted(touched & known),
            "untouched_documents": sorted(known - touched - selection),
            "whole_pool_sources": whole,
            "targeted_sources": targeted,
        },
    }
    body["suite_sha256"] = digest(body)
    return body


# --- scan stage ----------------------------------------------------------------------------------


def scan_references(roots: list[Path], held: list[dict[str, Any]], manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """One walk over ``roots`` producing the scan180 and scan_dr maps (same logic as the WRITE_SITE_READERS scratch scans)."""
    held_keys = passage_keys([{"id": item["document_hash"][:24], "text": item["context"]} for item in held])
    held_hashes = {item["document_hash"][:24].encode(): item["document_hash"][:24] for item in held}
    documents, _ = secondary_documents(manifest)
    dr_keys = passage_keys(documents)
    dr_ids = {d["id"].encode(): d["id"] for d in documents}
    refs180: dict[str, set[str]] = defaultdict(set)
    refs_dr: dict[str, dict[str, set[str]]] = defaultdict(lambda: {"docs": set(), "question_indices": set()})
    for root in roots:
        for directory, subdirectories, files in os.walk(root):
            subdirectories[:] = [name for name in subdirectories if name not in SKIP_DIRECTORIES]
            for name in files:
                if not name.endswith(TEXT_SUFFIXES):
                    continue
                path = Path(directory) / name
                try:
                    data = path.read_bytes()
                except OSError:
                    continue
                text = data.decode("utf-8", "ignore").replace("\\n", " ")
                source = f"{root.name}/" + "/".join(path.relative_to(root).parts[:2])
                found = matching_passages(text, held_keys) | {s for h, s in held_hashes.items() if h in data}
                if found:
                    refs180[source] |= found
                found = matching_passages(text, dr_keys) | {i for b, i in dr_ids.items() if b in data}
                indices = {b":".join(t).decode() for t in SCAN_VALIDATION_INDEX.findall(data)}
                if found or indices:
                    refs_dr[source]["docs"] |= found
                    refs_dr[source]["question_indices"] |= indices
    scan180 = {key: sorted(value) for key, value in refs180.items()}
    scan_dr = {key: {"docs": sorted(value["docs"]), "n_docs": len(value["docs"]),
                     "n_question_indices": len(value["question_indices"]),
                     "question_indices": sorted(value["question_indices"])[:50]} for key, value in refs_dr.items()}
    return scan180, scan_dr


# --- CLI -----------------------------------------------------------------------------------------


def write_frozen(path: Path, body: dict[str, Any]) -> None:
    if path.exists() and json.loads(path.read_text())["suite_sha256"] != body["suite_sha256"]:
        raise ValueError(f"{path} exists with a different suite_sha256; frozen inputs are not overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(path, body)


def summary(selection: dict[str, Any], secondary: dict[str, Any], exclusions: dict[str, Any]) -> dict[str, Any]:
    return {
        "selection": {"suite_sha256": selection["suite_sha256"], "documents": len(selection["documents"]),
                      "questions": len(selection["held_out_query_ids"]), "manifest_splits": selection["manifest_splits"],
                      "targeted_prior_sources": exclusions["selection_prior_references"]["targeted_sources"],
                      "confirmation": selection["confirmation"]["suite_sha256"]},
        "secondary": {"suite_sha256": secondary["suite_sha256"], "families": secondary["families"],
                      "d2l_selection_excluded": len(secondary["excluded"]["d2l_selection_documents"])},
        "exclusions": {"suite_sha256": exclusions["suite_sha256"],
                       "drop_ropes_touched": len(exclusions["drop_ropes"]["touched_documents"]),
                       "drop_ropes_untouched": len(exclusions["drop_ropes"]["untouched_documents"])},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("scan", "freeze"))
    parser.add_argument("--manifest", type=Path, default=Path("inputs/qa_protocol/manifest/pinned-v1/manifest.json"))
    parser.add_argument("--squad", type=Path, default=Path("inputs/qa_protocol/sources/pinned-v1/squad.jsonl"))
    parser.add_argument("--drop", type=Path, default=Path("inputs/qa_protocol/sources/pinned-v1/drop.jsonl"))
    parser.add_argument("--scan180", type=Path, default=Path("inputs/write_site_readers/scans/scan180-v1.json"))
    parser.add_argument("--scan-dr", type=Path, default=Path("inputs/write_site_readers/scans/scan_dr-v1.json"))
    parser.add_argument("--distractor_curves-suite", type=Path, default=DISTRACTOR_CURVES_SUITE)
    parser.add_argument("--distractor_curves-plan", type=Path, default=DISTRACTOR_CURVES_PLAN)
    parser.add_argument("--output-dir", type=Path, default=Path("inputs/write_site_readers"))
    parser.add_argument("--version", default="v1")
    parser.add_argument("--seed", default="write_site_readers-v1")
    args = parser.parse_args()
    manifest = load_frozen_manifest(args.manifest)
    held = held_out_contexts(args.squad)
    if args.stage == "scan":
        for path in (args.scan180, args.scan_dr):
            if path.exists():
                raise FileExistsError(f"{path} exists; reference scans are frozen inputs")
        scan180, scan_dr = scan_references([Path("artifacts"), Path("inputs")], held, manifest)
        args.scan180.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(args.scan180, scan180)
        _atomic_json(args.scan_dr, scan_dr)
        print(json.dumps({"scan180": {k: len(v) for k, v in scan180.items()},
                          "scan_dr": {k: v["n_docs"] for k, v in scan_dr.items()}}, indent=2, sort_keys=True))
        return
    scan180 = json.loads(args.scan180.read_text())
    scan_dr = json.loads(args.scan_dr.read_text())
    exclusions = freeze_exclusions(manifest, held, scan180, scan_dr)
    selection = freeze_selection(manifest, held, confirmation_pointer(args.distractor_curves_suite, args.distractor_curves_plan), seed=args.seed)
    _, rows = secondary_documents(manifest)
    wanted = {source_query_id(row["query_id"]) for group in rows.values() for row in group if row["family"] == "drop"}
    secondary = freeze_secondary(manifest, scan_dr, drop_answer_types(args.drop, wanted), seed=args.seed)
    for name, body in (("exclusions", exclusions), ("selection", selection), ("secondary", secondary)):
        write_frozen(args.output_dir / f"{name}-{args.version}.json", body)
    print(json.dumps({**summary(selection, secondary, exclusions), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
