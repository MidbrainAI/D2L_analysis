"""PASSAGE_FIT_SUITE/A3: freeze the never-inspected SQuAD retrieval suite and the A2 document sets.

Two stages:

* ``scan`` records every QA_PROTOCOL document that any prior artifact references, by
  source, so "never inspected" is a frozen input rather than an implicit rescan.
  Bulk sweeps -- sources referencing >= ``BULK_SWEEP_DOCUMENTS`` documents, i.e.
  training corpora and whole-split scoring such as reader training runs,
  COMPILER_FINETUNING/JOINT_FINETUNING training contracts, QA protocol audits and self-study corpora -- are listed with counts but
  not excluded: they trained on or scored documents without selecting or
  inspecting individual ones.  Evaluation suites, development sets, and fixture
  audits (all below ~1,300 documents) are excluded.
* ``freeze`` hash-orders the eligible QA_PROTOCOL-validation SQuAD documents (original
  SQuAD train; >= 5 natural questions; not referenced) under a new seed and
  interleaves length quartiles so every prefix is length-balanced.  The first
  160 are the A2 final documents, the first 200 the A3 suite, and the remainder
  a labeled extension reserve.  A disjoint QA_PROTOCOL-train SQuAD pool supplies the
  A2 development documents.  Exact-402 is kept as a continuity suite, with
  DROP/ROPES split into non-numeric passage-span answers versus the rest.

The QA_PROTOCOL test split is never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from qa_protocol_protocol import _atomic_json, digest, load_frozen_manifest

SCHEMA = "passage_fit_suite-suite-v1"
DOC_ID = re.compile(rb"qa_protocol-doc-[0-9a-f]{24}")
HEX = re.compile(rb"[0-9a-f]{24,64}")
TEXT_SUFFIXES = (".json", ".jsonl", ".log", ".md", ".txt", ".csv")
# A source referencing this many documents is a training corpus or bulk scoring sweep.
BULK_SWEEP_DOCUMENTS = 2000
NUMERIC = re.compile(r"^[\s\d.,%$£€+\-/:]+$")


def squad_documents(manifest: dict[str, Any], split: str) -> list[dict[str, Any]]:
    return [d for d in manifest["documents"] if d["family"] == "squad" and d["split"] == split]


def questions_by_document(manifest: dict[str, Any]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in manifest["rows"]:
        if row["split"] != "test":
            grouped[row["document_id"]].append(row["query_id"])
    return {key: sorted(value) for key, value in grouped.items()}


def _artifact_source(root: Path, path: Path) -> str:
    parts = path.relative_to(root).parts
    return "/".join(parts[:2])


WORD = re.compile(r"[a-z0-9]+")
SHINGLE = 10


def matching_passages(text: str, keys: dict[tuple[str, ...], str]) -> set[str]:
    words = WORD.findall(text.lower())
    return {keys[window] for i in range(len(words) - SHINGLE + 1) if (window := tuple(words[i:i + SHINGLE])) in keys}


def passage_keys(documents: list[dict[str, Any]]) -> dict[tuple[str, ...], str]:
    """Two 10-word windows per passage (start and middle); either occurring marks the passage referenced."""
    keys: dict[tuple[str, ...], str] = {}
    for document in documents:
        words = WORD.findall(document["text"].lower())
        if len(words) >= 2 * SHINGLE:
            for start in (0, len(words) // 2):
                keys[tuple(words[start:start + SHINGLE])] = document["id"]
    return keys


def scan_references(artifacts: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Map artifact source -> referenced QA_PROTOCOL document IDs (by ID, content hash, or passage text)."""
    candidates = [d for d in manifest["documents"] if d["split"] != "test"]
    known = {d["id"] for d in candidates}
    by_hash = {d["content_hash"][:24]: d["id"] for d in candidates}
    keys = passage_keys(candidates)
    references: dict[str, set[str]] = defaultdict(set)
    for directory, _, files in os.walk(artifacts):
        if "third_party" in Path(directory).parts:
            continue
        for name in files:
            if not name.endswith(TEXT_SUFFIXES):
                continue
            path = Path(directory) / name
            data = path.read_bytes()
            source = _artifact_source(artifacts, path)
            found = {match.decode() for match in DOC_ID.findall(data)} & known
            if not found:
                # Pre-QA_PROTOCOL artifacts identify passages by content hash or carry their text.
                found = {by_hash[h[:24].decode()] for h in set(HEX.findall(data)) if h[:24].decode() in by_hash}
                text = data.decode("utf-8", "ignore").replace("\\n", " ")
                found |= matching_passages(text, keys)
            references[source] |= found
    sweeps = {key: len(value) for key, value in references.items() if len(value) >= BULK_SWEEP_DOCUMENTS}
    targeted = {key: sorted(value) for key, value in references.items() if value and key not in sweeps}
    return {
        "schema": SCHEMA + "-exclusions",
        "manifest_sha256": manifest["manifest_sha256"],
        "bulk_sweep_threshold": BULK_SWEEP_DOCUMENTS,
        "targeted": targeted,
        "bulk_sweeps": sweeps,
        "excluded_documents": sorted(set().union(*map(set, targeted.values())) if targeted else set()),
    }


def hash_rank(seed: str, document_id: str) -> str:
    return hashlib.sha256(f"{seed}:{document_id}".encode()).hexdigest()


def quartile_cuts(tokens: list[int]) -> list[int]:
    ordered = sorted(tokens)
    return [ordered[len(ordered) * q // 4] for q in (1, 2, 3)]


def stratum(tokens: int, cuts: list[int]) -> int:
    return sum(tokens >= cut for cut in cuts)


def interleaved_order(documents: list[dict[str, Any]], seed: str, cuts: list[int]) -> list[dict[str, Any]]:
    """Hash order within each length quartile, then round-robin across quartiles."""
    strata: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for document in sorted(documents, key=lambda d: hash_rank(seed, d["id"])):
        strata[stratum(document["source_tokens"], cuts)].append(document)
    ordered, index = [], 0
    while any(index < len(strata[s]) for s in range(4)):
        ordered.extend(strata[s][index] for s in range(4) if index < len(strata[s]))
        index += 1
    return ordered


def answer_is_passage_span(answers: list[str], text: str) -> bool:
    normalized = " ".join(text.lower().split())
    return any(
        not NUMERIC.match(answer) and " ".join(answer.lower().split()) in normalized
        for answer in answers if answer.strip()
    )


def continuity_suite(manifest: dict[str, Any], exact402: dict[str, Any], questions: dict[str, list[str]]) -> dict[str, Any]:
    documents = {d["id"]: d for d in manifest["documents"]}
    rows_by_query = {row["query_id"]: row for row in manifest["rows"]}
    squad_docs = sorted({r["document_id"] for r in exact402["rows"] if r["dataset"] == "squad"})
    used = {r["query_id"] for r in exact402["rows"]}
    secondary = {"drop": {"span": [], "other": []}, "ropes": {"span": [], "other": []}}
    for row in exact402["rows"]:
        if row["dataset"] in secondary:
            answers = list(rows_by_query[row["query_id"]]["answers"])
            kind = "span" if answer_is_passage_span(answers, documents[row["document_id"]]["text"]) else "other"
            secondary[row["dataset"]][kind].append(row["query_id"])
    return {
        "exact402_squad_documents": squad_docs,
        "exact402_squad_questions": sorted(r["query_id"] for r in exact402["rows"] if r["dataset"] == "squad"),
        "exact402_squad_unused_natural_questions": sorted(q for d in squad_docs for q in questions[d] if q not in used),
        "secondary": {family: {kind: sorted(ids) for kind, ids in parts.items()} for family, parts in secondary.items()},
    }


def freeze_suite(
    manifest: dict[str, Any],
    exclusions: dict[str, Any],
    exact402: dict[str, Any],
    *,
    seed: str,
    a2_final: int = 160,
    a3_size: int = 200,
    reserve: int = 200,
    dev_pool: int = 100,
    a2_dev: int = 24,
    min_questions: int = 5,
) -> dict[str, Any]:
    if exclusions["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("exclusion scan was made against a different manifest")
    if not a2_dev <= dev_pool or not a2_final <= a3_size:
        raise ValueError("nested set sizes must be ordered")
    excluded = set(exclusions["excluded_documents"])
    questions = questions_by_document(manifest)

    def eligible(split: str) -> list[dict[str, Any]]:
        return [d for d in squad_documents(manifest, split)
                if d["id"] not in excluded and len(questions.get(d["id"], ())) >= min_questions]

    validation, train = eligible("validation"), eligible("train")
    cuts = quartile_cuts([d["source_tokens"] for d in validation])
    ordered = interleaved_order(validation, seed, cuts)[: a3_size + reserve]
    dev = interleaved_order(train, seed, cuts)[:dev_pool]
    if len(ordered) < a3_size + reserve or len(dev) < dev_pool:
        raise ValueError("not enough eligible documents")

    def entry(document: dict[str, Any], order: int, role: str) -> dict[str, Any]:
        return {"document_id": document["id"], "content_hash": document["content_hash"], "order": order,
                "role": role, "source_tokens": document["source_tokens"],
                "length_stratum": stratum(document["source_tokens"], cuts), "questions": questions[document["id"]]}

    suite_entries = [
        entry(d, i, "a2_final" if i < a2_final else "a3_only" if i < a3_size else "reserve")
        for i, d in enumerate(ordered)
    ]
    dev_entries = [entry(d, i, "a2_dev" if i < a2_dev else "dev_pool") for i, d in enumerate(dev)]
    body = {
        "schema": SCHEMA,
        "experiment": "PASSAGE_FIT_SUITE",
        "manifest_sha256": manifest["manifest_sha256"],
        "exclusions_sha256": digest(exclusions),
        "seed": seed,
        "eligibility": {"family": "squad", "min_natural_questions": min_questions, "excluded_references": len(excluded),
                        "eligible_validation_documents": len(validation), "eligible_train_documents": len(train)},
        "length_quartile_cuts": cuts,
        "sizes": {"a2_final": a2_final, "a3": a3_size, "reserve": reserve, "dev_pool": dev_pool, "a2_dev": a2_dev},
        "suite": suite_entries,
        "development": dev_entries,
        "continuity": continuity_suite(manifest, exact402, questions),
        "test_split_read": False,
    }
    body["suite_sha256"] = digest(body)
    return body


def summary(suite: dict[str, Any]) -> dict[str, Any]:
    roles = Counter(e["role"] for e in suite["suite"] + suite["development"])
    questions = {role: sum(len(e["questions"]) for e in suite["suite"] + suite["development"] if e["role"] == role)
                 for role in roles}
    strata = Counter((e["role"], e["length_stratum"]) for e in suite["suite"])
    secondary = {f: {k: len(v) for k, v in parts.items()} for f, parts in suite["continuity"]["secondary"].items()}
    return {"suite_sha256": suite["suite_sha256"], "documents_by_role": dict(roles), "questions_by_role": questions,
            "length_cuts": suite["length_quartile_cuts"], "strata": {f"{r}:{s}": n for (r, s), n in sorted(strata.items())},
            "secondary": secondary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("scan", "freeze"))
    parser.add_argument("--manifest", type=Path, default=Path("inputs/qa_protocol/manifest/pinned-v1/manifest.json"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--exclusions", type=Path, default=Path("inputs/passage_fit_suite/exclusions-v1.json"))
    parser.add_argument("--exact402", type=Path, default=Path("artifacts/activation_transplant/activation_transplantd-fixed-validation402-v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("inputs/passage_fit_suite/suite"))
    parser.add_argument("--seed", default="passage_fit_suite-a3-v1")
    args = parser.parse_args()
    manifest = load_frozen_manifest(args.manifest)
    if args.stage == "scan":
        if args.exclusions.exists():
            raise FileExistsError(f"{args.exclusions} exists; exclusion scans are frozen inputs")
        args.exclusions.parent.mkdir(parents=True, exist_ok=True)
        result = scan_references(args.artifacts, manifest)
        _atomic_json(args.exclusions, result)
        print(json.dumps({"sha256": digest(result), "excluded": len(result["excluded_documents"]),
                          "targeted_sources": len(result["targeted"]),
                          "bulk_sweeps": result["bulk_sweeps"]}, indent=2))
        return
    exclusions = json.loads(args.exclusions.read_text())
    suite = freeze_suite(manifest, exclusions, json.loads(args.exact402.read_text()), seed=args.seed)
    output = args.output_dir / suite["suite_sha256"][:16]
    output.mkdir(parents=True, exist_ok=True)
    path = output / "suite.json"
    if path.exists() and json.loads(path.read_text())["suite_sha256"] != suite["suite_sha256"]:
        raise ValueError("existing suite differs")
    _atomic_json(path, suite)
    print(json.dumps({**summary(suite), "output": str(path)}, indent=2))


if __name__ == "__main__":
    main()
