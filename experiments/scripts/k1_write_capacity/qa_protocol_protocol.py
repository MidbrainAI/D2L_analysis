"""Immutable QA_PROTOCOL data contracts and stage planning helpers.

This module intentionally has no torch or upstream D2L dependency.  Manifest
freezing can therefore run on a CPU host before any model or GPU run is
started.  The GPU runners use the same digests and row order.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

FAMILIES = ("squad", "drop", "ropes")
SPLITS = ("train", "validation", "test")
MANIFEST_SCHEMA = 1
RECIPE_SCHEMA = 1
EVAL_SUITE_SCHEMA = 1


def digest(value: Any) -> str:
    raw = value if isinstance(value, bytes) else json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def text_hash(value: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", value).strip().encode("utf-8")).hexdigest()


def answer_values(value: Any) -> list[str]:
    if isinstance(value, dict):
        value = value.get("text", value.get("spans", value.get("answers", [])))
    if isinstance(value, str):
        value = [value]
    return [" ".join(str(item).split()) for item in value or [] if "".join(str(item).split())]


def raw_example(row: dict[str, Any], family: str) -> tuple[str, str, list[str]]:
    context = (
        f"{row.get('background', '')}\n{row.get('situation', '')}"
        if family == "ropes" and ("background" in row or "situation" in row)
        else row.get("context", row.get("passage", row.get("document", row.get("text", ""))))
    )
    question = row.get("question", row.get("input", row.get("query", "")))
    answers = answer_values(row.get("answers", row.get("answers_spans", row.get("answer", []))))
    if not isinstance(context, str) or not context.strip():
        raise ValueError("empty context")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("empty question")
    if not answers:
        raise ValueError("empty canonical answer")
    return " ".join(context.split()), " ".join(question.split()), answers


def source_spec(value: str) -> tuple[str, Path]:
    try:
        family, raw_path = value.split("=", 1)
    except ValueError as error:
        raise ValueError("source must be FAMILY=JSONL_PATH") from error
    family = family.lower()
    if family not in FAMILIES:
        raise ValueError(f"unsupported QA_PROTOCOL family: {family}")
    path = Path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return family, path


def load_chunk_audit(path: Path) -> dict[str, dict[str, int]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    entries = value.get("documents", value)
    if not isinstance(entries, dict):
        raise ValueError("chunk audit must be an object or contain a documents object")
    result = {}
    for key, item in entries.items():
        if not isinstance(item, dict) or "chunks" not in item or "source_tokens" not in item:
            raise ValueError(f"chunk audit entry {key!r} lacks chunks/source_tokens")
        result[str(key)] = {"chunks": int(item["chunks"]), "source_tokens": int(item["source_tokens"])}
    return result


def _read_rows(family: str, path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    retained, rejected = [], []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        try:
            context, question, answers = raw_example(raw, family)
            split = str(raw.get("split", "train")).lower()
            if split not in SPLITS:
                raise ValueError(f"unsupported split {split!r}")
            source_id = str(raw.get("id", f"{family}:{line_number}"))
            query_id = f"{family}:{source_id}"
            if query_id in seen_ids:
                raise ValueError("duplicate query id")
            seen_ids.add(query_id)
            document_hash = text_hash(context)
            retained.append({
                "query_id": query_id,
                "document_id": f"qa_protocol-doc-{document_hash[:24]}",
                "document_hash": document_hash,
                "family": family,
                "split": split,
                "question": question,
                "answers": answers,
                "answer_hash": digest(answers),
                "source_file": str(path),
                "source_line": line_number,
                "context": context,
            })
        except (TypeError, ValueError, KeyError) as error:
            rejected.append({"family": family, "source_file": str(path), "source_line": line_number, "reason": str(error)})
    return retained, rejected


def freeze_manifest(
    sources: list[str],
    chunk_audit: Path,
    output: Path,
    *,
    source_revisions: list[str] = (),
    tokenizer_revision: str = "",
    prompt_template_version: str = "",
    min_validation_rows: int = 1,
    min_test_rows: int = 1,
) -> dict[str, Any]:
    """Freeze canonical rows and document-disjoint splits without downloading data."""
    if not sources:
        raise ValueError("at least one --source FAMILY=JSONL_PATH is required")
    if not tokenizer_revision or not prompt_template_version:
        raise ValueError("tokenizer revision and prompt-template version must be pinned")
    revisions = {}
    for value in source_revisions:
        family, revision = value.split("=", 1) if "=" in value else ("", "")
        if family not in FAMILIES or not revision:
            raise ValueError("source revision must be FAMILY=REVISION")
        revisions[family] = revision
    if set(revisions) != {spec.split("=", 1)[0].lower() for spec in sources}:
        raise ValueError("every source family needs one pinned source revision")
    audit = load_chunk_audit(chunk_audit)
    rows, rejected = [], []
    source_digests = []
    families = set()
    for spec in sorted(sources):
        family, path = source_spec(spec)
        families.add(family)
        source_digests.append({"family": family, "path": str(path), "sha256": digest(path.read_bytes())})
        accepted, bad = _read_rows(family, path)
        rows.extend(accepted)
        rejected.extend(bad)
    source_digests.sort(key=lambda item: (item["family"], item["path"]))

    by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_document[row["document_hash"]].append(row)
    for document_hash, document_rows in by_document.items():
        splits = {row["split"] for row in document_rows}
        if len(splits) != 1:
            for row in document_rows:
                rejected.append({**row, "reason": "document appears in multiple splits"})
            continue
        entry = audit.get(document_hash) or audit.get(document_rows[0]["document_id"])
        if entry is None:
            for row in document_rows:
                rejected.append({**row, "reason": "missing one-chunk audit"})
            continue
        if entry["chunks"] != 1:
            for row in document_rows:
                rejected.append({**row, "reason": f"requires one compiler chunk, got {entry['chunks']}"})
            continue
        for row in document_rows:
            row["source_tokens"] = entry["source_tokens"]

    rejected_hashes = {row["document_hash"] for row in rejected if row.get("document_hash")}
    rows = [row for row in rows if row["document_hash"] not in rejected_hashes]
    if families != set(FAMILIES):
        missing = sorted(set(FAMILIES) - families)
        raise ValueError(f"QA_PROTOCOL requires all families; missing {', '.join(missing)}")
    if not rows:
        raise ValueError("no rows survived the QA_PROTOCOL audit")
    counts = {
        family: {split: sum(row["family"] == family and row["split"] == split for row in rows) for split in SPLITS}
        for family in FAMILIES
    }
    for family, family_counts in counts.items():
        if family_counts["validation"] < min_validation_rows:
            raise ValueError(f"{family} has {family_counts['validation']} validation rows; need {min_validation_rows}")
        if family_counts["test"] < min_test_rows:
            raise ValueError(f"{family} has {family_counts['test']} test rows; need {min_test_rows}")
    rows.sort(key=lambda row: (SPLITS.index(row["split"]), row["family"], row["document_hash"], row["query_id"]))
    document_rows = {}
    for row in rows:
        document_rows.setdefault(row["document_hash"], {
            "id": row["document_id"], "content_hash": row["document_hash"], "family": row["family"],
            "split": row["split"], "text": row["context"], "source_tokens": row["source_tokens"],
        })
    documents = sorted(document_rows.values(), key=lambda document: document["content_hash"])
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "experiment": "QA_PROTOCOL",
        "documents": documents,
        "rows": [{key: value for key, value in row.items() if key != "context"} for row in rows],
        "counts": {
            "rows_by_family_split": counts,
            "documents_by_split": dict(Counter(document["split"] for document in documents)),
        },
        "source_digests": source_digests,
        "chunk_audit_sha256": digest(chunk_audit.read_bytes()),
        "source_revisions": revisions,
        "source_revision_hashes": {family: digest(revision) for family, revision in revisions.items()},
        "tokenizer_revision": tokenizer_revision,
        "tokenizer_revision_sha256": digest(tokenizer_revision),
        "prompt_template_version": prompt_template_version,
        "prompt_template_sha256": digest(prompt_template_version),
    }
    manifest["manifest_sha256"] = digest(manifest)
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "manifest.json", manifest)
    _atomic_json(output / "provenance.json", {
        "schema_version": 1, "manifest_sha256": manifest["manifest_sha256"], "source_digests": source_digests,
        "chunk_audit": str(chunk_audit), "plan_order": "split,family,document_hash,query_id",
        "retained_rows": len(rows), "rejected_rows": len(rejected),
    })
    _write_jsonl(output / "source_rows.jsonl", rows)
    _write_jsonl(output / "rejected_rows.jsonl", rejected)
    return manifest


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def load_frozen_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = value.pop("manifest_sha256", None)
    if expected != digest(value):
        raise ValueError("QA_PROTOCOL manifest digest mismatch")
    value["manifest_sha256"] = expected
    if value.get("experiment") != "QA_PROTOCOL" or value.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("not an QA_PROTOCOL manifest")
    rows = value.get("rows", [])
    if len({row["query_id"] for row in rows}) != len(rows):
        raise ValueError("QA_PROTOCOL manifest contains duplicate query IDs")
    for split in SPLITS:
        hashes = {row["document_hash"] for row in rows if row["split"] == split}
        for other in SPLITS:
            if split < other and hashes & {row["document_hash"] for row in rows if row["split"] == other}:
                raise ValueError("QA_PROTOCOL document split overlap")
    return value


def balanced_evaluation_rows(
    manifest: dict[str, Any], split: str, *, per_family: int, seed: int
) -> list[dict[str, Any]]:
    """Select rows deterministically while maximizing document coverage first."""
    if split not in {"validation", "test"}:
        raise ValueError("evaluation suites may only use validation or test rows")
    if per_family < 1:
        raise ValueError("per-family evaluation size must be positive")
    selected = []
    for family in FAMILIES:
        by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows_for_split(manifest, split):
            if row["family"] == family:
                by_document[row["document_hash"]].append(row)
        ordered_documents = sorted(
            by_document,
            key=lambda document_hash: (digest([seed, split, family, document_hash]), document_hash),
        )
        for document_rows in by_document.values():
            document_rows.sort(key=lambda row: (digest([seed, row["query_id"]]), row["query_id"]))
        family_rows = []
        depth = 0
        while len(family_rows) < per_family:
            added = False
            for document_hash in ordered_documents:
                document_rows = by_document[document_hash]
                if depth < len(document_rows):
                    family_rows.append(document_rows[depth])
                    added = True
                    if len(family_rows) == per_family:
                        break
            if not added:
                raise ValueError(f"{family} {split} has fewer than {per_family} eligible rows")
            depth += 1
        selected.extend(family_rows)
    return selected


def rows_for_split(manifest: dict[str, Any], split: str) -> list[dict[str, Any]]:
    return [row for row in manifest["rows"] if row["split"] == split]


def evaluation_identity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    identity_rows = [
        [row["query_id"], row["document_id"], row["document_hash"], row["answer_hash"], row["family"], row["split"]]
        for row in rows
    ]
    return {
        "rows": identity_rows,
        "digest": digest(identity_rows),
        "counts_by_family": dict(Counter(row["family"] for row in rows)),
        "documents_by_family": {
            family: len({row["document_hash"] for row in rows if row["family"] == family})
            for family in FAMILIES
        },
    }


def freeze_evaluation_suite(
    manifest_path: Path,
    output: Path,
    *,
    split: str,
    per_family: int,
    seed: int = 174,
) -> dict[str, Any]:
    manifest = load_frozen_manifest(manifest_path)
    if split not in {"validation", "test"}:
        raise ValueError("evaluation suite split must be validation or test")
    selected_rows = balanced_evaluation_rows(manifest, split, per_family=per_family, seed=seed)
    suite = {
        "schema_version": EVAL_SUITE_SCHEMA,
        "experiment": "QA_PROTOCOL-validation-suite" if split == "validation" else "QA_PROTOCOL-test-suite",
        "manifest_sha256": manifest["manifest_sha256"],
        "seed": seed,
        "algorithm": "seeded_document_hash_round_robin_v1",
        "split": split,
        "target_per_family": per_family,
        **evaluation_identity(selected_rows),
    }
    suite["suite_sha256"] = digest(suite)
    if output.exists():
        if json.loads(output.read_text(encoding="utf-8")) != suite:
            raise ValueError(f"immutable evaluation suite differs: {output}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(output, suite)
    return suite


def load_evaluation_suite(path: Path, manifest: dict[str, Any], *, expected_split: str) -> dict[str, Any]:
    suite = json.loads(path.read_text(encoding="utf-8"))
    expected = suite.pop("suite_sha256", None)
    if expected != digest(suite):
        raise ValueError("QA_PROTOCOL evaluation suite digest mismatch")
    suite["suite_sha256"] = expected
    expected_experiment = "QA_PROTOCOL-validation-suite" if expected_split == "validation" else "QA_PROTOCOL-test-suite"
    if suite.get("schema_version") != EVAL_SUITE_SCHEMA or suite.get("experiment") != expected_experiment:
        raise ValueError("not an QA_PROTOCOL evaluation suite")
    if suite.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("QA_PROTOCOL evaluation suite uses a different manifest")
    manifest_rows = {row["query_id"]: row for row in manifest["rows"]}
    identities = suite.get("rows", [])
    if suite.get("split") != expected_split or not identities:
        raise ValueError("QA_PROTOCOL evaluation suite is empty or uses the wrong split")
    reconstructed = []
    seen = set()
    for identity_row in identities:
        query_id = identity_row[0]
        row = manifest_rows.get(query_id)
        if row is None or evaluation_identity([row])["rows"][0] != identity_row:
            raise ValueError(f"QA_PROTOCOL evaluation suite row differs from manifest: {query_id}")
        if query_id in seen:
            raise ValueError(f"duplicate QA_PROTOCOL evaluation query: {query_id}")
        seen.add(query_id)
        reconstructed.append(row)
    identity = evaluation_identity(reconstructed)
    for key in ("digest", "counts_by_family", "documents_by_family"):
        if suite.get(key) != identity[key]:
            raise ValueError(f"QA_PROTOCOL evaluation suite {key} mismatch")
    return suite


def stage_contract(manifest: dict[str, Any], stage: str, **values: Any) -> dict[str, Any]:
    if stage not in {"baseline", "training", "selection", "test"}:
        raise ValueError(f"unknown QA_PROTOCOL stage: {stage}")
    return {
        "schema_version": RECIPE_SCHEMA, "experiment": f"QA_PROTOCOL-{stage}", "stage": stage,
        "manifest_sha256": manifest["manifest_sha256"], "representation": "rank-effect",
        "one_pass": stage == "training", "final_test_opened": stage == "test", **values,
    }


def validate_resume(saved: dict[str, Any], contract: dict[str, Any]) -> None:
    if saved.get("contract") != contract:
        raise ValueError("QA_PROTOCOL resume contract differs: manifest, row order, or runtime recipe changed")
