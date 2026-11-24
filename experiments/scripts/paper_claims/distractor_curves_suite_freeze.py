"""DISTRACTOR_CURVES: freeze an untouched SQuAD-validation suite for the paper's distractor and reader claims.

The PASSAGE_FIT_SUITE suite consists of SQuAD *train* passages, which the released D2L writer was
meta-trained on, so it cannot confirm native-D2L or reader claims. This freeze draws
from SQuAD *validation* paragraphs (the QA_PROTOCOL ``test`` split) instead.

Three stages:

* ``titles`` records the SQuAD validation article title of every question index from
  the pinned dataset revision (the QA_PROTOCOL sources carry no titles).
* ``scan`` records every SQuAD-validation paragraph that any artifact or input
  references: by QA_PROTOCOL document ID, content-hash prefix, ``squad:validation:<n>`` query
  index, or a 10-word passage window. Unlike PASSAGE_FIT_SUITE, bulk sweeps are excluded too, except
  sources that reference nearly every paragraph (raw source copies and whole-split
  sweeps), which are listed with counts. Paragraphs holding any of the first 1,000
  validation questions are excluded because D2L selected its checkpoint on them.
* ``freeze`` partitions articles into dev and test by hash order, then within each
  article takes source documents (at least ``min_questions`` natural questions) and
  same-article distractor documents. Random distractors for a source come from other
  articles of the same partition.

Dev and test share no article. Nothing here loads a model.
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
from passage_fit_suite_suite_freeze import TEXT_SUFFIXES, matching_passages, passage_keys

SCHEMA = "distractor_curves-suite-v1"
DOC_ID = re.compile(rb"qa_protocol-doc-[0-9a-f]{24}")
HEX = re.compile(rb"[0-9a-f]{24,64}")
VALIDATION_INDEX = re.compile(rb"squad:(?:squad:)?validation:(\d+)")
D2L_SELECTION_QUESTIONS = 1000  # upstream definitions.py: validation=split="validation[:1000]"
SQUAD_REVISION = "7b6d24c440a36b6815f21b70d25016731768db1f"
# Sources at or above this fraction of all validation paragraphs are raw copies or whole-split sweeps.
WHOLE_SPLIT_FRACTION = 0.9
SKIP_DIRECTORIES = {"third_party", "distractor_curves"}


def question_index(query_id: str) -> int:
    match = re.fullmatch(r"squad:(?:squad:)?validation:(\d+)", query_id)
    if match is None:
        raise ValueError(f"not a SQuAD validation query ID: {query_id}")
    return int(match.group(1))


def validation_documents(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    documents = [d for d in manifest["documents"] if d["family"] == "squad" and d["split"] == "test"]
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest["rows"]:
        if row["family"] == "squad" and row["split"] == "test":
            rows[row["document_id"]].append(row)
    return documents, {key: sorted(value, key=lambda r: question_index(r["query_id"])) for key, value in rows.items()}


def article_of(rows: list[dict[str, Any]], titles: list[str]) -> str:
    names = {titles[question_index(row["query_id"])] for row in rows}
    if len(names) != 1:
        raise ValueError(f"paragraph spans several articles: {sorted(names)}")
    return names.pop()


def scan_references(roots: list[Path], manifest: dict[str, Any]) -> dict[str, Any]:
    """Map source -> referenced SQuAD-validation paragraphs (ID, hash, question index, or passage text)."""
    documents, rows = validation_documents(manifest)
    known = {d["id"] for d in documents}
    by_hash = {d["content_hash"][:24]: d["id"] for d in documents}
    by_index = {question_index(row["query_id"]): document_id for document_id, group in rows.items() for row in group}
    keys = passage_keys(documents)
    references: dict[str, set[str]] = defaultdict(set)
    for root in roots:
        for directory, subdirectories, files in os.walk(root):
            subdirectories[:] = [name for name in subdirectories if name not in SKIP_DIRECTORIES]
            for name in files:
                if not name.endswith(TEXT_SUFFIXES):
                    continue
                path = Path(directory) / name
                data = path.read_bytes()
                source = f"{root.name}/" + "/".join(path.relative_to(root).parts[:2])
                found = {match.decode() for match in DOC_ID.findall(data)} & known
                found |= {by_hash[h[:24].decode()] for h in set(HEX.findall(data)) if h[:24].decode() in by_hash}
                found |= {by_index[int(n)] for n in VALIDATION_INDEX.findall(data) if int(n) in by_index}
                found |= matching_passages(data.decode("utf-8", "ignore").replace("\\n", " "), keys)
                references[source] |= found
    threshold = int(WHOLE_SPLIT_FRACTION * len(documents))
    whole = {key: len(value) for key, value in references.items() if len(value) >= threshold}
    targeted = {key: sorted(value) for key, value in references.items() if value and key not in whole}
    selection = sorted({by_index[n] for n in range(D2L_SELECTION_QUESTIONS) if n in by_index})
    return {
        "schema": SCHEMA + "-exclusions",
        "manifest_sha256": manifest["manifest_sha256"],
        "validation_documents": len(documents),
        "whole_split_threshold": threshold,
        "whole_split_sources": whole,
        "targeted": targeted,
        "d2l_selection_documents": selection,
        "excluded_documents": sorted(set(selection).union(*map(set, targeted.values()))),
    }


def hash_rank(seed: str, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def freeze_suite(
    manifest: dict[str, Any],
    exclusions: dict[str, Any],
    titles: list[str],
    *,
    seed: str,
    dev_sources: int = 40,
    test_sources: int = 200,
    min_questions: int = 5,
    max_sources_per_article: int = 8,
    min_same_article_distractors: int = 7,
) -> dict[str, Any]:
    """Partition untouched paragraphs into article-disjoint dev/test sources and distractor pools."""
    if exclusions["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("exclusion scan was made against a different manifest")
    documents, rows = validation_documents(manifest)
    excluded = set(exclusions["excluded_documents"])
    by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for document in documents:
        if document["id"] not in excluded:
            by_article[article_of(rows[document["id"]], titles)].append(document)
    # An article supplies sources only if enough untouched paragraphs remain for its same-article bank.
    usable = {}
    for article, members in by_article.items():
        ordered = sorted(members, key=lambda d: hash_rank(seed, d["id"]))
        candidates = [d for d in ordered if len(rows[d["id"]]) >= min_questions]
        count = min(max_sources_per_article, len(candidates), len(ordered) - min_same_article_distractors)
        if count > 0:
            sources = candidates[:count]
            chosen = {d["id"] for d in sources}
            usable[article] = (sources, [d for d in ordered if d["id"] not in chosen])
    articles = sorted(usable, key=lambda a: hash_rank(seed, a))
    partitions: dict[str, list[str]] = {"dev": [], "test": []}
    totals = {"dev": 0, "test": 0}
    targets = {"dev": dev_sources, "test": test_sources}
    for article in articles:
        role = "dev" if totals["dev"] < targets["dev"] else "test"
        if totals[role] >= targets[role]:
            break
        partitions[role].append(article)
        totals[role] += len(usable[article][0])
    if totals["dev"] < dev_sources or totals["test"] < test_sources:
        raise ValueError(f"not enough untouched articles: {totals}")

    def entry(document: dict[str, Any], article: str, role: str) -> dict[str, Any]:
        return {"document_id": document["id"], "content_hash": document["content_hash"], "article": article,
                "role": role, "source_tokens": document["source_tokens"],
                "questions": [row["query_id"] for row in rows[document["id"]]]}

    entries = []
    for partition, members in partitions.items():
        for article in members:
            sources, distractors = usable[article]
            entries += [entry(d, article, f"{partition}_source") for d in sources]
            entries += [entry(d, article, f"{partition}_distractor") for d in distractors]
    # Trim the last article's surplus so each partition has exactly its target number of sources.
    for partition in ("dev", "test"):
        surplus = totals[partition] - targets[partition]
        if surplus:
            last = partitions[partition][-1]
            drop = [e for e in entries if e["article"] == last and e["role"] == f"{partition}_source"][-surplus:]
            for item in drop:
                item["role"] = f"{partition}_distractor"
    body = {
        "schema": SCHEMA,
        "experiment": "DISTRACTOR_CURVES",
        "manifest_sha256": manifest["manifest_sha256"],
        "exclusions_sha256": digest(exclusions),
        "titles_revision": SQUAD_REVISION,
        "seed": seed,
        "eligibility": {"family": "squad", "split": "validation (QA_PROTOCOL test)", "min_natural_questions": min_questions,
                        "max_sources_per_article": max_sources_per_article,
                        "min_same_article_distractors": min_same_article_distractors,
                        "excluded_references": len(excluded), "usable_articles": len(usable)},
        "articles": partitions,
        "documents": entries,
    }
    body["suite_sha256"] = digest(body)
    return body


def summary(suite: dict[str, Any]) -> dict[str, Any]:
    roles = Counter(e["role"] for e in suite["documents"])
    questions = Counter()
    for item in suite["documents"]:
        questions[item["role"]] += len(item["questions"])
    return {"suite_sha256": suite["suite_sha256"], "articles": {k: len(v) for k, v in suite["articles"].items()},
            "documents_by_role": dict(roles), "questions_by_role": dict(questions)}


def read_titles(parquet: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet).to_pylist()
    return {"revision": SQUAD_REVISION, "parquet_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
            "titles": [row["title"] for row in table]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("titles", "scan", "freeze"))
    parser.add_argument("--manifest", type=Path, default=Path("inputs/qa_protocol/manifest/pinned-v1/manifest.json"))
    parser.add_argument("--parquet", type=Path, help="SQuAD validation parquet at the pinned revision (titles stage)")
    parser.add_argument("--titles", type=Path, default=Path("inputs/distractor_curves/squad-validation-titles.json"))
    parser.add_argument("--exclusions", type=Path, default=Path("inputs/distractor_curves/exclusions-v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("inputs/distractor_curves/suite"))
    parser.add_argument("--seed", default="distractor_curves-v1")
    args = parser.parse_args()
    if args.stage == "titles":
        if args.parquet is None:
            raise SystemExit("--parquet is required")
        args.titles.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(args.titles, read_titles(args.parquet))
        return
    manifest = load_frozen_manifest(args.manifest)
    if args.stage == "scan":
        if args.exclusions.exists():
            raise FileExistsError(f"{args.exclusions} exists; exclusion scans are frozen inputs")
        args.exclusions.parent.mkdir(parents=True, exist_ok=True)
        result = scan_references([Path("artifacts"), Path("inputs")], manifest)
        _atomic_json(args.exclusions, result)
        print(json.dumps({"sha256": digest(result), "excluded": len(result["excluded_documents"]),
                          "targeted_sources": len(result["targeted"]),
                          "whole_split_sources": result["whole_split_sources"]}, indent=2))
        return
    titles = json.loads(args.titles.read_text())["titles"]
    suite = freeze_suite(manifest, json.loads(args.exclusions.read_text()), titles, seed=args.seed)
    output = args.output_dir / suite["suite_sha256"][:16]
    output.mkdir(parents=True, exist_ok=True)
    path = output / "suite.json"
    if path.exists() and json.loads(path.read_text())["suite_sha256"] != suite["suite_sha256"]:
        raise ValueError("existing suite differs")
    _atomic_json(path, suite)
    print(json.dumps({**summary(suite), "output": str(path)}, indent=2))


if __name__ == "__main__":
    main()
