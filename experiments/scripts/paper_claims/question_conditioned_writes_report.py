"""Reduce a question-conditioned write run (Table 9): per-K tables and paired source-document intervals against
native, normalized, the post-block reader, and ICL. The baseline rows come from the distractor-curve run, paired
by query and bank, since both runs use the same banks.

Containment (a normalized gold answer occurs in the response) is reported beside token F1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from distractor_curves_core import K_VALUES
from distractor_curves_report import add_containment, contrast, load_rows, mean, with_seed_mean

PRIMARY = "question_conditioned_writes"


def question_conditioned_writes_rows(state: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((state / "units").glob("question_conditioned_writes-test-*.json")):
        for source in json.loads(path.read_text())["result"]["sources"]:
            rows.extend(source["rows"])
    return rows


def merge_v5(rows: list[dict[str, Any]], v5_state: Path) -> tuple[int, list[str]]:
    """Attach v5's ICL and per-seed reader responses/scores to QUESTION_CONDITIONED_WRITES rows with the same (query, bank).

    The seed-mean reader arm is formed after containment is added (``with_seed_mean``)."""
    v5, _, units = load_rows(v5_state, "gemma2")
    seeds = sorted({name for unit in units for name in unit["readers"] if name.startswith("reader_s")})
    table = {(r["query_id"], r["bank_id"]): r for r in v5}
    matched = 0
    for row in rows:
        other = table.get((row["query_id"], row["bank_id"]))
        if other is None:
            continue
        matched += 1
        for arm in ("icl", *seeds):
            if arm in other["responses"]:
                row["scores"][arm] = dict(other["scores"][arm])
                row["responses"][arm] = other["responses"][arm]
    return matched, seeds


def report(state: Path, v5_state: Path, articles: dict[str, str], answers: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    rows = question_conditioned_writes_rows(state)
    matched, seeds = merge_v5(rows, v5_state)
    add_containment(rows, answers)
    with_seed_mean(rows, seeds)
    arms = sorted({a for r in rows for a in r["scores"]})
    table = {}
    for k in K_VALUES:
        group = [r for r in rows if r["condition"] == "random" and r["k"] == k]
        if group:
            table[f"random:{k}"] = {"questions": len(group),
                                    "f1": {a: mean(group, a) for a in arms if mean(group, a) is not None},
                                    "contains": {a: mean(group, a, "contains") for a in arms
                                                 if mean(group, a) is not None}}
    pairs = [(PRIMARY, "native"), (PRIMARY, "normalized_b0.5"), (PRIMARY, "reader_mean"), (PRIMARY, "icl")]
    by_source = lambda r: r["source_id"]  # noqa: E731
    by_article = lambda r: articles[r["source_id"]]  # noqa: E731
    contrasts = {}
    for k in K_VALUES:
        group = [r for r in rows if r["condition"] == "random" and r["k"] == k]
        for a, b in pairs:
            for metric in ("f1", "contains"):
                result = contrast(group, a, b, by_source, metric)
                if result is not None:
                    key = f"random:{k}:{a}-{b}" + ("" if metric == "f1" else ":contains")
                    contrasts[key] = {**result, "article_cluster": contrast(group, a, b, by_article, metric)}
    return {"rows": len(rows), "v5_matched_rows": matched, "table": table, "contrasts": contrasts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--v5-state", type=Path, default=Path("artifacts/distractor_curves/state-v5"))
    parser.add_argument("--suite", type=Path, default=Path("inputs/distractor_curves/suite/14b15feae84cd72b/suite.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    articles = {d["document_id"]: d["article"] for d in json.loads(args.suite.read_text())["documents"]}
    manifest = json.loads(Path("inputs/qa_protocol/manifest/pinned-v1/manifest.json").read_text())
    answers = {row["query_id"]: tuple(row["answers"]) for row in manifest["rows"] if row["family"] == "squad"}
    result = report(args.state, args.v5_state, articles, answers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"rows": result["rows"], "v5_matched": result["v5_matched_rows"]}, indent=2))


if __name__ == "__main__":
    main()
