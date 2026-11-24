"""Reduce the distractor curves (Figure 2, Tables 10--11): per-K tables and paired source-document bootstrap contrasts.

Reads ``<state>/units/*.json`` and ``<state>/expansions/*.json`` from a completed run
state directory. Intervals resample source documents (paired: every arm is scored on the
same questions and banks); article-cluster intervals are a sensitivity check. Answer
containment (a normalized gold answer occurs in the response) is a secondary metric that
does not penalize sentence-style answers the way token F1 does.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from distractor_curves_core import K_VALUES

RESAMPLES = 10_000


def load_rows(state: Path, model: str) -> tuple[list[dict], list[dict], list[dict]]:
    """Merge the test-base and test-reader units into one row per (question, bank)."""
    rows: dict[tuple[str, str], dict] = {}
    per_question: dict[str, dict] = {}
    units = []
    for stage in ("test-base", "test-reader"):
        for path in sorted((state / "units").glob(f"{model}-{stage}-*.json")):
            entry = json.loads(path.read_text())
            units.append(entry["unit"])
            for source in entry["result"]["sources"]:
                for row in source["rows"]:
                    merged = rows.setdefault((row["query_id"], row["bank_id"]),
                                             {**row, "responses": {}, "scores": {}})
                    merged["responses"].update(row["responses"])
                    merged["scores"].update(row["scores"])
                for question in source["per_question"]:
                    merged = per_question.setdefault(question["query_id"], {
                        "query_id": question["query_id"], "source_id": source["source_id"],
                        "responses": {}, "scores": {}})
                    merged["responses"].update(question["responses"])
                    merged["scores"].update(question["scores"])
    return list(rows.values()), list(per_question.values()), units


def selection_of(state: Path, model: str) -> dict:
    selection = {}
    for stage in ("test-base", "test-reader"):
        path = state / "expansions" / f"{model}-{stage}.json"
        if path.exists():
            selection.update(json.loads(path.read_text())["units"][0]["selection"])
    return selection


def with_seed_mean(rows: list[dict], seeds: list[str], suffix: str = "") -> None:
    """Add the mean over presented seeds as arm ``reader_mean``."""
    for row in rows:
        names = [f"{seed}{suffix}" for seed in seeds]
        if all(name in row["scores"] for name in names):
            metrics = set.intersection(*(set(row["scores"][n]) for n in names))
            row["scores"][f"reader_mean{suffix}"] = {
                metric: sum(row["scores"][n][metric] for n in names) / len(names) for metric in metrics}


def contains(response: str, answers) -> float:
    from d2l_scale_core import normalize

    text = " " + " ".join(normalize(response)) + " "
    return float(any(" " + " ".join(normalize(a)) + " " in text for a in answers if normalize(a)))


def add_containment(rows: list[dict], answers: dict[str, tuple[str, ...]]) -> None:
    for row in rows:
        for arm, response in row["responses"].items():
            row["scores"].setdefault(arm, {})["contains"] = contains(response, answers[row["query_id"]])


def cell(rows: list[dict], condition: str, k: int) -> list[dict]:
    return [r for r in rows if r["condition"] == condition and r["k"] == k]


def bootstrap(differences: dict[str, list[float]], seed: int = 0) -> tuple[float, float, float]:
    """Mean over clusters of within-cluster mean differences, with a percentile interval."""
    clusters = [sum(v) / len(v) for v in differences.values()]
    estimate = sum(clusters) / len(clusters)
    rng = random.Random(seed)
    draws = sorted(sum(rng.choice(clusters) for _ in clusters) / len(clusters) for _ in range(RESAMPLES))
    return estimate, draws[int(0.025 * RESAMPLES)], draws[int(0.975 * RESAMPLES) - 1]


def contrast(rows: list[dict], a: str, b: str, cluster_of, metric: str = "f1") -> dict[str, Any] | None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if a in row["scores"] and b in row["scores"]:
            grouped[cluster_of(row)].append(row["scores"][a][metric] - row["scores"][b][metric])
    if not grouped:
        return None
    estimate, low, high = bootstrap(grouped)
    return {"difference": estimate, "low": low, "high": high, "clusters": len(grouped),
            "excludes_zero": low > 0 or high < 0}


def mean(rows: list[dict], arm: str, metric: str = "f1") -> float | None:
    values = [r["scores"][arm][metric] for r in rows if arm in r["scores"]]
    return sum(values) / len(values) if values else None


def report(state: Path, model: str, articles: dict[str, str],
           answers: dict[str, tuple[str, ...]] | None = None) -> dict[str, Any]:
    rows, per_question, units = load_rows(state, model)
    if answers is not None:
        add_containment(rows, answers)
    selection = selection_of(state, model)
    seeds = sorted({name for unit in units for name in unit["readers"] if name.startswith("reader_s")})
    with_seed_mean(rows, seeds)
    beta = selection["beta"]["beta"]
    normalized = f"normalized_b{beta}"
    arms = sorted({a for r in rows for a in r["scores"]})
    by_source = lambda r: r["source_id"]  # noqa: E731
    by_article = lambda r: articles[r["source_id"]]  # noqa: E731
    table = {}
    for k in K_VALUES:
        group = cell(rows, "random", k)
        table[f"random:{k}"] = {
            "questions": len(group),
            "f1": {arm: mean(group, arm) for arm in arms if mean(group, arm) is not None},
            "em": {arm: mean(group, arm, "em") for arm in arms if mean(group, arm) is not None},
            **({"contains": {arm: mean(group, arm, "contains") for arm in arms if mean(group, arm) is not None}}
               if answers is not None else {}),
        }
    contrasts = {}
    pairs = [("reader_mean", "native"), ("reader_mean", normalized), ("icl", "native"), (normalized, "native")]
    pairs += [(seed, normalized) for seed in seeds] + [(seed, "native") for seed in seeds]
    for k in K_VALUES:
        group = cell(rows, "random", k)
        for a, b in pairs:
            result = contrast(group, a, b, by_source)
            if result is not None:
                contrasts[f"random:{k}:{a}-{b}"] = {**result, "article_cluster": contrast(group, a, b, by_article)}
    return {"model": model, "selection": selection, "seeds": seeds, "normalized_arm": normalized,
            "table": table, "contrasts": contrasts,
            "preregistered_reader_rule": preregistered_reader_rule(contrasts, seeds, normalized)}


def preregistered_reader_rule(contrasts: dict[str, Any], seeds: list[str], normalized: str) -> dict[str, Any]:
    """Preregistered: seed-mean interval vs normalized excludes zero (positive) at K=1, 4, 8; seeds agree."""
    checks = {}
    for k in (1, 4, 8):
        reference = "native" if k == 1 else normalized
        entry = contrasts.get(f"random:{k}:reader_mean-{reference}")
        signs = [contrasts.get(f"random:{k}:{seed}-{reference}", {}).get("difference", 0) > 0 for seed in seeds]
        checks[str(k)] = {"reference": reference, "positive_interval": bool(entry and entry["low"] > 0),
                          "seeds_positive": all(signs)}
    return {"checks": checks, "holds": all(c["positive_interval"] and c["seeds_positive"] for c in checks.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--suite", type=Path, default=Path("inputs/distractor_curves/suite/14b15feae84cd72b/suite.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    articles = {d["document_id"]: d["article"] for d in json.loads(args.suite.read_text())["documents"]}
    manifest = json.loads(Path("inputs/qa_protocol/manifest/pinned-v1/manifest.json").read_text())
    answers = {row["query_id"]: tuple(row["answers"]) for row in manifest["rows"] if row["family"] == "squad"}
    result = {model: report(args.state, model, articles, answers) for model in ("gemma2", "qwen4")
              if (args.state / "expansions" / f"{model}-test-base.json").exists()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for model, value in result.items():
        print(model, json.dumps(value["preregistered_reader_rule"]), json.dumps(value["selection"]["beta"]))


if __name__ == "__main__":
    main()
