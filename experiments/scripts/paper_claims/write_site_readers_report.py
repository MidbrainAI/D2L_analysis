"""Reduce the write-site reader runs: arm selection on development data and the test-set confirmation.

Subcommands (CPU, reading experiment state directories):

* ``selection --state <selection state>``: the preregistered selection rule. Per arm, ``best`` or
  ``last`` by macro over K of mean F1 on random banks pooled over the development suites (ties to
  ``best``). Arms are ranked by macro over K in {1, 4, 8} of (arm minus reference), with reference
  ``native`` at K=1 and ``normalized_b0.5`` at K=4 and 8; ties go to the arm with fewer reader FLOPs
  per token at K=8. The output records every arm's tables and per-arm compute accounting.
* ``confirm --state <confirmation state>``: the distractor-curve test-base rows (ICL, native,
  normalized) merged with the confirmation reader rows, then reduced with the distractor-curve
  statistics: seed mean, paired source-document bootstrap contrasts with an article-cluster
  sensitivity check, the preregistered reader rule, and answer containment.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from distractor_curves_core import K_VALUES
from distractor_curves_plan import mean_f1
from distractor_curves_report import add_containment, cell, contrast, mean, preregistered_reader_rule, with_seed_mean

ARMS = ("A", "B", "C", "D")
SELECTION_K = (1, 4, 8)
NORMALIZED = "normalized_b0.5"
MODEL_WIDTH, MODEL_INPUT_WIDTH, MODEL_LAYERS = 2304, 9216, 26
SLOTS_PER_DOCUMENT = 8
BACKBONE_FLOPS_PER_TOKEN = 2 * 2.6e9
ACCOUNTING_K = (1, 8, 16)
DISTRACTOR_CURVES_STATE = Path("artifacts/distractor_curves/state-v5")
DISTRACTOR_CURVES_SUITE = Path("inputs/distractor_curves/suite/14b15feae84cd72b/suite.json")
MANIFEST = Path("inputs/qa_protocol/manifest/pinned-v1/manifest.json")
SELECTION_CHOICE = Path("artifacts/write_site_readers/selection-choice.json")


# -- rows ------------------------------------------------------------------------------------
def load_units(state: Path, pattern: str) -> list[dict[str, Any]]:
    return [json.loads(path.read_text()) for path in sorted((state / "units").glob(pattern))]


def merge_rows(entries: Iterable[dict[str, Any]]) -> tuple[list[dict], list[dict], list[dict]]:
    """One row per (question, bank) with every unit's responses and scores merged (as ``distractor_curves_report.load_rows``)."""
    rows: dict[tuple[str, str], dict] = {}
    per_question: dict[str, dict] = {}
    units = []
    for entry in entries:
        unit = entry["unit"]
        units.append(unit)
        for source in entry["result"]["sources"]:
            for row in source["rows"]:
                merged = rows.setdefault((row["query_id"], row["bank_id"]),
                                         {**row, "suite": unit.get("suite"), "responses": {}, "scores": {}})
                merged["responses"].update(row["responses"])
                merged["scores"].update(row["scores"])
            for question in source["per_question"]:
                merged = per_question.setdefault(question["query_id"], {
                    "query_id": question["query_id"], "source_id": source["source_id"], "responses": {}, "scores": {}})
                merged["responses"].update(question["responses"])
                merged["scores"].update(question["scores"])
    return list(rows.values()), list(per_question.values()), units


def arms_in(rows: list[dict]) -> list[str]:
    return sorted({arm for row in rows for arm in row["scores"]})


def table(rows: list[dict], cells: Iterable[tuple[str, int]], *, contains: bool = False) -> dict[str, Any]:
    arms = arms_in(rows)
    out = {}
    for condition, k in cells:
        group = cell(rows, condition, k)
        present = [arm for arm in arms if mean(group, arm) is not None]
        out[f"{condition}:{k}"] = {"questions": len(group), "f1": {arm: mean(group, arm) for arm in present},
                                   "em": {arm: mean(group, arm, "em") for arm in present}}
        if contains:
            out[f"{condition}:{k}"]["contains"] = {arm: mean(group, arm, "contains") for arm in present}
    return out


def random_cells() -> list[tuple[str, int]]:
    return [("random", k) for k in K_VALUES]


# -- selection ---------------------------------------------------------------------------------
def paired_difference(rows: list[dict], a: str, b: str) -> float | None:
    values = [r["scores"][a]["f1"] - r["scores"][b]["f1"] for r in rows if a in r["scores"] and b in r["scores"]]
    return sum(values) / len(values) if values else None


def reference_for(k: int) -> str:
    return "native" if k == 1 else NORMALIZED


def choose_checkpoint(rows: list[dict], arm: str) -> dict[str, Any]:
    """``best`` or ``last`` by macro over K of mean F1 on random banks; ties to ``best``."""
    best, last = mean_f1(rows, f"{arm}_best"), mean_f1(rows, f"{arm}_last")
    return {"checkpoint": "best" if best >= last else "last", "macro_f1": {"best": best, "last": last}}


def arm_score(rows: list[dict], reader: str) -> dict[str, Any]:
    """Macro over K in {1, 4, 8} of the paired (reader minus reference) mean F1 on random banks."""
    per_k = {}
    for k in SELECTION_K:
        group = cell(rows, "random", k)
        per_k[str(k)] = {"reference": reference_for(k), "difference": paired_difference(group, reader, reference_for(k))}
    missing = [k for k, value in per_k.items() if value["difference"] is None]
    if missing:
        raise ValueError(f"{reader}: no paired rows against the reference at K={missing}")
    return {"per_k": per_k, "score": sum(value["difference"] for value in per_k.values()) / len(per_k)}


def select(rows: list[dict], flops_at_k8: dict[str, float], arms: Iterable[str] = ARMS) -> dict[str, Any]:
    """The preregistered selection over merged selection-base and selection-reader rows."""
    results = {}
    for arm in arms:
        if not any(f"{arm}_best" in row["scores"] for row in rows):
            results[arm] = {"trained": False}
            continue
        choice = choose_checkpoint(rows, arm)
        reader = f"{arm}_{choice['checkpoint']}"
        results[arm] = {"trained": True, **choice, "reader": reader, **arm_score(rows, reader),
                        "tie_break_flops_k8": flops_at_k8[arm]}
    ranked = sorted((arm for arm, value in results.items() if value["trained"]),
                    key=lambda arm: (-results[arm]["score"], flops_at_k8[arm], arm))
    if not ranked:
        raise ValueError("no trained arm has selection rows")
    winner = ranked[0]
    return {"arms": results, "ranking": ranked,
            "selected": {"arm": winner, "checkpoint": results[winner]["checkpoint"], "reader": results[winner]["reader"],
                         "score": results[winner]["score"]},
            "reference": {str(k): reference_for(k) for k in SELECTION_K},
            "tie_break": "fewer reader FLOPs per token at K=8"}


def reader_module(arm: str):
    """A CPU instance of the arm's reader at the training recipe's shapes, for accounting only."""
    from write_site_readers_readers import make_reader

    model = SimpleNamespace(config=SimpleNamespace(hidden_size=MODEL_WIDTH, intermediate_size=MODEL_INPUT_WIDTH))
    return make_reader(arm, model, 0, MODEL_LAYERS)  # the rank-effect arm ignores the latent width


def accounting(arm: str, trained_parameters: int | None = None) -> dict[str, Any]:
    from write_site_readers_readers import parameter_count, per_token_flops

    reader = reader_module(arm)
    parameters = parameter_count(reader)
    if trained_parameters is not None and trained_parameters != parameters:
        raise ValueError(f"arm {arm}: trained reader has {trained_parameters} parameters, the recipe shape {parameters}")
    flops = {}
    for k in ACCOUNTING_K:
        value = per_token_flops(reader, SLOTS_PER_DOCUMENT * k, MODEL_LAYERS, MODEL_WIDTH, MODEL_INPUT_WIDTH)
        flops[str(k)] = {**value, "reader_over_backbone": value["reader"] / BACKBONE_FLOPS_PER_TOKEN}
    return {"parameters": parameters, "per_token_flops": flops, "backbone_flops_per_token": BACKBONE_FLOPS_PER_TOKEN,
            "slots_per_document": SLOTS_PER_DOCUMENT, "layers": MODEL_LAYERS, "width": MODEL_WIDTH,
            "input_width": MODEL_INPUT_WIDTH}


def selection_report(state: Path) -> dict[str, Any]:
    rows, _, units = merge_rows(load_units(state, "selection-base-*.json") + load_units(state, "selection-reader-*.json"))
    if not rows:
        raise ValueError(f"no selection rows under {state}")
    trained = {}
    for entry in load_units(state, "train-*-seed*.json"):
        result = entry["result"]
        trained[result["arm"]] = result.get("accounting", {}).get("reader_parameters")
    accounts = {arm: accounting(arm, trained.get(arm)) for arm in ARMS}
    choice = select(rows, {arm: accounts[arm]["per_token_flops"]["8"]["reader"] for arm in ARMS})
    suites = sorted({row["suite"] for row in rows if row["suite"]})
    tables = {suite: table([r for r in rows if r["suite"] == suite], random_cells()) for suite in suites}
    tables["pooled"] = table(rows, random_cells())
    return {**choice, "accounting": accounts, "table": tables,
            "rows": {suite: sum(r["suite"] == suite for r in rows) for suite in suites}, "units": len(units),
            "state": str(state)}


# -- confirmation ------------------------------------------------------------------------------
def contrasts_for(rows: list[dict], seeds: list[str], normalized: str, cells: Iterable[tuple[str, int]],
                  cluster_of: Callable[[dict], str], article_of: Callable[[dict], str] | None = None) -> dict[str, Any]:
    pairs = [("reader_mean", "native"), ("reader_mean", normalized), ("icl", "native"), (normalized, "native")]
    pairs += [(seed, normalized) for seed in seeds] + [(seed, "native") for seed in seeds]
    out = {}
    for condition, k in cells:
        group = cell(rows, condition, k)
        for a, b in pairs:
            result = contrast(group, a, b, cluster_of)
            if result is not None:
                out[f"{condition}:{k}:{a}-{b}"] = result if article_of is None else \
                    {**result, "article_cluster": contrast(group, a, b, article_of)}
    return out


def confirm_report(rows: list[dict], per_question: list[dict], units: list[dict], articles: dict[str, str],
                   answers: dict[str, tuple[str, ...]] | None = None, normalized: str = NORMALIZED) -> dict[str, Any]:
    seeds = sorted({name for unit in units for name in unit["readers"] if name.startswith("reader_s")})
    if not seeds:
        raise ValueError("no confirm-reader seeds in the merged units")
    if answers is not None:
        add_containment(rows, answers)
    with_seed_mean(rows, seeds)
    by_source = lambda r: r["source_id"]  # noqa: E731
    by_article = lambda r: articles[r["source_id"]]  # noqa: E731
    cells = random_cells()
    contrasts = contrasts_for(rows, seeds, normalized, cells, by_source, by_article)
    return {"seeds": seeds, "normalized_arm": normalized, "table": table(rows, cells, contains=answers is not None),
            "contrasts": contrasts,
            "preregistered_reader_rule": preregistered_reader_rule(contrasts, seeds, normalized),
            "rows": len(rows), "questions": len(per_question)}


def confirmation(state: Path, distractor_curves_state: Path, suite_path: Path, manifest_path: Path | None) -> dict[str, Any]:
    base = load_units(distractor_curves_state, "gemma2-test-base-*.json")
    reader = load_units(state, "confirm-reader-*.json")
    if not base or not reader:
        raise ValueError(f"need DISTRACTOR_CURVES test-base units ({len(base)}) and confirm-reader units ({len(reader)})")
    rows, per_question, units = merge_rows(base + reader)
    articles = {d["document_id"]: d["article"] for d in json.loads(suite_path.read_text())["documents"]}
    answers = None
    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_text())
        answers = {row["query_id"]: tuple(row["answers"]) for row in manifest["rows"] if row["family"] == "squad"}
    return {**confirm_report(rows, per_question, units, articles, answers), "distractor_curves_state": str(distractor_curves_state),
            "state": str(state), "base_units": len(base), "reader_units": len(reader)}


# -- CLI --------------------------------------------------------------------------------------------
def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("selection", "confirm"):
        command = sub.add_parser(name)
        command.add_argument("--state", type=Path, required=True, help="experiment state directory")
        command.add_argument("--output", type=Path)
    sub.choices["confirm"].add_argument("--distractor_curves-state", type=Path, default=DISTRACTOR_CURVES_STATE)
    sub.choices["confirm"].add_argument("--suite", type=Path, default=DISTRACTOR_CURVES_SUITE)
    sub.choices["confirm"].add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    if args.command == "selection":
        result = selection_report(args.state)
        write_json(args.output or SELECTION_CHOICE, result)
        print(json.dumps({"selected": result["selected"], "ranking": result["ranking"],
                          "scores": {arm: value.get("score") for arm, value in result["arms"].items()}}, indent=2))
    elif args.command == "confirm":
        result = confirmation(args.state, args.distractor_curves_state, args.suite, args.manifest)
        write_json(args.output or args.state / "confirm-report.json", result)
        print(json.dumps(result["preregistered_reader_rule"], indent=2))


if __name__ == "__main__":
    main()
