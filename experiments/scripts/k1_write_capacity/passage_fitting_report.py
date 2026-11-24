"""PASSAGE_FITTING report: reduce a experiment state directory into one summary JSON plus a printed table.

Document-mean natural-question F1 per (stage, rank, coverage, lr, checkpoint), paired document-cluster bootstrap
of the difference to native, gap closure toward the source teacher, F1 by natural-question stratum (blind-elicited,
oracle-elicited, unelicited), retention exact match against base (1.0 by construction) and native (step 0),
wrong-document F1 against the paired document's bias-only F1, probe coverage, selection, and unit costs.
``--exclude`` drops documents (e.g. smoke-inspected ones) for a sensitivity reading.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from passage_fitting_core import bootstrap_delta, gap_closure


def mean(values) -> float | None:
    values = list(values)
    return float(np.mean(values)) if values else None


def f1s(rows: list[dict]) -> list[float]:
    return [r["score"]["qa_f1"] for r in rows]


def load_units(state: Path) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted((state / "units").glob("*.json"))}


def coverage_summary(preps: dict[str, dict]) -> dict:
    strata = defaultdict(int)
    for prep in preps.values():
        for value in prep["natural_strata"].values():
            strata[value] += 1
    total = sum(strata.values())
    attempts = [a for p in preps.values() for a in p.get("oracle_attempts", {}).values()]
    return {
        "documents": len(preps),
        "blind_rows_mean": mean(p["blind_rows"] for p in preps.values()),
        "blind_rows_min_max": [min(p["blind_rows"] for p in preps.values()), max(p["blind_rows"] for p in preps.values())],
        "oracle_rows_mean": mean(p["oracle_rows"] for p in preps.values()),
        "sentences_short_of_spans_fraction": mean(p["sentences_short_of_spans"] / max(p["sentences"], 1)
                                                  for p in preps.values()),
        "natural_strata": {k: {"questions": v, "fraction": v / total} for k, v in sorted(strata.items())},
        "oracle_attempts": len(attempts),
        "oracle_covered": sum(a["covered"] for a in attempts),
        "oracle_relaxed_covered": sum(a.get("covered_relaxed", a["covered"]) for a in attempts),
        "oracle_rejected_overlap_fraction": (sum(a.get("rejected_overlap_strict", a.get("rejected_overlap", 0))
                                                 for a in attempts) / max(sum(a["candidates"] for a in attempts), 1)),
        "oracle_relaxed_rows_mean": mean(p["oracle_relaxed_rows"] for p in preps.values() if "oracle_relaxed_rows" in p),
        "max_oracle_overlap": max((p["max_oracle_overlap"] for p in preps.values()), default=0.0),
        "max_relaxed_overlap": max((p.get("max_relaxed_overlap", 0.0) for p in preps.values()), default=0.0),
    }


def summarize(state: Path, exclude: set[str]) -> dict:
    units = load_units(state)
    preps = {u["result"]["document_id"]: u["result"] for k, u in units.items() if k.startswith("prep.")}
    preps = {d: p for d, p in preps.items() if d not in exclude}
    fits = [u for k, u in units.items() if k.startswith("fit.") and u["result"]["document_id"] not in exclude]
    baseline = {d: {name: mean(f1s(p["baseline"][name])) for name in ("teacher", "native", "bias_only")}
                for d, p in preps.items()}
    cells: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for fit in fits:
        r = fit["result"]
        cells[(fit["unit"]["stage"], r["rank"], r["coverage"], r["lr"])][r["document_id"]] = r
    report_cells = []
    for (stage, rank, coverage, lr), by_doc in sorted(cells.items()):
        documents = sorted(by_doc)
        steps = sorted({int(s) for r in by_doc.values() for s in r["evaluations"]})
        for step in steps:
            docs = [d for d in documents if str(step) in by_doc[d]["evaluations"]]
            arm = {d: mean(f1s(by_doc[d]["evaluations"][str(step)]["natural"])) for d in docs}
            native = {d: baseline[d]["native"] for d in docs}
            teacher = {d: baseline[d]["teacher"] for d in docs}
            strata_scores = defaultdict(lambda: {"arm": [], "native": []})
            for d in docs:
                strata = preps[d]["natural_strata"]
                native_by_query = {n["query_id"]: n["score"]["qa_f1"] for n in preps[d]["baseline"]["native"]}
                for row in by_doc[d]["evaluations"][str(step)]["natural"]:
                    stratum = strata[row["query_id"]]
                    strata_scores[stratum]["arm"].append(row["score"]["qa_f1"])
                    strata_scores[stratum]["native"].append(native_by_query[row["query_id"]])
            evaluations = [by_doc[d]["evaluations"][str(step)] for d in docs]
            step_zero = [by_doc[d]["evaluations"]["0"] for d in docs]
            cell = {
                "stage": stage, "rank": rank, "coverage": coverage, "lr": lr, "step": step, "documents": len(docs),
                "f1": mean(arm.values()), "native": mean(native.values()), "teacher": mean(teacher.values()),
                "delta_vs_native_ci95": bootstrap_delta({d: (arm[d], native[d]) for d in docs}, seed=195) if len(docs) > 1 else None,
                "gap_closure": gap_closure(mean(native.values()), mean(arm.values()), mean(teacher.values())),
                "fit_agreement": mean(e["fit"]["agreement_f1"] for e in evaluations),
                "fit_kl": mean(e["fit"]["kl"] for e in evaluations),
                "retention": mean(e["retention_exact"] for e in evaluations),
                "retention_native": mean(e["retention_exact"] for e in step_zero),
                "strata": {k: {"questions": len(v["arm"]), "f1": mean(v["arm"]), "native": mean(v["native"])}
                           for k, v in sorted(strata_scores.items())},
            }
            if step == max(steps):
                wrong = [(by_doc[d]["wrongdoc"]["mean_f1"], baseline.get(by_doc[d]["wrongdoc"]["document_id"], {}).get("bias_only"))
                         for d in docs if by_doc[d].get("wrongdoc")]
                if wrong:
                    cell["wrongdoc_f1"] = mean(w for w, _ in wrong)
                    cell["wrongdoc_pair_bias_only_f1"] = mean(b for _, b in wrong if b is not None)
            report_cells.append(cell)
    costs = defaultdict(list)
    for key, unit in units.items():
        costs[key.split(".")[0] if not key.startswith("fit.") else f"fit.{unit['unit']['stage']}"].append(unit["wall_seconds"])
    expansion = state / "expansions" / "final.json"
    selection = None
    if expansion.exists():
        selection = {str(u["args"]["rank"]): u.get("selection") for u in json.loads(expansion.read_text())["units"]}
        selection = {k: {kk: v[kk] for kk in ("lr", "step", "gate_met", "agreement", "kl", "documents")}
                     for k, v in selection.items() if v}
    return {"excluded": sorted(exclude), "coverage": coverage_summary(preps) if preps else None,
            "baseline": {name: mean(b[name] for b in baseline.values()) for name in ("teacher", "native", "bias_only")},
            "selection": selection, "cells": report_cells,
            "unit_seconds": {k: {"units": len(v), "mean": mean(v), "total": float(sum(v))} for k, v in sorted(costs.items())}}


def fmt(value, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("state", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--exclude", nargs="*", default=[])
    args = parser.parse_args()
    report = summarize(args.state, set(args.exclude))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("coverage", "baseline", "selection", "unit_seconds")}, indent=2))
    print(f"{'stage':6} {'rank':>4} {'cov':6} {'lr':>7} {'step':>4} {'n':>3} {'F1':>6} {'nat':>6} {'tchr':>6} "
          f"{'dCI95':>15} {'close':>6} {'agree':>6} {'KL':>7} {'ret':>5} {'ret0':>5} strata(F1/native)")
    for c in report["cells"]:
        ci = c["delta_vs_native_ci95"]
        strata = " ".join(f"{k[:5]}:{fmt(v['f1'], 2)}/{fmt(v['native'], 2)}({v['questions']})" for k, v in c["strata"].items())
        wrong = f" wrong {fmt(c['wrongdoc_f1'], 2)} vs bias {fmt(c.get('wrongdoc_pair_bias_only_f1'), 2)}" if "wrongdoc_f1" in c else ""
        print(f"{c['stage']:6} {c['rank']:>4} {c['coverage']:6} {c['lr']:>7.0e} {c['step']:>4} {c['documents']:>3} "
              f"{fmt(c['f1'])} {fmt(c['native'])} {fmt(c['teacher'])} "
              f"{('[' + fmt(ci[0]) + ',' + fmt(ci[1]) + ']') if ci else '-':>15} {fmt(c['gap_closure'], 2):>6} "
              f"{fmt(c['fit_agreement'])} {fmt(c['fit_kl'], 4):>7} {fmt(c['retention'], 2):>5} {fmt(c['retention_native'], 2):>5} "
              f"{strata}{wrong}")


if __name__ == "__main__":
    main()
