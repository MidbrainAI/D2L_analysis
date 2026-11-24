"""Predicted-gate post-hoc analysis: what the stored per-rank probe logits say about gate calibration and ranking.

The ``predicted`` units store, per row and arm, the probe's per-rank logits at the last prompt position
(``scores_last``). This reads them back with the bank membership from the unit's banks and reports, per
scorer and K: the logit distributions of source and distractor ranks, how often the source document has
the highest document-mean logit, and the mean gate on source and distractor ranks under alternative gate
functions (the run's ``sigmoid``, shifted sigmoids, max-relative, bank-centred, document max-relative).
Run from ``experiments/``:

    uv run python scripts/paper_claims/attenuation_schedules_gate_calibration.py --state artifacts/predicted_gates/state --output artifacts/predicted_gates/gate-calibration.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path
from typing import Any, Callable

RPD = 8


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def gate_functions(ranks_per_document: int = RPD) -> dict[str, Callable[[list[float]], list[float]]]:
    def shifted(c: float):
        return lambda flat: [sigmoid(x + c) for x in flat]

    def document_max_relative(flat: list[float]) -> list[float]:
        means = [st.mean(flat[j:j + ranks_per_document]) for j in range(0, len(flat), ranks_per_document)]
        top = max(means)
        return [math.exp(means[i // ranks_per_document] - top) for i in range(len(flat))]

    return {"sigmoid(x)": shifted(0.0), "sigmoid(x+2)": shifted(2.0), "sigmoid(x+3)": shifted(3.0),
            "exp(x-max)": lambda flat: [math.exp(x - max(flat)) for x in flat],
            "sigmoid(2(x-mean))": lambda flat: [sigmoid(2.0 * (x - st.mean(flat))) for x in flat],
            "doc exp(mean-max)": document_max_relative}


def load_rows(state: Path, arms: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((state / "units").glob("predicted-*.json")):
        record = json.loads(path.read_text())
        unit = record["unit"]
        banks = {b["bank_id"]: b for s in unit["sources"] for b in unit["banks"][s]}
        for source in record["result"]["sources"]:
            for row in source["rows"]:
                members = [i == row["source_id"] for i in banks[row["bank_id"]]["document_ids"]]
                for arm in arms:
                    if arm in row.get("gates", {}):
                        rows.append({"arm": arm, "k": row["k"], "logits": row["gates"][arm]["scores_last"],
                                     "source": members.index(True)})
    return rows


def analyze(rows: list[dict[str, Any]], ranks_per_document: int = RPD) -> dict[str, Any]:
    out: dict[str, Any] = {}
    functions = gate_functions(ranks_per_document)
    for arm in sorted({r["arm"] for r in rows}):
        for k in sorted({r["k"] for r in rows if r["arm"] == arm}):
            selected = [r for r in rows if r["arm"] == arm and r["k"] == k]
            source_logits, distractor_logits, top = [], [], 0
            gates: dict[str, dict[str, list[float]]] = {name: {"source": [], "distractor": []} for name in functions}
            for r in selected:
                flat, si = r["logits"], r["source"]
                lo, hi = si * ranks_per_document, (si + 1) * ranks_per_document
                source_logits += flat[lo:hi]
                distractor_logits += flat[:lo] + flat[hi:]
                means = [st.mean(flat[j:j + ranks_per_document]) for j in range(0, len(flat), ranks_per_document)]
                top += means[si] >= max(means)
                for name, fn in functions.items():
                    g = fn(flat)
                    gates[name]["source"] += g[lo:hi]
                    gates[name]["distractor"] += g[:lo] + g[hi:]
            mean = lambda v: (st.mean(v) if v else None)  # noqa: E731
            out.setdefault(arm, {})[str(k)] = {
                "rows": len(selected), "source_logit_mean": mean(source_logits), "distractor_logit_mean": mean(distractor_logits),
                "source_logit_sd": st.pstdev(source_logits) if source_logits else None,
                "distractor_logit_sd": st.pstdev(distractor_logits) if len(distractor_logits) > 1 else None,
                "source_is_top_document": top / len(selected),
                "gates": {name: {"source": mean(v["source"]), "distractor": mean(v["distractor"])} for name, v in gates.items()}}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--arms", default="predicted_B6-b_rows-L6-prompt_mean-soft,predicted_B8-b_rows-L8-prompt_mean-soft")
    args = parser.parse_args()
    report = analyze(load_rows(args.state, tuple(args.arms.split(","))))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=1) + "\n")
    for arm, by_k in report.items():
        for k, e in by_k.items():
            print(f"{arm} K={k}: source logit {e['source_logit_mean']:+.2f}, distractor "
                  f"{e['distractor_logit_mean'] if e['distractor_logit_mean'] is None else round(e['distractor_logit_mean'], 2)}, "
                  f"source is top document {e['source_is_top_document']:.2f}")
            for name, g in e["gates"].items():
                print(f"    {name:20s} source {g['source']:.2f} distractor {g['distractor'] if g['distractor'] is None else round(g['distractor'], 2)}")


if __name__ == "__main__":
    main()
