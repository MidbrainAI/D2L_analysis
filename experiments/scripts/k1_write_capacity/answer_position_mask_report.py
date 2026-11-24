"""Reduce the answer-position mask results (Table 3, right panel).

Per condition: macro-over-family F1 and answer containment, gap closure toward the teacher against the
matched baseline, and paired document-cluster intervals.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

RESAMPLES = 10_000


def bootstrap(by_cluster: dict[str, list[float]], seed: int = 0) -> list[float]:
    clusters = np.asarray([np.mean(v) for v in by_cluster.values()])
    rng = np.random.default_rng(seed)
    draws = clusters[rng.integers(0, len(clusters), (RESAMPLES, len(clusters)))].mean(1)
    return np.quantile(draws, [0.025, 0.975]).tolist()


def closure(native: float, arm: float, teacher: float) -> float | None:
    return None if abs(teacher - native) < 1e-9 else (arm - native) / (teacher - native)


def units(state: Path, prefix: str) -> list[dict[str, Any]]:
    return [json.loads(p.read_text())["result"] for p in sorted((state / "units").glob(f"{prefix}*.json"))]


def contains(response: str, answers) -> float:
    from d2l_scale_core import normalize

    text = " " + " ".join(normalize(response)) + " "
    return float(any(" " + " ".join(normalize(a)) + " " in text for a in answers if normalize(a)))


def report_answer_position_mask(state: Path) -> dict[str, Any]:
    rows_by_suite: dict[str, list[dict]] = defaultdict(list)
    for result in units(state, "answer_position_mask-"):
        rows_by_suite[result["suite"]].extend(result["rows"])
    out = {}
    for suite, rows in rows_by_suite.items():
        conditions = sorted(rows[0]["scores"])

        def macro(metric_of):
            by_family: dict[str, list[float]] = defaultdict(list)
            for row in rows:
                by_family[row["dataset"]].append(metric_of(row))
            return float(np.mean([np.mean(v) for v in by_family.values()]))

        table = {}
        for condition in conditions:
            f1 = macro(lambda r, c=condition: r["scores"][c]["qa_f1"])
            cont = macro(lambda r, c=condition: contains(r["responses"][c], r["answers"]))
            table[condition] = {"macro_f1": f1, "macro_contains": cont}
        teacher = table["teacher"]["macro_f1"]
        for condition in conditions:
            # Matched baselines: mask arms without the adapter close the no-read -> teacher gap; arms with the
            # native adapter close the native -> teacher gap.
            base = "native" if condition.endswith("native") else "no_read"
            deltas: dict[str, list[float]] = defaultdict(list)
            for row in rows:
                deltas[row["document_id"]].append(row["scores"][condition]["qa_f1"] - row["scores"][base]["qa_f1"])
            table[condition].update(baseline=base,
                                    gap_closure=closure(table[base]["macro_f1"], table[condition]["macro_f1"], teacher),
                                    delta_vs_baseline_ci95=bootstrap(deltas))

        def label(value):
            return (None if value is None else "answer_positions_suffice" if value >= 0.75 else
                    "question_rows_needed" if value < 0.25 else "intermediate")
        decision = {"no_adapter": label(table["answer_read_all"]["gap_closure"]),
                    "with_native": label(table["answer_read_all_native"]["gap_closure"])}
        out[suite] = {"questions": len(rows), "conditions": table, "decision": decision}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = report_answer_position_mask(args.state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({suite: value["decision"] for suite, value in result.items()}, indent=2))

if __name__ == "__main__":
    main()
