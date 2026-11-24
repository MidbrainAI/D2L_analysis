"""Verify the published PersonaMem-v2 test results from label-only scores."""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from pathlib import Path
from typing import Any

BASELINE_ARMS = (
    "none", "full_history", "episodic", "icl_pk", "icl_pk_episodic", "d2l_joint_episodic",
)
FOLLOWUP_ARMS = ("d2l_entries_native_episodic", "d2l_entries_norm_episodic")
LABELS = frozenset("abcd")


def load_rows(state: Path, arms: tuple[str, ...]) -> list[dict[str, Any]]:
    """Accept only the public score schema, never provider exports or free text."""
    rows = []
    paths = sorted((state / "units").glob("*.json"))
    for path in paths:
        if not re.fullmatch(r"score-shard-\d{3}\.json", path.name):
            raise ValueError("Unexpected PersonaMem score shard name")
        shard = json.loads(path.read_text())
        if not isinstance(shard, list):
            raise ValueError("Invalid PersonaMem score shard")
        for row in shard:
            if set(row) != {"question_id", "persona_id", "correct_key", "arms"}:
                raise ValueError("Unexpected PersonaMem score fields")
            persona, question = row["persona_id"], row["question_id"]
            if not re.fullmatch(r"[0-9]+", persona) or not re.fullmatch(fr"p{persona}_q[0-9]+", question):
                raise ValueError("Invalid PersonaMem benchmark ID")
            if row["correct_key"] not in LABELS or set(row["arms"]) != set(arms):
                raise ValueError("Invalid PersonaMem labels or conditions")
            for score in row["arms"].values():
                if (set(score) != {"pred", "prompt_tokens"} or score["pred"] not in LABELS
                        or type(score["prompt_tokens"]) is not int or score["prompt_tokens"] < 0):
                    raise ValueError("Unexpected PersonaMem score payload")
            rows.append(row)
    if len(rows) != 4878 or len({r["question_id"] for r in rows}) != 4878 or len({r["persona_id"] for r in rows}) != 195:
        raise ValueError("Unexpected PersonaMem test support")
    return rows


def correct(row: dict[str, Any], arm: str) -> float:
    return float(row["arms"][arm]["pred"] == row["correct_key"])


def interval(rows: list[dict[str, Any]], a: str, b: str, resamples: int = 4000) -> dict[str, float]:
    by_persona: dict[str, list[float]] = {}
    for row in rows:
        by_persona.setdefault(row["persona_id"], []).append(correct(row, a) - correct(row, b))
    personas = sorted(by_persona)
    sums = {p: (sum(v), len(v)) for p, v in by_persona.items()}

    def mean(sample):
        return sum(sums[p][0] for p in sample) / sum(sums[p][1] for p in sample)

    rng = random.Random(213)
    draws = sorted(mean([rng.choice(personas) for _ in personas]) for _ in range(resamples))
    return {"difference": mean(personas), "low": draws[int(0.025 * resamples)],
            "high": draws[int(0.975 * resamples) - 1], "personas": len(personas)}


def summarize(rows: list[dict[str, Any]], arms) -> dict[str, Any]:
    table = {}
    for arm in arms:
        memory = [r["arms"][arm]["prompt_tokens"] - r["arms"]["none"]["prompt_tokens"] for r in rows]
        table[arm] = {"accuracy": sum(correct(r, arm) for r in rows) / len(rows),
                      "memory_tokens_median": statistics.median(memory), "memory_tokens_mean": sum(memory) / len(memory)}
    return {"questions": len(rows), "personas": len({r["persona_id"] for r in rows}), "arms": table}


def report_from_state(state: Path) -> dict[str, Any]:
    test = load_rows(state, BASELINE_ARMS)
    return {"test": summarize(test, BASELINE_ARMS[1:]),
            "contrasts": {"icl_pk_episodic-episodic": interval(test, "icl_pk_episodic", "episodic")}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = report_from_state(args.state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
