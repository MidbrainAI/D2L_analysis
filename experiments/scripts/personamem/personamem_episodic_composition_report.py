"""Reduce the matched PersonaMem-v2 episodic-composition follow-up."""
import argparse
import json
from pathlib import Path

from personamem_report import BASELINE_ARMS, FOLLOWUP_ARMS, interval, load_rows, summarize

NATIVE = "d2l_entries_native_episodic"
NORMALIZED = "d2l_entries_norm_episodic"
JOINT = "d2l_joint_episodic"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-state", type=Path, required=True)
    parser.add_argument("--followup-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = {r["question_id"]: r for r in load_rows(args.baseline_state, BASELINE_ARMS)}
    rows = load_rows(args.followup_state, FOLLOWUP_ARMS)
    if len(rows) != len(baseline) or {r["question_id"] for r in rows} != baseline.keys():
        raise ValueError("PersonaMem question support differs")
    for row in rows:
        base = baseline[row["question_id"]]
        if any(row[k] != base[k] for k in ("persona_id", "correct_key")):
            raise ValueError(f"PersonaMem row metadata differs: {row['question_id']}")
        row["arms"] = {**base["arms"], **row["arms"]}
    report = {
        "test": summarize(rows, FOLLOWUP_ARMS),
        "contrasts": {f"{a}-{JOINT}": interval(rows, a, JOINT) for a in (NATIVE, NORMALIZED)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
