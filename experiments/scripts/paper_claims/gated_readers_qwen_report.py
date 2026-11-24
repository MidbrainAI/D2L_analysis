"""Matched Qwen gated-reader scores and paired source-document intervals."""
import hashlib
import json
from pathlib import Path

import numpy as np
from distractor_curves_report import add_containment
from gated_readers_test_report import load_rows

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/gated_readers/qwen-gated-v1"


def score_cells(files, baseline_files, answers):
    assert len(files) == len(baseline_files) == 40
    gated, baseline = load_rows(files), load_rows(baseline_files)
    assert gated.keys() == baseline.keys() and len(gated) == 5635
    for rows in (gated, baseline):
        add_containment(list(rows.values()), answers)
    sources = sorted({key[0] for key in gated})
    assert len(sources) == 200
    samples = np.random.default_rng(218).integers(0, 200, (10000, 200))
    cells = {}
    for k in (1, 2, 4, 8, 16):
        keys = sorted(key for key in gated if key[3] == k)
        assert len(keys) == 1127
        cell = {"questions": len(keys), "contrasts": {}}
        for metric in ("f1", "em", "contains"):
            cell[metric] = float(np.mean([gated[key]["scores"]["gated"][metric] for key in keys]))
            cell["contrasts"][metric] = {}
            for name in ("native", "normalized"):
                arm = "native" if name == "native" or k == 1 else "normalized_b0.5"
                by_source = {s: [] for s in sources}
                for key in keys:
                    by_source[key[0]].append(gated[key]["scores"]["gated"][metric]
                                             - baseline[key]["scores"][arm][metric])
                values = np.array([np.mean(by_source[s]) for s in sources])
                cell["contrasts"][metric][name] = {
                    "difference": float(values.mean()),
                    "ci95": np.quantile(values[samples].mean(axis=1), [.025, .975]).tolist(),
                    "baseline_question_mean": float(np.mean([baseline[key]["scores"][arm][metric] for key in keys])),
                }
        cells[str(k)] = cell
    return {"cells": cells, "sources": len(sources), "rows": len(gated)}


def main():
    files = sorted((OUT / "state/units").glob("*gated-test*.json"))
    baseline_files = sorted((ROOT / "artifacts/distractor_curves/state-v5/units").glob("qwen4-test-base-*.json"))
    manifest = json.loads((ROOT / "inputs/qa_protocol/manifest/pinned-v1/manifest.json").read_text())
    answers = {r["query_id"]: tuple(r["answers"]) for r in manifest["rows"] if r["family"] == "squad"}
    scores = score_cells(files, baseline_files, answers)
    train_file = next((OUT / "state/units").glob("*train*.json"))
    plan = ROOT / "inputs/gated_readers/qwen-gated-v1.json"
    report = {**scores,
              "training": json.loads(train_file.read_text())["result"],
              "bootstrap": {"unit": "source document", "seed": 218, "resamples": 10000},
              "input_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [*files, *baseline_files, train_file, plan, OUT / "best.pt"]}}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(scores["cells"], indent=2))


if __name__ == "__main__":
    main()
