"""Summarize GATED_READERS test confirmation with matched source-cluster contrasts."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/gated_readers/test-confirmation-v1"


def load_rows(files):
    rows = {}
    for path in sorted(files):
        for source in json.loads(path.read_text())["result"]["sources"]:
            for row in source["rows"]:
                if row["condition"] != "random":
                    continue
                key = (row["source_id"], row["query_id"], row["bank_id"], row["k"])
                assert key not in rows, key
                rows[key] = row
    return rows


def score_cells(files, baseline_files):
    assert len(files) == len(baseline_files) == 40
    gated = load_rows(files)
    baseline = load_rows(baseline_files)
    assert gated.keys() == baseline.keys()
    assert len(gated) == 5635
    sources = sorted({key[0] for key in gated})
    assert len(sources) == 200
    samples = np.random.default_rng(218).integers(0, len(sources), (10000, len(sources)))
    cells = {}
    for k in (1, 2, 4, 8, 16):
        keys = sorted(key for key in gated if key[3] == k)
        assert len(keys) == 1127
        cell = {"questions": len(keys)}
        for metric in ("f1", "em"):
            cell[metric] = float(np.mean([gated[key]["scores"]["gated"][metric] for key in keys]))
        contrasts = {}
        for name in ("native", "normalized"):
            arm = "native" if name == "native" or k == 1 else "normalized_b0.5"
            differences = defaultdict(list)
            for key in keys:
                differences[key[0]].append(gated[key]["scores"]["gated"]["f1"] -
                                           baseline[key]["scores"][arm]["f1"])
            values = np.array([np.mean(differences[source]) for source in sources])
            draws = values[samples].mean(axis=1)
            contrasts[name] = {"difference": float(values.mean()),
                               "ci95": np.quantile(draws, [.025, .975]).tolist(),
                               "baseline_question_mean": float(np.mean([
                                   baseline[key]["scores"][arm]["f1"] for key in keys]))}
        cell["paired_document_mean_contrasts"] = contrasts
        cells[str(k)] = cell
    return {"cells": cells, "sources": len(sources), "rows": len(gated)}


def main():
    files = list((OUT / "state/units").glob("*.json"))
    baseline_files = list((ROOT / "artifacts/distractor_curves/state-v5/units").glob("gemma2-test-base-*.json"))
    scores = score_cells(files, baseline_files)
    plan = ROOT / "inputs/gated_readers/test-confirmation-v1.json"
    report = {
        "experiment": "GATED_READERS-test-confirmation", **scores, "seed": 218,
        "bootstrap": {"seed": 218, "resamples": 10000, "unit": "source document"},
        "plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
        "runtime": json.loads((OUT / "runtime.json").read_text()),
        "input_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(files + baseline_files)},
        "elapsed_seconds": max(p.stat().st_mtime for p in files) - (OUT / "runtime.json").stat().st_mtime,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(scores["cells"], indent=2))


if __name__ == "__main__":
    main()
