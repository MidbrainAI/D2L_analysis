"""Freeze and verify the gated-reader confirmation plan on Figure 2's test banks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "inputs/write_site_readers/plans/confirmation-v1.json"
CHECKPOINT = ROOT / "artifacts/layer_interventions/state/side/readers/gemma2/GB-lam1-seed218/best.pt"
PLAN = ROOT / "inputs/gated_readers/test-confirmation-v1.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    parent = json.loads(PARENT.read_text())
    for entry in parent["suites"].values():
        assert digest(ROOT / entry["path"]) == entry["sha256"], entry["path"]
    units = []
    for original in parent["units"]:
        if original["kind"] != "eval" or original["suite"] != "distractor_curves-test":
            continue
        unit = dict(original)
        unit.update(
            id=f"gated-test-{len(units):03d}", stage="gated-test",
            readers={"gated": str(CHECKPOINT.relative_to(ROOT))}, betas=[],
            include_icl=False, include_native=False,
        )
        unit["banks"] = {
            source: [bank for bank in banks if bank["condition"] == "random"]
            for source, banks in original["banks"].items()
        }
        for banks in unit["banks"].values():
            assert [bank["k"] for bank in banks] == [1, 2, 4, 8, 16]
        units.append(unit)
    assert units
    sources = [source for unit in units for source in unit["sources"]]
    assert len(sources) == len(set(sources))
    plan = {
        "schema": "unit-plan-v1", "experiment": "GATED_READERS-test-confirmation",
        "suites": parent["suites"], "stages": [{"name": "gated-test"}],
        "units": units,
        "provenance": {
            "parent_plan": str(PARENT.relative_to(ROOT)),
            "parent_sha256": digest(PARENT),
            "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
            "checkpoint_sha256": digest(CHECKPOINT),
            "selection": (
                "GB lam1 seed218, inline-validation best checkpoint (step 2500). "
                "Among GB/GBL best checkpoints, GB has higher mean development F1 "
                "over K=1,8 (.5145 vs .5085); BS is not a gated reader. "
                "No test-based selection. Prior DISTRACTOR_CURVES/WRITE_SITE_READERS test use disclosed."
            ),
            "baselines": "Reuse matched DISTRACTOR_CURVES/WRITE_SITE_READERS recorded rows; no regeneration.",
        },
    }
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if PLAN.exists():
        assert PLAN.read_text() == encoded, "Frozen plan differs"
    else:
        PLAN.write_text(encoded)
    print(json.dumps({"plan": str(PLAN), "sha256": digest(PLAN),
                      "sources": len(sources), "units": len(units),
                      "checkpoint_sha256": digest(CHECKPOINT)}, indent=2), flush=True)


def verify():
    plan = json.loads(PLAN.read_text())
    assert digest(CHECKPOINT) == plan["provenance"]["checkpoint_sha256"]
    assert digest(PARENT) == plan["provenance"]["parent_sha256"]
    for entry in plan["suites"].values():
        assert digest(ROOT / entry["path"]) == entry["sha256"]
    print("Plan inputs and checkpoint match their recorded hashes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "verify"])
    args = parser.parse_args()
    prepare() if args.action == "prepare" else verify()
