"""Freeze the single-recipe Qwen gated-reader training and matched test evaluation."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAN = Path("inputs/gated_readers/qwen-gated-v1.json")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    parent_path = ROOT / "inputs/layer_interventions/plans/layer_interventions-fast-v1.json"
    parent = json.loads(parent_path.read_text())
    matched_path = ROOT / "inputs/gated_readers/test-confirmation-v1.json"
    matched = json.loads(matched_path.read_text())
    expansion_path = ROOT / "artifacts/distractor_curves/state-v5/expansions/qwen4-test-base.json"
    expansion = json.loads(expansion_path.read_text())
    qwen_banks = {s: banks for u in expansion["units"] for s, banks in u["banks"].items()}
    train = copy.deepcopy(next(u for u in parent["units"] if u.get("arm") == "GB"))
    train.update(id="qwen4-train-GB-lam1-seed218", stage="train", model="qwen4",
                 max_document_occurrences=6)
    units = [train]
    for original in matched["units"]:
        unit = copy.deepcopy(original)
        for source, banks in unit["banks"].items():
            assert banks == [b for b in qwen_banks[source] if b["condition"] == "random"]
        unit.update(id=f"qwen4-gated-test-{len(units)-1:03d}", stage="test", model="qwen4",
                    readers={"gated": "side:readers/qwen4/GB-lam1-seed218/best.pt"})
        units.append(unit)
    assert len(units) == 41
    suites = {k: v for k, v in parent["suites"].items()
              if k in ("distractor_curves_suite", "selection")}
    for entry in suites.values():
        assert digest(ROOT / entry["path"]) == entry["sha256"]
    plan = {
        "schema": "unit-plan-v1", "experiment": "GATED_READERS-Qwen-gated",
        "stages": [{"name": "train"}, {"name": "test", "after": ["train"]}],
        "suites": suites, "units": units,
        "provenance": {
            "parents": {str(p.relative_to(ROOT)): digest(p)
                        for p in (parent_path, matched_path, expansion_path)},
            "selection": "GB lam1 seed218 fixed from Gemma; Qwen checkpoint selected by "
                         "inline validation macro F1, earlier checkpoint on ties, every 500 updates. "
                         "Previously opened DISTRACTOR_CURVES test; no test selection.",
            "latents": "artifacts/distractor_curves/state-v5/side/latents/qwen4/",
            "report": "F1, EM and answer containment; paired source-document bootstrap "
                      "against existing Qwen native and normalized beta .5 rows.",
        },
    }
    path = ROOT / PLAN
    encoded = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if path.exists():
        assert path.read_text() == encoded, "Frozen plan differs"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded)
    print(json.dumps({"plan": str(PLAN), "sha256": digest(path), "units": len(units)}, indent=2))


if __name__ == "__main__":
    prepare()
