"""Build the frozen plan for the answer-position document-read mask (Table 3, right panel), and stage its inputs.

One stage, ``answer_position_mask``: mask-oracle shards over the 402-question SQuAD/DROP/ROPES diagnostic set.
Each shard checks its fresh baseline responses against the recorded activation-transplant reference
before scoring the masks.

``--stage-inputs DIR`` copies the read-only reference the runner needs into DIR.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

SCHEMA = "unit-plan-v1"
SHARD = 26
CONFIRMATION = Path("inputs/paper-runs/activation-transplant-reference.json")
STAGE_ORDER = ("answer_position_mask",)


def chunks(values: list, size: int) -> list[list]:
    return [values[i:i + size] for i in range(0, len(values), size)]


def mask_units(query_ids: list[str], shard: int) -> list[dict[str, Any]]:
    return [{"id": f"answer_position_mask-exact402-{index:03d}", "stage": "answer_position_mask",
             "kind": "answer_position_mask", "suite": "exact402", "split": "validation", "query_ids": group,
             "cost": len(group) / shard}
            for index, group in enumerate(chunks(query_ids, shard))]


def build_plan(*, confirmation: dict[str, Any], inputs: str, smoke: bool = False) -> dict[str, Any]:
    query_ids = [row["query_id"] for row in confirmation["rows"]]
    if smoke:
        query_ids = query_ids[:2]
    config = {"smoke": smoke, "limits": {"max_new_tokens": 4} if smoke else None, "inputs": inputs,
              "confirmation_contract": confirmation["contract"]}
    return {"schema": SCHEMA, "experiment": "answer-position-mask", "config": config,
            "stages": [{"name": "answer_position_mask"}], "units": mask_units(query_ids, 1 if smoke else SHARD)}


def stage_inputs(destination: Path, plan_path: Path) -> None:
    (destination / "activation_transplant").mkdir(parents=True, exist_ok=True)
    shutil.copy2(CONFIRMATION, destination / "activation_transplant" / "reference.json")
    (destination / "plans").mkdir(exist_ok=True)
    target = destination / "plans" / plan_path.name
    if target.resolve() != plan_path.resolve():
        shutil.copy2(plan_path, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", default="artifacts/answer-position-mask-inputs",
                        help="experiments-relative directory holding the staged reference")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stage-inputs", type=Path, help="also assemble the staged inputs directory here")
    args = parser.parse_args()
    plan = build_plan(confirmation=json.loads(CONFIRMATION.read_text()), inputs=args.inputs, smoke=args.smoke)
    from experiment_plan import validate_plan

    validate_plan(plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n")
    if args.stage_inputs:
        stage_inputs(args.stage_inputs, args.output)
    print(json.dumps({"output": str(args.output), "units": len(plan["units"])}, indent=2))


if __name__ == "__main__":
    main()
