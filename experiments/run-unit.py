"""List, prepare, or execute one frozen experiment unit. No scheduling or retries."""

import argparse
import importlib
import json
import os
from pathlib import Path
import time

from experiment_plan import atomic_write, plan_digest, validate_plan


def read(path):
    return json.loads(path.read_text())


def load_units(plan, state):
    units = list(plan["units"])
    expanded = set()
    for stage in plan["stages"]:
        name = stage["name"]
        path = state / "expansions" / f"{name}.json"
        if "expand" not in stage or not path.is_file():
            continue
        entry = read(path)
        if entry["plan_sha256"] != plan_digest(plan) or entry["stage"] != name:
            raise ValueError(f"Expansion is not for this plan and stage: {path}")
        if any(u["stage"] != name for u in entry["units"]):
            raise ValueError(f"Wrong unit stage in {path}")
        units.extend(entry["units"])
        expanded.add(name)
    validate_plan({**plan, "units": units})
    return units, expanded


def expand(plan, units, expanded, state, name):
    stages = {s["name"]: s for s in plan["stages"]}
    if name not in stages or "expand" not in stages[name]:
        raise ValueError(f"No expansion hook for {name}")
    if name in expanded:
        raise ValueError(f"Expansion already exists for {name}")
    stage = stages[name]
    module, function = stage["expand"].split(":")
    prior = {}
    for dep in stage.get("after", []):
        if "expand" in stages[dep] and dep not in expanded:
            raise ValueError(f"Expand {dep} first")
        prior[dep] = {u["id"]: read(state / "units" / f"{u['id']}.json") for u in units if u["stage"] == dep}
    new_units = [{**u, "stage": name} for u in getattr(importlib.import_module(module), function)(plan, prior)]
    validate_plan({**plan, "units": units + new_units})
    atomic_write(state / "expansions" / f"{name}.json",
                 {"stage": name, "units": new_units, "plan_sha256": plan_digest(plan)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True, help="directory for results and shared side inputs")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true", help="list currently defined unit IDs by stage")
    action.add_argument("--expand", metavar="STAGE", help="materialize one stage from its prerequisite results")
    action.add_argument("--unit", metavar="ID", help="select exactly one unit")
    parser.add_argument("--runner", help="experiment module exposing setup and run")
    external = parser.add_mutually_exclusive_group()
    external.add_argument("--export-unit", type=Path, help="write the selected unit for a standalone trainer")
    external.add_argument("--import-result", type=Path, help="wrap a standalone trainer's JSON result for reducers")
    args = parser.parse_args()
    if not args.unit and (args.runner or args.export_unit or args.import_result):
        parser.error("--runner, --export-unit, and --import-result require --unit")
    plan = read(args.plan)
    validate_plan(plan)
    state = args.state_dir.resolve()
    units, expanded = load_units(plan, state)
    if args.list:
        for stage in plan["stages"]:
            name = stage["name"]
            if "expand" in stage and name not in expanded:
                print(f"{name}\t(expand after prerequisites)")
            for unit in units:
                if unit["stage"] == name:
                    print(f"{name}\t{unit['id']}")
        return
    if args.expand:
        expand(plan, units, expanded, state, args.expand)
        return
    unit = next((u for u in units if u["id"] == args.unit), None)
    if unit is None:
        parser.error(f"unknown unit {args.unit!r}; use --list")
    if args.export_unit:
        atomic_write(args.export_unit, {**unit, "plan_config": plan.get("config", {})})
        return
    path = state / "units" / f"{unit['id']}.json"
    if path.exists():
        parser.error(f"result already exists: {path}")
    record = {"unit": unit, "plan_sha256": plan_digest(plan)}
    if args.import_result:
        record.update(result=read(args.import_result), worker="external")
    else:
        if not args.runner:
            parser.error("--runner is required to execute a unit")
        os.environ["EXPERIMENT_STATE_DIR"] = str(state)
        runner = importlib.import_module(args.runner)
        context = runner.setup(plan, "direct")
        start = time.monotonic()
        try:
            record.update(result=runner.run(context, unit), worker="direct",
                          wall_seconds=time.monotonic() - start)
        finally:
            if hasattr(runner, "teardown"):
                runner.teardown(context)
    atomic_write(path, record)
    print(path)


if __name__ == "__main__":
    main()
