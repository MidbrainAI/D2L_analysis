"""Shared schema and result format for the paper's frozen experiment plans."""

import hashlib
import json
import os
from pathlib import Path


class GateError(RuntimeError):
    """A paper protocol check failed; the unit must not produce a valid result."""


def plan_digest(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_plan(plan):
    if plan.get("schema") != "unit-plan-v1":
        raise ValueError("plan schema must be unit-plan-v1")
    names = [s["name"] for s in plan["stages"]]
    if len(set(names)) != len(names) or any(not n or "/" in n or n.startswith(".") for n in names):
        raise ValueError("stage names must be unique, flat file names")
    for index, stage in enumerate(plan["stages"]):
        if any(dep not in names[:index] for dep in stage.get("after", [])):
            raise ValueError(f"stage {stage['name']} must follow only earlier stages")
    ids = [u["id"] for u in plan["units"]]
    if len(set(ids)) != len(ids) or any(not i or "/" in i or i.startswith(".") for i in ids):
        raise ValueError("unit IDs must be unique, flat file names")
    if any(unit["stage"] not in names for unit in plan["units"]):
        raise ValueError("unit refers to an unknown stage")
