import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"


@pytest.fixture
def run_unit(tmp_path):
    plan = {"schema": "unit-plan-v1", "config": {"value": 7},
            "stages": [{"name": "start"}, {"name": "next", "after": ["start"], "expand": "fixture_runner:expand"}],
            "units": [{"id": "first", "stage": "start"}]}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    state = tmp_path / "state"
    (tmp_path / "fixture_runner.py").write_text("""
import os
from experiment_plan import GateError

def setup(plan, worker):
    return plan["config"]

def run(context, unit):
    if unit["id"] == "second":
        raise GateError("test gate")
    return {"value": context["value"], "state_dir": os.environ["EXPERIMENT_STATE_DIR"]}

def expand(plan, prior):
    assert prior["start"]["first"]["result"]["value"] == plan["config"]["value"]
    return [{"id": "second"}]
""")

    def invoke(*args, check=True):
        return subprocess.run([
            sys.executable, "run-unit.py", "--plan", str(path), "--state-dir", str(state), *args,
        ], cwd=EXPERIMENTS, env={**os.environ, "PYTHONPATH": str(tmp_path)},
            capture_output=True, text=True, check=check)

    return state, invoke


def test_direct_unit_and_manual_expansion(run_unit):
    state, invoke = run_unit
    assert "start\tfirst" in invoke("--list").stdout
    assert not state.exists()
    invoke("--unit", "first", "--runner", "fixture_runner")
    result = json.loads((state / "units/first.json").read_text())
    assert result["result"] == {"value": 7, "state_dir": str(state)}
    assert not (state / "expansions").exists()
    assert invoke("--unit", "first", "--runner", "fixture_runner", check=False).returncode
    invoke("--expand", "next")
    assert "next\tsecond" in invoke("--list").stdout
    failure = invoke("--unit", "second", "--runner", "fixture_runner", check=False)
    assert failure.returncode and "test gate" in failure.stderr
    assert not (state / "units/second.json").exists()
    assert not (state / "claims").exists()


def test_external_training_result(run_unit, tmp_path):
    state, invoke = run_unit
    unit, result = tmp_path / "unit.json", tmp_path / "raw.json"
    invoke("--unit", "first", "--export-unit", str(unit))
    assert json.loads(unit.read_text()) == {"id": "first", "stage": "start", "plan_config": {"value": 7}}
    result.write_text('{"value": 7}')
    invoke("--unit", "first", "--import-result", str(result))
    assert json.loads((state / "units/first.json").read_text())["result"] == {"value": 7}
