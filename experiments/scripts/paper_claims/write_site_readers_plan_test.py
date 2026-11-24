"""CPU tests for the WRITE_SITE_READERS plans: stage order, banks, the selection-reader expansion, and the confirmation plan."""

import json

import pytest
from distractor_curves_core import K_VALUES
from write_site_readers_core import load_write_site_readers_suite, random_banks, sources
from write_site_readers_core_test import manifest, write_suite
from write_site_readers_plan import (
    ARMS,
    SELECTION_BETA,
    build_selection_plan,
    build_confirmation_plan,
    expand_selection_reader,
    trained_arms,
)
from experiment_plan import plan_digest, validate_plan

SUITES = {name: {"path": f"{name}.json", "sha256": "0" * 64} for name in ("selection", "secondary", "distractor_curves_suite",
                                                                            "distractor_curves_plan")}


def distractor_curves_like_banks(source: str, conditions=("random", "same_article")) -> list[dict]:
    banks = [{"bank_id": f"{source}:random:{k}", "condition": "random", "k": k,
              "document_ids": [f"x{i}" for i in range(k - 1)] + [source], "source_position": k - 1} for k in K_VALUES]
    banks.append({"bank_id": f"{source}:same_article:8", "condition": "same_article", "k": 8,
                  "document_ids": [source] + [f"y{i}" for i in range(7)], "source_position": 0})
    banks += [{"bank_id": f"{source}:copies:{k}", "condition": "copies", "k": k, "document_ids": [source] * k,
               "source_position": 0} for k in (2, 4, 8)]
    return [b for b in banks if b["condition"] in conditions]


def supports(tmp_path) -> dict:
    selection = load_write_site_readers_suite(write_suite(tmp_path / "sel", 20, None), manifest(20))
    secondary = load_write_site_readers_suite(write_suite(tmp_path / "sec", 20, "drop"), manifest(20))
    out = {"selection": {"sources": sources(selection, "selection_source")},
           "secondary": {"sources": sources(secondary, "drop_source")}}
    for name, suite in (("selection", selection), ("secondary", secondary)):
        out[name]["banks"] = {s: random_banks(suite, s) for s in out[name]["sources"]}
    for partition, count in (("dev", 7), ("test", 12)):
        ids = [f"distractor_curves-{partition}-{i}" for i in range(count)]
        out[f"distractor_curves-{partition}"] = {"sources": ids, "banks": {s: distractor_curves_like_banks(s) for s in ids}}
    return out


@pytest.fixture
def tmp_supports(tmp_path):
    (tmp_path / "sel").mkdir()
    (tmp_path / "sec").mkdir()
    return supports(tmp_path)


def check_eval_units(units: list[dict]) -> None:
    assert units
    for unit in units:
        assert unit["kind"] == "eval" and unit["suite"] in ("selection", "distractor_curves-dev", "distractor_curves-test", "secondary")
        assert set(unit["banks"]) == set(unit["sources"])
        for source, banks in unit["banks"].items():
            randoms = [b for b in banks if b["condition"] == "random"]
            assert sorted(b["k"] for b in randoms) == list(K_VALUES)
            for bank in banks:
                assert bank["document_ids"].count(source) == 1
                assert bank["document_ids"][bank["source_position"]] == source


def test_selection_plan_full_plan_stages_units_and_banks(tmp_supports):
    plan = build_selection_plan(tmp_supports, SUITES, smoke=False)
    validate_plan(plan)
    names = [s["name"] for s in plan["stages"]]
    assert names == ["regression", "train-A", "train-B", "train-C", "train-D", "selection-base",
                     "selection-reader"]
    assert "latents" not in names and all(u["kind"] != "latents" for u in plan["units"])
    reader_stage = plan["stages"][-1]
    assert reader_stage["after"] == ["train-A", "train-B", "train-C", "train-D"]
    assert reader_stage["expand"] == "write_site_readers_plan:expand_selection_reader"
    trains = [u for u in plan["units"] if u["kind"] == "train"]
    assert [u["arm"] for u in trains] == list(ARMS) and all(u["seed"] == 216 for u in trains)
    assert all(u["steps"] == 4362 and u["max_document_occurrences"] == 12 and u["validation_interval"] == 500
               and u["fresh_per_k"] == 40 and u["validation_limit"] is None
               and u["selection_suite"] == "inputs/write_site_readers/selection-v1.json" for u in trains)
    regression = next(u for u in plan["units"] if u["kind"] == "regression")
    assert regression["rows"] == 12 and regression["min_agreement"] == 0.75 and "readout_ladderc" in regression["reader"]
    base = [u for u in plan["units"] if u["stage"] == "selection-base"]
    check_eval_units(base)
    assert {u["suite"] for u in base} == {"selection", "distractor_curves-dev"}
    assert sum(len(u["sources"]) for u in base if u["suite"] == "selection") == 20
    assert sum(len(u["sources"]) for u in base if u["suite"] == "distractor_curves-dev") == 7
    assert all(u["betas"] == [SELECTION_BETA] and u["include_icl"] and u["include_native"] for u in base)
    assert all(u["readers"] == {} for u in base)
    assert all(b["condition"] == "random" for u in base for banks in u["banks"].values() for b in banks)
    assert plan["suites"] == SUITES and plan["config"]["bank_seed"] == "write_site_readers-banks-v1"
    assert plan["experiment"] == "WRITE_SITE_READERS" and plan["schema"] == "unit-plan-v1"
    plan_digest(plan)


def test_selection_plan_smoke_plan_has_a_latents_stage_and_short_training(tmp_supports):
    plan = build_selection_plan(tmp_supports, SUITES, smoke=True, reader_documents=[f"r{i}" for i in range(200)])
    validate_plan(plan)
    names = [s["name"] for s in plan["stages"]]
    assert names[:3] == ["regression", "latents", "train-A"]
    assert len([u for u in plan["units"] if u["kind"] == "latents"]) == 4
    assert all(s.get("after") == ["latents"] for s in plan["stages"] if s["name"].startswith("train-"))
    train = next(u for u in plan["units"] if u["kind"] == "train")
    assert (train["steps"], train["validation_interval"], train["validation_limit"], train["fresh_per_k"]) == (2, 1, 10, 1)
    base = [u for u in plan["units"] if u["stage"] == "selection-base"]
    assert sorted(u["suite"] for u in base) == ["distractor_curves-dev", "selection"] and all(len(u["sources"]) == 1 for u in base)
    with pytest.raises(ValueError):
        build_selection_plan(tmp_supports, SUITES, smoke=True)


def prior_for(arms_done: dict[str, bool]) -> dict:
    prior = {}
    for arm, ok in arms_done.items():
        unit = {"id": f"train-{arm}-seed216", "arm": arm, "seed": 216}
        result = {"arm": arm, "seed": 216, "checkpoints": {"best": "b", "last": "l"}} if ok else {}
        prior[f"train-{arm}"] = {unit["id"]: {"unit": unit, "result": result}}
    return prior


def test_selection_reader_expansion_skips_failed_arms(tmp_supports):
    plan = build_selection_plan(tmp_supports, SUITES, smoke=False)
    prior = prior_for({"A": True, "B": False, "C": True})  # D never reported
    assert trained_arms(plan, prior) == ["A", "C"]
    units = expand_selection_reader(plan, prior)
    check_eval_units(units)
    assert {u["suite"] for u in units} == {"selection", "distractor_curves-dev"}
    expected = {"A_best": "side:readers/gemma2/A-seed216/best.pt", "A_last": "side:readers/gemma2/A-seed216/last.pt",
                "C_best": "side:readers/gemma2/C-seed216/best.pt", "C_last": "side:readers/gemma2/C-seed216/last.pt"}
    assert all(u["readers"] == expected for u in units)
    assert all(not u["include_native"] and not u["include_icl"] and u["betas"] == [] for u in units)
    ids = {u["id"] for u in plan["units"]}
    assert not ids & {u["id"] for u in units}
    assert expand_selection_reader(plan, prior_for({"A": False})) == []


def test_confirmation_plan_fixes_arm_and_checkpoint(tmp_supports):
    choice = {"arm": "C", "checkpoint": "last"}
    plan = build_confirmation_plan(tmp_supports, SUITES, arm="C", checkpoint="last", smoke=False, selection_choice=choice)
    validate_plan(plan)
    names = [s["name"] for s in plan["stages"]]
    assert names == ["train-C-seed1216", "train-C-seed2216", "confirm-reader", "secondary-base"]
    assert plan["stages"][2]["after"] == ["train-C-seed1216", "train-C-seed2216"]
    assert "after" not in plan["stages"][3]
    assert plan["config"]["arm"] == "C" and plan["config"]["checkpoint"] == "last"
    assert plan["config"]["selection_choice"] == choice
    trains = [u for u in plan["units"] if u["kind"] == "train"]
    assert [(u["arm"], u["seed"]) for u in trains] == [("C", 1216), ("C", 2216)]
    readers = {f"reader_s{seed}": f"side:readers/gemma2/C-seed{seed}/last.pt" for seed in (216, 1216, 2216)}
    confirm = [u for u in plan["units"] if u["stage"] == "confirm-reader"]
    check_eval_units(confirm)
    assert all(u["suite"] == "distractor_curves-test" and u["readers"] == readers
               and not u["include_native"] and not u["include_icl"] for u in confirm)
    conditions = {b["condition"] for u in confirm for banks in u["banks"].values() for b in banks}
    assert conditions == {"random"}
    assert sum(len(u["sources"]) for u in confirm) == 12
    base = [u for u in plan["units"] if u["stage"] == "secondary-base"]
    check_eval_units(base)
    assert all(u["suite"] == "secondary" and u["readers"] == {} and u["betas"] == [0.5] and u["include_icl"]
               and u["include_native"] for u in base)
    assert plan["requires"][0]["path"] == "side/readers/gemma2/C-seed216/"
    with pytest.raises(ValueError):
        build_confirmation_plan(tmp_supports, SUITES, arm="E", checkpoint="best", smoke=False)


def test_confirmation_plan_smoke_adds_latents_and_one_source_each(tmp_supports):
    plan = build_confirmation_plan(tmp_supports, SUITES, arm="A", checkpoint="best", smoke=True, reader_documents=["r0", "r1"])
    validate_plan(plan)
    assert [s["name"] for s in plan["stages"]][:2] == ["latents", "train-A-seed1216"]
    assert plan["stages"][1]["after"] == ["latents"]
    for stage in ("confirm-reader", "secondary-base"):
        units = [u for u in plan["units"] if u["stage"] == stage]
        assert len(units) == 1 and len(units[0]["sources"]) == 1
    json.dumps(plan)  # serializable
