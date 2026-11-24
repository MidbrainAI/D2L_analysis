import pytest
from distractor_curves_core_test import toy_suite
from distractor_curves_plan import (
    build_plan,
    choose_beta,
    choose_checkpoint,
    expand_reader_dev,
    expand_test_base,
    expand_test_reader,
)
from experiment_plan import validate_plan


def suite_with_dev() -> dict:
    suite = toy_suite(articles=6, per_article=20)
    for document in suite["documents"].values():
        if document["article"] in ("article-0", "article-1"):
            document["role"] = document["role"].replace("test", "dev")
    suite["suite_sha256"] = "s"
    return suite


def row(arm_scores: dict, k: int, condition: str = "random") -> dict:
    return {"condition": condition, "k": k, "scores": {arm: {"f1": value} for arm, value in arm_scores.items()}}


def test_plan_is_valid_and_ordered():
    plan = build_plan(suite_path="suite.json", suite=suite_with_dev(), reader_documents=[f"r{i}" for i in range(1200)],
                      smoke=False)
    validate_plan(plan)
    names = [stage["name"] for stage in plan["stages"]]
    assert names.index("gemma2-train") < names.index("gemma2-dev-base") < names.index("gemma2-test-base")
    assert names.index("gemma2-test-reader") < names.index("timing")
    trains_by_model = {u["model"]: u["max_document_occurrences"] for u in plan["units"] if u["kind"] == "train"}
    assert trains_by_model == {"gemma2": 12, "qwen4": 6}
    trains = [u for u in plan["units"] if u["kind"] == "train"]
    assert sorted(u["seed"] for u in trains if u["model"] == "gemma2") == [199, 1199, 2199]
    dev = [u for u in plan["units"] if u["stage"] == "gemma2-dev-base"]
    assert dev and all(u["readers"] == {} for u in dev)
    assert all(b["condition"] == "random" for u in dev for banks in u["banks"].values() for b in banks)


def test_beta_rule_prefers_higher_dev_f1_and_breaks_ties_low():
    rows = [row({"native": 0.1, "normalized_b0.25": 0.3, "normalized_b0.5": 0.5, "normalized_b1.0": 0.5}, k)
            for k in (2, 4)] + [row({"normalized_b0.25": 0.9}, 1)]
    choice = choose_beta(rows, [0.25, 0.5, 1.0])
    assert choice["beta"] == 0.5 and choice["native_dev_f1"] == pytest.approx(0.1)


def test_checkpoint_rule_is_macro_over_k():
    rows = [row({"seed1_best": 0.9, "seed1_last": 0.2}, 1)] + [row({"seed1_best": 0.1, "seed1_last": 0.9}, 2)] * 5
    assert choose_checkpoint(rows, 1)["checkpoint"] == "last"
    tie = [row({"seed2_best": 0.5, "seed2_last": 0.5}, 1)]
    assert choose_checkpoint(tie, 2)["checkpoint"] == "best"


def test_expansions_use_the_selected_beta_and_checkpoints():
    plan = build_plan(suite_path="suite.json", suite=suite_with_dev(), reader_documents=["r0"], smoke=False)
    train = {"gemma2-train-seed199": {"result": {"seed": 199}}}
    dev_units = expand_reader_dev(plan, {"gemma2-train": train})
    assert dev_units and set(dev_units[0]["readers"]) == {"seed199_best", "seed199_last"}
    assert not dev_units[0]["include_native"]
    base = {"u": {"result": {"sources": [{"rows": [row({"native": 0.1, "normalized_b0.25": 0.2,
                                                       "normalized_b0.5": 0.4, "normalized_b0.75": 0.3,
                                                       "normalized_b1.0": 0.1}, 4)]}]}}}
    readers = {"u": {"result": {"sources": [{"rows": [row({"seed199_best": 0.3, "seed199_last": 0.6}, 1)]}]}}}
    base_units = expand_test_base(plan, {"gemma2-dev-base": base})
    assert base_units and all(u["betas"] == [0.5] and u["include_native"] for u in base_units)
    assert base_units[0]["readers"] == {}
    assert {b["condition"] for banks in base_units[0]["banks"].values() for b in banks} == {"random"}
    reader_units = expand_test_reader(plan, {"gemma2-dev-base": base, "gemma2-reader-dev": readers})
    assert reader_units[0]["readers"] == {"reader_s199": "side:readers/gemma2/seed199/last.pt"}
    assert not reader_units[0]["include_native"]
    assert {b["condition"] for banks in reader_units[0]["banks"].values() for b in banks} == {"random"}
