from answer_position_mask_plan import STAGE_ORDER, build_plan


def confirmation():
    return {"contract": {"x": 1}, "rows": [{"query_id": f"q{i}"} for i in range(30)]}


def test_plan_units_and_stages():
    from experiment_plan import validate_plan

    plan = build_plan(confirmation=confirmation(), inputs="artifacts/x")
    validate_plan(plan)
    assert [s["name"] for s in plan["stages"]] == list(STAGE_ORDER)
    assert [q for u in plan["units"] for q in u["query_ids"]] == [f"q{i}" for i in range(30)]
    assert {u["suite"] for u in plan["units"]} == {"exact402"}


def test_smoke_plan_is_small():
    plan = build_plan(confirmation=confirmation(), inputs="artifacts/x", smoke=True)
    assert sum(len(u["query_ids"]) for u in plan["units"]) == 2
