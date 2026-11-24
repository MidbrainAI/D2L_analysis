"""CPU tests for the LAYER_INTERVENTIONS/GATED_READERS plan builder (fake supports, no manifest)."""

import layer_interventions_plan as plan_module
from experiment_plan import validate_plan


def fake_supports(sources: int = 7):
    ids = [f"qa_protocol-doc-{i:024x}" for i in range(sources)]
    banks = {s: [{"bank_id": f"{s}:random:{k}", "condition": "random", "k": k,
                  "document_ids": [s] + [d for d in ids if d != s][: k - 1], "source_position": 0} for k in (1, 2, 4, 8, 16)]
             for s in ids}
    dev = {"d0": [{"bank_id": "d0:random:1", "condition": "random", "k": 1, "document_ids": ["d0"], "source_position": 0}]}
    return {"selection": {"sources": ids, "banks": banks}, "distractor_curves-dev": {"sources": ["d0"], "banks": dev}}


def test_fast_plan_trains_three_arms_first_and_runs_diagnostics_last():
    plan = plan_module.build(fake_supports(), {"selection": {"path": "x", "sha256": "y"}}, smoke=False)
    validate_plan(plan)
    names = [s["name"] for s in plan["stages"]]
    assert names == ["regression", "train-GB-lam1", "train-GBL-lam1", "train-BS-lam1", "gated-reader", "diagnostics"]
    assert plan_module.serial_order(plan) == ",".join(names)
    assert plan["config"]["variant"] == "fast" and plan["config"]["eval_ks"] == [1, 8]
    kinds = {u["kind"] for u in plan["units"]}
    assert kinds == {"regression", "oracle", "probe", "train"}
    oracles = [u for u in plan["units"] if u["kind"] == "oracle"]
    assert len(oracles) == 2 and oracles[0]["layer_from"] == [0, 8, 18] and oracles[0]["ks"] == [8, 16]
    probe = next(u for u in plan["units"] if u["kind"] == "probe")
    assert probe["priority"] == 10 and probe["train_plans"] == plan_module.PROBE_TRAIN_PLANS and len(probe["banks"]) == 7
    assert probe["test_ks"] == [8, 16]
    trains = [u for u in plan["units"] if u["kind"] == "train"]
    assert [(u["arm"], u["lam"]) for u in trains] == [("GB", 1.0), ("GBL", 1.0), ("BS", 1.0)]
    assert trains[0]["id"] == "train-GB-lam1-seed218" and trains[0]["steps"] == 4362
    assert plan["stages"][-2]["expand"] == "layer_interventions_plan:expand_gated_reader" and plan["stages"][-2]["after"] == names[1:4]


def test_full_plan_keeps_the_preregistered_ablation():
    plan = plan_module.build(fake_supports(), {"selection": {"path": "x", "sha256": "y"}}, smoke=False, full=True)
    validate_plan(plan)
    names = [s["name"] for s in plan["stages"]]
    assert names == ["regression", "diagnostics", "train-GB-lam0", "train-GB-lam1", "train-BS-lam1", "train-GBL-lam1",
                     "train-GC-lam1", "gated-reader"]
    oracles = [u for u in plan["units"] if u["kind"] == "oracle"]
    assert oracles[0]["layer_from"] == list(plan_module.ORACLE_LAYER_FROM) and oracles[0]["ks"] == [4, 8, 16]
    assert plan["config"]["eval_ks"] == [1, 2, 4, 8, 16] and plan["config"]["variant"] == "full"
    trains = [u for u in plan["units"] if u["kind"] == "train"]
    assert [(u["arm"], u["lam"]) for u in trains] == list(plan_module.FULL_ARMS)


def test_smoke_plan_adds_latents_and_shrinks_everything():
    plan = plan_module.build(fake_supports(), {}, smoke=True, reader_documents=["a", "b"])
    validate_plan(plan)
    assert plan["stages"][1]["name"] == "latents" and plan["stages"][2]["after"] == ["latents"]
    oracle = next(u for u in plan["units"] if u["kind"] == "oracle")
    assert oracle["layer_from"] == [0, 13] and oracle["ks"] == [4] and len(oracle["sources"]) == 1
    probe = next(u for u in plan["units"] if u["kind"] == "probe")
    assert probe["train_plans"] == 4 and probe["test_limit"] == 4
    assert all(u["steps"] == 2 for u in plan["units"] if u["kind"] == "train")


def test_expansion_evaluates_only_trained_arms_best_and_last_at_eval_ks():
    plan = plan_module.build(fake_supports(), {}, smoke=False)
    prior = {"train-GB-lam1": {"train-GB-lam1-seed218": {"result": {"checkpoints": {"best": "x", "last": "y"}}}},
             "train-GBL-lam1": {"train-GBL-lam1-seed218": {"result": {}}}}
    units = plan_module.expand_gated_reader(plan, prior)
    readers = units[0]["readers"]
    assert set(readers) == {"GB-lam1-seed218_best", "GB-lam1-seed218_last"}
    assert readers["GB-lam1-seed218_best"] == "side:readers/gemma2/GB-lam1-seed218/best.pt"
    assert {u["suite"] for u in units} == {"selection", "distractor_curves-dev"} and all(not u["include_native"] for u in units)
    selection = next(u for u in units if u["suite"] == "selection")
    assert {b["k"] for banks in selection["banks"].values() for b in banks} == {1, 8}
    assert selection["cost"] == 1.0 * len(selection["sources"]) * 2 / 5
    full = plan_module.build(fake_supports(), {}, smoke=False, full=True)
    full_units = plan_module.expand_gated_reader(full, prior)
    assert {b["k"] for u in full_units if u["suite"] == "selection" for banks in u["banks"].values() for b in banks} == {1, 2, 4, 8, 16}
    assert plan_module.expand_gated_reader(plan, {}) == []


def test_lam_tags_are_short():
    assert plan_module.lam_tag(0.0) == "0" and plan_module.lam_tag(1.0) == "1" and plan_module.lam_tag(0.3) == "0.3"


def test_runner_reader_directory_matches_plan_reader_path():
    """An earlier LAYER_INTERVENTIONS run failed every evaluation unit: the runner wrote ``GB-lam1.0-seed218`` and the plan
    looked for ``GB-lam1-seed218``."""
    import layer_interventions_runner

    for arm, lam in plan_module.FULL_ARMS:
        unit = {"arm": arm, "lam": lam, "seed": plan_module.SEED}
        assert plan_module.reader_path(arm, lam, "best").endswith(f"/{layer_interventions_runner.reader_name(unit)}/best.pt")


def test_probe_only_plan_has_one_stage_and_one_unit():
    plan = plan_module.build(fake_supports(), {}, smoke=False, probe_only=True)
    validate_plan(plan)
    assert [s["name"] for s in plan["stages"]] == ["diagnostics"]
    assert [u["kind"] for u in plan["units"]] == ["probe"] and plan["units"][0]["test_ks"] == [8, 16]
    assert plan["units"][0]["id"] == "probe-selection"


def test_response_only_oracle_plan_has_response_only_oracles_and_prompt_probes():
    plan = plan_module.build_response_only_oracle(fake_supports(), {"selection": {"path": "x", "sha256": "y"}}, smoke=False)
    validate_plan(plan)
    assert plan["experiment"] == "RESPONSE_ONLY_ORACLE" and [s["name"] for s in plan["stages"]] == ["diagnostics"]
    oracles = [u for u in plan["units"] if u["kind"] == "oracle"]
    assert oracles and all(u["layer_from"] == [0] and u["ks"] == [8, 16] and u["position_modes"] == ["decode", "decode_last"] for u in oracles)
    probes = [u for u in plan["units"] if u["kind"] == "probe"]
    assert [u["query_mode"] for u in probes] == ["last_prompt", "prompt_mean"]
    assert [u["id"] for u in probes] == ["probe-selection-last-prompt", "probe-selection-prompt-mean"]
    assert all(u["priority"] == 10 and u["test_ks"] == [8, 16] for u in probes)
    smoke = plan_module.build_response_only_oracle(fake_supports(), {}, smoke=True)
    assert [u["id"] for u in smoke["units"] if u["kind"] == "probe"] == ["probe-selection-last-prompt"]


def test_attenuation_schedules_plan_has_schedule_units_and_clean_prefill_probes():
    plan = plan_module.build_attenuation_schedules(fake_supports(), {"selection": {"path": "x"}}, smoke=False)
    assert plan["experiment"] == "ATTENUATION_SCHEDULES" and [s["name"] for s in plan["stages"]] == ["diagnostics"]
    schedule = [u for u in plan["units"] if u["kind"] == "schedule"]
    probes = [u for u in plan["units"] if u["kind"] == "probe"]
    assert schedule and all(u["ks"] == [1, 8] and u["schedules"] == ["none", "A", "B", "C"] and u["layer_from"] == 8 for u in schedule)
    assert len(schedule) + len(probes) == len(plan["units"])
    assert sorted(p["id"] for p in probes) == ["probe-clean-last-prompt", "probe-clean-prompt-mean"]
    assert all(p["silent"] and p["test_ks"] == [8] and p["key_types"] == ["b_rows", "latent", "abs_strength"] for p in probes)
    assert {p["query_mode"] for p in probes} == {"last_prompt", "prompt_mean"}
    smoke = plan_module.build_attenuation_schedules(fake_supports(), {"selection": {"path": "x"}}, smoke=True)
    assert [u for u in smoke["units"] if u["kind"] == "probe"][0]["train_plans"] == 4


def test_layer_six_ceiling_plan_runs_schedule_b_from_layer_six():
    plan = plan_module.build_layer_six_ceiling(fake_supports(), {"selection": {"path": "x"}}, smoke=False)
    assert [s["name"] for s in plan["stages"]] == ["ceiling"]
    assert plan["units"] and all(u["kind"] == "schedule" and u["schedules"] == ["B"] and u["layer_from"] == 6
                                 and u["ks"] == [1, 8] for u in plan["units"])


def test_predicted_gates_plan_fits_probes_then_runs_predicted_arms():
    plan = plan_module.build_predicted_gates(fake_supports(), {"selection": {"path": "x"}}, smoke=False, schedules=("A", "C"),
                                       scorers=("b_rows:8:last_prompt", "abs_strength:6:prompt_mean"), modes=("soft", "hard"))
    assert [s["name"] for s in plan["stages"]] == ["probe-fit", "predicted"] and plan["stages"][1]["after"] == ["probe-fit"]
    fits = [u for u in plan["units"] if u["kind"] == "probe-fit"]
    assert [u["id"] for u in fits] == ["probe-fit-b_rows-L8-last_prompt"] and fits[0]["silent"] and fits[0]["key_types"] == ["b_rows", "latent"]
    runs = [u for u in plan["units"] if u["kind"] == "predicted"]
    assert runs and all(u["ks"] == [1, 8] and len(u["arms"]) == 8 for u in runs)
    names = {a["name"] for a in runs[0]["arms"]}
    assert "A-b_rows-L8-last_prompt-soft" in names and "C-abs_strength-L6-prompt_mean-hard" in names
    only_strength = plan_module.build_predicted_gates(fake_supports(), {"selection": {"path": "x"}}, smoke=False, schedules=("A",),
                                                scorers=("abs_strength:6:prompt_mean",))
    assert [s["name"] for s in only_strength["stages"]] == ["predicted"]


def test_predicted_gates_schedule_b_arms_follow_gate_layers():
    plan = plan_module.build_predicted_gates(fake_supports(), {"selection": {"path": "x"}}, smoke=False, schedules=("B",),
                                       scorers=("b_rows:6:prompt_mean", "b_rows:8:prompt_mean"), modes=("soft", "hard"),
                                       pools=("rank", "document"), b_layers=(6, 8))
    runs = [u for u in plan["units"] if u["kind"] == "predicted"]
    names = sorted(a["name"] for a in runs[0]["arms"])
    assert len(names) == 12 and "B6-b_rows-L6-prompt_mean-soft" in names and "B8-b_rows-L6-prompt_mean-hard-document" in names
    assert not any(n.startswith("B6-b_rows-L8") for n in names)
    assert all(a["layer_from"] in (6, 8) for a in runs[0]["arms"])
    assert len([u for u in plan["units"] if u["kind"] == "probe-fit"]) == 2


def fake_supports_with_secondary(sources: int = 7, secondary: int = 6):
    supports = fake_supports(sources)
    ids = [f"qa_protocol-doc-{i:024x}" for i in range(100, 100 + secondary)]
    banks = {s: [{"bank_id": f"{s}:random:{k}", "condition": "random", "k": k,
                  "document_ids": [s, *[d for d in ids if d != s][: k - 1]], "source_position": 0} for k in (1, 2, 4, 8, 16)]
             for s in ids}
    supports["secondary"] = {"sources": ids, "banks": banks}
    return supports


def test_layer_replication_plan_runs_one_stage_per_model_with_cells_from_the_grid():
    suites = {"selection": {"path": "x", "sha256": "y"}, "secondary": {"path": "z", "sha256": "w"}}
    plan = plan_module.build_layer_replication(fake_supports_with_secondary(), suites, smoke=False)
    validate_plan(plan)
    assert [s["name"] for s in plan["stages"]] == ["gemma", "qwen"] and plan_module.serial_order(plan) == "gemma,qwen"
    assert plan["experiment"] == "LAYER_REPLICATION" and plan["config"]["variant"] == "layer_replication"
    by_stage = {name: [u for u in plan["units"] if u["stage"] == name] for name in ("gemma", "qwen")}
    assert all(u["model"] == "gemma2" for u in by_stage["gemma"]) and all(u["model"] == "qwen4" for u in by_stage["qwen"])
    # Gemma selection: completes the sweep (no base rows, no probes); Gemma secondary: three-point curves + probes
    gsel = [u for u in by_stage["gemma"] if u["suite"] == "selection"]
    assert {u["kind"] for u in gsel} == {"oracle", "schedule"}
    oracle = next(u for u in gsel if u["kind"] == "oracle")
    assert oracle["layer_from"] == [4, 13, 22] and oracle["ks"] == [8, 16]
    assert sorted(u["layer_from"] for u in gsel if u["kind"] == "schedule") == [4, 4, 13, 13]
    assert all(u["schedules"] == ["B"] and u["ks"] == [8] for u in gsel if u["kind"] == "schedule")
    gsec = [u for u in by_stage["gemma"] if u["suite"] == "secondary"]
    assert next(u for u in gsec if u["kind"] == "oracle")["layer_from"] == [0, 8, 18]
    assert sorted(u["id"] for u in gsec if u["kind"] == "probe") == ["probe-secondary-gemma2-clean", "probe-secondary-gemma2-response"]
    clean = next(u for u in gsec if u["id"] == "probe-secondary-gemma2-clean")
    assert clean["silent"] is True and clean["query_mode"] == "prompt_mean" and clean["test_ks"] == [8] and clean["priority"] == 10
    assert len(clean["banks"]) == 6
    # Qwen: its own ICL/native/normalized rows at K in {1, 8}, layers scaled to 36
    qwen = by_stage["qwen"]
    base = [u for u in qwen if u["kind"] == "eval"]
    assert base and base[0]["id"] == "selection-base-qwen4-000" and base[0]["betas"] == [0.5] and base[0]["include_icl"]
    assert all(b["k"] in (1, 8) for u in base for banks in u["banks"].values() for b in banks)
    assert next(u for u in qwen if u["kind"] == "oracle")["layer_from"] == [0, 11, 25]
    assert sorted({u["layer_from"] for u in qwen if u["kind"] == "schedule"}) == [8, 11]
    probes = [u for u in qwen if u["kind"] == "probe"]
    assert len(probes) == 2 and all(u["max_document_occurrences"] == 6 for u in probes)
    assert len({u["id"] for u in plan["units"]}) == len(plan["units"])


def test_layer_replication_smoke_plan_shrinks_every_cell():
    suites = {"selection": {"path": "x", "sha256": "y"}, "secondary": {"path": "z", "sha256": "w"}}
    plan = plan_module.build_layer_replication(fake_supports_with_secondary(), suites, smoke=True)
    validate_plan(plan)
    oracles = [u for u in plan["units"] if u["kind"] == "oracle"]
    assert all(len(u["layer_from"]) == 2 and u["ks"] == [4] and len(u["sources"]) == 1 for u in oracles)
    assert len([u for u in plan["units"] if u["kind"] == "probe"]) == 2
    assert all(u["train_plans"] == 4 for u in plan["units"] if u["kind"] == "probe")
