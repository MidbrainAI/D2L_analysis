from question_conditioned_writes_plan import build_plan, choose_checkpoint, expand_dev, expand_test


def fake_suite():
    documents = {f"d{i}": {"role": role, "article": "a", "text": "t"}
                 for i, role in enumerate(["dev_source"] * 2 + ["dev_distractor"] * 20 + ["test_source"] * 2
                                          + ["test_distractor"] * 20)}
    return {"documents": documents, "suite_sha256": "s",
            "questions": {d: [{"id": f"{d}q"}] for d, v in documents.items() if v["role"].endswith("source")}}


def test_plan_stages_and_units(monkeypatch):
    import question_conditioned_writes_plan

    monkeypatch.setattr(question_conditioned_writes_plan, "sources", lambda suite, p: sorted(d for d, v in suite["documents"].items()
                                                                      if v["role"] == f"{p}_source"))
    monkeypatch.setattr(question_conditioned_writes_plan, "source_banks", lambda suite, s, p, seed: [
        {"bank_id": f"{s}-{c}-{k}", "condition": c, "k": k, "document_ids": [s] * k}
        for c in ("random", "same_article", "copies") for k in (1, 2, 4, 8, 16)])
    from experiment_plan import validate_plan

    plan = build_plan(suite_path="s.json", suite=fake_suite(), query_ids=[f"q{i}" for i in range(4100)],
                      latents="artifacts/x")
    validate_plan(plan)
    assert [s["name"] for s in plan["stages"]] == ["question_conditioned_writes-gates", "question_conditioned_writes-teacher", "question_conditioned_writes-train", "question_conditioned_writes-dev", "question_conditioned_writes-test"]
    assert sum(len(u["query_ids"]) for u in plan["units"] if u["kind"] == "teacher") == 4100
    dev = expand_dev(plan, {})
    assert set(dev[0]["encoders"]) == {"question_conditioned_writes_best", "question_conditioned_writes_last"} and "include_removal" not in dev[0]
    assert {b["condition"] for banks in dev[0]["banks"].values() for b in banks} == {"random"}


def test_choose_checkpoint_macro_over_k():
    rows = [{"condition": "random", "k": k, "scores": {"question_conditioned_writes_best": {"f1": 0.5}, "question_conditioned_writes_last": {"f1": 0.6 if k == 1 else 0.5},
                                                      "native": {"f1": 0.4}}} for k in (1, 2)]
    assert choose_checkpoint(rows)["checkpoint"] == "last"
    tie = [{"condition": "random", "k": 1, "scores": {"question_conditioned_writes_best": {"f1": 0.5}, "question_conditioned_writes_last": {"f1": 0.5}, "native": {"f1": 0}}}]
    assert choose_checkpoint(tie)["checkpoint"] == "best"


def test_expand_test_uses_selected_checkpoint(monkeypatch):
    import question_conditioned_writes_plan

    monkeypatch.setattr(question_conditioned_writes_plan, "sources", lambda suite, p: sorted(d for d, v in suite["documents"].items()
                                                                      if v["role"] == f"{p}_source"))
    monkeypatch.setattr(question_conditioned_writes_plan, "source_banks", lambda suite, s, p, seed: [
        {"bank_id": f"{s}-random-{k}", "condition": "random", "k": k, "document_ids": [s] * k} for k in (1, 8)])
    plan = build_plan(suite_path="s.json", suite=fake_suite(), query_ids=["q"], latents="x")
    rows = [{"condition": "random", "k": 1, "scores": {"question_conditioned_writes_best": {"f1": 0.2}, "question_conditioned_writes_last": {"f1": 0.4},
                                                       "native": {"f1": 0.1}}}]
    units = expand_test(plan, {"question_conditioned_writes-dev": {"u": {"result": {"sources": [{"rows": rows}]}}}})
    assert units[0]["encoders"] == {"question_conditioned_writes": "encoders/seed210/last.pt"}
    assert units[0]["beta"] == 0.5 and "include_removal" not in units[0]


def test_k1_plan_skips_teacher_and_evaluates_k1(monkeypatch):
    import question_conditioned_writes_plan

    monkeypatch.setattr(question_conditioned_writes_plan, "sources", lambda suite, p: sorted(d for d, v in suite["documents"].items()
                                                                      if v["role"] == f"{p}_source"))
    monkeypatch.setattr(question_conditioned_writes_plan, "source_banks", lambda suite, s, p, seed: [
        {"bank_id": f"{s}-{c}-{k}", "condition": c, "k": k, "document_ids": [s] * k}
        for c in ("random", "same_article") for k in (1, 2, 4, 8, 16)])
    plan = build_plan(suite_path="s.json", suite=fake_suite(), query_ids=["q"], latents="x", k_values=[1],
                      teacher_dir="artifacts/t", experiment="SINGLE_DOCUMENT_WRITES")
    assert [s["name"] for s in plan["stages"]] == ["question_conditioned_writes-gates", "question_conditioned_writes-train", "question_conditioned_writes-dev", "question_conditioned_writes-test"]
    assert plan["config"]["train"]["k_values"] == [1] and not any(u["kind"] == "teacher" for u in plan["units"])
    gate = next(u for u in plan["units"] if u["kind"] == "gates")
    assert len(gate["document_ids"]) == 1
    rows = [{"condition": "random", "k": 1, "scores": {"question_conditioned_writes_best": {"f1": 0.5}, "question_conditioned_writes_last": {"f1": 0.4},
                                                       "native": {"f1": 0.4}}}]
    test = expand_test(plan, {"question_conditioned_writes-dev": {"u": {"result": {"sources": [{"rows": rows}]}}}})
    banks = [b for unit in test for bs in unit["banks"].values() for b in bs]
    assert {b["k"] for b in banks} == {1} and {b["condition"] for b in banks} == {"random"}


def test_rows_plan_adds_latents_and_one_pass(monkeypatch):
    import question_conditioned_writes_plan

    monkeypatch.setattr(question_conditioned_writes_plan, "sources", lambda suite, p: sorted(d for d, v in suite["documents"].items()
                                                                      if v["role"] == f"{p}_source"))
    monkeypatch.setattr(question_conditioned_writes_plan, "source_banks", lambda suite, s, p, seed: [
        {"bank_id": f"{s}-random-1", "condition": "random", "k": 1, "document_ids": [s]}])
    plan = build_plan(suite_path="s.json", suite=fake_suite(), query_ids=[f"q{i}" for i in range(650)],
                      latents="unused", k_values=[1], experiment="SINGLE_DOCUMENT_WRITES", rows="inputs/single_document_writes/rows.json",
                      document_ids=[f"d{i}" for i in range(1200)], train_questions=640, batch_size=64,
                      learning_rate=2e-4, validation_interval=250)
    assert [s["name"] for s in plan["stages"]] == ["question_conditioned_writes-gates", "question_conditioned_writes-latents", "question_conditioned_writes-teacher", "question_conditioned_writes-train",
                                                   "question_conditioned_writes-dev", "question_conditioned_writes-test"]
    assert plan["config"]["latents"] is None and plan["config"]["train"]["steps"] == 10
    assert plan["config"]["train"]["batch_size"] == 64
    assert sum(len(u["document_ids"]) for u in plan["units"] if u["kind"] == "latents") == 1200
    assert next(s for s in plan["stages"] if s["name"] == "question_conditioned_writes-train")["after"] == ["question_conditioned_writes-latents", "question_conditioned_writes-teacher"]


def test_contrastive_plan_reuses_latents_and_teacher(monkeypatch):
    import question_conditioned_writes_plan

    monkeypatch.setattr(question_conditioned_writes_plan, "sources", lambda suite, p: sorted(d for d, v in suite["documents"].items()
                                                                      if v["role"] == f"{p}_source"))
    monkeypatch.setattr(question_conditioned_writes_plan, "source_banks", lambda suite, s, p, seed: [
        {"bank_id": f"{s}-random-1", "condition": "random", "k": 1, "document_ids": [s]}])
    plan = build_plan(suite_path="s.json", suite=fake_suite(), query_ids=["q"], latents="artifacts/single_document_writes-latents",
                      k_values=[1], experiment="CONTRASTIVE_WRITES", rows="inputs/single_document_writes/rows.json", document_ids=["d0"],
                      train_questions=640, batch_size=64, learning_rate=2e-4, teacher_dir="artifacts/single_document_writes-teacher",
                      compile_latents=False, contrastive_weight=1.0, margin=0.1)
    assert [s["name"] for s in plan["stages"]] == ["question_conditioned_writes-gates", "question_conditioned_writes-train", "question_conditioned_writes-dev", "question_conditioned_writes-test"]
    assert plan["config"]["latents"] == "artifacts/single_document_writes-latents"
    assert plan["config"]["train"]["contrastive_weight"] == 1.0 and plan["config"]["train"]["margin"] == 0.1
