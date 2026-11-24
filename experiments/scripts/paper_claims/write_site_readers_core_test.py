import json

from distractor_curves_core import K_VALUES
from write_site_readers_core import distractor_curves_banks, load_write_site_readers_suite, random_banks, sources


def manifest(n: int):
    docs = [{"id": f"qa_protocol-doc-{i:024d}", "text": f"text {i}"} for i in range(n)]
    rows = [{"query_id": f"q{i}", "question": f"why {i}?", "answers": [f"a{i}"]} for i in range(n)]
    return {"documents": docs, "rows": rows}


def write_suite(tmp_path, n: int, family: str | None):
    items = []
    for i in range(n):
        item = {"document_id": f"qa_protocol-doc-{i:024d}", "role": f"{family or 'selection'}_source", "questions": [f"q{i}"]}
        if family:
            item.update({"family": family, "untouched": i % 2 == 0})
        items.append(item)
    path = tmp_path / "suite.json"
    path.write_text(json.dumps({"suite_sha256": "x", "schema": "write_site_readers", "documents": items}))
    return path


def test_suite_loads_with_family_as_prompt_dataset(tmp_path):
    suite = load_write_site_readers_suite(write_suite(tmp_path, 20, "drop"), manifest(20))
    doc = suite["documents"]["qa_protocol-doc-" + "0" * 24]
    assert doc["family"] == "drop" and doc["untouched"] is True and doc["text"] == "text 0"
    question = suite["questions"][doc["id"]][0]
    assert question["dataset"] == "drop" and question["answers"] == ("a0",) and question["input"] == "why 0?"
    assert sources(suite, "drop_source") == sorted(suite["documents"])
    plain = load_write_site_readers_suite(write_suite(tmp_path, 20, None), manifest(20))
    assert all(q["dataset"] == "squad" for qs in plain["questions"].values() for q in qs)


def test_random_banks_are_nested_same_role_and_hold_the_source_once(tmp_path):
    suite = load_write_site_readers_suite(write_suite(tmp_path, 20, "ropes"), manifest(20))
    source = "qa_protocol-doc-" + "0" * 23 + "3"
    banks = random_banks(suite, source)
    assert [b["k"] for b in banks] == list(K_VALUES)
    previous: set[str] = set()
    for bank in banks:
        ids = bank["document_ids"]
        assert len(ids) == bank["k"] == len(set(ids)) and ids.count(source) == 1
        assert ids[bank["source_position"]] == source
        distractors = set(ids) - {source}
        assert previous <= distractors  # nested
        previous = distractors
        assert all(suite["documents"][i]["role"] == "ropes_source" for i in ids)
    assert random_banks(suite, source) == banks  # deterministic


def test_distractor_curves_banks_filter_conditions():
    plan = {"config": {"banks": {"dev": {"s": [{"condition": "random", "k": 1}, {"condition": "copies", "k": 2},
                                              {"condition": "same_article", "k": 8}]}}}}
    assert distractor_curves_banks(plan, "dev") == {"s": [{"condition": "random", "k": 1}]}
    assert len(distractor_curves_banks(plan, "dev", ("random", "same_article"))["s"]) == 2
