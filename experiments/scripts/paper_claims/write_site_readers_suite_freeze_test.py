import json
from pathlib import Path

import pytest
from qa_protocol_protocol import digest
from write_site_readers_suite_freeze import (
    capped_questions,
    distractor_pool,
    freeze_exclusions,
    freeze_secondary,
    freeze_selection,
    hash_rank,
    held_out_contexts,
    nonempty_answers,
    selection_documents,
    source_query_id,
    touched_documents,
    whole_pool_threshold,
)

CONFIRMATION = {"suite": "s", "suite_sha256": "x", "banks_plan": "p", "plan_sha256": "y", "note": "n"}


def content_hash(label: str) -> str:
    return (label * 64)[:64]


def write_squad_jsonl(path: Path) -> None:
    # HF indices out of file order; contexts A (0,1,5), B (2,3), C (4), D (6) with a validation row interleaved.
    rows = [
        (3, "B"), (0, "A"), (2, "B"), (5, "A"), (4, "C"), (1, "A"), (6, "D"),
    ]
    lines = [json.dumps({"id": f"squad:train:{i}", "document_id": content_hash(c), "context": f"context {c}",
                         "question": "q", "answers": {"text": ["a"]}, "split": "train"}) for i, c in rows]
    lines.append(json.dumps({"id": "squad:validation:0", "document_id": content_hash("V"), "context": "context V",
                             "question": "q", "answers": {"text": ["a"]}, "split": "test"}))
    path.write_text("\n".join(lines) + "\n")


def selection_manifest(labels=("A", "B", "C"), indices=((0, 1, 5), (2, 3), (4,))) -> dict:
    documents, rows = [], []
    for label, group in zip(labels, indices, strict=True):
        identifier = "qa_protocol-doc-" + content_hash(label)[:24]
        documents.append({"id": identifier, "content_hash": content_hash(label), "family": "squad", "split": "train",
                          "text": f"context {label}", "source_tokens": 50})
        rows += [{"query_id": f"squad:squad:train:{i}", "document_id": identifier, "family": "squad", "split": "train",
                  "question": "q", "answers": ["a"]} for i in group]
    return {"manifest_sha256": "m", "documents": documents, "rows": rows}


def secondary_manifest() -> dict:
    documents, rows = [], []
    spec = {"drop": [("d0", (10, 950, 960)), ("d1", (1000, 1001, 1002, 1003)), ("d2", (2000, 2001))],
            "ropes": [("r0", (5,)), ("r1", (900, 901, 902, 903, 904, 905, 906)), ("r2", (1200,))]}
    for family, members in spec.items():
        for label, indices in members:
            identifier = f"qa_protocol-doc-{label}"
            documents.append({"id": identifier, "content_hash": content_hash(label), "family": family, "split": "test",
                              "text": f"passage {label}", "source_tokens": 80})
            rows += [{"query_id": f"{family}:{family}:validation:{i}", "document_id": identifier, "family": family,
                      "split": "test", "question": "q", "answers": [f"a{i}"]} for i in indices]
    # A non-test row and a SQuAD row must be ignored.
    rows.append({"query_id": "drop:drop:train:1", "document_id": "qa_protocol-doc-x", "family": "drop", "split": "train",
                 "question": "q", "answers": ["a"]})
    documents.append({"id": "qa_protocol-doc-x", "content_hash": content_hash("x"), "family": "drop", "split": "train",
                      "text": "t", "source_tokens": 1})
    return {"manifest_sha256": "m", "documents": documents, "rows": rows}


def answer_types_for(manifest: dict) -> dict:
    return {source_query_id(r["query_id"]): ["number"] if r["query_id"].endswith("0") else ["span"]
            for r in manifest["rows"] if r["family"] == "drop"}


def test_held_out_contexts_dedupes_on_context_in_hf_order(tmp_path):
    path = tmp_path / "squad.jsonl"
    write_squad_jsonl(path)
    held = held_out_contexts(path, count=3)
    assert [h["context"] for h in held] == ["context A", "context B", "context C"]
    assert [h["hf_train_index"] for h in held] == [0, 2, 4]
    assert held[0]["question_indices"] == [0, 1, 5]  # the late index-5 row still attaches to A
    assert held[0]["document_hash"] == content_hash("A")
    with pytest.raises(ValueError):
        held_out_contexts(path, count=10)


def test_selection_documents_maps_to_manifest_and_checks_questions(tmp_path):
    path = tmp_path / "squad.jsonl"
    write_squad_jsonl(path)
    held = held_out_contexts(path, count=3)
    entries = selection_documents(selection_manifest(), held)
    assert [e["document_id"] for e in entries] == ["qa_protocol-doc-" + content_hash(c)[:24] for c in "ABC"]
    assert entries[0]["questions"] == ["squad:squad:train:0", "squad:squad:train:1", "squad:squad:train:5"]
    assert all(e["role"] == "selection_source" for e in entries)
    missing = selection_manifest(labels=("A", "B"), indices=((0, 1, 5), (2, 3)))
    with pytest.raises(ValueError, match="not in the manifest"):
        selection_documents(missing, held)
    no_questions = selection_manifest(indices=((0, 1, 5), (2, 3), ()))
    with pytest.raises(ValueError, match="no manifest questions"):
        selection_documents(no_questions, held)
    wrong = selection_manifest(indices=((0, 1), (2, 3), (4,)))
    with pytest.raises(ValueError, match="differ from the source rows"):
        selection_documents(wrong, held)


def test_freeze_selection_records_held_out_ids_and_is_reproducible(tmp_path):
    path = tmp_path / "squad.jsonl"
    write_squad_jsonl(path)
    held = held_out_contexts(path, count=3)
    first = freeze_selection(selection_manifest(), held, CONFIRMATION, seed="s")
    second = freeze_selection(selection_manifest(), held, CONFIRMATION, seed="s")
    assert first["suite_sha256"] == second["suite_sha256"]
    body = {k: v for k, v in first.items() if k != "suite_sha256"}
    assert digest(body) == first["suite_sha256"]
    assert len(first["held_out_document_ids"]) == 3 and len(first["held_out_query_ids"]) == 6
    assert first["confirmation"] == CONFIRMATION and first["manifest_splits"] == {"train": 3}


def test_secondary_excludes_selection_and_labels_untouched():
    manifest = secondary_manifest()
    total = 6
    scan = {"whole": {"docs": [d["id"] for d in manifest["documents"] if d["split"] == "test"], "n_docs": total},
            "targeted": {"docs": ["qa_protocol-doc-d1", "qa_protocol-doc-r2"], "n_docs": 2},
            "empty": {"docs": [], "n_docs": 0}}
    suite = freeze_secondary(manifest, scan, answer_types_for(manifest), seed="s")
    ids = {e["document_id"]: e for e in suite["documents"]}
    assert set(ids) == {"qa_protocol-doc-d1", "qa_protocol-doc-d2", "qa_protocol-doc-r1", "qa_protocol-doc-r2"}  # d0 (10) and r0 (5) are < 900
    assert suite["excluded"]["d2l_selection_documents"] == ["qa_protocol-doc-d0", "qa_protocol-doc-r0"]
    assert ids["qa_protocol-doc-d1"]["untouched"] is False and ids["qa_protocol-doc-d2"]["untouched"] is True
    assert ids["qa_protocol-doc-r2"]["untouched"] is False and ids["qa_protocol-doc-r1"]["untouched"] is True
    assert suite["scan_sources"] == {"whole_pool": {"whole": total}, "targeted": {"targeted": 2}}
    assert suite["families"]["drop"]["untouched_documents"] == 1 and suite["families"]["ropes"]["untouched_documents"] == 1
    assert suite["families"]["drop"]["d2l_selection_excluded"] == 1
    assert ids["qa_protocol-doc-d1"]["role"] == "drop_source" and ids["qa_protocol-doc-r1"]["role"] == "ropes_source"


def test_question_caps_are_hash_ordered_and_deterministic():
    manifest = secondary_manifest()
    scan = {}
    suite = freeze_secondary(manifest, scan, answer_types_for(manifest), seed="s")
    again = freeze_secondary(manifest, scan, answer_types_for(manifest), seed="s")
    assert suite["suite_sha256"] == again["suite_sha256"]
    ids = {e["document_id"]: e for e in suite["documents"]}
    d1 = ids["qa_protocol-doc-d1"]
    assert len(d1["questions"]) == 3 and d1["questions_uncapped"] == 4
    all_ids = [f"drop:drop:validation:{i}" for i in (1000, 1001, 1002, 1003)]
    assert d1["questions"] == sorted(all_ids, key=lambda q: hash_rank("s", q))[:3]
    assert capped_questions("s", all_ids, 3) == d1["questions"]
    assert capped_questions("other", all_ids, 3) != d1["questions"] or hash_rank("s", "x") != hash_rank("other", "x")
    r1 = ids["qa_protocol-doc-r1"]
    assert len(r1["questions"]) == 5 and r1["questions_uncapped"] == 7
    assert set(d1["answer_types"]) == set(d1["questions"])
    assert suite["families"]["drop"]["answer_type_counts_uncapped"] == {"number": 2, "span": 4}


def test_distractor_pools_are_same_family():
    manifest = secondary_manifest()
    suite = freeze_secondary(manifest, {}, answer_types_for(manifest), seed="s")
    assert distractor_pool(suite, "qa_protocol-doc-d1") == ["qa_protocol-doc-d2"]
    assert distractor_pool(suite, "qa_protocol-doc-r1") == ["qa_protocol-doc-r2"]
    for entry in suite["documents"]:
        pool = distractor_pool(suite, entry["document_id"])
        assert entry["document_id"] not in pool
        assert all(entry["family"] == next(e["family"] for e in suite["documents"] if e["document_id"] == p) for p in pool)


def test_answers_must_be_nonempty():
    nonempty_answers({"query_id": "q", "answers": ["x"]})
    nonempty_answers({"query_id": "q", "answers": {"text": ["x"]}})
    for answers in ([], [""], {"text": []}, {"spans": [" "]}):
        with pytest.raises(ValueError, match="empty answers"):
            nonempty_answers({"query_id": "q", "answers": answers})
    manifest = secondary_manifest()
    manifest["rows"][3]["answers"] = [""]  # a d1 row; d0 is selection-excluded and never checked
    with pytest.raises(ValueError, match="empty answers"):
        freeze_secondary(manifest, {}, answer_types_for(manifest), seed="s")


def test_touched_documents_and_threshold_accept_both_scan_formats():
    assert whole_pool_threshold(180) == 162 and whole_pool_threshold(782) == 704
    listed = {"whole": ["a", "b", "c"], "one": ["b"], "none": []}
    touched, whole, targeted = touched_documents(listed, 3)
    assert touched == {"b"} and whole == {"whole": 3} and targeted == {"one": 1}
    nested = {"whole": {"docs": ["a", "b", "c"]}, "one": {"docs": ["c"]}}
    assert touched_documents(nested, 3)[0] == {"c"}


def test_exclusions_file_discloses_both_pools(tmp_path):
    path = tmp_path / "squad.jsonl"
    write_squad_jsonl(path)
    held = held_out_contexts(path, count=3)
    scan180 = {"copy": [content_hash(c)[:24] for c in "ABC"], "relational_readout": [content_hash("B")[:24]]}
    manifest = secondary_manifest()
    scan_dr = {"targeted": {"docs": ["qa_protocol-doc-d1"], "n_docs": 1}}
    exclusions = freeze_exclusions(manifest, held, scan180, scan_dr)
    prior = exclusions["selection_prior_references"]
    assert prior["whole_pool_sources"] == {"copy": 3} and prior["targeted_sources"] == {"relational_readout": 1}
    assert prior["referenced_documents"] == ["qa_protocol-doc-" + content_hash("B")[:24]]
    dr = exclusions["drop_ropes"]
    assert dr["d2l_selection_documents"] == ["qa_protocol-doc-d0", "qa_protocol-doc-r0"]
    assert dr["touched_documents"] == ["qa_protocol-doc-d1"]
    assert dr["untouched_documents"] == ["qa_protocol-doc-d2", "qa_protocol-doc-r1", "qa_protocol-doc-r2"]
    assert exclusions["scan180_sha256"] == digest(scan180)
    assert exclusions["suite_sha256"] == digest({k: v for k, v in exclusions.items() if k != "suite_sha256"})


def test_source_query_id_strips_doubled_prefix():
    assert source_query_id("drop:drop:validation:3846") == "drop:validation:3846"
    assert source_query_id("ropes:ropes:validation:1") == "ropes:validation:1"
    assert source_query_id("drop:validation:3") == "drop:validation:3"
