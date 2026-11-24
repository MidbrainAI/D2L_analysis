from __future__ import annotations

import hashlib
import json

import pytest
from qa_protocol_protocol import digest
from passage_fit_suite_suite_freeze import (
    answer_is_passage_span,
    freeze_suite,
    interleaved_order,
    quartile_cuts,
    scan_references,
)


def _document(index: int, split: str, tokens: int, family: str = "squad") -> dict:
    content = hashlib.sha256(str(index).encode()).hexdigest()
    words = " ".join(f"word{index}x{j}" for j in range(30))
    return {"id": f"qa_protocol-doc-{content[:24]}", "content_hash": content, "family": family, "split": split,
            "source_tokens": tokens, "text": f"Passage {index} about Paris and 1999. {words}"}


def _manifest(n_validation: int = 60, n_train: int = 40) -> dict:
    documents = [_document(i, "validation", 100 + i) for i in range(n_validation)]
    documents += [_document(1000 + i, "train", 100 + i) for i in range(n_train)]
    documents.append(_document(5000, "test", 120))
    documents.append(_document(6000, "validation", 130, family="drop"))
    rows = []
    for document in documents:
        count = 4 if document["source_tokens"] == 103 else 5
        for q in range(count):
            rows.append({"query_id": f"{document['family']}:{document['id']}:{q}", "document_id": document["id"],
                         "split": document["split"], "family": document["family"],
                         "answers": ["Paris"] if q % 2 else ["1999"]})
    manifest = {"documents": documents, "rows": rows}
    manifest["manifest_sha256"] = digest(manifest)
    return manifest


def _exact402(manifest: dict) -> dict:
    squad = [d for d in manifest["documents"] if d["family"] == "squad" and d["split"] == "validation"][:2]
    drop = next(d for d in manifest["documents"] if d["family"] == "drop")
    rows = [{"query_id": f"squad:{d['id']}:0", "document_id": d["id"], "dataset": "squad"} for d in squad]
    rows += [{"query_id": f"drop:{drop['id']}:{q}", "document_id": drop["id"], "dataset": "drop"} for q in (0, 1)]
    return {"rows": rows}


def _exclusions(manifest: dict, excluded: list[str]) -> dict:
    return {"manifest_sha256": manifest["manifest_sha256"], "targeted": {"x": excluded}, "excluded_documents": excluded}


def test_freeze_is_nested_disjoint_and_deterministic():
    manifest = _manifest()
    exact = _exact402(manifest)
    excluded = [row["document_id"] for row in exact["rows"]]
    kwargs = dict(seed="s", a2_final=8, a3_size=10, reserve=10, dev_pool=8, a2_dev=4)
    suite = freeze_suite(manifest, _exclusions(manifest, excluded), exact, **kwargs)
    assert suite == freeze_suite(manifest, _exclusions(manifest, excluded), exact, **kwargs)
    roles = [e["role"] for e in suite["suite"]]
    assert roles == ["a2_final"] * 8 + ["a3_only"] * 2 + ["reserve"] * 10
    suite_ids = {e["document_id"] for e in suite["suite"]}
    dev_ids = {e["document_id"] for e in suite["development"]}
    assert not suite_ids & dev_ids and not suite_ids & set(excluded)
    assert all(len(e["questions"]) >= 5 for e in suite["suite"] + suite["development"])
    by_id = {d["id"]: d for d in manifest["documents"]}
    assert {by_id[i]["split"] for i in suite_ids} == {"validation"}
    assert {by_id[i]["split"] for i in dev_ids} == {"train"}
    assert suite["test_split_read"] is False
    body = {k: v for k, v in suite.items() if k != "suite_sha256"}
    assert suite["suite_sha256"] == digest(body)
    assert freeze_suite(manifest, _exclusions(manifest, excluded), exact, **{**kwargs, "seed": "t"})["suite"] != suite["suite"]


def test_interleaving_balances_every_prefix():
    documents = [_document(i, "validation", 100 + i) for i in range(40)]
    cuts = quartile_cuts([d["source_tokens"] for d in documents])
    ordered = interleaved_order(documents, "s", cuts)
    first = [sum(d["source_tokens"] >= c for c in cuts) for d in ordered[:8]]
    assert sorted(first) == [0, 0, 1, 1, 2, 2, 3, 3]


def test_freeze_rejects_foreign_exclusions_and_shortfall():
    manifest = _manifest()
    exact = _exact402(manifest)
    with pytest.raises(ValueError, match="different manifest"):
        freeze_suite(manifest, {"manifest_sha256": "x", "excluded_documents": []}, exact, seed="s")
    with pytest.raises(ValueError, match="not enough"):
        freeze_suite(manifest, _exclusions(manifest, []), exact, seed="s", a2_final=10, a3_size=50, reserve=50)


def test_continuity_splits_secondary_span_answers():
    manifest = _manifest()
    exact = _exact402(manifest)
    suite = freeze_suite(manifest, _exclusions(manifest, []), exact, seed="s", a2_final=4, a3_size=4,
                         reserve=4, dev_pool=4, a2_dev=2)
    secondary = suite["continuity"]["secondary"]["drop"]
    assert len(secondary["span"]) == 1 and len(secondary["other"]) == 1  # "Paris" span, "1999" numeric
    assert len(suite["continuity"]["exact402_squad_documents"]) == 2
    assert answer_is_passage_span(["Paris"], "in paris") and not answer_is_passage_span(["1999"], "1999")
    assert not answer_is_passage_span(["London"], "Paris")


def test_scan_finds_ids_hashes_and_text_and_separates_sweeps(tmp_path, monkeypatch):
    manifest = _manifest()
    docs = [d for d in manifest["documents"] if d["split"] == "validation" and d["family"] == "squad"]
    (tmp_path / "scanner_fixture").mkdir()
    (tmp_path / "scanner_fixture" / "a.json").write_text(json.dumps({"doc": docs[0]["id"]}))
    (tmp_path / "scanner_fixture" / "b.jsonl").write_text(json.dumps({"hash": docs[1]["content_hash"]}))
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "c.log").write_text("context: " + docs[2]["text"].upper())
    monkeypatch.setattr("passage_fit_suite_suite_freeze.BULK_SWEEP_DOCUMENTS", 50)
    sweep = tmp_path / "qa_protocol" / "qa_protocola-cross-family-local-v1"
    sweep.mkdir(parents=True)
    (sweep / "all.json").write_text(json.dumps([d["id"] for d in docs]))
    test_doc = next(d for d in manifest["documents"] if d["split"] == "test")
    (tmp_path / "scanner_fixture" / "t.json").write_text(json.dumps({"doc": test_doc["id"]}))
    result = scan_references(tmp_path, manifest)
    assert set(result["excluded_documents"]) == {docs[0]["id"], docs[1]["id"], docs[2]["id"]}
    assert result["bulk_sweeps"] == {"qa_protocol/qa_protocola-cross-family-local-v1": len(docs)}
