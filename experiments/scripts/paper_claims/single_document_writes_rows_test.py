from single_document_writes_rows import split_rows


def test_validation_documents_held_out_whole_and_gold_only():
    rows = [{"family": "squad", "document_id": f"d{i % 10}", "row_id": f"r{i}", "question": "q", "answers": ["a"],
             "kind": "gold" if (i // 10) % 2 else "synthetic"} for i in range(100)]
    rows.append({"family": "drop", "document_id": "x", "row_id": "x", "question": "q", "answers": [], "kind": "gold"})
    data = split_rows(rows, 2)
    train_docs = {r["document"]["id"] for r in data["train"]}
    validation_docs = {r["document"]["id"] for r in data["validation"]}
    assert len(validation_docs) == 2 and not train_docs & validation_docs and "x" not in train_docs
    assert all(r["query"]["kind"] == "gold" for r in data["validation"])
    assert len(data["train"]) == 80 and len(data["validation"]) == 10
