from distractor_curves_core import K_VALUES, source_banks


def toy_suite(articles: int = 4, per_article: int = 12) -> dict:
    documents = {}
    for a in range(articles):
        for d in range(per_article):
            identifier = f"doc-{a}-{d}"
            role = "test_source" if d < 2 else "test_distractor"
            documents[identifier] = {"id": identifier, "text": f"text {identifier}", "article": f"article-{a}",
                                     "role": role}
    return {"documents": documents, "questions": {}}


def test_banks_are_nested_and_hold_the_source_once():
    suite = toy_suite()
    banks = {(b["condition"], b["k"]): b for b in source_banks(suite, "doc-0-0", "test", "seed")}
    previous: set[str] = set()
    for k in K_VALUES:
        bank = banks[("random", k)]
        assert len(bank["document_ids"]) == k
        assert bank["document_ids"].count("doc-0-0") == 1
        assert bank["document_ids"][bank["source_position"]] == "doc-0-0"
        distractors = set(bank["document_ids"]) - {"doc-0-0"}
        assert previous <= distractors
        assert all(suite["documents"][i]["article"] != "article-0" for i in distractors)
        previous = distractors


def test_banks_are_deterministic():
    suite = toy_suite()
    first = source_banks(suite, "doc-2-1", "test", "seed")
    assert first == source_banks(suite, "doc-2-1", "test", "seed")
    assert first != source_banks(suite, "doc-2-1", "test", "other-seed")
    assert {bank["condition"] for bank in first} == {"random"}
