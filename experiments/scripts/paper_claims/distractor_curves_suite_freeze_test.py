import pytest
from distractor_curves_suite_freeze import freeze_suite, question_index


def toy_manifest(articles: int = 10, per_article: int = 20) -> tuple[dict, list[str]]:
    documents, rows, titles = [], [], []
    for a in range(articles):
        for d in range(per_article):
            identifier = f"qa_protocol-doc-{a:02d}{d:02d}"
            documents.append({"id": identifier, "family": "squad", "split": "test", "text": f"passage {a} {d}",
                              "content_hash": f"{a:02d}{d:02d}" * 8, "source_tokens": 100 + d})
            for _ in range(5 if d % 2 == 0 else 3):
                rows.append({"query_id": f"squad:squad:validation:{len(titles)}", "document_id": identifier,
                             "family": "squad", "split": "test", "question": "q", "answers": ["a"]})
                titles.append(f"Article_{a}")
    return {"manifest_sha256": "m", "documents": documents, "rows": rows}, titles


def test_question_index_parses_both_formats():
    assert question_index("squad:squad:validation:12") == 12
    assert question_index("squad:validation:3") == 3
    with pytest.raises(ValueError):
        question_index("squad:train:3")


def test_freeze_partitions_articles_and_respects_exclusions():
    manifest, titles = toy_manifest()
    excluded = ["qa_protocol-doc-0000", "qa_protocol-doc-0102"]
    exclusions = {"manifest_sha256": "m", "excluded_documents": excluded}
    suite = freeze_suite(manifest, exclusions, titles, seed="s", dev_sources=8, test_sources=20,
                         max_sources_per_article=4, min_same_article_distractors=7)
    roles = {}
    for item in suite["documents"]:
        roles.setdefault(item["role"], []).append(item)
    assert len(roles["dev_source"]) == 8 and len(roles["test_source"]) == 20
    assert not set(suite["articles"]["dev"]) & set(suite["articles"]["test"])
    assert not {i["document_id"] for i in suite["documents"]} & set(excluded)
    assert all(len(i["questions"]) >= 5 for i in roles["dev_source"] + roles["test_source"])
    for item in roles["test_source"]:
        same = [d for d in roles["test_distractor"] if d["article"] == item["article"]]
        assert len(same) >= 7


def test_freeze_refuses_a_foreign_exclusion_scan():
    manifest, titles = toy_manifest()
    with pytest.raises(ValueError):
        freeze_suite(manifest, {"manifest_sha256": "other", "excluded_documents": []}, titles, seed="s")
