import pytest
from distractor_curves_core import K_VALUES
from write_site_readers_reader_training import (
    better,
    filtered_rows,
    fresh_subset,
    held_out_hashes,
    is_held_out,
    planned_steps,
    pools,
    terminator_id,
)


def payload():
    docs = [{"id": f"{i:02d}" + "a" * 62, "contenthash": f"{i:02d}" + "a" * 62, "dataset": "squad",
             "split": "train" if i < 6 else "validation"} for i in range(10)]
    return docs


def test_held_out_matching_uses_the_content_hash_prefix():
    docs = payload()
    selection = {"held_out_document_ids": ["qa_protocol-doc-" + docs[1]["contenthash"][:24],
                                           "qa_protocol-doc-" + docs[7]["contenthash"][:24]]}
    held = held_out_hashes(selection)
    assert [is_held_out(d, held) for d in docs] == [i in (1, 7) for i in range(10)]
    latents = {d["id"]: None for d in docs}
    pool = pools(latents, {d["id"]: d for d in docs}, held)
    assert docs[1]["id"] not in pool["train"] and docs[7]["id"] not in pool["validation"]
    assert len(pool["train"]) == 5 and len(pool["validation"]) == 3


def test_filtered_rows_drop_every_question_on_a_held_out_document(monkeypatch):
    docs = payload()
    rows = {"train": [{"document": {"id": docs[i % 6]["id"]}, "query": {"id": f"q{i}", "input": "why?"}}
                      for i in range(12)],
            "validation": [{"document": {"id": docs[6 + i % 4]["id"]}, "query": {"id": f"v{i}", "input": "why?"}}
                           for i in range(8)]}
    rows["train"].append({"document": {"id": docs[2]["id"]}, "query": {"id": "long", "input": " " * 3000 + "why?"}})
    monkeypatch.setattr("write_site_readers_reader_training.training_rows", lambda payload: rows)
    held = held_out_hashes({"held_out_document_ids": ["qa_protocol-doc-" + docs[0]["contenthash"][:24],
                                                      "qa_protocol-doc-" + docs[9]["contenthash"][:24]]})
    out = filtered_rows({"documents": docs}, held)
    assert len(out["train"]) == 10 and all(r["document"]["id"] != docs[0]["id"] for r in out["train"])
    assert all(r["query"]["id"] != "long" for r in out["train"])  # the over-long question is dropped
    assert len(out["validation"]) == 6 and all(r["document"]["id"] != docs[9]["id"] for r in out["validation"])


def test_fresh_subset_is_the_first_n_plans_per_k_in_order():
    plans = [{"k": K_VALUES[i % 5], "id": i} for i in range(50)]
    fresh = fresh_subset(plans, per_k=3)
    assert len(fresh) == 15
    assert [p["id"] for p in fresh if p["k"] == 1] == [0, 5, 10]


def test_better_prefers_higher_fresh_f1_and_keeps_the_earlier_tie():
    assert better({"fresh_macro_f1": 0.5}, None)
    assert better({"fresh_macro_f1": 0.6}, {"fresh_macro_f1": 0.5})
    assert not better({"fresh_macro_f1": 0.5}, {"fresh_macro_f1": 0.5})
    assert not better({"fresh_macro_f1": None}, {"fresh_macro_f1": 0.1})


def test_planned_steps_shrink_to_the_rows_that_remain():
    assert planned_steps(52_344, 4362) == 4362
    assert planned_steps(51_650, 4362) == 4304  # one pass over 12 per update
    assert planned_steps(30, 4362) == 2
    with pytest.raises(ValueError):
        planned_steps(5, 4362)


class _Tok:
    def __init__(self, vocab, eos):
        self._vocab, self.eos_token_id = vocab, eos

    def get_vocab(self):
        return self._vocab


def test_terminator_prefers_end_of_turn_and_falls_back_to_eos():
    assert terminator_id(_Tok({"<eos>": 1, "<end_of_turn>": 107}, 1)) == 107
    assert terminator_id(_Tok({"<|im_end|>": 151645}, 151645)) == 151645
    with pytest.raises(ValueError):
        terminator_id(_Tok({}, None))
