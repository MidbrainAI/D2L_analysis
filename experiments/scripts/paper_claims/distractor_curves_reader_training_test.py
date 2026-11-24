from collections import Counter

from distractor_curves_core import K_VALUES
from distractor_curves_reader_training import bank_for, k_schedule, microbatches, update_plans, validation_plans


def rows(count: int) -> list[dict]:
    return [{"document": {"id": f"doc{i % 50}"}, "query": {"id": f"q{i}", "answers": ["a"], "dataset": "squad"}}
            for i in range(count)]


def test_k_schedule_is_balanced_and_seeded():
    schedule = k_schedule(100, 7)
    assert Counter(schedule) == {k: 20 for k in K_VALUES}
    assert schedule == k_schedule(100, 7) and schedule != k_schedule(100, 8)


def test_banks_hold_the_source_once_with_distinct_distractors():
    pool = [f"doc{i}" for i in range(50)]
    for row in rows(30):
        for k in K_VALUES:
            bank = bank_for(row, k, pool, 3)
            assert len(bank) == k == len(set(bank))
            assert bank.count(row["document"]["id"]) == 1


def test_updates_are_k_homogeneous_and_microbatches_respect_the_budget():
    pool = [f"doc{i}" for i in range(50)]
    updates = update_plans(rows(12 * 10), 10, 5, pool)
    assert len(updates) == 10 and all(len(u) == 12 and len({p["k"] for p in u}) == 1 for u in updates)
    seen = [p["query"]["id"] for u in updates for p in u]
    assert len(seen) == len(set(seen))
    for update in updates:
        for batch in microbatches(update, 12):
            assert len(batch) * update[0]["k"] <= max(12, update[0]["k"])


def test_validation_plans_cycle_k():
    plans = validation_plans(rows(10), [f"doc{i}" for i in range(50)])
    assert [p["k"] for p in plans] == list(K_VALUES) * 2
