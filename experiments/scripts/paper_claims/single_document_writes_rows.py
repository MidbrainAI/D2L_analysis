"""Freeze SINGLE_DOCUMENT_WRITES's training and validation questions from the self-study SQuAD rows (gold plus synthetic).

Every self-study SQuAD row lies in the QA_PROTOCOL train split, so none overlaps the DISTRACTOR_CURVES suite. A seeded set of
``--validation-documents`` passages is held out whole: their gold questions form the validation set and
their synthetic questions are dropped, so validation passages are never written during training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SEED = "single_document_writes-validation-v1"


def rank(document_id: str) -> str:
    return hashlib.sha256(f"{SEED}:{document_id}".encode()).hexdigest()


def split_rows(rows: list[dict], validation_documents: int) -> dict[str, list[dict]]:
    squad = [r for r in rows if r["family"] == "squad"]
    held = set(sorted({r["document_id"] for r in squad}, key=rank)[:validation_documents])
    def row(r):
        return {"query": {"id": r["row_id"], "input": r["question"], "dataset": "squad", "answers": r["answers"]
                          if "answers" in r else [], "kind": r["kind"]}, "document": {"id": r["document_id"]}}
    return {"train": [row(r) for r in squad if r["document_id"] not in held],
            "validation": [row(r) for r in squad if r["document_id"] in held and r["kind"] == "gold"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path,
                        default=Path("artifacts/qa_protocol/self-study/row-plan.json"))
    parser.add_argument("--validation-documents", type=int, default=150)
    parser.add_argument("--output", type=Path, default=Path("inputs/single_document_writes/rows-v1.json"))
    args = parser.parse_args()
    data = split_rows(json.loads(args.plan.read_text())["x_rows"], args.validation_documents)
    body = json.dumps(data, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({**data, "source": str(args.plan), "seed": SEED,
                                       "sha256": hashlib.sha256(body.encode()).hexdigest()}, sort_keys=True))
    print(json.dumps({split: len(v) for split, v in data.items()}
                     | {"train_documents": len({r["document"]["id"] for r in data["train"]})}))


if __name__ == "__main__":
    main()
