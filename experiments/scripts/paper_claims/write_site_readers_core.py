"""WRITE_SITE_READERS core: the frozen WRITE_SITE_READERS suites in DISTRACTOR_CURVES's suite shape, their banks, and arms for every reader kind.

Three evaluation supports (see ``inputs/write_site_readers/``):

* **selection** (``selection-v1.json``): the 180 SQuAD-train contexts the released writer never
  saw, plus DISTRACTOR_CURVES's dev split. Arm and checkpoint selection happen here and nowhere else.
* **confirmation**: DISTRACTOR_CURVES's test split with its persisted banks (``inputs/distractor_curves/plans/full-v2.json``),
  opened once for the selected arm's seeds; a disclosed second look.
* **secondary** (``secondary-v1.json``): DROP and ROPES test documents minus D2L's selection rows,
  cross-family evidence with the untouched DROP cell flagged.

Banks for the WRITE_SITE_READERS suites are nested random banks at every K with distractors drawn from the
same role (and therefore the same family), the source at a hashed position, as in ``distractor_curves_core``.
There is no same-article or copies condition: the suites carry no article grouping.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from distractor_curves_core import K_VALUES, MAX_RANDOM_DISTRACTORS, Arms, _insert, bank_representation, hash_rank

GATED_ARMS = ("GB", "GC", "GBL", "BS")  # GATED_READERS arms, loaded through gated_readers_gated_readers

SELECTION_SUITE = Path("inputs/write_site_readers/selection-v1.json")
SECONDARY_SUITE = Path("inputs/write_site_readers/secondary-v1.json")
DISTRACTOR_CURVES_SUITE = Path("inputs/distractor_curves/suite/14b15feae84cd72b/suite.json")
DISTRACTOR_CURVES_PLAN = Path("inputs/distractor_curves/plans/full-v2.json")
MANIFEST = Path("inputs/qa_protocol/manifest/pinned-v1/manifest.json")
BANK_SEED = "write_site_readers-banks-v1"


# -- suites (CPU) -------------------------------------------------------------------
def load_write_site_readers_suite(suite_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Documents and questions of a frozen WRITE_SITE_READERS suite, with text from the QA_PROTOCOL manifest.

    Same shape as ``distractor_curves_core.load_suite``: ``documents[id] = {id, text, article, role, family,
    untouched}`` and ``questions[id] = [{id, document_id, input, answers, split, dataset}]``. The
    prompt dataset is the family (``squad``, ``drop``, ``ropes``), which selects the D2L template.
    """
    suite = json.loads(suite_path.read_text())
    texts = {d["id"]: d["text"] for d in manifest["documents"]}
    rows = {row["query_id"]: row for row in manifest["rows"]}
    documents, questions = {}, {}
    for item in suite["documents"]:
        family = item.get("family", "squad")
        documents[item["document_id"]] = {"id": item["document_id"], "text": texts[item["document_id"]],
                                          "article": None, "role": item["role"], "family": family,
                                          "untouched": bool(item.get("untouched", False))}
        questions[item["document_id"]] = [
            {"id": query_id, "document_id": item["document_id"], "input": rows[query_id]["question"],
             "answers": tuple(rows[query_id]["answers"]), "split": "test", "dataset": family}
            for query_id in item["questions"]
        ]
    return {"suite_sha256": suite["suite_sha256"], "schema": suite["schema"], "documents": documents,
            "questions": questions}


def sources(suite: dict[str, Any], role: str) -> list[str]:
    return sorted(d["id"] for d in suite["documents"].values() if d["role"] == role)


def random_banks(suite: dict[str, Any], source: str, seed: str = BANK_SEED) -> list[dict[str, Any]]:
    """Nested random banks at every K; distractors are the other documents of the source's role."""
    documents = suite["documents"]
    role = documents[source]["role"]
    pool = sorted((d["id"] for d in documents.values() if d["role"] == role and d["id"] != source),
                  key=lambda i: hash_rank(seed, source, i))
    if len(pool) < MAX_RANDOM_DISTRACTORS:
        raise ValueError(f"not enough distractors for {source}")
    banks = []
    for k in K_VALUES:
        bank, position = _insert(source, pool, k, seed, "random")
        banks.append({"bank_id": f"{source}:random:{k}", "condition": "random", "k": k,
                      "document_ids": bank, "source_position": position})
    return banks


def distractor_curves_banks(plan: dict[str, Any], partition: str, conditions: tuple[str, ...] = ("random",)) -> dict[str, list]:
    """DISTRACTOR_CURVES's persisted banks for one partition, so the confirmation reuses the exact cells."""
    banks = plan["config"]["banks"][partition]
    return {source: [b for b in group if b["condition"] in conditions] for source, group in banks.items()}


# -- readers and arms (GPU) ------------------------------------------------------------
def load_any_reader(path: Path, backend, layers) -> tuple[str, Any]:
    """("write_site_readers", (arm, reader)) for WRITE_SITE_READERS checkpoints; DISTRACTOR_CURVES's ("rank"|"legacy", reader) otherwise."""
    import torch
    from distractor_curves_core import load_reader
    from write_site_readers_readers import load_write_site_readers_reader

    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("arm", "") in GATED_ARMS:
        from gated_readers_gated_readers import load_gated_reader

        return "write_site_readers", load_gated_reader(path, backend, layers)
    if "arm" in saved:
        return "write_site_readers", load_write_site_readers_reader(path, backend, layers)
    return load_reader(path, backend, layers)


class WRITE_SITE_READERSArms(Arms):
    """DISTRACTOR_CURVES's arms plus WRITE_SITE_READERS readers (any arm) through the write-site or post-block installer."""

    def reader(self, kind: str, reader, query: dict, records: list[dict[str, Any]]) -> str:
        if kind != "write_site_readers":
            return super().reader(kind, reader, query, records)
        from write_site_readers_readers import generate_with_reader

        arm, module = reader
        compiled = bank_representation(self.backend, records)
        return generate_with_reader(self.backend, self.layers, arm, module, compiled, query["input"], query["dataset"])
