"""Dependency-free data, accounting, caching, and scoring primitives for D2L.

The executable harness imports this module on Python 3.10+ without torch,
transformers, datasets, or the upstream repository.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import string
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
SOURCE_COMMIT = "baa85db4d5df9b29d618af432d6ebf28b3ad5a29"
HF_REVISION = "7514da3cf18439265aef805f2a00170199e44456"
MODEL_IDS = {
    "gemma2": "google/gemma-2-2b-it",
    "qwen4": "Qwen/Qwen3-4B-Instruct-2507",
    "mistral7": "mistralai/Mistral-7B-Instruct-v0.2",
    "gemma-demo": "google/gemma-2-2b-it",
    "smoke": "fake",
}
CHECKPOINT_FILES = {
    "gemma2": "gemma_2b_d2l/checkpoint-20000/pytorch_model.bin",
    "qwen4": "qwen_4b_d2l/checkpoint-20000/pytorch_model.bin",
    "mistral7": "mistral_7b_d2l/checkpoint-20000/pytorch_model.bin",
    "gemma-demo": "gemma_demo/checkpoint-80000/pytorch_model.bin",
}
DATASETS = ("squad", "drop", "ropes", "longbench/qasper_e", "longbench/2wikimqa_e", "longbench/multifieldqa_en_e")


def digest(value: Any) -> str:
    raw = (
        value
        if isinstance(value, bytes)
        else json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()
    )
    return hashlib.sha256(raw).hexdigest()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def raw_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Document:
    id: str
    contenthash: str
    text: str
    split: str
    dataset: str
    rawhash: str = ""

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError(f"document {self.id!r} has empty text")
        if any(character in self.id for character in ("/", "\\")) or self.id in {".", ".."}:
            raise ValueError("Document identifiers must be safe single path components")
        if self.contenthash and self.contenthash != content_hash(self.text):
            raise ValueError("Document content hash mismatch")
        object.__setattr__(self, "contenthash", self.contenthash or content_hash(self.text))
        object.__setattr__(self, "rawhash", self.rawhash or raw_hash(self.text))


@dataclass(frozen=True)
class Query:
    id: str
    document_id: str | None
    input: str
    answers: tuple[str, ...]
    split: str
    dataset: str
    relevant_document_ids: tuple[str, ...] = ()


def _first(row: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return default


def _answers(row: dict[str, Any]) -> tuple[str, ...]:
    value = _first(row, "answers", "answer", "gold", "answers_spans", default=[])
    if isinstance(value, dict):
        value = _first(value, "text", "spans", "answers", default=[])
    if isinstance(value, str):
        return (value,)
    return tuple(str(x) for x in (value or []) if str(x))


def parse_jsonl(path: Path, *, default_dataset: str | None = None) -> tuple[list[Document], list[Query]]:
    """Import canonical JSONL and common QA rows, rejecting conflicting IDs."""
    docs: dict[str, Document] = {}
    queries: list[Query] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        dataset = str(row.get("dataset", default_dataset or "custom"))
        if "text" in row and "input" not in row and "question" not in row and "query" not in row:
            text, did = str(row["text"]), str(row.get("id", content_hash(str(row["text"]))[:16]))
            doc = Document(id=did, contenthash="", text=text, split=str(row.get("split", "train")), dataset=dataset)
            if did in docs and docs[did] != doc:
                raise ValueError(f"conflicting document id: {did}")
            docs[did] = doc
            continue
        context = _first(row, "context", "passage", "document", "text", default="")
        if isinstance(context, dict):
            # LongBench and nested QA exports commonly use this shape.
            context = _first(context, "text", "context", default="")
        text = str(context)
        did = row.get("document_id") or (content_hash(text)[:16] if text else None)
        if text:
            did = str(did)
            doc = Document(
                id=did,
                contenthash="",
                text=text,
                split=str(row.get("document_split", row.get("split", "test"))),
                dataset=dataset,
            )
            if did in docs and docs[did] != doc:
                raise ValueError(f"conflicting document id: {did}")
            docs[did] = doc
        qid = str(row.get("id", f"{dataset}:{number}"))
        if any(q.id == qid for q in queries):
            raise ValueError(f"duplicate query id: {qid}")
        relevant = tuple(str(x) for x in row.get("relevant_document_ids", row.get("relevant_documents", [])) or [])
        queries.append(
            Query(
                qid,
                str(did) if did is not None else None,
                str(_first(row, "input", "question", "query", "prompt", default="")),
                _answers(row),
                str(row.get("split", "test")),
                dataset,
                relevant,
            )
        )
    validate_manifest(docs.values(), queries)
    return list(docs.values()), queries


def split_by_document_hash(documents: Iterable[Document], validation_fraction: float = 0.2) -> list[Document]:
    return [
        Document(
            id=d.id,
            contenthash=d.contenthash,
            text=d.text,
            split="validation" if int(d.contenthash[:8], 16) / 0xFFFFFFFF < validation_fraction else "train",
            dataset=d.dataset,
            rawhash=d.rawhash,
        )
        for d in documents
    ]


def validate_no_overlap(train: Iterable[Document], other: Iterable[Document]) -> None:
    overlap = {d.contenthash for d in train} & {d.contenthash for d in other}
    if overlap:
        raise ValueError(f"document content overlap across held-out split: {len(overlap)}")


def validate_manifest(documents: Iterable[Document], queries: Iterable[Query]) -> None:
    docs = list(documents)
    by_id: dict[str, Document] = {}
    for d in docs:
        if d.id in by_id and by_id[d.id] != d:
            raise ValueError(f"conflicting document id: {d.id}")
        by_id[d.id] = d
    hashes: dict[str, str] = {}
    for d in docs:
        if d.contenthash in hashes and hashes[d.contenthash] != d.id:
            raise ValueError(f"duplicate document content: {d.id} and {hashes[d.contenthash]}")
        hashes[d.contenthash] = d.id
    qids: set[str] = set()
    for q in queries:
        if q.id in qids:
            raise ValueError(f"duplicate query id: {q.id}")
        qids.add(q.id)
        if q.document_id is not None and q.document_id not in by_id:
            raise ValueError(f"query {q.id} references unknown document {q.document_id}")
        if any(x not in by_id for x in q.relevant_document_ids):
            raise ValueError(f"query {q.id} references unknown relevant document")
        if q.document_id and (q.split != by_id[q.document_id].split or q.dataset != by_id[q.document_id].dataset):
            raise ValueError(f"Query {q.id} split/dataset differs from its source document")
        if not q.input.strip() or not q.answers:
            raise ValueError(f"Query {q.id} needs nonempty input and answer references")
    splits = {d.split for d in docs}
    for left in splits:
        for right in splits:
            if left < right:
                validate_no_overlap([d for d in docs if d.split == left], [d for d in docs if d.split == right])


def manifest(
    documents: Iterable[Document], queries: Iterable[Query], *, metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    docs, qs = list(documents), list(queries)
    validate_manifest(docs, qs)
    return {
        "schema_version": SCHEMA_VERSION,
        "documents": [asdict(x) for x in docs],
        "queries": [asdict(x) for x in qs],
        "metadata": metadata or {},
    }


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n")
        temporary = Path(handle.name)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_journal(path: Path) -> list[dict]:
    """Recover only an interrupted last write; never hide a corrupt complete row."""
    if not path.exists():
        return []
    rows = []
    with path.open("rb+") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                handle.seek(offset)
                handle.truncate()
                break
            rows.append(json.loads(line))
    if len({row["job_key"] for row in rows}) != len(rows):
        raise ValueError("Duplicate result journal jobs")
    return rows


def normalize(text: str) -> list[str]:
    # Pinned upstream eval_utils.normalize_answer uses humanized punctuation
    # as word separators (not the dataset vendors' task-specific scorers).
    text = re.sub(r"[\u00AD\u180E\u200B-\u200F\u202A-\u202E\u2060\u2066-\u2069\uFEFF]", "", text)
    text = re.sub(r"[\u2013\u2014]+", "-", text)
    text = re.sub(r"[\u201c\u201d\u00ab\u00bb\u201e]", '"', text)
    text = re.sub(r"[\u2018\u2019\u02bc]", "'", text)
    text = text.replace("\u2026", "...").replace("\u00a0", " ").lower()
    text = " ".join(re.split(f"[{re.escape(string.punctuation)}]", text))
    return [x for x in text.split() if x not in {"a", "an", "the"}]


def qa_f1(prediction: str, answers: Iterable[str]) -> float:
    p = normalize(prediction)
    best = 0.0
    for answer in answers:
        a = normalize(answer)
        common = sum(min(p.count(t), a.count(t)) for t in set(p))
        if common:
            precision, recall = common / len(p), common / len(a)
            best = max(best, 2 * precision * recall / (precision + recall))
    return best


def rouge_l(prediction: str, reference: str) -> float:
    """Report stemmed Rouge-L F1 separately from upstream closed-QA token F1."""
    from rouge_score.rouge_scorer import RougeScorer

    return float(RougeScorer(["rougeL"], use_stemmer=True).score(reference, prediction)["rougeL"].fmeasure)


def score_response(prediction: str, answers: Iterable[str]) -> dict[str, float | None]:
    refs = tuple(x for x in answers if x)
    if not refs:
        raise ValueError("Scoring requires reference answers")
    return {
        "qa_f1": qa_f1(prediction, refs),
        "rouge_l": max(rouge_l(prediction, x) for x in refs),
        "exact_match": float(any(normalize(prediction) == normalize(answer) for answer in refs)),
    }


def deterministic_bank(documents: Iterable[Document], size: int, seed: int) -> list[Document]:
    docs = sorted(documents, key=lambda d: d.contenthash)
    if size > len(docs):
        raise ValueError(f"requested bank size {size} exceeds document pool {len(docs)}")
    random.Random(seed).shuffle(docs)
    return docs[:size]


def source_anchored_bank(pool: Iterable[Document], anchor: Document, size: int, seed: int) -> list[Document]:
    """Return [anchor + deterministic same-dataset distractors] with no ID use."""
    docs = sorted((d for d in pool if d.contenthash != anchor.contenthash), key=lambda d: d.contenthash)
    if size < 1:
        raise ValueError("anchor bank size must be positive")
    if size > len(docs) + 1:
        raise ValueError(f"requested anchor bank size {size} exceeds pool {len(docs) + 1}")
    random.Random(digest([anchor.contenthash, seed])).shuffle(docs)
    bank = docs[: size - 1]
    random.Random(digest([anchor.contenthash, seed, "position"])).shuffle(bank)
    position = random.Random(digest([anchor.contenthash, seed, "slot"])).randrange(size)
    bank.insert(position, anchor)
    return bank


def anchor_queries(
    documents: Iterable[Document], queries: Iterable[Query], *, max_anchors: int, max_queries_per_anchor: int, seed: int
) -> list[tuple[Document, list[Query]]]:
    docs = {d.id: d for d in documents}
    grouped: dict[str, list[Query]] = {}
    for q in queries:
        if q.document_id in docs:
            grouped.setdefault(docs[q.document_id].contenthash, []).append(q)
    anchors = sorted(
        (docs[q.document_id] for q in queries if q.document_id in docs), key=lambda d: (d.dataset, d.contenthash)
    )
    unique: dict[str, Document] = {d.contenthash: d for d in anchors}
    selected = [
        d
        for dataset in sorted({d.dataset for d in unique.values()})
        for d in sorted(
            (d for d in unique.values() if d.dataset == dataset), key=lambda d: digest([d.contenthash, seed])
        )[:max_anchors]
    ]
    return [(d, sorted(grouped[d.contenthash], key=lambda q: q.id)[:max_queries_per_anchor]) for d in selected]
