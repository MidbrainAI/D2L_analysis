"""Build the CPU-only tokenizer/chunk and gold-response audits for QA_PROTOCOL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from d2l_scale import load_config
from d2l_scale_backend import MODEL_IDS, verify_source
from qa_protocol_protocol import FAMILIES, digest, raw_example, text_hash


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def template_digest(tokenizer: Any) -> str:
    return hashlib.sha256(str(tokenizer.chat_template).encode("utf-8")).hexdigest()


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    return list(ids)


def _eos_ids(tokenizer: Any) -> list[int]:
    eos = tokenizer.eos_token_id
    return list(eos) if isinstance(eos, (tuple, list)) else [eos]


def build_audits(
    source_paths: dict[str, Path],
    tokenizer: Any,
    *,
    model_id: str,
    model_revision: str,
    tokenizer_revision: str,
    context_chunk_len: int,
    chat_template_sha256: str,
    upstream_source: dict[str, Any],
    source_provenance: dict[str, Any] | None = None,
    tokenize_context: Callable,
    split_context: Callable,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if set(source_paths) != set(FAMILIES):
        raise ValueError("QA_PROTOCOL tokenizer audit requires exactly SQuAD, DROP, and ROPES sources")
    source_digests = [
        {"family": family, "path": str(path), "sha256": file_digest(path)}
        for family, path in sorted(source_paths.items())
    ]
    documents: dict[str, dict[str, Any]] = {}
    gold_rows: dict[str, dict[str, Any]] = {}
    for family, path in sorted(source_paths.items()):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            context, question, answers = raw_example(raw, family)
            document_hash = text_hash(context)
            document = documents.setdefault(document_hash, {"family": family, "text": context, "source_rows": []})
            if document["text"] != context:
                raise ValueError(f"canonical document hash collision at {path}:{line_number}")
            document["source_rows"].append({"family": family, "path": str(path), "line": line_number})
            query_id = f"{family}:{raw.get('id', f'{family}:{line_number}') }"
            if query_id in gold_rows:
                raise ValueError(f"duplicate query id: {query_id}")
            answer = answers[0]
            response_ids = _token_ids(tokenizer, answer)
            response_ids.extend(_eos_ids(tokenizer))
            if not response_ids:
                raise ValueError(f"empty gold response tokenization: {query_id}")
            gold_rows[query_id] = {
                "query_id": query_id,
                "document_hash": document_hash,
                "family": family,
                "split": str(raw.get("split", "train")),
                "answer_hash": digest([answer]),
                "answer": answer,
                "answer_tokens_without_eos": len(response_ids) - len(_eos_ids(tokenizer)),
                "eos_tokens": len(_eos_ids(tokenizer)),
                "response_tokens": len(response_ids),
                "question_hash": digest(question),
            }

    for document_hash, document in sorted(documents.items()):
        tokenized = tokenize_context({"context": [document["text"]]}, tokenizer)
        ids = list(tokenized["ctx_ids"][0])
        chunks = split_context(
            {"ctx_ids": ids}, model_name_or_path=model_id, num_chunk_probs=None,
            max_chunk_len=context_chunk_len, min_chunk_len=0, max_num_split=None, is_train=False,
        )["ctx_ids"]
        if not chunks:
            raise ValueError(f"upstream splitter returned no chunks for {document_hash}")
        document.update({"source_tokens": len(ids), "chunks": len(chunks), "chunk_token_counts": [len(chunk) for chunk in chunks]})

    contract = {
        "schema_version": 1, "experiment": "QA_PROTOCOL-cpu-tokenizer-audit", "model_id": model_id,
        "model_revision": model_revision, "tokenizer_revision": tokenizer_revision,
        "context_chunk_len": context_chunk_len, "chat_template_sha256": chat_template_sha256,
        "upstream_source": upstream_source, "source_digests": source_digests,
        "source_provenance": source_provenance or {},
        "document_hash_semantics": "qa_protocol_protocol.text_hash(normalized_whitespace)",
        "gold_answer_semantics": "first_canonical_answer_plus_all_eos_token_ids",
    }
    chunk_audit = {"contract": contract, "documents": {
        document_hash: {"family": document["family"], "source_tokens": document["source_tokens"],
                        "chunks": document["chunks"], "chunk_token_counts": document["chunk_token_counts"],
                        "source_rows": sorted(document["source_rows"], key=lambda row: (row["path"], row["line"]))}
        for document_hash, document in sorted(documents.items())
    }}
    gold_audit = {"contract": contract, "rows": {query_id: gold_rows[query_id] for query_id in sorted(gold_rows)}}
    chunk_audit["audit_sha256"] = digest(chunk_audit)
    gold_audit["audit_sha256"] = digest(gold_audit)
    return chunk_audit, gold_audit


def write_immutable(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != value:
            raise ValueError(f"immutable audit differs: {path}; use a new output directory")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--tokenizer-path", type=Path)
    args = parser.parse_args()
    root = Path(__file__).parents[2]
    config = load_config(root, args.config)
    source = (root / config["paths"]["source"]).resolve()
    upstream = verify_source(source)
    model_id = MODEL_IDS[config["variant"]]
    model_revision = config["model"].get("revision")
    if not model_revision:
        raise ValueError("QA_PROTOCOL tokenizer audit requires a pinned model revision in the config")
    source_paths = {family: args.source_dir / f"{family}.jsonl" for family in FAMILIES}
    if any(not path.is_file() for path in source_paths.values()):
        raise FileNotFoundError("source directory must contain squad.jsonl, drop.jsonl, and ropes.jsonl")
    source_provenance_path = args.source_dir / "provenance.json"
    source_provenance = json.loads(source_provenance_path.read_text(encoding="utf-8")) if source_provenance_path.exists() else {}
    sys.path.insert(0, str(source / "src"))
    with UpstreamCwd(source):
        from ctx_to_lora.data.processing import split_too_long_ctx, tokenize_ctx_text
        from ctx_to_lora.model_loading import get_tokenizer

        if args.tokenizer_path:
            tokenizer_path = args.tokenizer_path.resolve()
            if not tokenizer_path.is_dir():
                raise FileNotFoundError(tokenizer_path)
            tokenizer = get_tokenizer(str(tokenizer_path))
            template_path = source / "chat_templates" / f"{model_id}.jinja"
            tokenizer.chat_template = template_path.read_text(encoding="utf-8").replace("    ", "").replace("\n", "")
        else:
            tokenizer = get_tokenizer(model_id, {"revision": args.tokenizer_revision})
    chunk_audit, gold_audit = build_audits(
        source_paths, tokenizer, model_id=model_id, model_revision=model_revision,
        tokenizer_revision=args.tokenizer_revision, context_chunk_len=config["model"]["context_chunk_len"],
        chat_template_sha256=template_digest(tokenizer), upstream_source=upstream,
        tokenize_context=tokenize_ctx_text, split_context=split_too_long_ctx,
        source_provenance=source_provenance,
    )
    write_immutable(args.output / "chunk-audit.json", chunk_audit)
    write_immutable(args.output / "gold-token-audit.json", gold_audit)
    print(json.dumps({"status": "audited", "output": str(args.output),
                      "chunk_audit_sha256": chunk_audit["audit_sha256"], "gold_audit_sha256": gold_audit["audit_sha256"]}, indent=2))


class UpstreamCwd:
    def __init__(self, path: Path):
        self.path, self.previous = path, None

    def __enter__(self):
        self.previous = Path.cwd()
        os.chdir(self.path)

    def __exit__(self, *_):
        os.chdir(self.previous)


if __name__ == "__main__":
    main()
