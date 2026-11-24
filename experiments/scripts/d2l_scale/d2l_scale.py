"""Configuration, manifest, and download helpers shared by the D2L runners."""

from __future__ import annotations

import json
from pathlib import Path

from d2l_scale_core import MODEL_IDS, Document, Query, content_hash, manifest, write_json_atomic


def load_config(root: Path, name: str) -> dict:
    explicit = Path(name)
    path = (
        explicit
        if explicit.is_absolute()
        else root / explicit
        if (root / explicit).is_file()
        else root / "inputs/configs/d2l_scaling" / (name if name.endswith(".json") else name + ".json")
    )
    cfg = json.loads(path.read_text())
    if "extends" in cfg:
        base = json.loads((path.parent / cfg.pop("extends")).read_text())
        for key, value in cfg.items():
            base[key] = {**base.get(key, {}), **value} if isinstance(value, dict) else value
        cfg = base
    variant = cfg["variant"]
    cfg.setdefault("model", {})["id"] = MODEL_IDS[variant]
    run_name = cfg.get("run_name", variant)
    checkpoint = {"gemma2": "gemma_2b_d2l", "qwen4": "qwen_4b_d2l", "mistral7": "mistral_7b_d2l"}[variant]
    defaults = {
        "source": "artifacts/third_party/doc-to-lora/source",
        "checkpoint": f"artifacts/third_party/doc-to-lora/checkpoints/{checkpoint}/checkpoint-20000/pytorch_model.bin",
        "manifest": f"artifacts/d2l_scaling/{run_name}/manifest.json",
        "cache": f"artifacts/d2l_scaling/{run_name}/cache",
        "router": f"artifacts/d2l_scaling/{run_name}/router.pt",
        "output": f"artifacts/d2l_scaling/{run_name}/results",
    }
    cfg["paths"] = {**defaults, **cfg.get("paths", {})}
    if any(n < 1 for n in cfg["scaling"]["bank_sizes"]):
        raise ValueError("Candidate counts must be positive")
    return cfg


def paths(root, cfg):
    return {key: (root / value).resolve() for key, value in cfg["paths"].items()}


def raw_example(row: dict, dataset: str) -> tuple[str, str, object]:
    # Match the upstream preprocessing context, including ROPES's single newline.
    context = (
        f"{row['background']}\n{row['situation']}" if dataset == "ropes" else row.get("context", row.get("passage"))
    )
    question = row.get("question", row.get("input"))
    answers = row.get("answers", row.get("answers_spans"))
    if not isinstance(context, str) or not context.strip() or not isinstance(question, str) or answers is None:
        raise ValueError(f"Unsupported or empty {dataset} row; no silent skipping")
    return context, question, answers


def download_manifest(root: Path, cfg: dict) -> Path:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    target = paths(root, cfg)["manifest"].with_name("source_rows.jsonl")
    if target.exists():
        return target  # frozen snapshot; explicit new run path for a new download
    revisions, rows, train_seen = {}, [], set()
    api = HfApi()
    provenance, loaded_sources = [], []
    for spec in cfg["data"]["sources"]:
        enabled = (
            cfg["data"].get("router_training_datasets", cfg["data"]["datasets"])
            if spec.get("role") == "train"
            else cfg["data"]["datasets"]
        )
        if spec["dataset"] not in enabled:
            continue
        repo = spec["path"]
        revision = spec.get("revision") or revisions.get(repo)
        if revision is None:
            revision = api.dataset_info(repo).sha
        revisions[repo] = revision
        kwargs = {"path": repo, "split": spec["split"], "revision": revision}
        if spec.get("name"):
            kwargs["name"] = spec["name"]
        if spec.get("trust_remote_code"):
            kwargs["trust_remote_code"] = True
        dataset = load_dataset(**kwargs)
        provenance.append({**spec, "resolved_revision": revision, "fingerprint": dataset._fingerprint})
        loaded_sources.append((spec, dataset))

    # Official QA releases can repeat contexts across train and validation.
    # Reserve evaluation documents before applying train limits so the frozen
    # manifest remains document-disjoint without dropping evaluation cases.
    evaluation_hashes = set()
    for spec, dataset in loaded_sources:
        if spec.get("role", "test") == "train":
            continue
        for row in dataset:
            context, _, _ = raw_example(row, spec["dataset"])
            evaluation_hashes.add(content_hash(context))

    canonical_contexts, excluded_train_documents = {}, set()
    for spec, dataset in loaded_sources:
        role = spec.get("role", "test")
        seen = set()
        for index, row in enumerate(dataset):
            context, question, answers = raw_example(row, spec["dataset"])
            hashed = content_hash(context)
            if role == "train" and hashed in evaluation_hashes:
                excluded_train_documents.add(hashed)
                continue
            context = canonical_contexts.setdefault(hashed, context)
            limit = (
                cfg["data"].get("max_train_documents", 256)
                if role == "train"
                else cfg["data"].get("max_eval_documents", 256)
            )
            if hashed not in seen and len(seen) >= limit:
                continue
            seen.add(hashed)
            split = "test"
            if role == "train":
                train_seen.add(hashed)
                split = (
                    "validation"
                    if int(hashed[:8], 16) / 2**32 < cfg["data"].get("validation_fraction", 0.2)
                    else "train"
                )
            elif hashed in train_seen:
                raise ValueError("Train/evaluation context overlap: choose disjoint inputs before proceeding")
            rows.append(
                {
                    "id": f"{spec['dataset']}:{spec['split']}:{index}",
                    "document_id": hashed,
                    "context": context,
                    "question": question,
                    "answers": answers,
                    "dataset": spec["dataset"],
                    "split": split,
                }
            )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in rows))
    temporary.replace(target)
    write_json_atomic(
        target.with_suffix(".provenance.json"),
        {
            "sources": provenance,
            "training_documents_excluded_for_test_overlap": len(excluded_train_documents),
        },
    )
    return target


def load_manifest(path):
    value = json.loads(path.read_text())
    manifest([Document(**d) for d in value["documents"]], [Query(**q) for q in value["queries"]])
    return value
