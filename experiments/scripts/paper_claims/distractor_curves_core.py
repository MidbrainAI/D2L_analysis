"""Distractor-curve core: suite loading, persisted banks, bank compilation, and fresh-generation arms.

Every arm is greedy batch-1 generation with the frozen D2L prompt, so rows compare with
ONLINE_BATCH_READER/READOUT_LADDER. Banks are built from the frozen DISTRACTOR_CURVES suite and persisted in the plan, so every
reported cell can be reconstructed. Native D2L composition is the released one: generated
factors concatenated along rank with the shared head bias appended (equal to summing the
per-document updates). Normalization multiplies each document's A factor by ``K**-beta``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

K_VALUES = (1, 2, 4, 8, 16)
BETA_GRID = (0.25, 0.5, 0.75, 1.0)
MAX_NEW_TOKENS = 32
MAX_RANDOM_DISTRACTORS = max(K_VALUES) - 1


def hash_rank(*parts: object) -> str:
    return hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()


# -- suite and banks (CPU) -------------------------------------------------------
def load_suite(suite_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Documents and natural questions of the frozen suite, with text from the QA_PROTOCOL manifest."""
    suite = json.loads(suite_path.read_text())
    texts = {d["id"]: d["text"] for d in manifest["documents"]}
    rows = {row["query_id"]: row for row in manifest["rows"]}
    documents = {}
    questions = {}
    for item in suite["documents"]:
        documents[item["document_id"]] = {"id": item["document_id"], "text": texts[item["document_id"]],
                                          "article": item["article"], "role": item["role"]}
        questions[item["document_id"]] = [
            {"id": query_id, "document_id": item["document_id"], "input": rows[query_id]["question"],
             "answers": tuple(rows[query_id]["answers"]), "split": "test", "dataset": "squad"}
            for query_id in item["questions"]
        ]
    return {"suite_sha256": suite["suite_sha256"], "documents": documents, "questions": questions}


def sources(suite: dict[str, Any], partition: str) -> list[str]:
    return sorted(d["id"] for d in suite["documents"].values() if d["role"] == f"{partition}_source")


def _insert(source: str, distractors: list[str], k: int, seed: str, condition: str) -> tuple[list[str], int]:
    position = int(hash_rank(seed, source, condition, k), 16) % k
    bank = list(distractors[: k - 1])
    bank.insert(position, source)
    return bank, position


def source_banks(suite: dict[str, Any], source: str, partition: str, seed: str) -> list[dict[str, Any]]:
    """Nested random banks at every K, with distractors from other articles in the same partition."""
    documents = suite["documents"]
    article = documents[source]["article"]
    pool = [d for d in documents.values() if d["role"] == f"{partition}_distractor"]
    other = sorted((d["id"] for d in pool if d["article"] != article), key=lambda i: hash_rank(seed, source, i))
    if len(other) < MAX_RANDOM_DISTRACTORS:
        raise ValueError(f"not enough distractors for {source}")
    banks = []
    for k in K_VALUES:
        bank, position = _insert(source, other, k, seed, "random")
        banks.append({"bank_id": f"{source}:random:{k}", "condition": "random", "k": k,
                      "document_ids": bank, "source_position": position})
    return banks


# -- compilation (GPU) -----------------------------------------------------------
def compile_documents(backend, texts: list[str], chunk_batch: int = 8) -> list[dict[str, Any]]:
    """Generated factors and pre-head latents per document; one compiler chunk per document."""
    import torch
    from official_d2l_k1_representation_ladder import capture_pre_head_latents
    from official_d2l_online_batch_distillation import pad_sequences

    chunks = []
    for text in texts:
        ids = backend.tokenize_context({"context": [text]}, backend.tokenizer)["ctx_ids"][0]
        split = backend.split_context({"ctx_ids": ids}, model_name_or_path=backend.model_id, num_chunk_probs=None,
                                      max_chunk_len=backend.config["model"]["context_chunk_len"], min_chunk_len=0,
                                      max_num_split=None, is_train=False)["ctx_ids"]
        if len(split) != 1:
            raise ValueError("DISTRACTOR_CURVES documents must compile as one chunk")
        chunks.append(torch.tensor(split[0], dtype=torch.long))
    records = []
    with torch.inference_mode():
        for start in range(0, len(chunks), chunk_batch):
            ctx_ids, ctx_mask = pad_sequences(chunks[start:start + chunk_batch], backend.tokenizer.pad_token_id,
                                              device=backend.model.device)
            with capture_pre_head_latents(backend.model.hypernet) as captured:
                generated, layernorms = backend.model.generate_weights(ctx_ids, ctx_mask, None)
            if layernorms:
                raise ValueError("layernorm adaptation is outside the frozen protocol")
            latents = torch.cat(captured, dim=0)[:, :, 0]
            for index in range(ctx_ids.shape[0]):
                records.append({
                    "generated": {m: {key: value[index:index + 1].clone() for key, value in part.items()}
                                  for m, part in generated.items()},
                    "latent": latents[index:index + 1].clone(),
                    "source_tokens": int(ctx_mask[index].sum()),
                })
    return records


def factors_from_latents(backend, latents):
    """Regenerate generated A/B from cached normalized pre-head latents (exact head replay)."""
    hypernet = backend.model.hypernet
    return hypernet._to_lora_dict(hypernet.head(latents.unsqueeze(2)))


def head_bias(backend) -> dict:
    return {m: {key: value.to(backend.model.device) for key, value in part.items()}
            for m, part in backend.head_bias.items()}


def concatenate(records: list[dict[str, Any]]) -> dict:
    import torch

    modules = records[0]["generated"]
    return {m: {key: torch.cat([r["generated"][m][key] for r in records], dim=0) for key in ("A", "B")}
            for m in modules}


def bank_state(backend, records: list[dict[str, Any]], scale: float = 1.0) -> dict:
    """Native merged state for one bank; an empty bank is the shared bias alone."""
    import torch

    bias = head_bias(backend)
    if not records:
        return {m: {key: value.unsqueeze(0) for key, value in part.items()} for m, part in bias.items()}
    generated = concatenate(records)
    scalers = None if scale == 1.0 else torch.full((len(records),), scale, device=backend.model.device)
    with torch.inference_mode():
        state = backend.merge(generated, torch.tensor([len(records)], device=backend.model.device, dtype=torch.int32),
                              lora_bias=bias, scalers=scalers, bias_scaler=1.0)
    return {m: {key: value.detach().clone() for key, value in part.items()} for m, part in state.items()}


def bank_representation(backend, records: list[dict[str, Any]]):
    """Native state plus per-rank generated factors for a rank-effect reader."""
    import torch
    from official_d2l_k1_representation_ladder import CompiledRepresentationBatch

    bias = head_bias(backend)
    factors = bank_state(backend, records)
    if records:
        generated = {m: {key: value.transpose(0, 1).reshape(value.shape[1], -1, value.shape[-1]).unsqueeze(0)
                         for key, value in part.items()} for m, part in concatenate(records).items()}
    else:  # READOUT_LADDER's no-effect control: zero document ranks beside the live bias
        generated = {m: {key: torch.zeros_like(value).unsqueeze(0) for key, value in part.items()}
                     for m, part in bias.items()}
    latent_shape = (1, generated["down_proj"]["A"].shape[1], 1, backend.model.hypernet.config.latent_size)
    document_latents = torch.stack([r["latent"][0] for r in records]).unsqueeze(0) if records else None
    return CompiledRepresentationBatch(
        factors=factors,
        generated_factors=generated,
        bias_factors={m: {key: value.unsqueeze(0) for key, value in part.items()} for m, part in bias.items()},
        pre_head_latents=torch.zeros(latent_shape, device=backend.model.device),
        document_latents=document_latents,
    )


# -- readers and arms (GPU) ------------------------------------------------------
def load_reader(path: Path, backend, layers):
    """Return ("rank", reader) for READOUT_LADDER-style readers or ("legacy", reader) for SQUAD_POSTBLOCK_READER-QA_MATCHED_BANK_POSTBLOCK_READER transitions."""
    import torch
    from official_d2l_k1_representation_ladder import make_representation_transition
    from post_block_transitions import make_state_transition

    model = backend.model.base_model
    saved = torch.load(path, map_location=model.device, weights_only=False)
    if "representation" in saved:
        if saved["representation"] != "rank-effect" or saved["execution_site"] != "post-block":
            raise ValueError(f"DISTRACTOR_CURVES evaluates rank-effect post-block readers only: {path}")
        reader = make_representation_transition("rank-effect", model.config.hidden_size,
                                                 backend.model.hypernet.config.latent_size, len(layers),
                                                 saved["transition_bottleneck"]).to(model.device)
        kind = "rank"
    else:
        reader = make_state_transition(saved.get("transition_type", "shared"), model.config.hidden_size,
                                       len(layers), saved.get("transition_bottleneck", 32)).to(model.device)
        kind = "legacy"
    reader.load_state_dict(saved["transition"])
    reader.eval()
    return kind, reader


class Arms:
    """Batch-1 greedy generation for every condition, sharing one loaded backend."""

    def __init__(self, backend):
        from ctx_to_lora.utils import get_layers

        self.backend = backend
        self.layers = get_layers(backend.model.base_model)
        self.scaling = backend.model.base_model.peft_config["default"].lora_alpha

    def plain(self, query: dict, *, state=None, context: str | None = None) -> str:
        return self.backend.generate(query["input"], state=state, context=context, dataset=query["dataset"],
                                     max_new_tokens=MAX_NEW_TOKENS)["text"]

    def reader(self, kind: str, reader, query: dict, records: list[dict[str, Any]]) -> str:
        if kind == "rank":
            from official_d2l_k1_representation_evaluation import generate_reader_condition

            return generate_reader_condition(self.backend, self.layers, bank_representation(self.backend, records),
                                             reader, "rank-effect", self.scaling, query["input"], MAX_NEW_TOKENS,
                                             query["dataset"])
        from official_d2l_online_reader_evaluation import generate_reader_condition as legacy

        return legacy(self.backend, self.layers, bank_state(self.backend, records), reader, self.scaling,
                      query["input"], MAX_NEW_TOKENS, query["dataset"])


def score(response: str, answers) -> dict[str, float]:
    from d2l_scale_core import score_response

    values = score_response(response, answers)
    return {"f1": values["qa_f1"], "em": values["exact_match"]}


def evaluate_source(arms: Arms, suite: dict[str, Any], source: str, banks: list[dict[str, Any]], *,
                    betas: list[float], readers: dict[str, tuple[str, Any]],
                    include_icl: bool = True, include_native: bool = True) -> dict[str, Any]:
    """All arms for every natural question of one source document across its banks."""
    documents = suite["documents"]
    needed = sorted({i for bank in banks for i in bank["document_ids"]})
    compiled = dict(zip(needed, compile_documents(arms.backend, [documents[i]["text"] for i in needed]), strict=True))
    questions = suite["questions"][source]
    per_question = [{"query_id": query["id"], "responses": {"base": arms.plain(query)}}
                    for query in questions] if include_native else []
    rows = []
    for bank in banks:
        records = [compiled[i] for i in bank["document_ids"]]
        k = bank["k"]
        for query in questions:
            responses = {}
            if include_icl:
                context = "\n\n".join(documents[i]["text"] for i in bank["document_ids"])
                responses["icl"] = arms.plain(query, context=context)
            if include_native:
                responses["native"] = arms.plain(query, state=bank_state(arms.backend, records))
                for beta in betas if k > 1 else []:
                    responses[f"normalized_b{beta}"] = arms.plain(
                        query, state=bank_state(arms.backend, records, k ** -beta))
            for name, (kind, reader) in readers.items():
                responses[name] = arms.reader(kind, reader, query, records)
            rows.append({"query_id": query["id"], "source_id": source, "bank_id": bank["bank_id"],
                         "condition": bank["condition"], "k": k, "responses": responses})
    answers = {q["id"]: q["answers"] for q in questions}
    for row in per_question + rows:
        row["scores"] = {name: score(text, answers[row["query_id"]]) for name, text in row["responses"].items()}
    return {"source_id": source, "per_question": per_question, "rows": rows,
            "compiled_documents": len(needed),
            "source_tokens": {i: compiled[i]["source_tokens"] for i in needed}}
