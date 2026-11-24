#!/usr/bin/env python
"""ANSWER_ATTRIBUTION: write-salience and latent-decodability diagnostic on EXACT_400's exact 400 K=1 questions.

No training of readers or writers. For every held-out document the released D2L
writer is run once. Three families of per-question measurements are produced:

1. Question/answer-side lexical features (CPU only).
2. Write salience: teacher-forced log-probability of the passage tokens under the
   native D2L state versus the clean base model, read out at the gold answer span,
   plus a free recitation of the passage under the native state.
3. Latent decodability: a linear kernel-ridge probe from the frozen 26x8x512
   pre-LoRA document latent to passage token presence, fitted on training-split
   documents and applied to the held-out documents.

These are then related to the READOUT_LADDER per-question fresh outcomes (teacher, native,
summed, rank, latent readers). Mechanism evidence only: 57 documents, 400 queries.
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from d2l_scale import load_config, load_manifest, paths  # noqa: E402
from official_d2l_k1_representation_evaluation import query_ids_from_artifact  # noqa: E402
from official_d2l_k1_representation_ladder import compile_k1_representation_batch  # noqa: E402
from reader_banks import build_dataset_local_plans  # noqa: E402
from reader_banks import split_rows  # noqa: E402

RECITE_PROMPT = "Repeat the passage you have memorized, word for word."
STOPWORDS = set(
    "the a an of in on at to for from by with and or is are was were be been being this that these those it its "
    "as which who whom whose what when where why how did do does done has have had not no than then there their "
    "they them he she his her we you your i me my our us also into over under about after before during while".split()
)


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def content_words(text: str) -> set[str]:
    return {w for w in normalize_answer(text).split() if w not in STOPWORDS and len(w) > 1}


def sentence_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for match in re.finditer(r"(?<=[.!?])\s+", text):
        spans.append((start, match.start()))
        start = match.end()
    spans.append((start, len(text)))
    return spans


def locate_answer(passage: str, answers: list[str]) -> tuple[str, int, int] | None:
    lowered = passage.lower()
    best = None
    for answer in answers:
        answer = answer.strip()
        if not answer:
            continue
        index = lowered.find(answer.lower())
        if index >= 0 and (best is None or index < best[1]):
            best = (answer, index, index + len(answer))
    return best


def auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return None
    order = np.argsort(np.concatenate([positives, negatives]), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    combined = np.concatenate([positives, negatives])[order]
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and combined[j + 1] == combined[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    rank_sum = ranks[: len(positives)].sum()
    return float((rank_sum - len(positives) * (len(positives) + 1) / 2.0) / (len(positives) * len(negatives)))


def bootstrap_auc(scores: np.ndarray, labels: np.ndarray, seed: int, resamples: int = 1000) -> list[float] | None:
    rng = np.random.default_rng(seed)
    values = []
    n = len(scores)
    for _ in range(resamples):
        idx = rng.integers(0, n, n)
        value = auc(scores[idx], labels[idx])
        if value is not None:
            values.append(value)
    if len(values) < resamples // 2:
        return None
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    from scipy.stats import spearmanr

    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return None
    return float(spearmanr(x, y).statistic)


class StateInstaller:
    """Replicates the native factor installation used by OfficialBackend.generate."""

    def __init__(self, backend):
        self.backend = backend

    def __call__(self, state):
        return _Installed(self.backend, state)


class _Installed:
    def __init__(self, backend, state):
        self.backend = backend
        self.state = state

    def __enter__(self):
        model = self.backend.model
        model.reset()
        if self.state is not None:
            model.patch_lora_forward()
            loaded = {
                module: {key: value.to(model.device) for key, value in part.items()}
                for module, part in self.state.items()
            }
            self.backend.install(
                model.base_model,
                model.hypernet.layer_indices,
                loaded,
                torch.ones(1, device=model.device, dtype=torch.int32),
            )
        return self

    def __exit__(self, *_):
        self.backend.model.reset()


def chat_prefix(backend, prompt: str) -> torch.Tensor:
    return backend.tokenizer.apply_chat_template(
        [{"role": "system", "content": ""}, {"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        add_special_tokens=False,
    )[0]


def passage_logprobs(backend, prefix: torch.Tensor, passage_ids: list[int], state) -> np.ndarray:
    ids = torch.cat([prefix, torch.tensor(passage_ids, dtype=torch.long)]).unsqueeze(0).to(backend.model.device)
    with StateInstaller(backend)(state), torch.inference_mode():
        logits = backend.model.base_model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
    logprobs = torch.log_softmax(logits[0, :-1].float(), dim=-1)
    targets = ids[0, 1:]
    token_logprobs = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return token_logprobs[prefix.shape[0] - 1 :].cpu().numpy()


def recite(backend, prefix: torch.Tensor, state, max_new_tokens: int) -> str:
    ids = prefix.unsqueeze(0).to(backend.model.device)
    with StateInstaller(backend)(state), torch.inference_mode():
        output = backend.model.base_model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=backend.tokenizer.pad_token_id,
        )
    return backend.tokenizer.decode(output[0, ids.shape[1] :], skip_special_tokens=True).strip()


def token_f1(prediction: str, reference: str) -> float:
    p = normalize_answer(prediction).split()
    r = normalize_answer(reference).split()
    if not p or not r:
        return float(p == r)
    common = Counter(p) & Counter(r)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def compile_documents(backend, document_ids: list[str], documents: dict, batch_size: int, chunk_batch: int):
    """Yield (document_id, native_state_or_None, latent_vector) in deterministic order."""
    for start in range(0, len(document_ids), batch_size):
        batch_ids = document_ids[start : start + batch_size]
        plans = [{"k": 1, "bank_document_ids": [doc_id], "query": {"id": f"probe:{doc_id}"}} for doc_id in batch_ids]
        try:
            compiled, _ = compile_k1_representation_batch(backend, plans, documents, chunk_batch)
        except ValueError as error:
            if "one compiler chunk" not in str(error):
                raise
            for doc_id in batch_ids:
                yield doc_id, None, None
            continue
        latents = compiled.pre_head_latents.float().reshape(len(batch_ids), -1).cpu().numpy()
        for index, doc_id in enumerate(batch_ids):
            state = None
            if len(batch_ids) == 1:
                state = compiled.factors
            yield doc_id, state, latents[index]


def fit_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    lambdas: list[float],
    device: torch.device,
    seed: int,
) -> dict:
    """Linear kernel ridge from latents to multi-hot token presence with internal lambda selection."""
    rng = np.random.default_rng(seed)
    n = train_x.shape[0]
    permutation = rng.permutation(n)
    holdout = permutation[: max(1, n // 10)]
    fit = permutation[max(1, n // 10) :]

    def solve(x_fit, y_fit, x_eval, lam):
        xf = torch.tensor(x_fit, device=device, dtype=torch.float32)
        mean = xf.mean(dim=0, keepdim=True)
        scale = xf.std(dim=0, keepdim=True) + 1e-6
        xf = (xf - mean) / scale
        xe = (torch.tensor(x_eval, device=device, dtype=torch.float32) - mean) / scale
        yf = torch.tensor(y_fit, device=device, dtype=torch.float32)
        y_mean = yf.mean(dim=0, keepdim=True)
        kernel = xf @ xf.T
        kernel_eval = xe @ xf.T
        alpha = torch.linalg.solve(kernel + lam * torch.eye(kernel.shape[0], device=device), yf - y_mean)
        return (kernel_eval @ alpha + y_mean).cpu().numpy()

    selection = {}
    best_lambda = None
    best_auc = -1.0
    for lam in lambdas:
        predicted = solve(train_x[fit], train_y[fit], train_x[holdout], lam)
        value = auc(predicted.reshape(-1), train_y[holdout].reshape(-1).astype(int))
        selection[str(lam)] = value
        if value is not None and value > best_auc:
            best_auc = value
            best_lambda = lam
    predicted_test = solve(train_x, train_y, test_x, best_lambda)
    return {"selected_lambda": best_lambda, "internal_holdout_auc": selection, "test_scores": predicted_test}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="gemma2_squad_matched_distill_scale60k_seed141")
    parser.add_argument(
        "--readout_ladder-artifact",
        type=Path,
        default=Path("artifacts/online_encoding/readout_ladder-exact_400-heldout400-representation-evaluation.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-train-documents", type=int, default=2000)
    parser.add_argument("--probe-min-document-frequency", type=int, default=3)
    parser.add_argument("--probe-max-document-fraction", type=float, default=0.2)
    parser.add_argument("--probe-lambdas", default="1,10,100,1000,10000")
    parser.add_argument("--recite-tokens", type=int, default=256)
    parser.add_argument("--compiler-chunk-batch-size", type=int, default=8)
    parser.add_argument("--max-heldout-documents", type=int)
    parser.add_argument("--seed", type=int, default=182)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    root = Path(__file__).parents[2]
    cfg = load_config(root, args.config)
    payload = load_manifest(paths(root, cfg)["manifest"])
    documents = {document["id"]: document for document in payload["documents"]}
    rows = split_rows(payload)["test"]
    query_ids = query_ids_from_artifact(args.readout_ladder_artifact)
    by_query = {row["query"]["id"]: row for row in rows}
    rows = [by_query[query_id] for query_id in query_ids]
    _, plans = build_dataset_local_plans(rows, documents, split="test", limit=None, fixed_k=1)
    readout_ladder = json.loads(args.readout_ladder_artifact.read_text())
    readout_ladder_rows = {row["query_id"]: row for row in readout_ladder["rows"]}
    if set(readout_ladder_rows) != set(query_ids):
        raise ValueError("READOUT_LADDER artifact rows do not match recovered query IDs")

    heldout_ids = []
    for plan in plans:
        doc_id = plan["bank_document_ids"][0]
        if doc_id not in heldout_ids:
            heldout_ids.append(doc_id)
    if args.max_heldout_documents:
        heldout_ids = heldout_ids[: args.max_heldout_documents]
        plans = [plan for plan in plans if plan["bank_document_ids"][0] in set(heldout_ids)]
    heldout_set = set(heldout_ids)
    train_ids = sorted(
        doc_id
        for doc_id, document in documents.items()
        if document["split"] == "train" and doc_id not in heldout_set
    )[: args.probe_train_documents]

    from d2l_scale_backend import OfficialBackend

    backend = OfficialBackend(cfg, root)
    tokenizer = backend.tokenizer
    report = {
        "schema_version": 1,
        "experiment": "ANSWER_ATTRIBUTION-write-salience-latent-decodability",
        "status": "running",
        "contract": {
            "config": args.config,
            "readout_ladder_artifact": str(args.readout_ladder_artifact),
            "query_ids": query_ids,
            "heldout_document_ids": heldout_ids,
            "probe_train_document_count": len(train_ids),
            "probe_train_document_ids_sha256": __import__("hashlib").sha256("\n".join(train_ids).encode()).hexdigest(),
            "recite_prompt": RECITE_PROMPT,
            "recite_tokens": args.recite_tokens,
            "probe_lambdas": args.probe_lambdas,
            "seed": args.seed,
        },
        "documents": {},
        "rows": [],
        "probe": {},
        "summary": {},
        "accounting": {},
    }
    started = time.perf_counter()
    try:
        prefix = chat_prefix(backend, RECITE_PROMPT)

        # ---- held-out documents: native state, salience, recitation, latent ----
        heldout_latents = {}
        document_info = {}
        for doc_id, state, latent in compile_documents(backend, heldout_ids, documents, 1, args.compiler_chunk_batch_size):
            if state is None:
                raise ValueError(f"held-out document {doc_id} does not compile to one chunk")
            passage = documents[doc_id]["text"]
            encoded = tokenizer(passage, add_special_tokens=False, return_offsets_mapping=True)
            passage_ids = encoded["input_ids"]
            offsets = encoded["offset_mapping"]
            lp_base = passage_logprobs(backend, prefix, passage_ids, None)
            lp_state = passage_logprobs(backend, prefix, passage_ids, state)
            recitation = recite(backend, prefix, state, args.recite_tokens)
            recitation_base = recite(backend, prefix, None, args.recite_tokens)
            heldout_latents[doc_id] = latent
            document_info[doc_id] = {
                "passage_tokens": len(passage_ids),
                "offsets": offsets,
                "passage_ids": passage_ids,
                "lp_base": lp_base,
                "lp_state": lp_state,
                "recitation": recitation,
                "recitation_base": recitation_base,
            }
            report["documents"][doc_id] = {
                "passage_tokens": len(passage_ids),
                "mean_logprob_base": float(lp_base.mean()),
                "mean_logprob_state": float(lp_state.mean()),
                "mean_gain": float((lp_state - lp_base).mean()),
                "recitation": recitation,
                "recitation_base": recitation_base,
                "recitation_passage_f1": token_f1(recitation, passage),
                "recitation_base_passage_f1": token_f1(recitation_base, passage),
            }
        print(f"held-out documents done: {len(document_info)} in {time.perf_counter() - started:.1f}s", flush=True)

        # ---- per-question features ----
        for plan in plans:
            query = plan["query"]
            doc_id = plan["bank_document_ids"][0]
            info = document_info[doc_id]
            passage = documents[doc_id]["text"]
            readout_ladder_row = readout_ladder_rows[query["id"]]
            located = locate_answer(passage, query["answers"])
            row = {
                "query_id": query["id"],
                "document_id": doc_id,
                "question": query["input"],
                "answers": query["answers"],
                "outcomes": {
                    name: {"qa_f1": readout_ladder_row["scores"][name]["qa_f1"], "exact_match": readout_ladder_row["scores"][name]["exact_match"]}
                    for name in ("teacher", "base", "native_d2l", "summed", "rank", "latent")
                },
                "responses": {name: readout_ladder_row["responses"][name] for name in ("teacher", "base", "native_d2l", "rank")},
                "features": {},
            }
            features = row["features"]
            features["passage_tokens"] = info["passage_tokens"]
            features["question_tokens"] = len(tokenizer(query["input"], add_special_tokens=False)["input_ids"])
            features["answer_located"] = located is not None
            spans = sentence_spans(passage)
            if located is not None:
                answer, start, end = located
                features["answer_text"] = answer
                features["answer_char_position"] = start / max(1, len(passage))
                sentence_index = next((i for i, (s, e) in enumerate(spans) if s <= start < e), len(spans) - 1)
                features["answer_sentence_index"] = sentence_index
                features["answer_sentence_fraction"] = sentence_index / max(1, len(spans) - 1) if len(spans) > 1 else 0.0
                features["answer_occurrences"] = passage.lower().count(answer.lower())
                features["answer_is_numeric"] = any(ch.isdigit() for ch in answer)
                words = answer.split()
                features["answer_words"] = len(words)
                features["answer_capitalized_fraction"] = sum(w[:1].isupper() for w in words) / max(1, len(words))
                sentence_text = passage[spans[sentence_index][0] : spans[sentence_index][1]]
                q_words = content_words(query["input"])
                s_words = content_words(sentence_text)
                features["question_sentence_jaccard"] = len(q_words & s_words) / max(1, len(q_words | s_words))
                features["question_sentence_overlap"] = len(q_words & s_words) / max(1, len(q_words))
                token_indices = [i for i, (s, e) in enumerate(info["offsets"]) if e > start and s < end]
                features["answer_token_count"] = len(token_indices)
                if token_indices:
                    gain = info["lp_state"] - info["lp_base"]
                    features["span_logprob_base"] = float(info["lp_base"][token_indices].mean())
                    features["span_logprob_state"] = float(info["lp_state"][token_indices].mean())
                    features["span_logprob_state_min"] = float(info["lp_state"][token_indices].min())
                    features["span_gain"] = float(gain[token_indices].mean())
                    features["span_gain_relative"] = float(gain[token_indices].mean() - gain.mean())
                    features["span_first_token_gain"] = float(gain[token_indices[0]])
                    features["span_first_token_logprob_state"] = float(info["lp_state"][token_indices[0]])
                    features["span_state_prob"] = float(np.exp(info["lp_state"][token_indices]).mean())
                recitation_norm = normalize_answer(info["recitation"])
                features["recitation_contains_answer"] = any(
                    normalize_answer(a) and normalize_answer(a) in recitation_norm for a in query["answers"]
                )
                features["recitation_base_contains_answer"] = any(
                    normalize_answer(a) and normalize_answer(a) in normalize_answer(info["recitation_base"])
                    for a in query["answers"]
                )
                features["answer_token_ids"] = [info["passage_ids"][i] for i in token_indices]
            report["rows"].append(row)
        print(f"question features done: {len(report['rows'])}", flush=True)

        # ---- probe: compile training documents and fit ----
        train_latents = []
        train_token_sets = []
        kept_train_ids = []
        probe_started = time.perf_counter()
        for doc_id, _, latent in compile_documents(backend, train_ids, documents, 8, args.compiler_chunk_batch_size):
            if latent is None:
                continue
            kept_train_ids.append(doc_id)
            train_latents.append(latent.astype(np.float32))
            train_token_sets.append(set(tokenizer(documents[doc_id]["text"], add_special_tokens=False)["input_ids"]))
            if len(kept_train_ids) % 200 == 0:
                print(f"probe compile {len(kept_train_ids)}/{len(train_ids)}", flush=True)
        compile_seconds = time.perf_counter() - probe_started
        n_train = len(kept_train_ids)
        document_frequency = Counter()
        for token_set in train_token_sets:
            document_frequency.update(token_set)
        vocabulary = sorted(
            token
            for token, frequency in document_frequency.items()
            if frequency >= args.probe_min_document_frequency
            and frequency <= args.probe_max_document_fraction * n_train
            and tokenizer.decode([token]).strip()
            and not all(ch in string.punctuation or ch.isspace() for ch in tokenizer.decode([token]))
        )
        vocab_index = {token: i for i, token in enumerate(vocabulary)}
        train_y = np.zeros((n_train, len(vocabulary)), dtype=np.float32)
        for i, token_set in enumerate(train_token_sets):
            for token in token_set:
                j = vocab_index.get(token)
                if j is not None:
                    train_y[i, j] = 1.0
        train_x = np.stack(train_latents)
        test_x = np.stack([heldout_latents[doc_id] for doc_id in heldout_ids]).astype(np.float32)
        lambdas = [float(v) for v in args.probe_lambdas.split(",")]
        probe = fit_probe(train_x, train_y, test_x, lambdas, backend.model.device, args.seed)
        scores = probe["test_scores"]  # [heldout, vocab]
        heldout_index = {doc_id: i for i, doc_id in enumerate(heldout_ids)}

        # per-document probe sanity: present-vs-absent AUC on held-out docs
        doc_auc = {}
        for doc_id in heldout_ids:
            present = np.zeros(len(vocabulary), dtype=int)
            for token in set(document_info[doc_id]["passage_ids"]):
                j = vocab_index.get(token)
                if j is not None:
                    present[j] = 1
            doc_auc[doc_id] = auc(scores[heldout_index[doc_id]], present)
            report["documents"][doc_id]["probe_present_vs_absent_auc"] = doc_auc[doc_id]

        # per-question answer-token decodability: percentile of answer tokens' scores within the doc's vocab scores
        ranked = np.argsort(np.argsort(scores, axis=1), axis=1) / max(1, len(vocabulary) - 1)
        shift = max(1, len(heldout_ids) // 2)
        for row in report["rows"]:
            tokens = [vocab_index[t] for t in row["features"].get("answer_token_ids", []) if t in vocab_index]
            row["features"]["answer_tokens_in_probe_vocab"] = len(tokens)
            if tokens:
                i = heldout_index[row["document_id"]]
                control = (i + shift) % len(heldout_ids)
                row["features"]["probe_answer_percentile_mean"] = float(ranked[i, tokens].mean())
                row["features"]["probe_answer_percentile_min"] = float(ranked[i, tokens].min())
                row["features"]["probe_answer_percentile_control_mean"] = float(ranked[control, tokens].mean())
        report["probe"] = {
            "train_documents": n_train,
            "vocabulary_size": len(vocabulary),
            "selected_lambda": probe["selected_lambda"],
            "internal_holdout_auc": probe["internal_holdout_auc"],
            "heldout_present_vs_absent_auc_mean": float(np.mean([v for v in doc_auc.values() if v is not None])),
            "compile_seconds": compile_seconds,
            "latent_dimension": int(train_x.shape[1]),
        }
        print(f"probe done: vocab {len(vocabulary)}, lambda {probe['selected_lambda']}, heldout AUC {report['probe']['heldout_present_vs_absent_auc_mean']:.4f}", flush=True)

        # ---- analysis ----
        rows = [row for row in report["rows"] if row["features"].get("answer_located")]
        feature_names = [
            "answer_char_position", "answer_sentence_fraction", "answer_occurrences", "answer_words",
            "answer_capitalized_fraction", "question_sentence_jaccard", "question_sentence_overlap",
            "passage_tokens", "question_tokens",
            "span_logprob_base", "span_logprob_state", "span_logprob_state_min", "span_gain",
            "span_gain_relative", "span_first_token_gain", "span_first_token_logprob_state", "span_state_prob",
            "probe_answer_percentile_mean", "probe_answer_percentile_min", "probe_answer_percentile_control_mean",
        ]
        binary_names = ["answer_is_numeric", "recitation_contains_answer", "recitation_base_contains_answer"]

        def ok(row, name, threshold=0.5):
            return row["outcomes"][name]["qa_f1"] >= threshold

        groups = {
            "teacher_ok": [r for r in rows if ok(r, "teacher")],
        }
        teacher_ok = groups["teacher_ok"]
        contrasts = {
            "native_success_given_teacher_ok": (teacher_ok, lambda r: ok(r, "native_d2l")),
            "rank_success_given_teacher_ok": (teacher_ok, lambda r: ok(r, "rank")),
            "rank_rescue_given_native_fail": (
                [r for r in teacher_ok if not ok(r, "native_d2l")],
                lambda r: ok(r, "rank"),
            ),
            "base_success_given_teacher_ok": (teacher_ok, lambda r: ok(r, "base")),
        }
        analysis = {"counts": {name: len(subset) for name, (subset, _) in contrasts.items()}, "auc": {}, "spearman": {}, "binary": {}}
        for cname, (subset, label_fn) in contrasts.items():
            labels_all = np.array([1 if label_fn(r) else 0 for r in subset])
            analysis["auc"][cname] = {"positives": int(labels_all.sum()), "negatives": int(len(labels_all) - labels_all.sum()), "features": {}}
            for fname in feature_names:
                pairs = [(r["features"][fname], 1 if label_fn(r) else 0) for r in subset if fname in r["features"]]
                if len(pairs) < 10:
                    continue
                values = np.array([p[0] for p in pairs], dtype=np.float64)
                labels = np.array([p[1] for p in pairs])
                value = auc(values, labels)
                analysis["auc"][cname]["features"][fname] = {
                    "auc": value,
                    "ci95": bootstrap_auc(values, labels, args.seed) if value is not None else None,
                    "n": len(pairs),
                    "mean_positive": float(values[labels == 1].mean()) if labels.sum() else None,
                    "mean_negative": float(values[labels == 0].mean()) if (labels == 0).sum() else None,
                }
            for fname in binary_names:
                pairs = [(bool(r["features"][fname]), 1 if label_fn(r) else 0) for r in subset if fname in r["features"]]
                if pairs:
                    analysis["binary"].setdefault(cname, {})[fname] = {
                        "rate_positive": float(np.mean([p[0] for p in pairs if p[1] == 1])) if any(p[1] == 1 for p in pairs) else None,
                        "rate_negative": float(np.mean([p[0] for p in pairs if p[1] == 0])) if any(p[1] == 0 for p in pairs) else None,
                    }
        for target in ("native_d2l", "rank", "teacher", "base"):
            analysis["spearman"][target] = {}
            for fname in feature_names:
                pairs = [(r["features"][fname], r["outcomes"][target]["qa_f1"]) for r in rows if fname in r["features"]]
                if len(pairs) >= 10:
                    analysis["spearman"][target][fname] = spearman(
                        np.array([p[0] for p in pairs], dtype=np.float64), np.array([p[1] for p in pairs], dtype=np.float64)
                    )
        analysis["group_means"] = {}
        for gname, predicate in {
            "teacher_ok_native_ok": lambda r: ok(r, "teacher") and ok(r, "native_d2l"),
            "teacher_ok_native_fail_rank_ok": lambda r: ok(r, "teacher") and not ok(r, "native_d2l") and ok(r, "rank"),
            "teacher_ok_native_fail_rank_fail": lambda r: ok(r, "teacher") and not ok(r, "native_d2l") and not ok(r, "rank"),
            "teacher_fail": lambda r: not ok(r, "teacher"),
        }.items():
            subset = [r for r in rows if predicate(r)]
            analysis["group_means"][gname] = {"n": len(subset)}
            for fname in feature_names + binary_names:
                values = [float(r["features"][fname]) for r in subset if fname in r["features"]]
                if values:
                    analysis["group_means"][gname][fname] = float(np.mean(values))
        report["summary"] = {
            "rows": len(report["rows"]),
            "answer_located": len(rows),
            "analysis": analysis,
            "document_mean_gain": float(np.mean([d["mean_gain"] for d in report["documents"].values()])),
            "document_recitation_passage_f1_mean": float(np.mean([d["recitation_passage_f1"] for d in report["documents"].values()])),
            "document_recitation_base_passage_f1_mean": float(np.mean([d["recitation_base_passage_f1"] for d in report["documents"].values()])),
        }
        report["status"] = "completed"
    finally:
        report["accounting"] = {
            "wall_seconds": time.perf_counter() - started,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
            "backward_passes": 0,
            "torch": torch.__version__,
        }
        for doc in report["documents"].values():
            doc.pop("offsets", None)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        backend.close()
    print(json.dumps(report["summary"]["analysis"]["auc"], indent=1)[:4000])


if __name__ == "__main__":
    main()
