# ruff: noqa: B023 -- benchmark closures run within the iteration that defines them
"""SERVING_COST: per-request serving cost of internalized memory (D2L, D2L plus reader) versus ICL.

Two parts:

* ``account`` (CPU, exact arithmetic) counts, per arm and as functions of memory length
  L and document count K: prefill positions, decode-time attended positions, MACs per
  request and per generated token, per-user persisted and resident bytes, compile MACs,
  and break-even queries per memory. Arms: ICL, ICL with a reused prefix cache, native
  D2L, and D2L plus the rank-effect reader (as implemented, and with the reader's key and
  value projections factored through the static B rows, which the reader's layer norm
  makes exact up to its epsilon).
* ``benchmark`` (GPU, the ``timing`` unit of the distractor-curve plan) measures compile versus prefill time, batch-1
  time to first token and decode time per arm, and multi-user decode throughput with
  per-user state (left-padded per-user contexts against per-user LoRA factors).

Counting conventions: one MAC per multiply-add; embeddings and the LM head are counted;
softmax, norms, and activations are not. Attention to a memory position costs
``2 * heads * head_dim`` MACs per layer (scores and values). Gemma-2 local layers attend
to at most ``sliding_window`` positions.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

BF16 = 2
FP32 = 4


@dataclass(frozen=True)
class ModelShape:
    name: str
    layers: int
    hidden: int
    intermediate: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    sliding_window: int | None  # applies to every other layer (Gemma-2) when set
    rank: int = 8  # D2L rank per compiled chunk, also the shared bias rank
    latent: int = 512  # D2L pre-head latent width
    reader_bottleneck: int = 32


GEMMA2_2B = ModelShape("gemma-2-2b-it", 26, 2304, 9216, 8, 4, 256, 256000, 4096)
QWEN3_4B = ModelShape("Qwen3-4B-Instruct-2507", 36, 2560, 9728, 32, 8, 128, 151936, None)
SHAPES = {"gemma2": GEMMA2_2B, "qwen4": QWEN3_4B}


# -- exact accounting (CPU) --------------------------------------------------------
def linear_macs_per_token(m: ModelShape) -> int:
    attention = m.hidden * (m.heads + 2 * m.kv_heads) * m.head_dim + m.heads * m.head_dim * m.hidden
    mlp = 3 * m.hidden * m.intermediate
    return m.layers * (attention + mlp) + m.hidden * m.vocab


def attended(m: ModelShape, positions: int) -> int:
    """Total attended positions summed over layers for one query token at this context length."""
    if m.sliding_window is None:
        return m.layers * positions
    local = m.layers // 2
    return (m.layers - local) * positions + local * min(positions, m.sliding_window)


def attention_macs(m: ModelShape, positions: int) -> int:
    return 2 * m.heads * m.head_dim * attended(m, positions)


def kv_bytes(m: ModelShape, positions: int) -> int:
    return 2 * m.kv_heads * m.head_dim * BF16 * attended(m, positions)


def lora_macs_per_token(m: ModelShape, rank: int) -> int:
    return m.layers * rank * (m.intermediate + m.hidden)


def reader_macs_per_token(m: ModelShape, slots: int, factored: bool) -> int:
    """Rank-effect ActiveMemoryTransition per token, all layers.

    As implemented: rank effects (slots x hidden outer products), their key and value
    projections, and the slot softmax read. Factored: keys and values of each rank are
    precomputed from its B row once per state, so the per-token cost is the read only.
    """
    b = m.reader_bottleneck
    fixed = 4 * m.hidden * b + b * m.hidden  # state, query, bias-effect, layer, output projections
    read = 2 * slots * b
    if factored:
        return m.layers * (fixed + read)
    effects = slots * m.hidden
    projections = 2 * slots * m.hidden * b
    return m.layers * (fixed + read + effects + projections)


def reader_state_macs(m: ModelShape, slots: int) -> int:
    """One-off per-state precompute for the factored reader: K/V of every B row."""
    return m.layers * 2 * slots * m.hidden * m.reader_bottleneck


def request_cost(m: ModelShape, arm: str, *, memory_tokens: int, documents: int, question_tokens: int,
                 output_tokens: int) -> dict[str, Any]:
    """Per-request cost for one arm; memory is L tokens over K documents (one chunk each)."""
    linear = linear_macs_per_token(m)
    rank = m.rank * (documents + 1)  # documents plus the shared bias
    slots = m.rank * documents
    if arm in ("icl", "icl_prefix_cache"):
        prefill_positions = memory_tokens + question_tokens if arm == "icl" else question_tokens
        first = 0 if arm == "icl" else memory_tokens
        prefill_attention = sum(attention_macs(m, p + 1) for p in range(first, memory_tokens + question_tokens))
        prefill = prefill_positions * linear + prefill_attention
        context = memory_tokens + question_tokens
        decode_attention = sum(attention_macs(m, context + t + 1) for t in range(output_tokens))
        decode = output_tokens * linear + decode_attention
        per_user_bytes = kv_bytes(m, memory_tokens)
        return {"arm": arm, "prefill_positions": prefill_positions,
                "decode_attended_positions_per_token": attended(m, context + output_tokens),
                "prefill_macs": prefill, "decode_macs": decode,
                "decode_macs_per_token": decode / output_tokens, "request_macs": prefill + decode,
                "persisted_kv_slots": memory_tokens if arm == "icl_prefix_cache" else 0,
                "per_user_resident_bytes": per_user_bytes, "per_user_persisted_bytes": per_user_bytes
                if arm == "icl_prefix_cache" else memory_tokens * 4, "state_precompute_macs": 0}
    if arm not in ("d2l", "d2l_reader", "d2l_reader_factored"):
        raise ValueError(f"unknown arm {arm}")
    extra = lora_macs_per_token(m, rank)
    if arm != "d2l":
        extra += reader_macs_per_token(m, slots, factored=arm == "d2l_reader_factored")
    prefill_attention = sum(attention_macs(m, p + 1) for p in range(question_tokens))
    prefill = question_tokens * (linear + extra) + prefill_attention
    decode_attention = sum(attention_macs(m, question_tokens + t + 1) for t in range(output_tokens))
    decode = output_tokens * (linear + extra) + decode_attention
    factor_bytes = m.layers * rank * (m.intermediate + m.hidden) * BF16
    return {"arm": arm, "prefill_positions": question_tokens,
            "decode_attended_positions_per_token": attended(m, question_tokens + output_tokens),
            "prefill_macs": prefill, "decode_macs": decode, "decode_macs_per_token": decode / output_tokens,
            "request_macs": prefill + decode, "persisted_kv_slots": 0,
            "per_user_resident_bytes": factor_bytes,
            "per_user_persisted_bytes": m.layers * m.rank * m.latent * FP32 * documents,
            "state_precompute_macs": reader_state_macs(m, slots) if arm == "d2l_reader_factored" else 0}


def prefill_macs(m: ModelShape, memory_tokens: int) -> int:
    return memory_tokens * linear_macs_per_token(m) + sum(attention_macs(m, p + 1) for p in range(memory_tokens))


def break_even(arm: tuple[int, int], base: tuple[int, int]) -> float | None:
    """Queries after which (per-request, one-off) ``arm`` costs no more than ``base``; 0 if never worse."""
    (arm_request, arm_once), (base_request, base_once) = arm, base
    if arm_once <= base_once and arm_request <= base_request:
        return 0.0
    if arm_request >= base_request:
        return None
    return (arm_once - base_once) / (base_request - arm_request)


def compile_macs(m: ModelShape, memory_tokens: int) -> int:
    """Upper bound: one full forward of the context encoder over the memory (the Perceiver is smaller)."""
    return prefill_macs(m, memory_tokens)


def account(m: ModelShape, *, lengths: list[int], documents: list[int], question_tokens: int, output_tokens: int,
            document_tokens: int) -> dict[str, Any]:
    arms = ("icl", "icl_prefix_cache", "d2l", "d2l_reader", "d2l_reader_factored")
    by_length = [{"memory_tokens": L, "documents": 1, "arms": [request_cost(m, a, memory_tokens=L, documents=1,
                  question_tokens=question_tokens, output_tokens=output_tokens) for a in arms],
                  "compile_macs": compile_macs(m, L)} for L in lengths]
    by_count = [{"memory_tokens": K * document_tokens, "documents": K,
                 "arms": [request_cost(m, a, memory_tokens=K * document_tokens, documents=K,
                                       question_tokens=question_tokens, output_tokens=output_tokens) for a in arms],
                 "compile_macs": K * compile_macs(m, document_tokens)} for K in documents]
    for block in by_length + by_count:
        # One-off cost per memory: D2L compiles it; prefix-cached ICL prefills it once to build the cache.
        one_off = {"icl": 0, "icl_prefix_cache": prefill_macs(m, block["memory_tokens"]),
                   **{arm: block["compile_macs"] for arm in ("d2l", "d2l_reader", "d2l_reader_factored")}}
        for arm in block["arms"]:
            arm["one_off_macs"] = one_off[arm["arm"]] + arm["state_precompute_macs"]
        costs = {a["arm"]: (a["request_macs"], a["one_off_macs"]) for a in block["arms"]}
        block["break_even_queries"] = {
            f"{arm}_vs_{base}": break_even(costs[arm], costs[base])
            for base in ("icl", "icl_prefix_cache") for arm in ("d2l", "d2l_reader", "d2l_reader_factored")}
    return {"model": asdict(m), "question_tokens": question_tokens, "output_tokens": output_tokens,
            "document_tokens": document_tokens, "by_memory_length": by_length, "by_document_count": by_count}


# -- measured benchmark (GPU) ---------------------------------------------------------
def _timed(function, repeats: int, warmup: int = 2) -> dict[str, float]:
    import torch

    for _ in range(warmup):
        function()
    values = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        function()
        torch.cuda.synchronize()
        values.append(time.perf_counter() - started)
    return {"median_seconds": statistics.median(values), "min_seconds": min(values), "repeats": repeats}


def memory_text(backend, texts: list[str], tokens: int) -> str:
    """Concatenate suite passages and cut to an exact compiler token length."""
    ids: list[int] = []
    for text in texts:
        ids += backend.tokenizer(text + "\n\n", add_special_tokens=False)["input_ids"]
        if len(ids) >= tokens:
            break
    if len(ids) < tokens:
        raise ValueError("not enough passage text for the requested memory length")
    return backend.tokenizer.decode(ids[:tokens])


def benchmark(backend, texts: list[str], *, lengths: list[int], documents: list[int], batch_sizes: list[int],
              output_tokens: int = 64, repeats: int = 5) -> dict[str, Any]:
    """Measured serving costs on the run's GPU; untrained reader of the presented shape."""
    import torch
    from ctx_to_lora.utils import get_layers
    from distractor_curves_core import bank_representation, bank_state, compile_documents
    from official_d2l_k1_representation_evaluation import generate_reader_condition
    from official_d2l_k1_representation_ladder import make_representation_transition

    model = backend.model.base_model
    layers = get_layers(model)
    scaling = model.peft_config["default"].lora_alpha
    question = "Which year is mentioned in the passage?"
    results: dict[str, Any] = {"device": torch.cuda.get_device_name(), "torch": torch.__version__,
                               "model": backend.model_id, "output_tokens": output_tokens, "rows": []}

    def generate(state=None, context=None, tokens=output_tokens):
        backend.model.reset()
        prompt = backend.templates["squad"].format(input=question)
        if context:
            prompt = context + "\n\n" + prompt
        ids = backend.tokenizer.apply_chat_template([{"role": "system", "content": ""},
                                                     {"role": "user", "content": prompt}], tokenize=True,
                                                    add_generation_prompt=True, return_tensors="pt",
                                                    add_special_tokens=False).to(model.device)
        try:
            if state is not None:
                backend.model.patch_lora_forward()
                backend.install(model, backend.model.hypernet.layer_indices, state,
                                torch.ones(1, device=model.device, dtype=torch.int32))
            with torch.inference_mode():
                model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False, max_new_tokens=tokens,
                               min_new_tokens=tokens, use_cache=True, pad_token_id=backend.tokenizer.pad_token_id)
        finally:
            backend.model.reset()

    def per_arm(name: str, run, **fields) -> None:
        first = _timed(lambda: run(1), repeats)
        full = _timed(lambda: run(output_tokens + 1), repeats)
        results["rows"].append({"arm": name, **fields, "time_to_first_token_seconds": first["median_seconds"],
                                "decode_seconds_per_token": (full["median_seconds"] - first["median_seconds"])
                                / output_tokens, "total_seconds": full["median_seconds"]})

    for length in lengths:
        text = memory_text(backend, texts, length)
        ids = backend.tokenize_context({"context": [text]}, backend.tokenizer)["ctx_ids"][0]
        chunk = torch.tensor([ids], device=model.device)
        compile_time = _timed(lambda: backend.model.generate_weights(chunk, torch.ones_like(chunk), None), repeats)
        with torch.inference_mode():
            prefill = _timed(lambda: model(chunk, use_cache=False), repeats)
        results["rows"].append({"arm": "compile_vs_prefill", "memory_tokens": len(ids),
                                "compile_seconds": compile_time["median_seconds"],
                                "prefill_seconds": prefill["median_seconds"]})
        record = compile_documents(backend, [text])
        state = bank_state(backend, record)
        per_arm("icl", lambda n: generate(context=text, tokens=n), memory_tokens=length, documents=1)
        per_arm("d2l", lambda n: generate(state=state, tokens=n), memory_tokens=length, documents=1)
    reader = make_representation_transition("rank-effect", model.config.hidden_size,
                                            backend.model.hypernet.config.latent_size, len(layers), 32).to(model.device)
    reader.eval()
    passage_records = compile_documents(backend, texts[: max(documents)])
    for k in documents:
        records = passage_records[:k]
        state = bank_state(backend, records)
        compiled = bank_representation(backend, records)
        context = "\n\n".join(texts[:k])
        per_arm("icl", lambda n: generate(context=context, tokens=n), memory_tokens=None, documents=k)
        per_arm("d2l", lambda n: generate(state=state, tokens=n), memory_tokens=None, documents=k)

        def with_reader(n):
            generate_reader_condition(backend, layers, compiled, reader, "rank-effect", scaling, question, n, "squad")

        first = _timed(lambda: with_reader(1), repeats)
        full = _timed(lambda: with_reader(output_tokens + 1), repeats)
        results["rows"].append({"arm": "d2l_reader", "documents": k,
                                "time_to_first_token_seconds": first["median_seconds"],
                                "decode_seconds_per_token": (full["median_seconds"] - first["median_seconds"])
                                / output_tokens, "total_seconds": full["median_seconds"],
                                "note": "reader decode does not stop at EOS; min_new_tokens not enforced"})
    results["rows"].extend(throughput(backend, texts, batch_sizes, output_tokens, repeats))
    return results


def throughput(backend, texts: list[str], batch_sizes: list[int], output_tokens: int, repeats: int) -> list[dict]:
    """Decode throughput with B users, each with their own memory (one ~150-token passage, K=4)."""
    import torch
    from distractor_curves_core import bank_state, compile_documents

    model = backend.model.base_model
    tokenizer = backend.tokenizer
    rows = []
    question = "Which year is mentioned in the passage?"
    for batch in batch_sizes:
        user_texts = ["\n\n".join(texts[(4 * u + j) % len(texts)] for j in range(4)) for u in range(batch)]
        prompts = []
        for text in user_texts:
            prompt = text + "\n\n" + backend.templates["squad"].format(input=question)
            prompts.append(tokenizer.apply_chat_template([{"role": "system", "content": ""},
                                                          {"role": "user", "content": prompt}], tokenize=False,
                                                         add_generation_prompt=True))
        tokenizer.padding_side = "left"
        icl = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        bare = tokenizer([tokenizer.apply_chat_template([{"role": "system", "content": ""},
                                                         {"role": "user",
                                                          "content": backend.templates["squad"].format(input=question)}],
                                                        tokenize=False, add_generation_prompt=True)] * batch,
                         return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        states = [bank_state(backend, compile_documents(backend, [texts[(4 * u + j) % len(texts)] for j in range(4)]))
                  for u in range(batch)]
        stacked = {m: {key: torch.cat([s[m][key] for s in states]) for key in ("A", "B")} for m in states[0]}

        def run_icl():
            backend.model.reset()
            with torch.inference_mode():
                model.generate(**icl, do_sample=False, max_new_tokens=output_tokens, min_new_tokens=output_tokens,
                               pad_token_id=tokenizer.pad_token_id)

        def run_d2l():
            backend.model.reset()
            backend.model.patch_lora_forward()
            backend.install(model, backend.model.hypernet.layer_indices, stacked,
                            torch.ones(batch, device=model.device, dtype=torch.int32))
            try:
                with torch.inference_mode():
                    model.generate(**bare, do_sample=False, max_new_tokens=output_tokens,
                                   min_new_tokens=output_tokens, pad_token_id=tokenizer.pad_token_id)
            finally:
                backend.model.reset()

        for name, run in (("icl", run_icl), ("d2l", run_d2l)):
            torch.cuda.reset_peak_memory_stats()
            timing = _timed(run, repeats)
            rows.append({"arm": f"throughput_{name}", "batch": batch, "documents_per_user": 4,
                         "prompt_positions": int((icl if name == "icl" else bare)["input_ids"].shape[1]),
                         "tokens_per_second": batch * output_tokens / timing["median_seconds"],
                         "peak_memory_bytes": torch.cuda.max_memory_allocated()})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("artifacts/serving_cost/accounting-v1.json"))
    parser.add_argument("--question-tokens", type=int, default=40)  # Mean D2L question length was 40.3 tokens.
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--document-tokens", type=int, default=150)
    args = parser.parse_args()
    lengths = [150, 256, 512, 1024, 2048, 4096]
    counts = [1, 2, 4, 8, 16]
    report = {"experiment": "SERVING_COST", "part": "exact accounting",
              "models": {name: account(shape, lengths=lengths, documents=counts, question_tokens=args.question_tokens,
                                       output_tokens=args.output_tokens, document_tokens=args.document_tokens)
                         for name, shape in SHAPES.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    gemma = report["models"]["gemma2"]["by_memory_length"][0]
    print(json.dumps({a["arm"]: {"request_macs": a["request_macs"], "per_user_resident_bytes":
                                 a["per_user_resident_bytes"]} for a in gemma["arms"]}, indent=2))


if __name__ == "__main__":
    main()
