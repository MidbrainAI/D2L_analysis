"""Per-passage LoRA fitting helpers: trainable writes, forward-KL targets, batching, decoding, and parity checks."""

from __future__ import annotations

import contextlib
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from official_d2l_k1_representation_ladder import compile_k1_representation_batch
from official_d2l_online_batch_distillation import (
    gather_response_logits,
    pad_sequences,
    render_prompt,
    response_position_mask,
    response_token_lengths,
)
from qa_protocol_protocol import digest
from reader_banks import compact_topk, sparse_coarsened_forward_kl

FAMILY = "squad"
RANK = 8
TOP_K = 128
BATCH = 16
ARM_SPEC = {
    "A": {"down": "d2l", "attention": False, "label": "rank-8 down_proj from D2L init"},
    "B": {"down": "random", "attention": False, "label": "rank-8 down_proj from random init"},
    "C": {"down": "d2l+random", "attention": False, "label": "rank-16 down_proj: D2L rows + 8 new rows"},
    "D": {"down": None, "attention": True, "label": "rank-8 attention K/V hooks; down_proj bias only"},
    "D2": {"down": "frozen_native", "attention": True, "label": "native down_proj frozen + attention K/V"},
}


def pad_targets(targets: list[dict], width: int) -> dict[str, torch.Tensor]:
    """Stack per-row compact targets ([R,k] indices/log_probs, [R] tail) with zero padding to width."""
    k = targets[0]["indices"].shape[-1]
    indices = torch.zeros(len(targets), width, k, dtype=torch.int32)
    log_probs = torch.zeros(len(targets), width, k, dtype=torch.bfloat16)
    tail = torch.zeros(len(targets), width, dtype=torch.float32)
    for i, t in enumerate(targets):
        r = t["indices"].shape[0]
        indices[i, :r], log_probs[i, :r], tail[i, :r] = t["indices"], t["log_probs"], t["tail_log_prob"]
    return {"indices": indices, "log_probs": log_probs, "tail_log_prob": tail}


def kaiming_rows(shape: tuple[int, int, int], generator: torch.Generator) -> torch.Tensor:
    tensor = torch.empty(*shape, dtype=torch.float32)
    for layer in tensor:
        torch.nn.init.kaiming_uniform_(layer, a=math.sqrt(5), generator=generator)
    return tensor


def gap_closure(native: float, arm: float, teacher: float) -> float | None:
    gap = teacher - native
    return None if abs(gap) < 1e-9 else (arm - native) / gap


def seed_int(*parts) -> int:
    return int(digest(list(parts))[:8], 16)


# ----------------------------------------------------------------------------- model-side primitives
class Trainable(torch.nn.Module):
    """Master fp32 factors per arm, installed directly: released D2L factors are fp32, so step 0 equals native/bias-only."""

    def __init__(self, arm: str, native_doc: dict, bias: dict, attention_shapes: dict | None, generator, device,
                 extra_rows: int = RANK):
        super().__init__()
        spec = ARM_SPEC[arm]
        self.arm, self.device = arm, device
        self.bias_a = bias["A"].float().to(device)  # [26, 8, 9216] fp32, frozen
        self.bias_b = bias["B"].float().to(device)
        self.frozen_a = self.frozen_b = None
        self.down_a = self.down_b = None
        layers, _, d_in = native_doc["A"].shape
        d_out = native_doc["B"].shape[-1]
        if spec["down"] == "d2l":
            a, b = native_doc["A"].float(), native_doc["B"].float()
        elif spec["down"] == "random":
            a, b = kaiming_rows((layers, RANK, d_in), generator), torch.zeros(layers, RANK, d_out)
        elif spec["down"] == "d2l+random":
            a = torch.cat([native_doc["A"].float().cpu(), kaiming_rows((layers, extra_rows, d_in), generator)], dim=1)
            b = torch.cat([native_doc["B"].float().cpu(), torch.zeros(layers, extra_rows, d_out)], dim=1)
        elif spec["down"] == "frozen_native":
            self.frozen_a, self.frozen_b = native_doc["A"].float().to(device), native_doc["B"].float().to(device)
            a = b = None
        else:
            a = b = None
        if a is not None:
            self.down_a = torch.nn.Parameter(a.to(device))
            self.down_b = torch.nn.Parameter(b.to(device))
        self.attention = None
        if spec["attention"]:
            if attention_shapes is None:
                raise ValueError("attention arms need projection shapes")
            self.attention = torch.nn.ParameterDict()
            for name, (in_features, out_features) in attention_shapes.items():
                self.attention[f"{name}_a"] = torch.nn.Parameter(
                    kaiming_rows((layers, RANK, in_features), generator).to(device))
                self.attention[f"{name}_b"] = torch.nn.Parameter(torch.zeros(layers, RANK, out_features, device=device))

    def down_state(self) -> dict:
        parts_a, parts_b = [], []
        if self.down_a is not None:
            parts_a.append(self.down_a)
            parts_b.append(self.down_b)
        if self.frozen_a is not None:
            parts_a.append(self.frozen_a)
            parts_b.append(self.frozen_b)
        parts_a.append(self.bias_a)
        parts_b.append(self.bias_b)
        return {"down_proj": {"A": torch.cat(parts_a, dim=1)[None], "B": torch.cat(parts_b, dim=1)[None]}}

    def attention_factors(self) -> dict | None:
        if self.attention is None:
            return None
        names = sorted({key[:-2] for key in self.attention})
        return {name: (self.attention[f"{name}_a"], self.attention[f"{name}_b"]) for name in names}

    def trainable_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.parameters())  # installed fp32 bytes

    def export(self) -> dict:
        return {name: p.detach().cpu() for name, p in self.named_parameters()}


def attention_hook(a: torch.Tensor, b: torch.Tensor, scale: float):
    def hook(_module, inputs, output):
        delta = (inputs[0].to(a.dtype) @ a.transpose(-1, -2) @ b) * scale
        return output + delta.to(output.dtype)
    return hook


@contextlib.contextmanager
def installed(backend, down_state: dict | None, attention: dict | None, scale: float):
    """Install a merged down_proj state through upstream and optional K/V hooks; always reset afterwards."""
    from ctx_to_lora.utils import get_layers

    model = backend.model
    handles = []
    model.reset()
    try:
        if down_state is not None:
            model.patch_lora_forward()
            backend.install(model.base_model, model.hypernet.layer_indices, down_state,
                            torch.ones(1, device=model.device, dtype=torch.int32))
        if attention:
            layers = get_layers(model.base_model)
            for name, (a, b) in attention.items():
                for index, layer in enumerate(layers):
                    module = getattr(layer.self_attn, f"{name}_proj")
                    handles.append(module.register_forward_hook(attention_hook(a[index], b[index], scale)))
        yield
    finally:
        for handle in handles:
            handle.remove()
        model.reset()


def attention_shapes(backend) -> dict[str, tuple[int, int]]:
    from ctx_to_lora.utils import get_layers

    attn = get_layers(backend.model.base_model)[0].self_attn
    return {name: (getattr(attn, f"{name}_proj").in_features, getattr(attn, f"{name}_proj").out_features)
            for name in ("k", "v")}


def logits_for(backend, ids: torch.Tensor, mask: torch.Tensor, down_state, attention, scale: float) -> torch.Tensor:
    with installed(backend, down_state, attention, scale):
        return backend.model.base_model(input_ids=ids, attention_mask=mask, use_cache=False).logits


def greedy(backend, prompt: torch.Tensor, down_state, attention, scale: float, budget: int, counter: dict,
           kind: str) -> tuple[str, int]:
    history: list[int] = []
    eos = backend.model.base_model.generation_config.eos_token_id
    stops = set(eos if isinstance(eos, list) else [eos])
    for _ in range(budget):
        ids = torch.cat([prompt, torch.tensor(history, dtype=torch.long, device=prompt.device)])[None]
        counter[f"{kind}_forwards"] = counter.get(f"{kind}_forwards", 0) + 1
        counter[f"{kind}_input_positions"] = counter.get(f"{kind}_input_positions", 0) + ids.numel()
        with torch.inference_mode():
            logits = logits_for(backend, ids, torch.ones_like(ids), down_state, attention, scale)
        token = int(logits[0, -1].float().argmax())
        history.append(token)
        if token in stops:
            break
    return backend.tokenizer.decode(history, skip_special_tokens=True).strip(), len(history)


def native_factors(backend, plan: dict, documents: dict) -> tuple[dict, dict, dict, dict]:
    """Native merged state plus split document rows and bias rows (bf16, device)."""
    compiled, accounting = compile_k1_representation_batch(backend, [plan], documents, 8)
    full = compiled.factors["down_proj"]
    doc = {"A": full["A"][0, :, :RANK].clone(), "B": full["B"][0, :, :RANK].clone()}
    bias = {"A": full["A"][0, :, RANK:].clone(), "B": full["B"][0, :, RANK:].clone()}
    if not torch.equal(bias["A"], compiled.bias_factors["down_proj"]["A"][0]):
        raise RuntimeError("bias rows are not the trailing rank block")
    return compiled.factors, doc, bias, accounting


def bias_only_state(bias: dict) -> dict:
    return {"down_proj": {"A": bias["A"][None], "B": bias["B"][None]}}


# ----------------------------------------------------------------------------- self-study
def _sample_batch_ids(backend, prompts: list, max_new_tokens: int, seed: int, temperature: float = 0.7):
    model = backend.model.base_model
    input_ids, attention_mask = pad_sequences(prompts, backend.tokenizer.pad_token_id, left=True, device=model.device)
    torch.manual_seed(seed)
    with torch.inference_mode():
        generated = model.generate(input_ids, attention_mask=attention_mask, do_sample=True, temperature=temperature,
                                   top_p=0.95, max_new_tokens=max_new_tokens, use_cache=True,
                                   pad_token_id=backend.tokenizer.pad_token_id)
    continuations = generated[:, input_ids.shape[1]:].detach().cpu()
    eos = backend.tokenizer.eos_token_id
    eos_ids = tuple(eos) if isinstance(eos, (tuple, list)) else (eos,)
    lengths = response_token_lengths(continuations, eos_ids, backend.tokenizer.pad_token_id)
    return [c[:n] for c, n in zip(continuations, lengths, strict=True)]


def decode_questions(backend, continuations) -> list[str]:
    return [backend.tokenizer.decode(c, skip_special_tokens=True).strip() for c in continuations]


def teacher_rows(backend, text: str, rows: list[dict], max_new_tokens: int, counter: dict) -> list[dict]:
    """Add source-visible greedy teacher answers and top-k replay targets to question rows, in place."""
    teacher_prompts = [render_prompt(backend, r["question"], FAMILY, text) for r in rows]
    student_prompts = [render_prompt(backend, r["question"], FAMILY) for r in rows]
    responses = []
    for start in range(0, len(rows), BATCH):
        responses.extend(_greedy_batch_ids(backend, teacher_prompts[start:start + BATCH], max_new_tokens))
    counter["teacher_answer_forwards"] = counter.get("teacher_answer_forwards", 0) + len(rows)
    replay = [torch.cat((p, r[:-1])) for p, r in zip(teacher_prompts, responses, strict=True)]
    for start in range(0, len(rows), 8):
        ids, mask = pad_sequences(replay[start:start + 8], backend.tokenizer.pad_token_id, device=backend.model.device)
        with torch.inference_mode():
            backend.model.reset()
            logits = backend.model.base_model(ids, attention_mask=mask, use_cache=False).logits
            gathered = gather_response_logits(logits, [p.numel() for p in teacher_prompts[start:start + 8]],
                                              [r.numel() for r in responses[start:start + 8]])
            compact = compact_topk(gathered, TOP_K)
        counter["teacher_replay_forwards"] = counter.get("teacher_replay_forwards", 0) + ids.shape[0]
        counter["teacher_replay_positions"] = counter.get("teacher_replay_positions", 0) + int(mask.sum())
        for offset, row in enumerate(rows[start:start + 8]):
            r = responses[start + offset].numel()
            row["target"] = {"indices": compact["indices"][offset, :r].clone(),
                             "log_probs": compact["log_probs"][offset, :r].clone(),
                             "tail_log_prob": compact["tail_log_prob"][offset, :r].clone()}
    for row, prompt, student, response in zip(rows, teacher_prompts, student_prompts, responses, strict=True):
        row.update(response_ids=response.clone(), response_text=backend.tokenizer.decode(response, skip_special_tokens=True).strip(),
                   student_prompt_ids=student.clone(), teacher_prompt_tokens=int(prompt.numel()))
    return rows


# ----------------------------------------------------------------------------- fitting
def make_batch(rows: list[dict], indices: list[int], pad_token_id: int, device) -> dict:
    chosen = [rows[i] for i in indices]
    replay = [torch.cat((r["student_prompt_ids"], r["response_ids"][:-1])) for r in chosen]
    ids, mask = pad_sequences(replay, pad_token_id, device=device)
    response_lengths = [int(r["response_ids"].numel()) for r in chosen]
    width = max(response_lengths)
    return {"ids": ids, "mask": mask, "prompt_lengths": [int(r["student_prompt_ids"].numel()) for r in chosen],
            "response_lengths": response_lengths, "targets": pad_targets([r["target"] for r in chosen], width),
            "position_mask": response_position_mask(response_lengths, device=device)}


def kl_loss(backend, trainable: Trainable, batch: dict, scale: float) -> torch.Tensor:
    logits = logits_for(backend, batch["ids"], batch["mask"], trainable.down_state(), trainable.attention_factors(), scale)
    gathered = gather_response_logits(logits, batch["prompt_lengths"], batch["response_lengths"])
    return sparse_coarsened_forward_kl(batch["targets"], gathered, batch["position_mask"])


def gradient_isolation(backend, trainable: Trainable) -> dict:
    own = [p.grad for p in trainable.parameters()]
    if any(g is None for g in own) or not all(torch.isfinite(g).all() for g in own) or not any(g.count_nonzero() for g in own):
        raise RuntimeError("trainable factors did not receive finite nonzero gradients")
    if any(p.grad is not None for p in backend.model.base_model.parameters()):
        raise RuntimeError("frozen backbone received gradients")
    if any(p.grad is not None for p in backend.model.hypernet.parameters()):
        raise RuntimeError("frozen hypernetwork received gradients")
    return {"trainable_tensors": len(own), "finite": True, "nonzero": True}


def step_zero_parity(backend, trainable: Trainable, reference_state: dict, prompt: torch.Tensor, scale: float,
                     *, strict: bool, budget: int) -> dict:
    """Step 0 must reproduce the reference merged state. Strict arms (identical tensors) require torch.equal
    logits; the rank-16 arm appends zero-B rows, which changes the fp32 GEMM reduction order, so it requires
    identical shared rows and an identical fresh greedy response while the logit difference is recorded."""
    ids = prompt[None]
    state = trainable.down_state()
    with torch.inference_mode():
        expected = logits_for(backend, ids, torch.ones_like(ids), reference_state, None, scale)
        actual = logits_for(backend, ids, torch.ones_like(ids), state, trainable.attention_factors(), scale)
    max_abs = float((expected - actual).abs().max())
    result = {"logits_exact": bool(torch.equal(expected, actual)), "max_abs": max_abs, "strict": strict}
    if strict:
        if not result["logits_exact"]:
            raise RuntimeError(f"step-zero parity failed for arm {trainable.arm}: max abs {max_abs}")
        return result
    ref_a, ref_b = reference_state["down_proj"]["A"][0], reference_state["down_proj"]["B"][0]
    a, b = state["down_proj"]["A"][0], state["down_proj"]["B"][0]
    shared = ref_a.shape[1] - RANK  # reference document rows precede its bias rows
    rows_equal = (torch.equal(a[:, :shared], ref_a[:, :shared]) and torch.equal(b[:, :shared], ref_b[:, :shared])
                  and torch.equal(a[:, -RANK:], ref_a[:, -RANK:]) and torch.equal(b[:, -RANK:], ref_b[:, -RANK:])
                  and torch.count_nonzero(b[:, shared:-RANK]) == 0)
    counter: dict = {}
    expected_text, _ = greedy(backend, prompt, reference_state, None, scale, budget, counter, "parity")
    actual_text, _ = greedy(backend, prompt, state, trainable.attention_factors(), scale, budget, counter, "parity")
    result.update(shared_rows_exact=bool(rows_equal), response_exact=expected_text == actual_text,
                  reference_response=expected_text, response=actual_text)
    if not rows_equal or expected_text != actual_text:
        raise RuntimeError(f"step-zero parity failed for arm {trainable.arm}: {result}")
    return result


def bootstrap_delta(by_doc: dict[str, tuple[float, float]], seed: int, resamples: int = 4000) -> list[float]:
    docs = sorted(by_doc)
    deltas = np.asarray([by_doc[d][0] - by_doc[d][1] for d in docs])
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(docs), (resamples, len(docs)))
    return np.quantile(deltas[index].mean(1), [0.025, 0.975]).tolist()


CATEGORY_REQUESTS = {
    "direct_factual_use": "Write one concrete question about a specific fact in the source. Output only the question.",
    "paraphrase": "Write one concrete question asking a respondent to paraphrase a specific source fact. Output only the question.",
    "summary_structure": "Write one concrete question asking for a concise summary of the source's main facts or structure. Output only the question.",
    "relationship_comparison": "Write one concrete question asking to compare or relate two specific source facts. Output only the question.",
    "inference": "Write one concrete question requiring a conservative inference supported by the source. Output only the question.",
    "source_insufficient": "Write one concrete question whose answer is not established by this source. Output only the question.",
}


def _source_prompt(category: str) -> str:
    if category not in CATEGORY_REQUESTS:
        raise ValueError(f"unknown SELF_STUDY probe category: {category}")
    return (
        "Use only the supplied source; do not use outside knowledge. "
        f"Category ({category}): {CATEGORY_REQUESTS[category]}"
    )


def _greedy_batch_ids(backend, prompts: list, max_new_tokens: int):
    from official_d2l_online_batch_distillation import (
        pad_sequences,
        response_token_lengths,
    )

    model = backend.model.base_model
    device = model.device
    input_ids, attention_mask = pad_sequences(
        prompts,
        backend.tokenizer.pad_token_id,
        left=True,
        device=device,
    )
    with backend.torch.inference_mode():
        generated = model.generate(
            input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=backend.tokenizer.pad_token_id,
        )
    continuations = generated[:, input_ids.shape[1] :].detach().cpu()
    eos = backend.tokenizer.eos_token_id
    eos_ids = tuple(eos) if isinstance(eos, (tuple, list)) else (eos,)
    lengths = response_token_lengths(continuations, eos_ids, backend.tokenizer.pad_token_id)
    return [continuation[:length] for continuation, length in zip(continuations, lengths, strict=True)]
