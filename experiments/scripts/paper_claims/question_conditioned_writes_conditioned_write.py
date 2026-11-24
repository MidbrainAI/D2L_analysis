"""Question-conditioned latent write (Table 9).

Each document is stored as D2L's normalized pre-head latent z (26 layers x 8 slots x 512). At request time,
inside the normal PK-free prefill, layer l's MLP input at the prompt rows (the question) conditions a shared
encoder that adds a residual to the bank's slots at that layer, Z'_l = Z_l + Enc(Z_l, u_l, l); D2L's frozen
per-layer head maps Z'_l to the layer's rank-8K factors, merged with the shared bias exactly as native D2L
merges, and the layer's down_proj applies them with D2L's own ``lora_forward``. The encoder's output
projection starts at zero, so update zero is native D2L (gate).

Conditioning never reads response rows (rows at or after the last prompt token are excluded from u), so the
teacher-forced training sequence cannot leak the answer into the write, and cache-free greedy generation
recomputes the same write at every step.

Training (SQuAD): DISTRACTOR_CURVES's mixed-K schedule and banks over READOUT_LADDER's fitting questions, with
KL to the ICL teacher that sees the whole bank, over the teacher's cached greedy response to the question
with its source document. Only the encoder is trained.
"""

from __future__ import annotations

import contextlib
import math
import time
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional

from distractor_curves_core import K_VALUES

FAMILY = "squad"
WIDTH = 128
HEADS = 4
BLOCKS = 2
EXAMPLES_PER_UPDATE = 12
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
VALIDATION_INTERVAL = 500
RESUME_INTERVAL = 100
MAX_NEW_TOKENS = 32
GATE_TOLERANCE = 1e-4


# ----------------------------------------------------------------------------- encoder
class Block(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.cross_norm, self.self_norm, self.mlp_norm = nn.LayerNorm(width), nn.LayerNorm(width), nn.LayerNorm(width)
        self.cross = nn.MultiheadAttention(width, heads, batch_first=True)
        self.self_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, slots, question, question_padding):
        x = self.cross_norm(slots)
        slots = slots + self.cross(x, question, question, key_padding_mask=question_padding, need_weights=False)[0]
        x = self.self_norm(slots)
        slots = slots + self.self_attention(x, x, x, need_weights=False)[0]
        return slots + self.mlp(self.mlp_norm(slots))


class WriteEncoder(nn.Module):
    """Residual on a layer's bank slots, from the question rows' states at that layer."""

    def __init__(self, latent_width: int, hidden_width: int, layers: int, width: int = WIDTH, heads: int = HEADS,
                 blocks: int = BLOCKS):
        super().__init__()
        self.latent_in = nn.Linear(latent_width, width)
        self.question_in = nn.Linear(hidden_width, width)
        self.layer_embedding = nn.Embedding(layers, width)
        self.blocks = nn.ModuleList(Block(width, heads) for _ in range(blocks))
        self.out_norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, latent_width)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.config = {"latent_width": latent_width, "hidden_width": hidden_width, "layers": layers, "width": width,
                       "heads": heads, "blocks": blocks}

    def forward(self, slots: torch.Tensor, question: torch.Tensor, question_padding: torch.Tensor,
                layer: int) -> torch.Tensor:
        """slots [B, S, latent], question [B, T, hidden], question_padding [B, T] (True = ignore)."""
        x = self.latent_in(functional.layer_norm(slots.float(), (slots.shape[-1],)))
        x = x + self.layer_embedding.weight[layer]
        q = self.question_in(functional.layer_norm(question.float(), (question.shape[-1],)))
        for block in self.blocks:
            x = block(x, q, question_padding)
        return slots + self.out(self.out_norm(x)).to(slots.dtype)


# ----------------------------------------------------------------------------- per-layer head
def question_rows(states: torch.Tensor, prompt_lengths: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Prompt rows strictly before each row's last prompt token (the first answer predictor is excluded).

    Returns states [B, T, H] (T = max prompt length - 1) and a padding mask (True = not a question row)."""
    width = max(prompt_lengths) - 1
    positions = torch.arange(width, device=states.device)[None]
    padding = positions >= (torch.tensor(prompt_lengths, device=states.device)[:, None] - 1)
    return states[:, :width], padding


def layer_factors(hypernet, slots: torch.Tensor, layer: int, module: str = "down_proj") -> tuple[torch.Tensor, torch.Tensor]:
    """Merged rank-(S + r) factors for one layer from slots [B, S, latent], as head + scalers + merge do."""
    weight = hypernet.head.weight[layer]  # [d_latent, d_lora]
    flat = torch.einsum("bsd,dl->bsl", slots.to(weight.dtype), weight)
    d_in, d_out = hypernet.d_in[module], hypernet.d_out[module]
    rank = hypernet.scaler_A[module].shape[2]
    copies = slots.shape[1] // rank
    scaler_a = hypernet.scaler_A[module][0, layer].repeat(copies, 1)  # [S, 1]
    scaler_b = hypernet.scaler_B[module][0, layer].repeat(copies, 1)
    a = flat[..., :d_in] * scaler_a
    b = flat[..., d_in:d_in + d_out] * scaler_b
    bias_a = hypernet.bias_A[module][layer].to(a.dtype).expand(slots.shape[0], -1, -1)
    bias_b = hypernet.bias_B[module][layer].to(b.dtype).expand(slots.shape[0], -1, -1)
    return torch.cat([a, bias_a], dim=1), torch.cat([b, bias_b], dim=1)


def bank_slots(latents: torch.Tensor, layer: int) -> torch.Tensor:
    """latents [B, K, layers, r, d] -> slots of one layer [B, K*r, d] in bank order (the merge's rank order)."""
    batch, k, _, rank, width = latents.shape
    return latents[:, :, layer].reshape(batch, k * rank, width)


# ----------------------------------------------------------------------------- in-pass controller
@contextlib.contextmanager
def conditioned(backend, encoder: WriteEncoder | None, latents: torch.Tensor, prompt_lengths: list[int], *,
                question_override: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
                capture: list | None = None):
    """Install the conditioned write on every adapted layer for one forward (or a cache-free greedy loop).

    ``encoder=None`` applies the stored latents unchanged (native D2L through the per-layer path).
    ``question_override`` supplies per-layer (question states, padding) from another prompt (wrong-question
    control); ``capture`` collects each layer's (question states, padding) from this prompt."""
    from ctx_to_lora.modeling.lora_layer import lora_forward
    from ctx_to_lora.utils import get_layers

    model = backend.model
    hypernet = model.hypernet
    scaling = model.peft_config.lora_alpha
    layers = get_layers(model.base_model)
    model.reset()
    stash: dict[int, torch.Tensor] = {}
    handles, originals = [], []

    def pre_hook(index):
        def hook(_module, args):
            stash[index] = args[0]
        return hook

    def forward(x, *, module, index):
        if question_override is not None:
            question, padding = question_override[index]
        else:
            question, padding = question_rows(stash[index], prompt_lengths)
        if capture is not None:
            capture.append((question.detach(), padding))
        slots = bank_slots(latents, index)
        if encoder is not None:
            slots = encoder(slots, question, padding, index)
        a, b = layer_factors(hypernet, slots, index)
        n_qs = torch.ones(x.shape[0], device=x.device, dtype=torch.int32)
        return lora_forward(x, n_qs, x.shape[0], a, b, 0.0, scaling, self=module)

    try:
        for index in hypernet.layer_indices:
            layer = layers[index]
            handles.append(layer.mlp.register_forward_pre_hook(pre_hook(index)))
            module = layer.mlp.down_proj
            originals.append((module, module.forward))
            module.forward = partial(forward, module=module, index=index)
        yield
    finally:
        for handle in handles:
            handle.remove()
        for module, original in originals:
            module.forward = original
        model.reset()


def bank_latents(latents: dict[str, torch.Tensor], banks: list[list[str]], device) -> torch.Tensor:
    return torch.stack([torch.stack([latents[i] for i in bank]) for bank in banks]).to(device)


# ----------------------------------------------------------------------------- generation
def greedy(backend, encoder, latents: torch.Tensor, prompt: torch.Tensor, *, question_override=None,
           budget: int = MAX_NEW_TOKENS) -> str:
    """Cache-free greedy decoding for one question; the write is recomputed from the prompt rows each step."""
    model = backend.model.base_model
    eos = model.generation_config.eos_token_id
    stops = set(eos if isinstance(eos, list) else [eos])
    history: list[int] = []
    device = model.device
    for _ in range(budget):
        ids = torch.cat([prompt.to(device), torch.tensor(history, dtype=torch.long, device=device)])[None]
        with torch.inference_mode(), conditioned(backend, encoder, latents, [prompt.numel()],
                                                 question_override=question_override):
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                           logits_to_keep=1).logits
        token = int(logits[0, -1].float().argmax())
        history.append(token)
        if token in stops:
            break
    return backend.tokenizer.decode(history, skip_special_tokens=True).strip()


# ----------------------------------------------------------------------------- gates (GPU or CPU smoke)
def gates(backend, encoder: WriteEncoder, latents: dict[str, torch.Tensor], document_ids: list[str],
          query: dict) -> dict[str, Any]:
    """Per-layer construction equals native merged factors (relative error within the head-replay tolerance;
    the per-layer einsum can differ from D2L's batched head in the last bits, HEAD_INVERSION); a zero-initialized encoder
    equals the native LoRA path fed the same replayed factors exactly, and native D2L in greedy response;
    changing response tokens leaves the write unchanged."""
    from passage_fitting_core import logits_for
    from distractor_curves_core import factors_from_latents, head_bias
    from official_d2l_online_batch_distillation import render_prompt

    device = backend.model.device
    bank = bank_latents(latents, [document_ids], device)
    with torch.no_grad():
        generated = factors_from_latents(backend, bank[0])
        state = backend.merge(generated, torch.tensor([len(document_ids)], device=device, dtype=torch.int32),
                              lora_bias=head_bias(backend), bias_scaler=1.0)
        native = state["down_proj"]
        replayed_a, replayed_b = native["A"].clone(), native["B"].clone()
        factor_error = 0.0
        for layer in backend.model.hypernet.layer_indices:
            a, b = layer_factors(backend.model.hypernet, bank_slots(bank, layer), layer)
            factor_error = max(factor_error,
                               float((a[0] - native["A"][0, layer]).abs().max() / native["A"][0, layer].abs().max()),
                               float((b[0] - native["B"][0, layer]).abs().max() / native["B"][0, layer].abs().max()))
            replayed_a[:, layer], replayed_b[:, layer] = a.to(replayed_a.dtype), b.to(replayed_b.dtype)
        replayed = {**state, "down_proj": {**native, "A": replayed_a, "B": replayed_b}}
    prompt = render_prompt(backend, query["input"], FAMILY)
    ids = prompt[None].to(device)
    with torch.inference_mode():
        alpha = float(backend.model.peft_config.lora_alpha)
        expected = logits_for(backend, ids, torch.ones_like(ids), replayed, None, alpha)
        native_logits = logits_for(backend, ids, torch.ones_like(ids), state, None, alpha)
        with conditioned(backend, encoder, bank, [prompt.numel()]):
            actual = backend.model.base_model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
    logit_error = float((expected.float() - actual.float()).abs().max())
    native_logit_error = float((native_logits.float() - actual.float()).abs().max())
    native_text = backend.generate(query["input"], state=state, dataset=query["dataset"],
                                   max_new_tokens=MAX_NEW_TOKENS)["text"]
    conditioned_text = greedy(backend, encoder, bank, prompt)
    # leakage: the write must not change when response rows change
    tail_a = torch.cat([prompt, torch.tensor([1, 2, 3])])[None].to(device)
    tail_b = torch.cat([prompt, torch.tensor([7, 8, 9])])[None].to(device)
    captured = []
    with torch.inference_mode():
        for tail in (tail_a, tail_b):
            capture: list = []
            with conditioned(backend, encoder, bank, [prompt.numel()], capture=capture):
                backend.model.base_model(input_ids=tail, attention_mask=torch.ones_like(tail), use_cache=False,
                                         logits_to_keep=1)
            captured.append(capture)
    leak = max(float((a[0] - b[0]).abs().max()) for a, b in zip(*captured, strict=True))
    result = {"factor_max_relative": factor_error, "logit_max_abs": logit_error,
              "native_factor_logit_max_abs": native_logit_error, "response_equal": native_text == conditioned_text,
              "question_state_change_with_response": leak, "native_response": native_text,
              "conditioned_response": conditioned_text}
    if factor_error > GATE_TOLERANCE or logit_error != 0.0 or native_text != conditioned_text or leak != 0.0:
        raise RuntimeError(f"QUESTION_CONDITIONED_WRITES gates failed: {result}")
    return result


# ----------------------------------------------------------------------------- teacher responses (GPU)
def teacher_responses(backend, rows: list[dict[str, Any]], documents: dict[str, dict], batch: int = 16) -> dict:
    """The source-visible teacher's greedy response (token ids, EOS-terminated) to each question."""
    from passage_fitting_core import _greedy_batch_ids
    from official_d2l_online_batch_distillation import render_prompt

    backend.model.reset()
    out = {}
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        prompts = [render_prompt(backend, r["query"]["input"], FAMILY, documents[r["document"]["id"]]["text"])
                   for r in chunk]
        for row, response in zip(chunk, _greedy_batch_ids(backend, prompts, MAX_NEW_TOKENS), strict=True):
            out[row["query"]["id"]] = response.clone()
    return out


# ----------------------------------------------------------------------------- loss and training
def kl_loss(backend, encoder, plans: list[dict], latents: dict, responses: dict, documents: dict) -> tuple[Any, int]:
    """Token-mean KL(teacher with the whole bank in context || conditioned student) over the cached response."""
    return student_kl(backend, encoder, plans, latents, responses, teacher_log_probs(backend, plans, responses, documents))


def teacher_log_probs(backend, plans: list[dict], responses: dict, documents: dict) -> torch.Tensor:
    """The bank-visible teacher's log-probabilities over each plan's cached response (no gradient)."""
    from official_d2l_online_batch_distillation import gather_response_logits, pad_sequences, render_prompt

    model = backend.model.base_model
    targets = [responses[p["query"]["id"]] for p in plans]
    teacher_prompts = [render_prompt(backend, p["query"]["input"], FAMILY,
                                     "\n\n".join(documents[i]["text"] for i in p["bank_document_ids"])) for p in plans]
    with torch.no_grad():
        backend.model.reset()
        ids, mask = pad_sequences([torch.cat((p, t[:-1])) for p, t in zip(teacher_prompts, targets, strict=True)],
                                  backend.tokenizer.pad_token_id, device=backend.model.device)
        return gather_response_logits(model(input_ids=ids, attention_mask=mask, use_cache=False).logits,
                                      [p.numel() for p in teacher_prompts],
                                      [int(t.numel()) for t in targets]).float().log_softmax(-1)


def partner_states(backend, encoder, latents: dict, plan: dict) -> list:
    """Per-layer question states of the plan's partner question (same source) under its own write, detached."""
    from official_d2l_online_batch_distillation import render_prompt

    bank = bank_latents(latents, [plan["bank_document_ids"]], backend.model.device)
    prompt = render_prompt(backend, plan["partner"]["input"], FAMILY)
    capture: list = []
    ids = prompt[None].to(backend.model.device)
    with torch.no_grad(), conditioned(backend, encoder, bank, [prompt.numel()], capture=capture):
        backend.model.base_model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1)
    return [(q.clone(), m.clone()) for q, m in capture]


def student_kl(backend, encoder, plans: list[dict], latents: dict, responses: dict, teacher: torch.Tensor, *,
               question_override: list | None = None) -> tuple[Any, int]:
    """Token-mean KL(teacher || conditioned student); ``question_override`` writes from another question's states."""
    from official_d2l_online_batch_distillation import gather_response_logits, pad_sequences, render_prompt, response_position_mask

    device = backend.model.device
    model = backend.model.base_model
    targets = [responses[p["query"]["id"]] for p in plans]
    response_lengths = [int(t.numel()) for t in targets]
    student_prompts = [render_prompt(backend, p["query"]["input"], FAMILY) for p in plans]
    prompt_lengths = [p.numel() for p in student_prompts]
    ids, mask = pad_sequences([torch.cat((p, t[:-1])) for p, t in zip(student_prompts, targets, strict=True)],
                              backend.tokenizer.pad_token_id, device=device)
    start = min(prompt_lengths) - 1
    bank = bank_latents(latents, [p["bank_document_ids"] for p in plans], device)
    with conditioned(backend, encoder, bank, prompt_lengths, question_override=question_override):
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=ids.shape[1] - start).logits
    student = gather_response_logits(logits, [n - start for n in prompt_lengths], response_lengths).float().log_softmax(-1)
    positions = response_position_mask(response_lengths, device=device)
    kl = (teacher.exp() * (teacher - student)).sum(-1)
    return kl.masked_select(positions).mean(), int(positions.sum())


def k_schedule(steps: int, seed: int, k_values: tuple[int, ...]) -> list[int]:
    """Every consecutive block of len(k_values) updates holds each K once, in a seeded order (DISTRACTOR_CURVES's scheme)."""
    import random

    rng = random.Random(seed)
    schedule: list[int] = []
    while len(schedule) < steps:
        block = list(k_values)
        rng.shuffle(block)
        schedule += block
    return schedule[:steps]


def partners(rows: list[dict], seed: object) -> list[dict | None]:
    """For each row, another row's query on the same source document (seeded), or None if it has no sibling."""
    from distractor_curves_reader_training import hash_rank

    by_document: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_document.setdefault(row["document"]["id"], []).append(index)
    out: list[dict | None] = []
    for index, row in enumerate(rows):
        siblings = [i for i in by_document[row["document"]["id"]] if i != index]
        out.append(rows[min(siblings, key=lambda i: hash_rank(seed, row["query"]["id"], rows[i]["query"]["id"]))]["query"]
                   if siblings else None)
    return out


def update_plans(rows: list[dict], steps: int, seed: int, pool: list[str],
                 k_values: tuple[int, ...] = K_VALUES, batch_size: int = EXAMPLES_PER_UPDATE) -> list[list[dict]]:
    """DISTRACTOR_CURVES's update plans (``batch_size`` questions per update, nested random banks), one pass, no repeats.

    Each plan carries ``partner``: another question on the same source (the contrastive wrong question)."""
    import random

    from distractor_curves_reader_training import bank_for

    if len(rows) < steps * batch_size:
        raise ValueError("not enough training questions for the requested updates")
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    paired = partners(rows, seed)
    return [[{"query": rows[i]["query"], "k": k, "bank_document_ids": bank_for(rows[i], k, pool, seed),
              "partner": paired[i]}
             for i in order[step * batch_size:(step + 1) * batch_size]]
            for step, k in enumerate(k_schedule(steps, seed, k_values))]


def validation_plans(rows: list[dict], pool: list[str], k_values: tuple[int, ...] = K_VALUES) -> list[dict]:
    from distractor_curves_reader_training import bank_for

    paired = partners(rows, "validation")
    return [{"query": row["query"], "k": k_values[i % len(k_values)],
             "bank_document_ids": bank_for(row, k_values[i % len(k_values)], pool, "validation"), "partner": paired[i]}
            for i, row in enumerate(rows)]


def encoder_payload(encoder: WriteEncoder, step: int, seed: int) -> dict[str, Any]:
    return {"encoder": {k: v.detach().cpu() for k, v in encoder.state_dict().items()}, "config": encoder.config,
            "selected_step": step, "seed": seed, "recipe": "conditioned-latent-write-kl-v1"}


def load_encoder(path: Path, device) -> WriteEncoder:
    saved = torch.load(path, map_location=device, weights_only=False)
    config = saved["config"]
    encoder = WriteEncoder(config["latent_width"], config["hidden_width"], config["layers"], config["width"],
                           config["heads"], config["blocks"]).to(device)
    encoder.load_state_dict(saved["encoder"])
    return encoder.eval()


def train(backend, data: dict[str, Any], latents: dict[str, torch.Tensor], responses: dict, output_dir: Path, *,
          seed: int, steps: int, validation_interval: int = VALIDATION_INTERVAL,
          validation_limit: int | None = None, k_values: tuple[int, ...] = K_VALUES,
          batch_size: int = EXAMPLES_PER_UPDATE, learning_rate: float = LEARNING_RATE,
          contrastive_weight: float = 0.0, margin: float = 0.1) -> dict[str, Any]:
    """Resumable encoder training (DDP-aware like DISTRACTOR_CURVES); writes best.pt and last.pt.

    With ``contrastive_weight`` > 0 (CONTRASTIVE_WRITES) each question's loss adds
    weight * relu(margin - (KL with its partner's write - KL with its own write)), the partner being another
    question on the same source; validation then also reports the wrong-question KL.

    ``data`` = {"documents": {id: {"text": ...}}, "rows": {"train": [...], "validation": [...]}} with rows
    {"query": {...}, "document": {"id": ...}}; validation documents are disjoint from training documents."""
    from distractor_curves_reader_training import all_reduce_sum, atomic_save, world

    documents = data["documents"]
    rows = {split: [r for r in values if r["document"]["id"] in latents and r["query"]["id"] in responses]
            for split, values in data["rows"].items()}
    train_pool = sorted({r["document"]["id"] for r in rows["train"]})
    validation_pool = sorted({r["document"]["id"] for r in rows["validation"]})
    if set(train_pool) & set(validation_pool):
        raise ValueError("validation documents overlap training documents")
    k_values = tuple(k_values)
    updates = update_plans(rows["train"], steps, seed, train_pool, k_values, batch_size)
    held_out = validation_plans(rows["validation"][:validation_limit], validation_pool, k_values)
    model = backend.model.base_model
    for parameter in backend.model.hypernet.parameters():
        parameter.requires_grad_(False)
    torch.manual_seed(seed)
    first = next(iter(latents.values()))
    encoder = WriteEncoder(first.shape[-1], model.config.hidden_size, first.shape[0]).to(model.device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    resume = output_dir / "resume.pt"
    state = {"step": 0, "history": [], "losses": [], "best": None, "best_kl": math.inf, "seconds": 0.0, "passes": 0}
    if resume.exists():
        saved = torch.load(resume, map_location=model.device, weights_only=False)
        if saved["seed"] != seed or saved["steps"] != steps:
            raise ValueError("resume state belongs to a different run")
        encoder.load_state_dict(saved["encoder"])
        optimizer.load_state_dict(saved["optimizer"])
        state = saved["state"]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    rank, size = world()

    def validate(step: int) -> None:
        encoder.eval()
        sums = {k: [0.0, 0.0] for k in k_values}
        wrong = [0.0, 0.0, 0.0]  # paired right KL * tokens, wrong KL * tokens, tokens
        with torch.no_grad():
            for plan in held_out[rank::size]:
                teacher = teacher_log_probs(backend, [plan], responses, documents)
                loss, count = student_kl(backend, encoder, [plan], latents, responses, teacher)
                sums[plan["k"]][0] += float(loss) * count
                sums[plan["k"]][1] += count
                if contrastive_weight > 0 and plan.get("partner"):
                    other, _ = student_kl(backend, encoder, [plan], latents, responses, teacher,
                                          question_override=partner_states(backend, encoder, latents, plan))
                    wrong[0] += float(loss) * count
                    wrong[1] += float(other) * count
                    wrong[2] += count
        flat = all_reduce_sum([v for k in k_values for v in sums[k]] + wrong, model.device)
        per_k = {k: (flat[2 * i], flat[2 * i + 1]) for i, k in enumerate(k_values)}
        total = sum(v[0] for v in per_k.values()) / max(1.0, sum(v[1] for v in per_k.values()))
        entry = {"step": step, "kl": total, "per_k": {str(k): v[0] / v[1] for k, v in per_k.items() if v[1]}}
        right_sum, wrong_sum, paired = flat[-3:]
        if paired:
            entry["paired_kl"], entry["wrong_question_kl"] = right_sum / paired, wrong_sum / paired
        state["history"].append(entry)
        if total < state["best_kl"]:
            state["best_kl"], state["best"] = total, encoder_payload(encoder, step, seed)
        if rank == 0:
            extra = f" wrong-question KL {entry['wrong_question_kl']:.4f}" if paired else ""
            print(f"[question_conditioned_writes seed={seed}] step {step} validation KL {total:.4f}{extra}", flush=True)

    def save_resume() -> None:
        if rank == 0:
            atomic_save({"seed": seed, "steps": steps, "encoder": encoder.state_dict(),
                         "optimizer": optimizer.state_dict(), "state": state}, resume)

    if state["step"] == 0 and not state["history"]:
        validate(0)
    started = time.perf_counter()
    for step in range(state["step"], steps):
        encoder.train()
        plans = updates[step]
        optimizer.zero_grad(set_to_none=True)
        mine = plans[rank::size] if size > 1 else plans
        total_tokens = sum(int(responses[p["query"]["id"]].numel()) for p in plans)
        loss_sum, hinge_sum, backward_passes = 0.0, 0.0, 0
        for plan in mine:
            teacher = teacher_log_probs(backend, [plan], responses, documents)
            loss, count = student_kl(backend, encoder, [plan], latents, responses, teacher)
            objective = loss
            if contrastive_weight > 0 and plan.get("partner"):
                other, _ = student_kl(backend, encoder, [plan], latents, responses, teacher,
                                      question_override=partner_states(backend, encoder, latents, plan))
                hinge = torch.relu(margin - (other - loss))
                objective = loss + contrastive_weight * hinge
                hinge_sum += float(hinge.detach()) * count
            (objective * (count / total_tokens)).backward()
            backward_passes += 2 if objective is not loss else 1
            loss_sum += float(loss.detach()) * count
        if step == state["step"] and step == 0:
            grads = [p.grad for p in encoder.parameters() if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):
                raise RuntimeError("encoder gradients missing or non-finite at the first update")
            if any(p.grad is not None for p in model.parameters()) or \
                    any(p.grad is not None for p in backend.model.hypernet.parameters()):
                raise RuntimeError("frozen model or compiler received gradients")
        if size > 1:
            import torch.distributed as dist

            for parameter in encoder.parameters():
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                dist.all_reduce(parameter.grad)
        loss_total, hinge_total, passes = all_reduce_sum([loss_sum, hinge_sum, float(backward_passes)], model.device)
        gradient_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()
        state["passes"] += int(passes)
        state["losses"].append({"step": step + 1, "k": plans[0]["k"], "kl": loss_total / total_tokens,
                                "hinge": hinge_total / total_tokens,
                                "gradient_norm": float(gradient_norm)})
        state["step"] = step + 1
        if state["step"] % validation_interval == 0 or state["step"] == steps:
            state["seconds"] += time.perf_counter() - started
            started = time.perf_counter()
            validate(state["step"])
            save_resume()
        elif state["step"] % RESUME_INTERVAL == 0:
            state["seconds"] += time.perf_counter() - started
            started = time.perf_counter()
            save_resume()
    if rank == 0:
        atomic_save(state["best"], output_dir / "best.pt")
        atomic_save(encoder_payload(encoder, steps, seed), output_dir / "last.pt")
    by_k = {str(k): [r["kl"] for r in state["losses"] if r["k"] == k] for k in k_values}
    return {"world_size": size, "seed": seed, "steps": steps, "best_step": state["best"]["selected_step"],
            "best_validation_kl": state["best_kl"], "validation_history": state["history"],
            "train_kl_by_k": {k: (sum(v) / len(v) if v else None) for k, v in by_k.items()},
            "loss_tail": state["losses"][-50:],
            "checkpoints": {"best": str(output_dir / "best.pt"), "last": str(output_dir / "last.pt")},
            "accounting": {"encoder_parameters": sum(p.numel() for p in encoder.parameters()), "updates": steps,
                           "batch_size": batch_size, "learning_rate": learning_rate,
                           "contrastive_weight": contrastive_weight, "margin": margin,
                           "training_questions": steps * batch_size,
                           "student_backward_passes": state["passes"], "wall_seconds": state["seconds"],
                           "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0}}


# ----------------------------------------------------------------------------- evaluation (DISTRACTOR_CURVES suite banks)
def evaluate_source(backend, suite: dict[str, Any], source: str, banks: list[dict[str, Any]],
                    encoders: dict[str, WriteEncoder], *, beta: float | None, budget: int = MAX_NEW_TOKENS,
                    question_limit: int | None = None) -> dict[str, Any]:
    """Native, normalized, and each encoder arm for one source."""
    from distractor_curves_core import bank_state, compile_documents, score
    from official_d2l_online_batch_distillation import render_prompt

    documents = suite["documents"]
    needed = sorted({i for bank in banks for i in bank["document_ids"]})
    records = dict(zip(needed, compile_documents(backend, [documents[i]["text"] for i in needed]), strict=True))
    latents = {i: records[i]["latent"][0] for i in needed}
    device = backend.model.device
    questions = suite["questions"][source][:question_limit]
    prompts = {q["id"]: render_prompt(backend, q["input"], FAMILY) for q in questions}
    rows = []
    for bank in banks:
        k = bank["k"]
        ids = bank["document_ids"]
        state = bank_state(backend, [records[i] for i in ids])
        stacked = bank_latents(latents, [ids], device)
        for query in questions:
            responses = {"native": backend.generate(query["input"], state=state, dataset=query["dataset"],
                                                    max_new_tokens=budget)["text"]}
            if beta is not None and k > 1:
                responses[f"normalized_b{beta}"] = backend.generate(
                    query["input"], state=bank_state(backend, [records[i] for i in ids], k ** -beta),
                    dataset=query["dataset"], max_new_tokens=budget)["text"]
            for name, encoder in encoders.items():
                responses[name] = greedy(backend, encoder, stacked, prompts[query["id"]], budget=budget)
            rows.append({"query_id": query["id"], "source_id": source, "bank_id": bank["bank_id"],
                         "condition": bank["condition"], "k": k, "responses": responses,
                         "scores": {n: score(t, query["answers"]) for n, t in responses.items()}})
    return {"source_id": source, "rows": rows}
