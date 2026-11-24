"""ANSWER_POSITION_MASK: answer-position document-read mask oracle (no training, source-visible).

The ICL (teacher) prompt is run with an attention-mask intervention: in selected decoder layers, the
question rows and/or the answer rows are blocked from attending to the document columns. The document's
own K/V and the prompt geometry stay the teacher's, so the only change is which rows may read the document
at which layers. This localizes where a bounded memory must be read (QUESTION_POSITION_TRANSPLANT transplanted teacher states; here
the read itself is restricted and generation is fresh).

Rows and columns, in teacher-prompt coordinates: document columns are the teacher positions not shared with
the student prompt (``document_span``); question rows are the shared positions after the document except the
last prompt token; answer rows are the last prompt token (the first answer predictor) and every generated
token. Rows before the document cannot see it causally and are untouched.

Memory: batch 1, cache-free greedy decoding (exact under arbitrary masks), and logits for the last position
only (``logits_to_keep=1``), so a step holds one [1, 1, L, L] mask clone per layer and one vocabulary row.
"""

from __future__ import annotations

import contextlib
import time
from typing import Any

import torch

LAYERS = 26
ALL = tuple(range(LAYERS))
NONE: tuple[int, ...] = ()
# question_layers / answer_layers: decoder layers (0..25) in which those rows may read the document.
CONDITIONS: dict[str, dict[str, Any]] = {
    "teacher": {"question_layers": ALL, "answer_layers": ALL, "native": False},
    "answer_read_all": {"question_layers": NONE, "answer_layers": ALL, "native": False},
    "answer_read_from_9": {"question_layers": NONE, "answer_layers": tuple(range(9, LAYERS)), "native": False},
    "answer_read_from_18": {"question_layers": NONE, "answer_layers": tuple(range(18, LAYERS)), "native": False},
    "answer_read_until_18": {"question_layers": NONE, "answer_layers": tuple(range(0, 18)), "native": False},
    "question_read_only": {"question_layers": ALL, "answer_layers": NONE, "native": False},
    "no_read": {"question_layers": NONE, "answer_layers": NONE, "native": False},
    "answer_read_all_native": {"question_layers": NONE, "answer_layers": ALL, "native": True},
    "no_read_native": {"question_layers": NONE, "answer_layers": NONE, "native": True},
}
MAX_NEW_TOKENS = 32
PARITY_MIN_AGREEMENT = 0.75  # fresh responses vs the recorded transplant reference


# ----------------------------------------------------------------------------- pure helpers (CPU-tested)
def row_sets(span: dict, teacher_prompt_len: int, history_len: int) -> dict[str, list[int]]:
    """Document columns, question rows, and answer rows of the teacher sequence prompt+history."""
    if not span["start"] < span["end"] <= teacher_prompt_len - 1:
        raise ValueError("document span must end before the last prompt token")
    return {"document": list(range(span["start"], span["end"])),
            "question": list(range(span["end"], teacher_prompt_len - 1)),
            "answer": list(range(teacher_prompt_len - 1, teacher_prompt_len + history_len))}


def blocked_rows(spec: dict, rows: dict[str, list[int]]) -> dict[int, list[int]]:
    """Per decoder layer, the rows that may not read the document (layers with nothing blocked are omitted)."""
    blocked = {}
    for layer in range(LAYERS):
        chosen = []
        if layer not in spec["question_layers"]:
            chosen += rows["question"]
        if layer not in spec["answer_layers"]:
            chosen += rows["answer"]
        if chosen:
            blocked[layer] = chosen
    return blocked


def causal_additive_mask(length: int, dtype: torch.dtype, device) -> torch.Tensor:
    mask = torch.full((length, length), torch.finfo(dtype).min, dtype=dtype, device=device).triu(1)
    return mask[None, None]


def block(mask: torch.Tensor, rows: list[int], columns: list[int]) -> torch.Tensor:
    """Copy of a [B, 1, Q, K] attention mask (additive float or boolean keep-mask) with rows x columns blocked."""
    out = mask.clone()
    r = torch.as_tensor(rows, device=mask.device)[:, None]
    c = torch.as_tensor(columns, device=mask.device)[None, :]
    if out.dtype == torch.bool:
        out[:, :, r, c] = False
    else:
        out[:, :, r, c] = torch.finfo(out.dtype).min
    return out


@contextlib.contextmanager
def document_mask(layers, blocked: dict[int, list[int]], columns: list[int]):
    """Replace each self-attention's attention_mask kwarg with a copy blocking rows x document columns."""
    handles = []

    def make(layer_index: int):
        def hook(module, args, kwargs):
            rows = blocked.get(layer_index)
            if not rows:
                return args, kwargs
            mask = kwargs.get("attention_mask")
            if mask is None:
                hidden = kwargs.get("hidden_states", args[0] if args else None)
                mask = causal_additive_mask(hidden.shape[1], hidden.dtype, hidden.device)
            kwargs["attention_mask"] = block(mask, rows, columns)
            return args, kwargs
        return hook

    try:
        for index, layer in enumerate(layers):
            handles.append(layer.self_attn.register_forward_pre_hook(make(index), with_kwargs=True))
        yield
    finally:
        for handle in handles:
            handle.remove()


# ----------------------------------------------------------------------------- model side
def last_logits(backend, ids: torch.Tensor, state) -> torch.Tensor:
    from answer_attribution_write_salience_diagnostic import StateInstaller

    with StateInstaller(backend)(state), torch.inference_mode():
        out = backend.model.base_model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                                       logits_to_keep=1)
    return out.logits[:, -1].float()


class Case:
    def __init__(self, backend, layers, plan: dict, documents: dict):
        from transplant_core import setup_case
        from transplant_core import document_span

        self.backend, self.layers = backend, layers
        self.compiled, self.student, self.teacher, self.compilation = setup_case(backend, plan, documents)
        self.span = document_span(self.teacher, self.student)
        self.forwards = 0
        self.positions = 0

    def step_logits(self, condition: str, history: list[int]) -> torch.Tensor:
        device = self.teacher.device
        tail = torch.tensor(history, dtype=torch.long, device=device)
        if condition == "native":
            ids = torch.cat([self.student, tail])[None]
            self.forwards += 1
            self.positions += ids.numel()
            return last_logits(self.backend, ids, self.compiled.factors)
        spec = CONDITIONS[condition]
        ids = torch.cat([self.teacher, tail])[None]
        rows = row_sets(self.span, self.teacher.numel(), len(history))
        state = self.compiled.factors if spec["native"] else None
        self.forwards += 1
        self.positions += ids.numel()
        with document_mask(self.layers, blocked_rows(spec, rows), rows["document"]):
            return last_logits(self.backend, ids, state)

    def generate(self, condition: str, budget: int = MAX_NEW_TOKENS) -> tuple[str, int]:
        eos = self.backend.model.base_model.generation_config.eos_token_id
        stops = set(eos if isinstance(eos, list) else [eos])
        history: list[int] = []
        for _ in range(budget):
            token = int(self.step_logits(condition, history).argmax())
            history.append(token)
            if token in stops:
                break
        return self.backend.tokenizer.decode(history, skip_special_tokens=True).strip(), len(history)

    def hook_gate(self) -> float:
        """The mask hook with nothing blocked must reproduce the plain teacher forward exactly."""
        ids = self.teacher[None]
        plain = last_logits(self.backend, ids, None)
        everything = {layer: [] for layer in range(LAYERS)}
        with document_mask(self.layers, everything, list(range(self.span["start"], self.span["end"]))):
            hooked = last_logits(self.backend, ids, None)
        # an all-rows-blocked control must change something, or the hook is not reaching attention
        rows = row_sets(self.span, self.teacher.numel(), 0)
        with document_mask(self.layers, {layer: rows["question"] + rows["answer"] for layer in range(LAYERS)},
                           rows["document"]):
            masked = last_logits(self.backend, ids, None)
        if torch.equal(masked, plain):
            raise RuntimeError("blocking every row did not change the logits; the mask hook is inert")
        return float((hooked - plain).abs().max())


def run_shard(backend, plans: list[dict], documents: dict, reference_rows: dict[str, dict] | None,
              conditions: tuple[str, ...] = tuple(CONDITIONS) + ("native",),
              budget: int = MAX_NEW_TOKENS) -> dict[str, Any]:
    """Every condition for each plan of one shard; returns rows plus gate and accounting records."""
    from ctx_to_lora.utils import get_layers
    from d2l_scale_core import score_response

    layers = get_layers(backend.model.base_model)
    if len(layers) != LAYERS:
        raise RuntimeError("unexpected layer count")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    rows, gate_max = [], 0.0
    forwards = positions = 0
    for plan in plans:
        q = plan["query"]
        case = Case(backend, layers, plan, documents)
        gate = case.hook_gate()
        gate_max = max(gate_max, gate)
        if gate != 0.0:
            raise RuntimeError(f"mask hook changed unmasked teacher logits by {gate} on {q['id']}")
        responses, tokens = {}, {}
        for condition in conditions:
            responses[condition], tokens[condition] = case.generate(condition, budget)
        reference = (reference_rows or {}).get(q["id"]) if budget == MAX_NEW_TOKENS else None
        parity = None if reference is None else {
            "teacher": responses["teacher"] == reference["responses"]["teacher"],
            "native": responses["native"] == reference["responses"]["native"]}
        rows.append({"query_id": q["id"], "document_id": q["document_id"], "dataset": q["dataset"],
                     "question": q["input"], "answers": list(q["answers"]), "responses": responses,
                     "scores": {k: score_response(v, q["answers"]) for k, v in responses.items()},
                     "generated_tokens": tokens, "span": case.span, "reference_parity": parity})
        forwards += case.forwards
        positions += case.positions
        del case
    checked = [r["reference_parity"] for r in rows if r["reference_parity"] is not None]
    agreement = {arm: (sum(p[arm] for p in checked) / len(checked) if checked else None) for arm in ("teacher", "native")}
    return {"rows": rows, "gates": {"hook_unblocked_max_abs": gate_max, "reference_agreement": agreement},
            "accounting": {"forwards": forwards, "input_positions": positions,
                           "wall_seconds": time.perf_counter() - started,
                           "peak_vram_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None}}
