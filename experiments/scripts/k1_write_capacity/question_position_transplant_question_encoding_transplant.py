"""Teacher-state transplants at the question positions (Table 3, left panel); no training.

Tests whether the teacher's advantage is carried by document-conditioned encoding of the question tokens
rather than by attention to document tokens. The source-visible teacher is run once per case and its
hidden states at the shared (non-document) prompt positions are installed into the PK-free student at
selected layer boundaries; answer positions always compute natively with no document present. Sweeping
which boundaries are replaced localizes where question-position contextualization matters.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from d2l_scale import load_config
from d2l_scale_backend import OfficialBackend, sha256
from d2l_scale_core import score_response, write_json_atomic
from qa_protocol_protocol import digest, load_evaluation_suite, load_frozen_manifest
from qa_protocol_runner_common import d2l_documents, d2l_plans
from answer_attribution_write_salience_diagnostic import StateInstaller
from transplant_core import analyze
from transplant_core import setup_case
from transplant_core import document_span, rest_mapping
from post_block_transitions import hidden_output, replace_hidden

LAYERS = 26
TOLERANCE = 1e-5
# boundary b in 1..26 is the output of decoder layer b-1 (b=26 is the final hidden state); the embedding
# boundary is identical in both models and is omitted. Question positions exclude the last prompt token,
# which is the first answer predictor, so answer computation stays native in the q_* conditions.
CONDITIONS = {
    "native": {"positions": "question", "boundaries": [], "source": "native"},
    "teacher_all_positions": {"positions": "all", "boundaries": list(range(1, LAYERS + 1)), "source": "teacher"},
    "q_from_0": {"positions": "question", "boundaries": list(range(1, LAYERS)), "source": "teacher"},
    "q_from_9": {"positions": "question", "boundaries": list(range(9, LAYERS)), "source": "teacher"},
    "q_from_18": {"positions": "question", "boundaries": list(range(18, LAYERS)), "source": "teacher"},
    "q_until_9": {"positions": "question", "boundaries": list(range(1, 9)), "source": "teacher"},
    "q_until_18": {"positions": "question", "boundaries": list(range(1, 18)), "source": "teacher"},
    "qh_from_0": {"positions": "question_history", "boundaries": list(range(1, LAYERS)), "source": "teacher"},
}


def position_sets(span: dict, teacher_prompt_len: int, student_prompt_len: int, history_len: int) -> dict:
    """Teacher/student index pairs for question (shared prompt), history, and answer-prediction positions."""
    t_prompt, s_prompt = rest_mapping(span, teacher_prompt_len, student_prompt_len)
    offset = teacher_prompt_len - student_prompt_len
    last_s = student_prompt_len - 1
    if last_s not in s_prompt:
        raise ValueError("last prompt token is not a shared position")
    question_s = [i for i in s_prompt if i != last_s]
    question_t = [t for t, i in zip(t_prompt, s_prompt, strict=True) if i != last_s]
    history_s = list(range(student_prompt_len, student_prompt_len + history_len))
    history_t = [i + offset for i in history_s]
    return {"question": (question_t, question_s), "history": (history_t, history_s),
            "question_history": ([*question_t, *history_t], [*question_s, *history_s]),
            "all": ([*question_t, last_s + offset, *history_t], [*question_s, last_s, *history_s]), "offset": offset}


@contextlib.contextmanager
def transplant_hooks(layers, boundaries: list[int], student_idx: list[int], source_hidden: list[torch.Tensor],
                     source_idx: list[int]):
    """At each boundary b, overwrite decoder layer b-1's output at student_idx with source_hidden[b][source_idx]."""
    handles = []
    try:
        for b in boundaries:
            def hook(_module, _inputs, output, b=b):
                hidden = hidden_output(output)
                patched = hidden.clone()
                patched[:, student_idx] = source_hidden[b][:, source_idx].to(hidden.dtype)
                return replace_hidden(output, patched)
            handles.append(layers[b - 1].register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


def forward_hidden(backend, layers, ids: torch.Tensor, state) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Logits plus raw boundary states: hidden[b] is the raw output of decoder layer b-1 (b=1..26), hidden[0] embeddings.

    Transformers' output_hidden_states returns the final-normed state as its last entry, so the raw output of the
    last layer is captured by a hook instead."""
    raw_last = {}

    def hook(_module, _inputs, output):
        raw_last["value"] = hidden_output(output).detach().clone()

    handle = layers[-1].register_forward_hook(hook)
    try:
        with StateInstaller(backend)(state), torch.inference_mode():
            out = backend.model.base_model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                                           output_hidden_states=True)
    finally:
        handle.remove()
    hidden = [h.detach().clone() for h in out.hidden_states[:LAYERS]] + [raw_last["value"]]
    if len(hidden) != LAYERS + 1:
        raise RuntimeError("unexpected hidden-state count")
    return out.logits, hidden


def student_logits(backend, layers, ids: torch.Tensor, state, spec: dict, hidden_by_source: dict, positions: dict,
                   answer: slice, counter: dict) -> torch.Tensor:
    counter["native_student_forwards"] = counter.get("native_student_forwards", 0) + 1
    counter["native_student_input_positions"] = counter.get("native_student_input_positions", 0) + ids.numel()
    if not spec["boundaries"]:
        return forward_hidden(backend, layers, ids, state)[0][:, answer].float()
    source_idx, student_idx = positions[spec["positions"]]
    source_hidden = hidden_by_source[spec["source"]]
    with StateInstaller(backend)(state), torch.inference_mode(), transplant_hooks(layers, spec["boundaries"],
                                                                                 student_idx, source_hidden, source_idx):
        out = backend.model.base_model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    return out.logits[:, answer].float()


class Case:
    def __init__(self, backend, layers, plan, documents):
        self.backend, self.layers = backend, layers
        self.compiled, self.student, self.teacher, self.compilation = setup_case(backend, plan, documents)
        self.span = document_span(self.teacher, self.student)
        self.teacher_cache = {}
        self.distinct_histories = 0
        self.controls = {"native_self_logits_max_abs": 0.0, "teacher_all_positions_logits_max_abs": 0.0}
        self.counter = backend.transplant_accounting

    def teacher_hidden(self, history):
        """Teacher answer-position logits and boundary states for this history; only the latest history is kept
        (each greedy loop extends one history, so a one-entry cache suffices and bounds memory)."""
        key = tuple(history)
        if key not in self.teacher_cache:
            ids = torch.cat([self.teacher, torch.tensor(history, dtype=torch.long, device=self.teacher.device)])[None]
            self.counter["source_teacher_forwards"] = self.counter.get("source_teacher_forwards", 0) + 1
            self.counter["source_teacher_input_positions"] = self.counter.get("source_teacher_input_positions", 0) + ids.numel()
            logits, hidden = forward_hidden(self.backend, self.layers, ids, None)
            t_answer = slice(self.teacher.numel() - 1, self.teacher.numel() + len(history))
            self.teacher_cache.clear()
            self.teacher_cache[key] = (logits[:, t_answer].float().clone(), hidden)
            self.distinct_histories += 1
            del logits
        return self.teacher_cache[key]

    def logits(self, history, condition):
        spec = CONDITIONS[condition]
        ids = torch.cat([self.student, torch.tensor(history, dtype=torch.long, device=self.student.device)])[None]
        answer = slice(self.student.numel() - 1, self.student.numel() + len(history))
        positions = position_sets(self.span, self.teacher.numel(), self.student.numel(), len(history))
        teacher_logits, teacher_hidden = self.teacher_hidden(history)
        native_logits, native_hidden = forward_hidden(self.backend, self.layers, ids, self.compiled.factors)
        hidden_by_source = {"teacher": teacher_hidden, "native": native_hidden}
        if condition == "native":
            # self-transplant gate: install the student's own states at question positions at every boundary
            spec = {"positions": "question", "boundaries": list(range(1, LAYERS + 1)), "source": "native"}
            positions_self = dict(positions)
            positions_self["question"] = (positions["question"][1], positions["question"][1])
            positions_self["offset"] = 0
            result = student_logits(self.backend, self.layers, ids, self.compiled.factors, spec, hidden_by_source,
                                    positions_self, answer, self.counter)
            error = float((result - native_logits[:, answer].float()).abs().max())
            self.controls["native_self_logits_max_abs"] = max(self.controls["native_self_logits_max_abs"], error)
            if error > TOLERANCE:
                raise RuntimeError(f"native self transplant failed: {error}")
            return result
        result = student_logits(self.backend, self.layers, ids, self.compiled.factors, spec, hidden_by_source, positions,
                                answer, self.counter)
        if condition == "teacher_all_positions":
            error = float((result - teacher_logits).abs().max())
            self.controls["teacher_all_positions_logits_max_abs"] = max(
                self.controls["teacher_all_positions_logits_max_abs"], error)
            if error > TOLERANCE:
                raise RuntimeError(f"all-position transplant does not reproduce teacher logits: {error}")
        return result

    def generate(self, condition, budget):
        history = []
        eos = self.backend.model.base_model.generation_config.eos_token_id
        stops = set(eos if isinstance(eos, list) else [eos])
        for _ in range(budget):
            logits = self.logits(history, condition)
            token = int(logits[0, -1].argmax())
            history.append(token)
            if token in stops:
                break
        return self.backend.tokenizer.decode(history, skip_special_tokens=True).strip(), len(history)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("inputs/qa_protocol/manifest/pinned-v1/manifest.json"))
    parser.add_argument("--suite", type=Path, default=Path("inputs/qa_protocol/validation/qa_protocol-v1/validation-suite.json"))
    parser.add_argument("--confirmation", type=Path, default=Path("artifacts/activation_transplant/activation_transplantd-fixed-validation402-v1.json"))
    parser.add_argument("--development-sample", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_new_tokens != 32 or (args.limit is not None and args.limit < 1):
        raise ValueError("require frozen 32-token budget and positive sample size")
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    if args.resume and not args.output.exists():
        raise FileNotFoundError(args.output)
    root = Path(__file__).parents[2]
    manifest = load_frozen_manifest(args.manifest)
    suite = load_evaluation_suite(args.suite, manifest, expected_split="validation")
    confirmation = json.loads(args.confirmation.read_text())
    if confirmation["status"] != "completed" or confirmation["contract"]["validation_suite_sha256"] != suite["suite_sha256"]:
        raise ValueError("completed matching exact-402 confirmation required")
    plans_by_id = {p["query"]["id"]: p for p in d2l_plans(manifest, "validation")}
    identities = (json.loads(args.development_sample.read_text())["contract"]["sample"]
                  if args.development_sample else confirmation["rows"])
    plans = [plans_by_id[r["query_id"]] for r in identities]
    if args.limit:
        plans = plans[:args.limit]
    contract = {"experiment": "QUESTION_POSITION_TRANSPLANT", "manifest_sha256": manifest["manifest_sha256"], "suite_sha256": suite["suite_sha256"],
                "confirmation_sha256": sha256(args.confirmation),
                "development_sample_sha256": sha256(args.development_sample) if args.development_sample else None,
                "conditions": {k: {kk: (vv if kk != "boundaries" else [min(vv), max(vv)] if vv else []) for kk, vv in v.items()}
                               for k, v in CONDITIONS.items()},
                "max_new_tokens": args.max_new_tokens, "k": 1, "training_updates": 0,
                "sample": [{"query_id": p["query"]["id"], "document_id": p["query"]["document_id"]} for p in plans],
                "transplant": "teacher hidden states at shared prompt (question) positions, optionally history positions, "
                              "installed at the output of decoder layer b-1 for each boundary b; answer positions native",
                "limitations": "source-visible oracle; repeatedly inspected validation; no bounded-state, locality, "
                               "multi-document, or compute claim"}
    report = {"status": "running", "contract": contract, "rows": [], "accounting": {"wall_seconds": 0., "replay": {}}}
    if args.resume:
        report = json.loads(args.output.read_text())
        if digest(report["contract"]) != digest(contract):
            raise ValueError("resume contract differs")
        expected = [(p["query"]["id"], p["query"]["document_id"]) for p in plans[:len(report["rows"])]]
        if [(r["query_id"], r["document_id"]) for r in report["rows"]] != expected:
            raise ValueError("resume rows differ from frozen prefix")
        if report["status"] == "completed":
            print(json.dumps(report["analysis"]["conditions"], indent=2))
            return
    write_json_atomic(args.output, report)
    started = time.perf_counter()
    prior_wall = report["accounting"]["wall_seconds"]
    backend = OfficialBackend(load_config(root, "gemma2_qa_protocol"), root)
    backend.transplant_accounting = report["accounting"]["replay"]
    baseline_rows = {r["query_id"]: r for r in confirmation["rows"]}
    try:
        from ctx_to_lora.utils import get_layers
        layers = get_layers(backend.model.base_model)
        if len(layers) != LAYERS:
            raise RuntimeError("unexpected layer count")
        documents = d2l_documents(manifest)
        report["runtime"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                             "gpu": torch.cuda.get_device_name(), "backend": backend.provenance}
        for index, plan in enumerate(plans):
            if index < len(report["rows"]):
                continue
            case = Case(backend, layers, plan, documents)
            q = plan["query"]
            baseline = baseline_rows[q["id"]]
            history = backend.tokenizer.encode(baseline["responses"]["teacher"], add_special_tokens=False)
            case.logits(history, "native")
            case.logits(history, "teacher_all_positions")
            responses = {"teacher": baseline["responses"]["teacher"]}
            counts = {}
            for condition in CONDITIONS:
                responses[condition], counts[condition] = case.generate(condition, args.max_new_tokens)
            matches = {"native": responses["native"] == baseline["responses"]["native"],
                       "teacher_all_positions": responses["teacher_all_positions"] == baseline["responses"]["teacher"]}
            if not matches["native"]:
                raise RuntimeError(f"fresh native mismatch {q['id']}")
            report["rows"].append({"query_id": q["id"], "document_id": q["document_id"], "dataset": q["dataset"],
                                   "question": q["input"], "answers": q["answers"], "responses": responses,
                                   "scores": {k: score_response(v, q["answers"]) for k, v in responses.items()},
                                   "controls": case.controls, "reference_match": matches, "span": case.span,
                                   "generated_tokens": counts, "compilation": case.compilation,
                                   "distinct_histories": case.distinct_histories})
            del case
            report["accounting"].update(wall_seconds=prior_wall + time.perf_counter() - started,
                                        replay=backend.transplant_accounting,
                                        peak_vram_bytes=max(report["accounting"].get("peak_vram_bytes", 0),
                                                            torch.cuda.max_memory_allocated()),
                                        compiler_document_occurrences=len(report["rows"]))
            write_json_atomic(args.output, report)
            if (index + 1) % 16 == 0 or len(plans) <= 24:
                print(json.dumps({"completed": index + 1, "total": len(plans),
                                  "f1": {k: round(v["qa_f1"], 3) for k, v in report["rows"][-1]["scores"].items()}}), flush=True)
        report["status"] = "completed"
        report["analysis"] = analyze(report, seed=193)
        report["analysis"]["parity"] = {
            "teacher_all_positions_matches_activation_transplantd_teacher": sum(r["reference_match"]["teacher_all_positions"] for r in report["rows"]),
            "rows": len(report["rows"]),
            "max_controls": {k: max(r["controls"][k] for r in report["rows"]) for k in report["rows"][0]["controls"]}}
        write_json_atomic(args.output, report)
        print(json.dumps({c: {"macro_f1": v["macro_f1"], "ci": v["document_cluster_delta_ci95"]}
                          for c, v in report["analysis"]["conditions"].items()}, indent=2))
        print(json.dumps(report["analysis"]["parity"]))
    except Exception as error:
        report["status"] = "failed"
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        report["accounting"].update(wall_seconds=prior_wall + time.perf_counter() - started,
                                    replay=backend.transplant_accounting)
        write_json_atomic(args.output, report)
        raise
    finally:
        backend.close()


if __name__ == "__main__":
    main()
