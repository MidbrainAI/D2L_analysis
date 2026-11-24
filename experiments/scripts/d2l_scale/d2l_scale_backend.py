"""Thin benchmark bridge to SakanaAI/doc-to-lora, not an inference implementation.

All context tokenization/chunking, compilation, LoRA composition, installation,
and model execution are delegated to the pinned upstream repository.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from d2l_scale_core import MODEL_IDS, SOURCE_COMMIT, write_json_atomic


class ContextOverflowError(ValueError):
    """An untruncated textual control exceeds the target context limit."""


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify_source(source: Path) -> dict:
    if shutil.which("git") is None:
        for parent in (source, *source.parents):
            manifest_path = parent / "_midbrain_manifest.json"
            if not manifest_path.is_file():
                continue
            relative = str(source.relative_to(parent)) or "."
            manifest = json.loads(manifest_path.read_text())
            matches = [
                repository
                for repository in manifest.get("git_repositories", [])
                if repository["path"] == relative
            ]
            if len(matches) != 1:
                continue
            provenance = matches[0]
            if provenance["commit"] != SOURCE_COMMIT or provenance["dirty"]:
                raise ValueError(
                    f"Expected clean upstream {SOURCE_COMMIT}; got {provenance['commit']}, "
                    f"dirty={provenance['dirty']}"
                )
            return {key: provenance[key] for key in ("commit", "dirty")}
        raise RuntimeError("git is unavailable and staged source provenance was not found")

    def git(*args):
        return subprocess.run(
            ["git", "-c", f"safe.directory={source}", *args],
            cwd=source,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    commit = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain"))
    if commit != SOURCE_COMMIT or dirty:
        raise ValueError(f"Expected clean upstream {SOURCE_COMMIT}; got {commit}, dirty={dirty}")
    return {
        "commit": commit,
        "dirty": dirty,
    }


def tensor_bytes(value) -> int:
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    return value.numel() * value.element_size()


def cpu(value):
    if isinstance(value, dict):
        return {key: cpu(item) for key, item in value.items()}
    return value.detach().cpu()


def resolve_model_revision(model_id: str, pinned_revision: str | None, revision_path: Path) -> str:
    """A configured revision is already frozen; only unpinned models need a local lock."""
    if pinned_revision:
        return pinned_revision
    if revision_path.exists():
        lock = json.loads(revision_path.read_text())
        if lock["model_id"] != model_id:
            raise ValueError("Model revision lock conflicts with configuration")
        return lock["revision"]
    from huggingface_hub import HfApi

    revision = HfApi().model_info(model_id).sha
    write_json_atomic(revision_path, {"model_id": model_id, "revision": revision})
    return revision


class OfficialBackend:
    def __init__(self, config: dict, workspace: Path, *, train_compiler: bool = False):
        self.config = config
        self.source = (workspace / config["paths"]["source"]).resolve()
        checkpoint = (workspace / config["paths"]["checkpoint"]).resolve()
        self.provenance = {
            "source": str(self.source),
            "checkpoint": str(checkpoint),
        }
        import torch
        import transformers

        self.torch = torch
        self.model_id = MODEL_IDS[config["variant"]]
        revision_path = workspace / config["paths"]["cache"] / "model_revision.json"
        revision = resolve_model_revision(self.model_id, config["model"].get("revision"), revision_path)
        self.provenance.update(
            {
                "model": self.model_id,
                "model_revision": revision,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "settings": config["model"],
            }
        )
        sys.path.insert(0, str(self.source / "src"))
        with self._upstream_cwd():
            from ctx_to_lora.data.definitions import EVAL_INTX_TEMPLATES
            from ctx_to_lora.data.processing import split_too_long_ctx, tokenize_ctx_text
            from ctx_to_lora.model_loading import get_tokenizer
            from ctx_to_lora.modeling import hypernet
            from ctx_to_lora.modeling.lora_layer import apply_lora_to_layers
            from ctx_to_lora.modeling.lora_merger import combine_lora

            self.templates, self.tokenize_context = EVAL_INTX_TEMPLATES, tokenize_ctx_text
            self.split_context, self.merge, self.install = split_too_long_ctx, combine_lora, apply_lora_to_layers
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if state["base_model_name_or_path"] != self.model_id:
                raise ValueError("D2L checkpoint does not target the configured base model")
            original_loader = hypernet.get_model

            def pinned_loader(model_name_or_path, *args, **kwargs):
                if model_name_or_path != self.model_id:
                    raise ValueError("This benchmark expects the same target and context encoder checkpoint")
                kwargs["model_kwargs"] = {**(kwargs.get("model_kwargs") or {}), "revision": revision}
                # "sdpa" keeps long in-context prompts (PERSONAMEM's 32K histories) within GPU memory;
                # upstream otherwise loads eager attention unless flash attention is requested.
                if config["model"].get("attention") == "sdpa":
                    kwargs["model_kwargs"]["attn_implementation"] = "sdpa"
                # Upstream defaults to device="cuda"; allow CPU smokes when no GPU is visible.
                # bitsandbytes NF4 has no CPU backend here, so CPU smokes load the context
                # encoder unquantized, and in fp32 because CPUs without native bf16 run bf16
                # matmuls ~1000x slower: functional checks only, not numeric GPU parity.
                if not torch.cuda.is_available():
                    kwargs.setdefault("device", "cpu")
                    kwargs["dtype"] = torch.float32
                    kwargs["model_kwargs"]["torch_dtype"] = torch.float32
                    self.provenance["cpu_fp32_models"] = True
                    if kwargs.get("use_q_lora"):
                        kwargs["use_q_lora"] = False
                        self.provenance["cpu_unquantized_ctx_encoder"] = True
                        print("WARNING: CPU run loads the D2L context encoder without NF4 quantization", flush=True)
                return original_loader(model_name_or_path, *args, **kwargs)

            # Upstream loads target and context encoder independently; pin both.
            hypernet.get_model = pinned_loader
            try:
                self.model = hypernet.ModulatedPretrainedModel.from_state_dict(
                    state,
                    train=train_compiler,
                    use_sequence_packing=False,
                    use_flash_attn=config["model"].get("attention", "flash_attention_2") == "flash_attention_2",
                    base_model_kwargs={"torch_dtype": getattr(torch, config["model"].get("dtype", "bfloat16"))},
                )
            finally:
                hypernet.get_model = original_loader
            self.tokenizer = get_tokenizer(self.model_id, {"revision": revision})
        self.model.eval().requires_grad_(False)
        if train_compiler:
            # Keep both frozen language models deterministic while differentiating
            # through native LoRA execution into the shared compiler only.
            self.model.hypernet.requires_grad_(True)
            self.model.hypernet.eval()
        self.model.enable_iterative_mode(True)
        self.model.reset()
        self.head_bias = cpu(self.model.hypernet.get_head_bias())
        self.provenance["tokenizer_template_sha256"] = hashlib.sha256(
            str(self.tokenizer.chat_template).encode()
        ).hexdigest()

    @contextlib.contextmanager
    def _upstream_cwd(self):
        previous = Path.cwd()
        os.chdir(self.source)
        try:
            yield
        finally:
            os.chdir(previous)

    def query_features(self, text: str) -> list[float]:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            raise ValueError("Cannot encode an empty query/document")
        with self.torch.inference_mode():
            embedding = self.model.base_model.get_input_embeddings()
            # Feature extraction is embedding lookup only, not another LM forward.
            parts = [
                embedding(self.torch.tensor(ids[i : i + 4096], device=embedding.weight.device)).float().sum(0)
                for i in range(0, len(ids), 4096)
            ]
            return (self.torch.stack(parts).sum(0) / len(ids)).cpu().tolist()

    def compile_document(self, text: str) -> dict:
        started = time.perf_counter()
        ids = self.tokenize_context({"context": [text]}, self.tokenizer)["ctx_ids"][0]
        chunks = self.split_context(
            {"ctx_ids": ids},
            model_name_or_path=self.model_id,
            num_chunk_probs=None,
            max_chunk_len=self.config["model"]["context_chunk_len"],
            min_chunk_len=0,
            max_num_split=None,
            is_train=False,
        )["ctx_ids"]
        raw = []
        with self.torch.inference_mode():
            for chunk in chunks:
                tokens = self.torch.tensor([chunk], device=self.model.device)
                weights, layernorms = self.model.generate_weights(tokens, self.torch.ones_like(tokens), None)
                if layernorms:
                    raise ValueError("Layernorm adaptation is not part of the frozen LoRA benchmark")
                raw.append(cpu(weights))
        combined_raw = {
            module: {key: self.torch.cat([entry[module][key] for entry in raw], dim=0) for key in ("A", "B")}
            for module in raw[0]
        }
        if any(t.ndim != 4 or t.shape[0] != len(chunks) for part in combined_raw.values() for t in part.values()):
            raise ValueError("Unexpected upstream raw-factor layout")
        return {
            "raw": combined_raw,
            "key": self.query_features(text),
            "nchunks": len(chunks),
            "source_tokens": len(ids),
            "chunk_token_counts": list(map(len, chunks)),
            "compile_seconds": time.perf_counter() - started,
            "raw_bytes": tensor_bytes(combined_raw),
        }

    def compose(self, records: list[dict], weights: list[float]) -> tuple[dict, dict]:
        if len(records) != len(weights) or any(not math.isfinite(w) or not 0 <= w <= 1 for w in weights):
            raise ValueError("Expected one finite [0,1] weight per document")
        selected = [(record, w) for record, w in zip(records, weights, strict=True) if w > 0]
        if not selected:
            state = {
                module: {key: value.unsqueeze(0) for key, value in part.items()}
                for module, part in self.head_bias.items()
            }
            chunks = 0
        else:
            chunks = sum(record["nchunks"] for record, _ in selected)
            sample = next(iter(selected[0][0]["raw"].values()))["A"]
            estimated_bytes = sum(tensor_bytes(record["raw"]) for record, _ in selected) + sum(
                value.numel() * sample.element_size() for part in self.head_bias.values() for value in part.values()
            )
            budget = self.config.get("scaling", {}).get("max_loaded_adapter_bytes")
            if budget and estimated_bytes > budget:
                raise ContextOverflowError(f"Projected adapter requires {estimated_bytes} bytes, above cap {budget}")
            raw = {
                module: {
                    key: self.torch.cat([record["raw"][module][key] for record, _ in selected], dim=0)
                    for key in ("A", "B")
                }
                for module in selected[0][0]["raw"]
            }
            scalers = self.torch.tensor([weight for record, weight in selected for _ in range(record["nchunks"])])
            state = self.merge(
                raw,
                self.torch.tensor([chunks], dtype=self.torch.int32),
                lora_bias=self.head_bias,
                scalers=scalers,
                bias_scaler=1.0,
            )
        loaded = tensor_bytes(state)
        budget = self.config.get("scaling", {}).get("max_loaded_adapter_bytes")
        if budget and loaded > budget:
            raise ContextOverflowError(f"Adapter requires {loaded} bytes, above configured cap {budget}")
        return state, {
            "selected_documents": len(selected),
            "selected_chunks": chunks,
            "loaded_bytes": loaded,
            "loaded_rank": next(iter(state.values()))["A"].shape[2],
            "bias_only": not selected,
        }

    def generate(self, question: str, *, state=None, context=None, dataset="squad", max_new_tokens=256) -> dict:
        prompt = self.templates[dataset].format(input=question)
        if context is not None:
            prompt = context.strip() + "\n\n" + prompt
        ids = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": ""}, {"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.model.device)
        limit = self.model.config.max_position_embeddings
        if ids.shape[1] + max_new_tokens > limit:
            raise ContextOverflowError(f"Untruncated prompt {ids.shape[1]} + output {max_new_tokens} exceeds {limit}")
        started = time.perf_counter()
        self.model.reset()
        try:
            if state is not None:
                self.model.patch_lora_forward()
                loaded = {
                    module: {key: value.to(self.model.device) for key, value in part.items()}
                    for module, part in state.items()
                }
                self.install(
                    self.model.base_model,
                    self.model.hypernet.layer_indices,
                    loaded,
                    self.torch.ones(1, device=self.model.device, dtype=self.torch.int32),
                )
            with self.torch.inference_mode():
                output = self.model.base_model.generate(
                    input_ids=ids,
                    attention_mask=self.torch.ones_like(ids),
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
        finally:
            self.model.reset()
        return {
            "text": self.tokenizer.decode(output[0, ids.shape[1] :], skip_special_tokens=True).strip(),
            "prompt_tokens": ids.shape[1],
            "output_tokens": output.shape[1] - ids.shape[1],
            "seconds": time.perf_counter() - started,
            "loaded_bytes": tensor_bytes(state) if state is not None else 0,
        }

    def close(self):
        del self.model
        self.torch.cuda.empty_cache()
