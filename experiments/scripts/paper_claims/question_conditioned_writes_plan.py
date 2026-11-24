"""Build the frozen plans for the question-conditioned writes (Table 9), and the checkpoint-selection rule.

Stages (serial; dependencies in brackets):

1. ``question_conditioned_writes-gates``: correctness gates on one development bank (fails without writing a result).
   ``question_conditioned_writes-latents`` (SINGLE_DOCUMENT_WRITES only): compile the training passages to cached D2L latents.
2. ``question_conditioned_writes-teacher``: the source-visible teacher's greedy responses for every training and validation question.
3. ``question_conditioned_writes-train`` [teacher]: the write encoder, data-parallel over every GPU.
4. ``question_conditioned_writes-dev`` [train]: development split, best and last checkpoint, random banks.
5. ``question_conditioned_writes-test`` [dev]: test split, selected checkpoint, with native and normalized
   (beta 0.5, the distractor-curve selection). Banks are the distractor-curve banks (seed ``distractor_curves-banks-v1``),
   so rows pair with the distractor-curve ICL and reader rows by (query, bank).

Selection: the checkpoint (best or last) with the higher development F1, macro over K on random banks (ties to
best), as DISTRACTOR_CURVES chose reader checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from distractor_curves_core import K_VALUES, source_banks, sources

SCHEMA = "unit-plan-v1"
BANK_SEED = "distractor_curves-banks-v1"
TRAIN_STEPS = 4362
SEED = 210
BETA = 0.5
SOURCES_PER_UNIT = 5
TEACHER_SHARD = 2000
LATENT_SHARD = 500


def chunks(values: list, size: int) -> list[list]:
    return [values[i:i + size] for i in range(0, len(values), size)]


def eval_units(config: dict[str, Any], partition: str, encoders: dict[str, str], *, tag: str,
               test: bool) -> list[dict[str, Any]]:
    units = []
    for index, group in enumerate(chunks(config["sources"][partition], config["sources_per_unit"])):
        units.append({"id": f"question_conditioned_writes-{tag}-{index:03d}", "kind": "eval", "partition": partition, "sources": group,
                      "banks": {s: [b for b in config["banks"][partition][s] if b["condition"] == "random"
                                    and b["k"] in config["k_values"]] for s in group},
                      "encoders": encoders, "beta": BETA if test else None, "budget": config["max_new_tokens"],
                      "question_limit": config["question_limit"], "cost": float(len(group))})
    return units


def build_plan(*, suite_path: str, suite: dict[str, Any], query_ids: list[str], latents: str,
               smoke: bool = False, k_values: list[int] | None = None, teacher_dir: str | None = None,
               experiment: str = "QUESTION_CONDITIONED_WRITES", rows: str | None = None, document_ids: list[str] | None = None,
               train_questions: int | None = None, batch_size: int | None = None,
               learning_rate: float | None = None, validation_interval: int = 500,
               compile_latents: bool = True, contrastive_weight: float = 0.0, margin: float = 0.1) -> dict[str, Any]:
    """``k_values`` sets both training and evaluation bank sizes (SINGLE_DOCUMENT_WRITES: [1]); ``teacher_dir`` reuses cached
    teacher responses and drops the teacher stage. ``rows`` (a frozen rows file) adds a latents stage over
    ``document_ids`` (unless ``compile_latents`` is False: reuse ``latents``) and trains one pass over
    ``train_questions`` in updates of ``batch_size``; ``contrastive_weight`` > 0 adds CONTRASTIVE_WRITES's wrong-question hinge."""
    selected = {p: sources(suite, p) for p in ("dev", "test")}
    k_values = list(k_values or K_VALUES)
    if smoke:
        selected = {p: ids[:1] for p, ids in selected.items()}
        k_values = [k for k in k_values if k <= 2]
    banks = {p: {s: source_banks(suite, s, p, BANK_SEED) for s in ids} for p, ids in selected.items()}
    train = {"seed": SEED, "steps": 3 if smoke else TRAIN_STEPS, "k_values": k_values,
             "validation_interval": 3 if smoke else validation_interval, "validation_limit": 5 if smoke else None}
    if rows:
        train["batch_size"], train["learning_rate"] = 4 if smoke else batch_size, learning_rate
        if contrastive_weight:
            train["contrastive_weight"], train["margin"] = contrastive_weight, margin
        if not smoke:
            train["steps"] = train_questions // batch_size  # one pass: every question used at most once
    config = {"smoke": smoke, "suite": suite_path, "latents": None if rows and compile_latents else latents, "rows": rows,
              "teacher_dir": teacher_dir,
              "bank_seed": BANK_SEED, "sources": selected,
              "banks": banks, "k_values": k_values, "sources_per_unit": SOURCES_PER_UNIT, "beta": BETA,
              "max_new_tokens": 4 if smoke else 32, "question_limit": 2 if smoke else None,
              "train": train}
    first_source = selected["dev"][0]
    gate_bank = min((b for b in banks["dev"][first_source] if b["condition"] == "random" and b["k"] in k_values),
                    key=lambda b: b["k"])
    before = (["question_conditioned_writes-latents"] if rows and compile_latents else []) + ([] if teacher_dir else ["question_conditioned_writes-teacher"])
    stages = [{"name": "question_conditioned_writes-gates"}] + [{"name": name} for name in before] + [
              {"name": "question_conditioned_writes-train", "after": before},
              {"name": "question_conditioned_writes-dev", "after": ["question_conditioned_writes-train"], "expand": "question_conditioned_writes_plan:expand_dev"},
              {"name": "question_conditioned_writes-test", "after": ["question_conditioned_writes-dev"], "expand": "question_conditioned_writes_plan:expand_test"}]
    units = [{"id": "question_conditioned_writes-gates", "stage": "question_conditioned_writes-gates", "kind": "gates", "source": first_source,
              "document_ids": gate_bank["document_ids"], "cost": 1.0}]
    if rows and compile_latents:
        units += [{"id": f"question_conditioned_writes-latents-{i:03d}", "stage": "question_conditioned_writes-latents", "kind": "latents", "shard": i,
                   "document_ids": group, "cost": len(group) / LATENT_SHARD}
                  for i, group in enumerate(chunks(sorted(document_ids), LATENT_SHARD))]
    if not teacher_dir:
        units += [{"id": f"question_conditioned_writes-teacher-{i:03d}", "stage": "question_conditioned_writes-teacher", "kind": "teacher", "shard": i,
                   "query_ids": group, "cost": len(group) / TEACHER_SHARD}
                  for i, group in enumerate(chunks(query_ids, TEACHER_SHARD))]
    units.append({"id": f"question_conditioned_writes-train-seed{SEED}", "stage": "question_conditioned_writes-train", "kind": "train", **config["train"],
                  "cost": 50.0})
    return {"schema": SCHEMA, "experiment": experiment, "config": config, "stages": stages, "units": units}


def expand_dev(plan: dict[str, Any], prior: dict[str, Any]) -> list[dict[str, Any]]:
    seed = plan["config"]["train"]["seed"]
    encoders = {f"question_conditioned_writes_{name}": f"encoders/seed{seed}/{name}.pt" for name in ("best", "last")}
    return eval_units(plan["config"], "dev", encoders, tag="dev", test=False)


def _rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for entry in results.values() for source in entry["result"]["sources"] for row in source["rows"]]


def choose_checkpoint(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def macro(arm):
        by_k: dict[int, list[float]] = {}
        for row in rows:
            if row["condition"] == "random" and arm in row["scores"]:
                by_k.setdefault(row["k"], []).append(row["scores"][arm]["f1"])
        return sum(sum(v) / len(v) for v in by_k.values()) / len(by_k)
    best, last, native = macro("question_conditioned_writes_best"), macro("question_conditioned_writes_last"), macro("native")
    return {"checkpoint": "best" if best >= last else "last", "dev_f1": {"best": best, "last": last, "native": native}}


def expand_test(plan: dict[str, Any], prior: dict[str, Any]) -> list[dict[str, Any]]:
    choice = choose_checkpoint(_rows(prior["question_conditioned_writes-dev"]))
    seed = plan["config"]["train"]["seed"]
    units = eval_units(plan["config"], "test", {"question_conditioned_writes": f"encoders/seed{seed}/{choice['checkpoint']}.pt"},
                       tag="test", test=True)
    return [{**unit, "selection": choice} for unit in units]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", type=Path, default=Path("inputs/distractor_curves/suite/14b15feae84cd72b/suite.json"))
    parser.add_argument("--latents", default="artifacts/question_conditioned_writes-latents", help="experiments-relative latent shard directory")
    parser.add_argument("--latent-ids", type=Path, help="local latent shard dir to restrict training rows (smoke)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--k-values", type=int, nargs="+", help="training and evaluation bank sizes (SINGLE_DOCUMENT_WRITES: 1)")
    parser.add_argument("--teacher-dir", help="experiments-relative directory of cached teacher responses (skips the stage)")
    parser.add_argument("--experiment", default="QUESTION_CONDITIONED_WRITES")
    parser.add_argument("--rows", type=Path, help="frozen rows file (SINGLE_DOCUMENT_WRITES); latents are compiled in the run")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--validation-interval", type=int, default=500)
    parser.add_argument("--reuse-latents", action="store_true", help="with --rows: mount --latents, skip the stage")
    parser.add_argument("--contrastive-weight", type=float, default=0.0)
    parser.add_argument("--margin", type=float, default=0.1)
    args = parser.parse_args()
    if args.rows:
        frozen = json.loads(args.rows.read_text())
        train = [r["query"]["id"] for r in frozen["train"]]
        validation = [r["query"]["id"] for r in frozen["validation"]]
        documents = sorted({r["document"]["id"] for split in ("train", "validation") for r in frozen[split]})
        if args.smoke:
            keep = set(documents[:12])
            train = [r["query"]["id"] for r in frozen["train"] if r["document"]["id"] in keep][:40]
            validation = [r["query"]["id"] for r in frozen["validation"]][:5]
            documents = sorted(keep | {r["document"]["id"] for r in frozen["validation"][:5]})
        from qa_protocol_protocol import load_frozen_manifest
        from distractor_curves_core import load_suite

        from experiment_plan import plan_digest, validate_plan

        suite = load_suite(args.suite, load_frozen_manifest(Path("inputs/qa_protocol/manifest/pinned-v1/manifest.json")))
        plan = build_plan(suite_path=str(args.suite), suite=suite, query_ids=train + validation,
                          latents=args.latents, smoke=args.smoke, k_values=args.k_values, experiment=args.experiment,
                          rows=str(args.rows), document_ids=documents, train_questions=len(train),
                          batch_size=args.batch_size, learning_rate=args.learning_rate,
                          validation_interval=args.validation_interval, teacher_dir=args.teacher_dir,
                          compile_latents=not args.reuse_latents, contrastive_weight=args.contrastive_weight,
                          margin=args.margin)
        validate_plan(plan)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n")
        print(json.dumps({"output": str(args.output), "sha256": plan_digest(plan), "units": len(plan["units"]),
                          "teacher_questions": len(train) + len(validation), "documents": len(documents),
                          "steps": plan["config"]["train"]["steps"]}, indent=2))
        return
    from qa_protocol_protocol import load_frozen_manifest
    from distractor_curves_core import load_suite
    from distractor_curves_reader_training import cache_documents, training_rows

    from d2l_scale import load_manifest
    from experiment_plan import plan_digest, validate_plan

    suite = load_suite(args.suite, load_frozen_manifest(Path("inputs/qa_protocol/manifest/pinned-v1/manifest.json")))
    payload = load_manifest(Path("artifacts/d2l_scaling/gemma2-squad-matched-distill-scale60k-seed141/manifest.json"))
    cached = set(cache_documents(payload))
    if args.latent_ids:
        import torch

        cached = {i for p in sorted(args.latent_ids.glob("*.pt")) for i in torch.load(p, weights_only=False)["ids"]}
    rows = training_rows(payload)
    train = [r["query"]["id"] for r in rows["train"] if r["document"]["id"] in cached]
    validation = [r["query"]["id"] for r in rows["validation"] if r["document"]["id"] in cached]
    if args.smoke:
        train, validation = train[:40], validation[:5]
    plan = build_plan(suite_path=str(args.suite), suite=suite, query_ids=train + validation, latents=args.latents,
                      smoke=args.smoke, k_values=args.k_values, teacher_dir=args.teacher_dir, experiment=args.experiment)
    validate_plan(plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": plan_digest(plan), "units": len(plan["units"]),
                      "teacher_questions": len(train) + len(validation)}, indent=2))


if __name__ == "__main__":
    main()
