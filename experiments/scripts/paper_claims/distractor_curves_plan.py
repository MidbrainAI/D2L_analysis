"""Build the frozen plan for the distractor curves on Gemma and Qwen (Figure 2, Tables 10--11) and the serving benchmark, with its selection rules.

Stages (dependencies in brackets):

1. ``<m>-latents``: compiler latents for every reader-training document.
2. ``<m>-train`` [latents]: the mixed-K rank-effect reader, three seeds on Gemma, one on Qwen.
3. ``<m>-dev-base``: dev split, every non-trained arm, including the normalization grid.
4. ``<m>-reader-dev`` [train]: dev split, best and last checkpoint of every seed.
5. ``<m>-test-base`` [dev-base]: test split, baseline arms with the selected
   normalization exponent; it does not wait for reader training.
6. ``<m>-test-reader`` [dev-base, reader-dev]: test split, the selected checkpoint of
   every seed. Test is opened once per arm.
7. ``timing`` [every test stage]: SERVING_COST's measured serving benchmark as one unit over both
   models, so the GPU it runs on carries no other work.

Selection rules, applied in code by the expand hooks and fixed before the test split was scored:

* ``beta``: the normalization exponent with the highest dev F1 over random banks at K >= 2
  (ties to the smaller exponent).
* checkpoint: per seed, ``best`` (lowest validation gold CE) or ``last``, whichever has the
  higher dev F1 over random banks at every K (ties to ``best``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from distractor_curves_core import BETA_GRID, K_VALUES, source_banks, sources

SCHEMA = "unit-plan-v1"
MODELS = ("gemma2", "qwen4")
SEEDS = {"gemma2": [199, 1199, 2199], "qwen4": [200]}
# The rank-effect K=1 reader of Table 2, used only by the smoke-plan regression check.
REGRESSION_READER = "artifacts/distractor_curves-inputs/readers/readout_ladderc-step-004365.pt"
TRAIN_STEPS = 4362  # one pass at 12 per update over READOUT_LADDER's 52,354 fitting questions (52,344 used)
SOURCES_PER_UNIT = 5


def chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[i:i + size] for i in range(0, len(values), size)]


def eval_units(config: dict[str, Any], model: str, partition: str, *, readers: dict[str, str],
               betas: list[float], include_native: bool, tag: str) -> list[dict[str, Any]]:
    banks = config["banks"][partition]
    chosen = config["sources"][partition]
    units = []
    for index, group in enumerate(chunks(chosen, config["sources_per_unit"])):
        units.append({
            "id": f"{model}-{tag}-{index:03d}", "kind": "eval", "model": model, "partition": partition,
            "sources": group,
            "banks": {s: banks[s] for s in group},
            "readers": readers, "betas": betas,
            "include_icl": include_native, "include_native": include_native, "cost": 1.0 * len(group),
        })
    return units


def build_plan(*, suite_path: Path, suite: dict[str, Any], reader_documents: list[str], smoke: bool,
               bank_seed: str = "distractor_curves-banks-v1") -> dict[str, Any]:
    partitions = ("dev", "test")
    selected = {p: sources(suite, p) for p in partitions}
    if smoke:
        selected = {p: ids[:1] for p, ids in selected.items()}
    banks = {p: {s: source_banks(suite, s, p, bank_seed) for s in ids} for p, ids in selected.items()}
    shard = 50 if smoke else 500
    documents = reader_documents
    config = {
        "smoke": smoke, "bank_seed": bank_seed, "sources": selected, "banks": banks,
        "sources_per_unit": SOURCES_PER_UNIT, "betas": list(BETA_GRID), "k_values": list(K_VALUES),
        "seeds": SEEDS,
        # Qwen at 12 peaked at 24.6 GB; 6 keeps it beside a co-located worker on a 40 GB A100.
        "train": {"steps": 20 if smoke else TRAIN_STEPS, "max_document_occurrences": {"gemma2": 12, "qwen4": 6},
                  "validation_interval": 10 if smoke else 500, "validation_limit": 10 if smoke else None},
    }
    stages: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []
    if smoke:
        stages.append({"name": "regression"})
        units.append({"id": "gemma2-regression", "stage": "regression", "kind": "regression", "model": "gemma2",
                      "artifact": "artifacts/distractor_curves-inputs/readout_ladder-exact_400-heldout400-representation-evaluation.json",
                      "reader": REGRESSION_READER, "rows": 12, "min_agreement": 0.75, "priority": 10})
    for model in MODELS:
        stages.append({"name": f"{model}-latents"})
        for index, group in enumerate(chunks(documents, shard)):
            units.append({"id": f"{model}-latents-{index:03d}", "stage": f"{model}-latents", "kind": "latents",
                          "model": model, "shard": index, "documents": group, "cost": len(group) / 500})
    for model in MODELS:
        stages.append({"name": f"{model}-train", "after": [f"{model}-latents"]})
        for seed in SEEDS[model][: 1 if smoke else None]:
            units.append({"id": f"{model}-train-seed{seed}", "stage": f"{model}-train", "kind": "train",
                          "model": model, "seed": seed, "steps": config["train"]["steps"],
                          "max_document_occurrences": config["train"]["max_document_occurrences"][model],
                          "validation_interval": config["train"]["validation_interval"],
                          "validation_limit": config["train"]["validation_limit"], "cost": 50.0})
    for model in MODELS:
        stages.append({"name": f"{model}-dev-base"})
        for unit in eval_units(config, model, "dev", readers={}, betas=list(BETA_GRID),
                               include_native=True, tag="dev-base"):
            units.append({**unit, "stage": f"{model}-dev-base"})
    for model in MODELS:
        stages.append({"name": f"{model}-reader-dev", "after": [f"{model}-train"],
                       "expand": "distractor_curves_plan:expand_reader_dev"})
    # Baseline arms do not wait for reader training, so a training failure cannot block them.
    for model in MODELS:
        stages.append({"name": f"{model}-test-base", "after": [f"{model}-dev-base"],
                       "expand": "distractor_curves_plan:expand_test_base"})
    for model in MODELS:
        stages.append({"name": f"{model}-test-reader", "after": [f"{model}-dev-base", f"{model}-reader-dev"],
                       "expand": "distractor_curves_plan:expand_test_reader"})
    # One timing unit, last: every other worker is idle by then, so no co-located load distorts it.
    stages.append({"name": "timing", "after": [f"{model}-test-{part}" for model in MODELS
                                               for part in ("base", "reader")]})
    units.append({"id": "timing", "stage": "timing", "kind": "timing", "models": list(MODELS),
                  "lengths": [256, 1024] if smoke else [128, 256, 512, 1024, 2048, 4096],
                  "documents": [1, 4] if smoke else [1, 2, 4, 8, 16],
                  "batch_sizes": [1, 4] if smoke else [1, 8, 16], "repeats": 2 if smoke else 5,
                  "output_tokens": 64, "cost": 4.0})
    return {"schema": SCHEMA, "experiment": "SERVING_COST-QWEN_REPLICATION", "suite": str(suite_path), "suite_sha256":
            suite["suite_sha256"], "config": config, "stages": stages, "units": units}


# -- selection rules (expand hooks) -------------------------------------------------
def _stage_model(plan: dict[str, Any], prior: dict[str, Any], suffix: str) -> str:
    names = [name for name in prior if name.endswith(suffix)]
    if len(names) != 1:
        raise ValueError(f"expected one {suffix} dependency, found {names}")
    return names[0][: -len(suffix)]


def _rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for entry in results.values() for source in entry["result"]["sources"] for row in source["rows"]]


def mean_f1(rows: list[dict[str, Any]], arm: str, *, min_k: int = 1) -> float:
    """Macro over K of mean F1 on random banks; rows lacking the arm are ignored."""
    by_k: dict[int, list[float]] = {}
    for row in rows:
        if row["condition"] == "random" and row["k"] >= min_k and arm in row["scores"]:
            by_k.setdefault(row["k"], []).append(row["scores"][arm]["f1"])
    if not by_k:
        raise ValueError(f"no dev rows for arm {arm}")
    return sum(sum(v) / len(v) for v in by_k.values()) / len(by_k)


def choose_beta(rows: list[dict[str, Any]], betas: list[float]) -> dict[str, Any]:
    values = {beta: mean_f1(rows, f"normalized_b{beta}", min_k=2) for beta in betas}
    best = max(values.values())
    return {"beta": min(beta for beta, value in values.items() if value == best), "dev_f1": values,
            "native_dev_f1": mean_f1(rows, "native", min_k=2)}


def choose_checkpoint(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    best, last = mean_f1(rows, f"seed{seed}_best"), mean_f1(rows, f"seed{seed}_last")
    return {"checkpoint": "best" if best >= last else "last", "dev_f1": {"best": best, "last": last}}


def expand_reader_dev(plan: dict[str, Any], prior: dict[str, Any]) -> list[dict[str, Any]]:
    model = _stage_model(plan, prior, "-train")
    readers = {}
    for entry in prior[f"{model}-train"].values():
        seed = entry["result"]["seed"]
        for name in ("best", "last"):
            readers[f"seed{seed}_{name}"] = f"side:readers/{model}/seed{seed}/{name}.pt"
    return eval_units(plan["config"], model, "dev", readers=readers, betas=[],
                      include_native=False, tag="reader-dev")


def expand_test_base(plan: dict[str, Any], prior: dict[str, Any]) -> list[dict[str, Any]]:
    """Test baseline arms: ICL, native, and the development-selected normalization."""
    model = _stage_model(plan, prior, "-dev-base")
    beta = choose_beta(_rows(prior[f"{model}-dev-base"]), plan["config"]["betas"])
    units = eval_units(plan["config"], model, "test", readers={}, betas=[beta["beta"]],
                       include_native=True, tag="test-base")
    return [{**unit, "selection": {"beta": beta}} for unit in units]


def expand_test_reader(plan: dict[str, Any], prior: dict[str, Any]) -> list[dict[str, Any]]:
    """Test arms of the presented reader seeds at their development-selected checkpoints."""
    model = _stage_model(plan, prior, "-reader-dev")
    reader_rows = _rows(prior[f"{model}-reader-dev"])
    seeds = sorted({int(name.split("_")[0][4:]) for row in reader_rows for name in row["scores"]})
    choices = {seed: choose_checkpoint(reader_rows, seed) for seed in seeds}
    readers = {f"reader_s{seed}": f"side:readers/{model}/seed{seed}/{choice['checkpoint']}.pt"
               for seed, choice in choices.items()}
    units = eval_units(plan["config"], model, "test", readers=readers, betas=[],
                       include_native=False, tag="test-reader")
    return [{**unit, "selection": {"checkpoints": {str(seed): c for seed, c in choices.items()}}} for unit in units]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    from d2l_scale import load_manifest
    from qa_protocol_protocol import load_frozen_manifest
    from distractor_curves_core import load_suite
    from distractor_curves_reader_training import cache_documents

    suite = load_suite(args.suite, load_frozen_manifest(Path("inputs/qa_protocol/manifest/pinned-v1/manifest.json")))
    reader_payload = load_manifest(Path("artifacts/d2l_scaling/gemma2-squad-matched-distill-scale60k-seed141/manifest.json"))
    documents = cache_documents(reader_payload)
    if args.smoke:  # enough training questions for the smoke's updates, plus validation documents
        split = {d["id"]: d["split"] for d in reader_payload["documents"]}
        documents = ([i for i in documents if split[i] == "train"][:150]
                     + [i for i in documents if split[i] == "validation"][:50])
    plan = build_plan(suite_path=args.suite, suite=suite, reader_documents=documents, smoke=args.smoke)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n")
    from experiment_plan import plan_digest, validate_plan

    validate_plan(plan)
    print(json.dumps({"output": str(args.output), "sha256": plan_digest(plan), "units": len(plan["units"]),
                      "stages": [s["name"] for s in plan["stages"]]}, indent=2))


if __name__ == "__main__":
    main()
