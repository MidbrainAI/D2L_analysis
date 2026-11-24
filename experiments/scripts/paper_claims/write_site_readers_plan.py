"""Build the frozen plans for the write-site deep rank reader (Table 12).

Selection plan (``selection``), stages in order:

1. ``regression``: a 12-row runtime check that reproduces recorded single-document reader responses.
2. ``train-A`` .. ``train-D``: one training unit per reader arm, seed 216. Needs the distractor-curve
   run's cached Gemma latents under the state directory's ``side/latents/gemma2/``.
3. ``selection-base``: ICL, native, and normalized beta 0.5 on the 180-document development set and
   the distractor-curve development split.
4. ``selection-reader`` [after every train stage]: ``best`` and ``last`` of every trained arm on the
   same sources (expand hook ``expand_selection_reader``).

``write_site_readers_report.py selection`` applies the preregistered selection rule to these results.

Confirmation plan (``confirmation --selection <choice>``) fixes the arm and checkpoint from that choice:

1. ``train-<arm>-seed1216``, ``train-<arm>-seed2216``: two more seeds.
2. ``confirm-reader`` [after both]: the three seeds' selected checkpoint on the distractor-curve test
   banks.
3. ``secondary-base``: ICL, native, and normalized beta 0.5 on the DROP/ROPES set; these rows are the
   matched baseline for the cross-dataset layer interventions (Figure 3b).

The confirmation plan ``requires`` seed 216's reader directory from the selection state.

``--smoke`` adds a ``latents`` stage, evaluates one source per suite, and shortens training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from distractor_curves_core import K_VALUES
from write_site_readers_core import (
    BANK_SEED,
    DISTRACTOR_CURVES_PLAN,
    DISTRACTOR_CURVES_SUITE,
    MANIFEST,
    SECONDARY_SUITE,
    SELECTION_SUITE,
    distractor_curves_banks,
    load_write_site_readers_suite,
    random_banks,
    sources,
)

SCHEMA = "unit-plan-v1"
EXPERIMENT = "WRITE_SITE_READERS"
MODEL = "gemma2"
ARMS = ("A", "B", "C", "D")
SELECTION_SEED = 216
CONFIRMATION_SEEDS = (216, 1216, 2216)
SELECTION_BETA = 0.5  # fixed from DISTRACTOR_CURVES's dev selection, not re-tuned
SOURCES_PER_UNIT = 5
TRAIN_STEPS = 4362
TRAIN = {"steps": TRAIN_STEPS, "max_document_occurrences": 12, "validation_interval": 500, "fresh_per_k": 40,
         "selection_suite": str(SELECTION_SUITE)}
SUITES = ("selection", "distractor_curves-dev", "distractor_curves-test", "secondary")
# The rank-effect single-document reader of Table 2, used only by the runtime regression check.
REGRESSION_READER = "artifacts/distractor_curves-inputs/readers/readout_ladderc-step-004365.pt"
REGRESSION_ARTIFACT = "artifacts/distractor_curves-inputs/readout_ladder-exact_400-heldout400-representation-evaluation.json"
READER_MANIFEST = Path("artifacts/d2l_scaling/gemma2-squad-matched-distill-scale60k-seed141/manifest.json")


def chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[i:i + size] for i in range(0, len(values), size)]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reader_path(arm: str, seed: int, checkpoint: str) -> str:
    return f"side:readers/{MODEL}/{arm}-seed{seed}/{checkpoint}.pt"


# -- evaluation supports -----------------------------------------------------------------
def collect_supports(manifest: dict[str, Any], *, bank_seed: str = BANK_SEED) -> dict[str, dict[str, Any]]:
    """Sources and banks of every suite: WRITE_SITE_READERS suites get fresh random banks, DISTRACTOR_CURVES partitions reuse
    the persisted banks of ``inputs/distractor_curves/plans/full-v2.json``."""
    distractor_curves_plan = json.loads(DISTRACTOR_CURVES_PLAN.read_text())
    selection = load_write_site_readers_suite(SELECTION_SUITE, manifest)
    secondary = load_write_site_readers_suite(SECONDARY_SUITE, manifest)
    supports = {
        "selection": {"sources": sources(selection, "selection_source")},
        "secondary": {"sources": sources(secondary, "drop_source") + sources(secondary, "ropes_source")},
    }
    for name, suite in (("selection", selection), ("secondary", secondary)):
        supports[name]["banks"] = {s: random_banks(suite, s, bank_seed) for s in supports[name]["sources"]}
    for partition in ("dev", "test"):
        banks = distractor_curves_banks(distractor_curves_plan, partition, ("random",))
        supports[f"distractor_curves-{partition}"] = {"sources": sorted(distractor_curves_plan["config"]["sources"][partition]), "banks": banks}
    return supports


def suite_records() -> dict[str, dict[str, str]]:
    """Paths and byte-level sha256 of every frozen input the plans depend on."""
    records = {}
    for name, path in (("selection", SELECTION_SUITE), ("secondary", SECONDARY_SUITE), ("distractor_curves_suite", DISTRACTOR_CURVES_SUITE),
                       ("distractor_curves_plan", DISTRACTOR_CURVES_PLAN)):
        record = {"path": str(path), "sha256": file_sha256(path)}
        content = json.loads(path.read_text())
        if "suite_sha256" in content:
            record["suite_sha256"] = content["suite_sha256"]
        records[name] = record
    return records


def smoke_reader_documents(payload: dict[str, Any], selection: dict[str, Any]) -> list[str]:
    """150 train + 50 validation reader-training documents, none held out for selection."""
    from distractor_curves_reader_training import cache_documents
    from write_site_readers_reader_training import held_out_hashes, is_held_out

    held = held_out_hashes(selection)
    documents = {d["id"]: d for d in payload["documents"]}
    usable = [i for i in cache_documents(payload) if not is_held_out(documents[i], held)]
    return ([i for i in usable if documents[i]["split"] == "train"][:150]
            + [i for i in usable if documents[i]["split"] == "validation"][:50])


# -- units -------------------------------------------------------------------------------
def eval_units(config: dict[str, Any], suite: str, *, tag: str, readers: dict[str, str],
               betas: list[float], include_icl: bool, include_native: bool) -> list[dict[str, Any]]:
    banks = config["banks"][suite]
    units = []
    for index, group in enumerate(chunks(config["sources"][suite], config["sources_per_unit"])):
        units.append({
            "id": f"{tag}-{suite}-{index:03d}", "kind": "eval", "model": MODEL, "suite": suite, "sources": group,
            "banks": {s: [b for b in banks[s] if b["condition"] == "random"] for s in group},
            "readers": readers, "betas": betas, "include_icl": include_icl, "include_native": include_native,
            "cost": 1.0 * len(group),
        })
    return units


def train_unit(config: dict[str, Any], arm: str, seed: int, stage: str) -> dict[str, Any]:
    return {"id": f"train-{arm}-seed{seed}", "stage": stage, "kind": "train", "model": MODEL, "arm": arm,
            "seed": seed, **config["train"], "cost": 50.0}


def regression_unit() -> dict[str, Any]:
    return {"id": f"{MODEL}-regression", "stage": "regression", "kind": "regression", "model": MODEL,
            "artifact": REGRESSION_ARTIFACT, "reader": REGRESSION_READER, "rows": 12,
            "min_agreement": 0.75, "priority": 10}


def latents_stage(documents: list[str], shard: int = 50) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    units = [{"id": f"latents-{index:03d}", "stage": "latents", "kind": "latents", "model": MODEL, "shard": index,
              "documents": group, "cost": len(group) / 500} for index, group in enumerate(chunks(documents, shard))]
    return {"name": "latents"}, units


def base_config(supports: dict[str, dict[str, Any]], suites: tuple[str, ...], *, smoke: bool,
                bank_seed: str) -> dict[str, Any]:
    chosen = {name: list(supports[name]["sources"]) for name in suites}
    if smoke:
        chosen = {name: ids[:1] for name, ids in chosen.items()}
    return {
        "smoke": smoke, "model": MODEL, "bank_seed": bank_seed, "beta": SELECTION_BETA, "k_values": list(K_VALUES),
        "sources_per_unit": SOURCES_PER_UNIT, "sources": chosen,
        "banks": {name: {s: supports[name]["banks"][s] for s in ids} for name, ids in chosen.items()},
        "train": {**TRAIN, "steps": 2 if smoke else TRAIN["steps"],
                  "validation_interval": 1 if smoke else TRAIN["validation_interval"],
                  "validation_limit": 10 if smoke else None, "fresh_per_k": 1 if smoke else TRAIN["fresh_per_k"]},
    }


def plan_envelope(config: dict[str, Any], stages: list[dict[str, Any]], units: list[dict[str, Any]],
                  suites: dict[str, dict[str, str]], run: str) -> dict[str, Any]:
    return {"schema": SCHEMA, "experiment": EXPERIMENT, "run": run, "suites": suites, "config": config,
            "stages": stages, "units": units}


# -- selection run ------------------------------------------------------------------------------
def build_selection_plan(supports: dict[str, dict[str, Any]], suites: dict[str, dict[str, str]], *, smoke: bool,
               reader_documents: list[str] | None = None, bank_seed: str = BANK_SEED) -> dict[str, Any]:
    config = base_config(supports, ("selection", "distractor_curves-dev"), smoke=smoke, bank_seed=bank_seed)
    config.update({"arms": list(ARMS), "seed": SELECTION_SEED})
    stages: list[dict[str, Any]] = [{"name": "regression"}]
    units: list[dict[str, Any]] = [regression_unit()]
    needs_latents: list[str] = []
    if smoke:
        if not reader_documents:
            raise ValueError("the smoke plan needs reader-training documents for its latents stage")
        stage, latent_units = latents_stage(reader_documents)
        stages.append(stage)
        units += latent_units
        needs_latents = ["latents"]
    for arm in ARMS:
        stages.append({"name": f"train-{arm}", **({"after": needs_latents} if needs_latents else {})})
        units.append(train_unit(config, arm, SELECTION_SEED, f"train-{arm}"))
    stages.append({"name": "selection-base"})
    for suite in ("selection", "distractor_curves-dev"):
        for unit in eval_units(config, suite, tag="selection-base", readers={},
                               betas=[SELECTION_BETA], include_icl=True, include_native=True):
            units.append({**unit, "stage": "selection-base"})
    stages.append({"name": "selection-reader", "after": [f"train-{arm}" for arm in ARMS],
                   "expand": "write_site_readers_plan:expand_selection_reader"})
    return plan_envelope(config, stages, units, suites, "selection")


def trained_arms(plan: dict[str, Any], prior: dict[str, Any]) -> list[str]:
    """Arms whose train stage finished with checkpoints. Expansion requires every
    dependency has a result, so this is defensive; a missing or checkpoint-less result skips the arm."""
    arms = []
    for arm in plan["config"]["arms"]:
        results = [entry.get("result") or {} for entry in prior.get(f"train-{arm}", {}).values()]
        if any(result.get("checkpoints") for result in results):
            arms.append(arm)
    return arms


def expand_selection_reader(plan: dict[str, Any], prior: dict[str, Any]) -> list[dict[str, Any]]:
    """Best and last checkpoint of every trained arm on the selection suite and DISTRACTOR_CURVES dev."""
    seed = plan["config"]["seed"]
    readers = {f"{arm}_{name}": reader_path(arm, seed, name) for arm in trained_arms(plan, prior)
               for name in ("best", "last")}
    if not readers:
        return []
    units = []
    for suite in ("selection", "distractor_curves-dev"):
        units += eval_units(plan["config"], suite, tag="selection-reader", readers=readers,
                            betas=[], include_icl=False, include_native=False)
    return units


# -- confirmation run ------------------------------------------------------------------------------
def build_confirmation_plan(supports: dict[str, dict[str, Any]], suites: dict[str, dict[str, str]], *, arm: str, checkpoint: str,
               smoke: bool, selection_choice: dict[str, Any] | None = None, reader_documents: list[str] | None = None,
               bank_seed: str = BANK_SEED) -> dict[str, Any]:
    if arm not in ARMS or checkpoint not in ("best", "last"):
        raise ValueError(f"invalid selection: arm {arm}, checkpoint {checkpoint}")
    config = base_config(supports, ("distractor_curves-test", "secondary"), smoke=smoke, bank_seed=bank_seed)
    config.update({"arm": arm, "checkpoint": checkpoint, "seeds": list(CONFIRMATION_SEEDS),
                   "selection_choice": selection_choice or {}})
    readers = {f"reader_s{seed}": reader_path(arm, seed, checkpoint) for seed in CONFIRMATION_SEEDS}
    stages: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []
    needs_latents: list[str] = []
    if smoke:
        if not reader_documents:
            raise ValueError("the smoke plan needs reader-training documents for its latents stage")
        stage, latent_units = latents_stage(reader_documents)
        stages.append(stage)
        units += latent_units
        needs_latents = ["latents"]
    train_stages = []
    for seed in CONFIRMATION_SEEDS[1:]:
        name = f"train-{arm}-seed{seed}"
        stages.append({"name": name, **({"after": needs_latents} if needs_latents else {})})
        units.append(train_unit(config, arm, seed, name))
        train_stages.append(name)
    stages.append({"name": "confirm-reader", "after": list(train_stages)})
    for unit in eval_units(config, "distractor_curves-test", tag="confirm-reader", readers=readers,
                           betas=[], include_icl=False, include_native=False):
        units.append({**unit, "stage": "confirm-reader"})
    stages.append({"name": "secondary-base"})
    for unit in eval_units(config, "secondary", tag="secondary-base", readers={},
                           betas=[SELECTION_BETA], include_icl=True, include_native=True):
        units.append({**unit, "stage": "secondary-base"})
    plan = plan_envelope(config, stages, units, suites, "confirmation")
    plan["requires"] = [{
        "path": f"side/readers/{MODEL}/{arm}-seed{CONFIRMATION_SEEDS[0]}/",
        "from": "the selection plan's state directory",
        "note": f"seed {CONFIRMATION_SEEDS[0]}'s {checkpoint}.pt is evaluated by confirm-reader; "
                "copy the directory into this plan's state directory before running it",
    }]
    return plan


# -- CLI -----------------------------------------------------------------------------------
def write_plan(plan: dict[str, Any], output: Path) -> None:
    from experiment_plan import plan_digest, validate_plan

    validate_plan(plan)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "sha256": plan_digest(plan), "units": len(plan["units"]),
                      "stages": [s["name"] for s in plan["stages"]]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="run", required=True)
    for name in ("selection", "confirmation"):
        command = sub.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--smoke", action="store_true")
    sub.choices["confirmation"].add_argument("--selection", type=Path, help="artifacts/write_site_readers/selection-choice.json")
    sub.choices["confirmation"].add_argument("--arm", choices=ARMS, help="override for smoke/testing")
    sub.choices["confirmation"].add_argument("--checkpoint", choices=("best", "last"), help="override for smoke/testing")
    args = parser.parse_args()
    from d2l_scale import load_manifest
    from qa_protocol_protocol import load_frozen_manifest

    manifest = load_frozen_manifest(MANIFEST)
    supports = collect_supports(manifest)
    suites = suite_records()
    reader_documents = None
    if args.smoke:
        reader_documents = smoke_reader_documents(load_manifest(READER_MANIFEST),
                                                  json.loads(SELECTION_SUITE.read_text()))
    if args.run == "selection":
        plan = build_selection_plan(supports, suites, smoke=args.smoke, reader_documents=reader_documents)
    else:
        choice: dict[str, Any] = {}
        if args.selection is not None:
            choice = json.loads(args.selection.read_text())
            choice = {"path": str(args.selection), "sha256": file_sha256(args.selection), **choice["selected"]}
        arm = args.arm or choice.get("arm")
        checkpoint = args.checkpoint or choice.get("checkpoint")
        if arm is None or checkpoint is None:
            parser.error("confirmation needs --selection or both --arm and --checkpoint")
        plan = build_confirmation_plan(supports, suites, arm=arm, checkpoint=checkpoint, smoke=args.smoke,
                          selection_choice=choice, reader_documents=reader_documents)
    write_plan(plan, args.output)


if __name__ == "__main__":
    main()
