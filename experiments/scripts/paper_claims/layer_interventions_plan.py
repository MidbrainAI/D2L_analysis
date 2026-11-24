"""Build the frozen plans for the layer interventions, gated readers, and membership probes (RQ2).

Default (``fast``: the shortest run that answers the distractor-immunity question). Serial stages,
results first:

1. ``regression``: DISTRACTOR_CURVES's 12-row READOUT_LADDER regression (runtime gate).
2. ``train-GB-lam1``, ``train-GBL-lam1``, ``train-BS-lam1`` (GATED_READERS): one exclusive data-parallel
   training each, seed 218, the WRITE_SITE_READERS contract (about 35 min each on 8 A100s for a B body).
   The inline validation (fresh F1 at every K on 200 questions every 500 updates) is the
   earliest immunity signal.
3. ``gated-reader`` [after the trainings]: ``best`` and ``last`` of every trained arm on the
   selection suite and DISTRACTOR_CURVES dev at K in ``FAST_EVAL_K`` = {1, 8} only (the immunity number is
   the largest evaluated K minus K=1; expand hook). ICL, native, and normalized rows are not rerun: the WRITE_SITE_READERS selection run
   measured them on the same banks (same bank seed), and ``layer_interventions_report`` merges them.
4. ``diagnostics`` (LAYER_INTERVENTIONS, last so the run can be stopped once the readers are judged): 36
   ``oracle`` units (five selection-suite sources each; random banks at K in {8, 16};
   layer-prefix oracles from layers 0/8/18) and one ``probe`` unit (priority 10, so it starts
   first and runs beside the oracles on its own GPU). The probe needs DISTRACTOR_CURVES's cached Gemma
   latents in ``state/side/latents/gemma2/`` before launch.

``--full`` restores the preregistered ablation: arms GB lam 0 and GC lam 1 as well, oracles from
layers 0/4/8/13/18/22 at K in {4, 8, 16}, evaluation at every K, diagnostics before training.
``--smoke`` adds a ``latents`` stage, evaluates one source, two oracle layers, two probe plans,
and two training updates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from write_site_readers_core import BANK_SEED
from write_site_readers_plan import (
    MANIFEST,
    MODEL,
    READER_MANIFEST,
    SELECTION_SUITE,
    base_config,
    chunks,
    collect_supports,
    eval_units,
    plan_envelope,
    regression_unit,
    smoke_reader_documents,
    suite_records,
    write_plan,
)
from layer_interventions_gates import ORACLE_K, ORACLE_LAYER_FROM, PROBE_K

EXPERIMENT = "LAYER_INTERVENTIONS"
SEED = 218
FULL_ARMS = (("GB", 0.0), ("GB", 1.0), ("BS", 1.0), ("GBL", 1.0), ("GC", 1.0))
ARMS = (("GB", 1.0), ("GBL", 1.0), ("BS", 1.0))  # fast default: the candidate, its latent-keyed variant, the loss-only control
FAST_ORACLE_LAYER_FROM = (0, 8, 18)
FAST_ORACLE_K = (8, 16)
FAST_EVAL_K = (1, 8)  # the immunity number is K=8 minus K=1
PROBE_TRAIN_PLANS = 1500


def lam_tag(lam: float) -> str:
    return f"{lam:g}"


def train_name(arm: str, lam: float, seed: int = SEED) -> str:
    return f"{arm}-lam{lam_tag(lam)}-seed{seed}"


def reader_path(arm: str, lam: float, checkpoint: str, seed: int = SEED) -> str:
    return f"side:readers/{MODEL}/{train_name(arm, lam, seed)}/{checkpoint}.pt"


def oracle_units(config: dict[str, Any], *, smoke: bool) -> list[dict[str, Any]]:
    banks = config["banks"]["selection"]
    layers = [0, 13] if smoke else list(config["oracle_layer_from"])
    ks = [4] if smoke else list(config["oracle_ks"])
    units = []
    for index, group in enumerate(chunks(config["sources"]["selection"], config["sources_per_unit"])):
        units.append({"id": f"oracle-selection-{index:03d}", "stage": "diagnostics", "kind": "oracle", "model": MODEL,
                      "suite": "selection", "sources": group, "banks": {s: banks[s] for s in group},
                      "layer_from": layers, "ks": ks, "cost": 0.5 * len(group) * len(layers) * len(ks)})
    return units


def probe_unit(config: dict[str, Any], *, smoke: bool) -> dict[str, Any]:
    return {"id": "probe-selection", "stage": "diagnostics", "kind": "probe", "model": MODEL, "suite": "selection",
            "banks": config["banks"]["selection"], "selection_suite": str(SELECTION_SUITE), "seed": SEED,
            "train_plans": 4 if smoke else PROBE_TRAIN_PLANS, "train_ks": list(PROBE_K),
            "test_ks": [4] if smoke else list(config["oracle_ks"]), "test_limit": 4 if smoke else None,
            "max_document_occurrences": 12, "priority": 10, "cost": 60.0}


def train_unit(config: dict[str, Any], arm: str, lam: float) -> dict[str, Any]:
    name = train_name(arm, lam)
    return {"id": f"train-{name}", "stage": f"train-{arm}-lam{lam_tag(lam)}", "kind": "train", "model": MODEL,
            "arm": arm, "lam": lam, "seed": SEED, **config["train"], "cost": 50.0}


def build(supports: dict[str, dict[str, Any]], suites: dict[str, dict[str, str]], *, smoke: bool,
          reader_documents: list[str] | None = None, full: bool = False, probe_only: bool = False) -> dict[str, Any]:
    """``probe_only``: one ``diagnostics`` stage holding only the probe unit, for rerunning the probe alone."""
    arms = FULL_ARMS if full else ARMS
    config = base_config(supports, ("selection", "distractor_curves-dev"), smoke=smoke, bank_seed=BANK_SEED)
    config.update({"arms": [{"arm": arm, "lam": lam, "name": train_name(arm, lam)} for arm, lam in arms], "seed": SEED,
                   "variant": "full" if full else "fast",
                   "oracle_layer_from": list(ORACLE_LAYER_FROM if full else FAST_ORACLE_LAYER_FROM),
                   "oracle_ks": list(ORACLE_K if full else FAST_ORACLE_K),
                   "eval_ks": list(config["k_values"] if full else FAST_EVAL_K)})
    if probe_only:
        return plan_envelope(config, [{"name": "diagnostics"}], [probe_unit(config, smoke=smoke)], suites, "layer_interventions")
    stages: list[dict[str, Any]] = [{"name": "regression"}]
    units: list[dict[str, Any]] = [regression_unit()]
    needs_latents: list[str] = []
    if smoke:
        if not reader_documents:
            raise ValueError("the smoke plan needs reader-training documents for its latents stage")
        from write_site_readers_plan import latents_stage

        stage, latent_units = latents_stage(reader_documents)
        stages.append(stage)
        units += latent_units
        needs_latents = ["latents"]
    diagnostics = {"name": "diagnostics", **({"after": needs_latents} if needs_latents else {})}
    diagnostic_units = oracle_units(config, smoke=smoke) + [probe_unit(config, smoke=smoke)]
    if full:
        stages.append(diagnostics)
        units += diagnostic_units
    train_stages = []
    for arm, lam in arms:
        unit = train_unit(config, arm, lam)
        stages.append({"name": unit["stage"], **({"after": needs_latents} if needs_latents else {})})
        units.append(unit)
        train_stages.append(unit["stage"])
    stages.append({"name": "gated-reader", "after": train_stages, "expand": "layer_interventions_plan:expand_gated_reader"})
    if not full:  # results first; the diagnostics come last
        stages.append(diagnostics)
        units += diagnostic_units
    return plan_envelope(config, stages, units, suites, "layer_interventions")


RESPONSE_ONLY_ORACLE_POSITION_MODES = ("decode", "decode_last")
RESPONSE_ONLY_ORACLE_QUERY_MODES = ("last_prompt", "prompt_mean")
RESPONSE_ONLY_ORACLE_K = (8, 16)


def build_response_only_oracle(supports: dict[str, dict[str, Any]], suites: dict[str, dict[str, str]], *, smoke: bool) -> dict[str, Any]:
    """RESPONSE_ONLY_ORACLE (no training): response-only oracles (source mask on generated positions only, prompt native K;
    ``decode`` and ``decode_last``) with ``from0`` as the in-run ceiling, and the membership probe keyed on the
    prompt summary (last prompt token; mean over prompt positions) instead of the response positions."""
    config = base_config(supports, ("selection",), smoke=smoke, bank_seed=BANK_SEED)
    config.update({"seed": SEED, "variant": "response_only_oracle", "oracle_layer_from": [0], "oracle_ks": list(RESPONSE_ONLY_ORACLE_K),
                   "position_modes": list(RESPONSE_ONLY_ORACLE_POSITION_MODES), "query_modes": list(RESPONSE_ONLY_ORACLE_QUERY_MODES)})
    units = []
    for unit in oracle_units(config, smoke=smoke):
        unit["position_modes"] = ["decode"] if smoke else list(RESPONSE_ONLY_ORACLE_POSITION_MODES)
        unit["cost"] = 0.5 * len(unit["sources"]) * len(unit["ks"]) * (len(unit["layer_from"]) + len(unit["position_modes"]))
        units.append(unit)
    for mode in (RESPONSE_ONLY_ORACLE_QUERY_MODES[:1] if smoke else RESPONSE_ONLY_ORACLE_QUERY_MODES):
        unit = probe_unit(config, smoke=smoke)
        unit.update({"id": f"probe-selection-{mode.replace('_', '-')}", "query_mode": mode})
        units.append(unit)
    plan = plan_envelope(config, [{"name": "diagnostics"}], units, suites, "response_only_oracle")
    plan["experiment"] = "RESPONSE_ONLY_ORACLE"
    return plan


ATTENUATION_SCHEDULES_K = (1, 8)  # the true-mask ceilings are K-independent by construction; both cells check that
ATTENUATION_SCHEDULES_QUERY_MODES = ("last_prompt", "prompt_mean")
ATTENUATION_SCHEDULES_KEY_TYPES = ("b_rows", "latent", "abs_strength")


def build_attenuation_schedules(supports: dict[str, dict[str, Any]], suites: dict[str, dict[str, str]], *, smoke: bool) -> dict[str, Any]:
    """ATTENUATION_SCHEDULES (no training): the ceilings of the three attenuation schedules (``attenuation_schedules_schedules``: A clean
    prefill + attenuated tail, B late attenuation from layer 8, C clean prefill + replayed cache) with the true
    membership mask beside the silent floor, at K in {1, 8} on the selection suite; plus the membership probe
    refit on the *clean* (bank-silent) prefill state with the prompt-summary queries, keys ``b_rows`` and
    ``latent`` and the parameter-free A-side strength scorer, scored at K=8."""
    from attenuation_schedules_schedules import DESIGN_B_LAYER_FROM, SCHEDULES

    config = base_config(supports, ("selection",), smoke=smoke, bank_seed=BANK_SEED)
    config.update({"seed": SEED, "variant": "attenuation_schedules", "schedules": list(SCHEDULES), "layer_from": DESIGN_B_LAYER_FROM,
                   "schedule_ks": list(ATTENUATION_SCHEDULES_K), "query_modes": list(ATTENUATION_SCHEDULES_QUERY_MODES), "key_types": list(ATTENUATION_SCHEDULES_KEY_TYPES),
                   "oracle_ks": [8]})
    banks = config["banks"]["selection"]
    ks = [1, 4] if smoke else list(ATTENUATION_SCHEDULES_K)
    units = []
    for index, group in enumerate(chunks(config["sources"]["selection"], config["sources_per_unit"])):
        units.append({"id": f"schedule-selection-{index:03d}", "stage": "diagnostics", "kind": "schedule", "model": MODEL,
                      "suite": "selection", "sources": group, "banks": {s: banks[s] for s in group},
                      "schedules": list(SCHEDULES), "layer_from": DESIGN_B_LAYER_FROM, "ks": ks,
                      "cost": 0.5 * len(group) * len(ks) * len(SCHEDULES)})
    for mode in (ATTENUATION_SCHEDULES_QUERY_MODES[:1] if smoke else ATTENUATION_SCHEDULES_QUERY_MODES):
        unit = probe_unit(config, smoke=smoke)
        unit.update({"id": f"probe-clean-{mode.replace('_', '-')}", "query_mode": mode, "silent": True,
                     "key_types": list(ATTENUATION_SCHEDULES_KEY_TYPES), "test_ks": [4] if smoke else [8]})
        units.append(unit)
    plan = plan_envelope(config, [{"name": "diagnostics"}], units, suites, "attenuation_schedules")
    plan["experiment"] = "ATTENUATION_SCHEDULES"
    return plan


def parse_scorer(spec: str) -> dict[str, Any]:
    """``key_type:layer:query_mode`` -> arm fields; the scorer name is the spec with ``-`` separators."""
    key_type, layer, query_mode = spec.split(":")
    return {"scorer": f"{key_type}-L{int(layer)}-{query_mode}", "key_type": key_type, "layer": int(layer), "query_mode": query_mode}


def build_predicted_gates(supports: dict[str, dict[str, Any]], suites: dict[str, dict[str, str]], *, smoke: bool,
                    schedules: tuple[str, ...], scorers: tuple[str, ...], modes: tuple[str, ...] = ("soft", "hard"),
                    pools: tuple[str, ...] = ("rank",), b_layers: tuple[int, ...] = (8,)) -> dict[str, Any]:
    """Predicted gates: the chosen schedules with gates predicted by the chosen clean-prefill scorers
    (``attenuation_schedules_predicted``). Stage ``probe-fit`` saves one probe per learned scorer recipe (silent collection,
    the scorer's query mode, keys ``b_rows`` and ``latent``); stage ``predicted`` [after it] runs
    every (schedule, scorer, mode, pool) arm on the selection suite at K in {1, 8}. Schedule B arms are made per
    ``b_layers`` gate layer with every scorer whose layer is at or below it."""
    from attenuation_schedules_schedules import DESIGN_B_LAYER_FROM

    config = base_config(supports, ("selection",), smoke=smoke, bank_seed=BANK_SEED)
    parsed = [parse_scorer(spec) for spec in scorers]
    arms = []
    for schedule in schedules:
        for layer_from in (b_layers if schedule == "B" else (DESIGN_B_LAYER_FROM,)):
            for p in parsed:
                if schedule == "B" and p["layer"] > layer_from:
                    continue
                for mode in modes:
                    for pool in pools:
                        label = f"B{layer_from}" if schedule == "B" else schedule
                        arms.append({"name": f"{label}-{p['scorer']}-{mode}" + ("" if pool == "rank" else f"-{pool}"),
                                     "schedule": schedule, "layer_from": layer_from, **p, "mode": mode, "pool": pool})
    config.update({"seed": SEED, "variant": "predicted-gates", "schedules": list(schedules), "scorers": parsed, "arms": arms,
                   "layer_from": DESIGN_B_LAYER_FROM, "b_layers": list(b_layers),
                   "schedule_ks": list(ATTENUATION_SCHEDULES_K), "oracle_ks": [8]})
    units, fits = [], []
    for scorer in parsed:
        if scorer["key_type"] == "abs_strength":
            continue
        unit = probe_unit(config, smoke=smoke)
        unit.update({"id": f"probe-fit-{scorer['scorer']}", "stage": "probe-fit", "kind": "probe-fit", "scorer": scorer["scorer"],
                     "query_mode": scorer["query_mode"], "silent": True, "key_types": ["b_rows", "latent"], "priority": 10})
        unit.pop("test_ks", None)
        units.append(unit)
        fits.append(unit["id"])
    banks = config["banks"]["selection"]
    ks = [1, 4] if smoke else list(ATTENUATION_SCHEDULES_K)
    groups = list(enumerate(chunks(config["sources"]["selection"], config["sources_per_unit"])))
    for index, group in groups:
            units.append({"id": f"schedule-selection-b{layer_from}-{index:03d}", "stage": "probe-fit" if fits else "predicted",
                          "kind": "schedule", "model": MODEL, "suite": "selection", "sources": group,
                          "banks": {s: banks[s] for s in group}, "schedules": ["B"], "layer_from": layer_from, "ks": ks,
                          "cost": 0.5 * len(group) * len(ks)})
    for index, group in groups:
        units.append({"id": f"predicted-selection-{index:03d}", "stage": "predicted", "kind": "predicted", "model": MODEL,
                      "suite": "selection", "sources": group, "banks": {s: banks[s] for s in group}, "arms": arms,
                      "layer_from": DESIGN_B_LAYER_FROM, "ks": ks, "cost": 0.5 * len(group) * len(ks) * len(arms)})
    stages = ([{"name": "probe-fit"}] if fits else []) + [{"name": "predicted", **({"after": ["probe-fit"]} if fits else {})}]
    plan = plan_envelope(config, stages, units, suites, "attenuation_schedules")
    plan["experiment"] = "PREDICTED_GATES"
    return plan


def build_layer_six_ceiling(supports: dict[str, dict[str, Any]], suites: dict[str, dict[str, str]], *, smoke: bool,
                            ceiling_layers: tuple[int, ...] = (6,)) -> dict[str, Any]:
    """True-mask ceilings of schedule B (bank silent below the gate layer, source only from it) at
    ``ceiling_layers`` on the selection suite at K in {1, 8}; responses ``schedule_B<L>``. Layer 6 is
    the "silent below 6" row of Table 16."""
    config = base_config(supports, ("selection",), smoke=smoke, bank_seed=BANK_SEED)
    config.update({"seed": SEED, "variant": "layer-six-ceiling", "ceiling_layers": list(ceiling_layers),
                   "schedule_ks": list(ATTENUATION_SCHEDULES_K), "oracle_ks": [8]})
    banks = config["banks"]["selection"]
    ks = [1, 4] if smoke else list(ATTENUATION_SCHEDULES_K)
    units = []
    for layer_from in ceiling_layers:
        for index, group in enumerate(chunks(config["sources"]["selection"], config["sources_per_unit"])):
            units.append({"id": f"schedule-selection-b{layer_from}-{index:03d}", "stage": "ceiling",
                          "kind": "schedule", "model": MODEL, "suite": "selection", "sources": group,
                          "banks": {s: banks[s] for s in group}, "schedules": ["B"], "layer_from": layer_from, "ks": ks,
                          "cost": 0.5 * len(group) * len(ks)})
    plan = plan_envelope(config, [{"name": "ceiling"}], units, suites, "attenuation_schedules")
    plan["experiment"] = "LAYER_SIX_CEILING"
    return plan


# -- LAYER_REPLICATION: generality of the layer localization -------------------------------------------
LAYER_REPLICATION_MODELS = ("gemma2", "qwen4")
LAYER_REPLICATION_LAYERS = {"gemma2": 26, "qwen4": 36}
# Gate layers per (suite, model). Gemma's selection cells complete the LAYER_INTERVENTIONS (0/8/18) and ATTENUATION_SCHEDULES (6/8) curves; the
# secondary (DROP + ROPES) cells repeat the three-point curves; Qwen3-4B's 36 layers use the same fractions
# as Gemma's 0/8/18 and 6/8 (8/26*36 = 11.1, 18/26*36 = 24.9, 6/26*36 = 8.3).
LAYER_REPLICATION_GRID: dict[tuple[str, str], dict[str, Any]] = {
    ("selection", "gemma2"): {"oracle": [4, 13, 22], "oracle_ks": [8, 16], "silent": [4, 13], "probes": [], "base": False},
    ("secondary", "gemma2"): {"oracle": [0, 8, 18], "oracle_ks": [8], "silent": [6, 8], "probes": ["response", "clean"],
                              "base": False},
    ("selection", "qwen4"): {"oracle": [0, 11, 25], "oracle_ks": [8], "silent": [8, 11], "probes": ["response", "clean"],
                             "base": True},
}
LAYER_REPLICATION_SILENT_K = (8,)  # the true-mask schedules are K-independent (ATTENUATION_SCHEDULES checked K=1 against K=8)
LAYER_REPLICATION_BASE_K = (1, 8)
LAYER_REPLICATION_OCCURRENCES = {"gemma2": 12, "qwen4": 6}  # DISTRACTOR_CURVES's per-model probe/training batch budget


def layer_replication_probe_unit(config: dict[str, Any], suite: str, model: str, kind: str, *, smoke: bool) -> dict[str, Any]:
    unit = probe_unit(config, smoke=smoke)
    unit.update({"id": f"probe-{suite}-{model}-{kind}", "stage": model_stage(model), "model": model, "suite": suite,
                 "banks": config["banks"][suite], "test_ks": [4] if smoke else [8],
                 "max_document_occurrences": LAYER_REPLICATION_OCCURRENCES[model]})
    if kind == "clean":
        unit.update({"query_mode": "prompt_mean", "silent": True, "key_types": list(ATTENUATION_SCHEDULES_KEY_TYPES)})
    else:
        unit.update({"query_mode": "response", "key_types": ["b_rows", "latent"]})
    return unit


def model_stage(model: str) -> str:
    return {"gemma2": "gemma", "qwen4": "qwen"}[model]


def build_layer_replication(supports: dict[str, dict[str, Any]], suites: dict[str, dict[str, str]], *, smoke: bool) -> dict[str, Any]:
    """LAYER_REPLICATION (no training): do the layer curves of the analysis paper hold on a second model and task family?
    Per (suite, model) cell of ``LAYER_REPLICATION_GRID``: layer-prefix oracles (native below L, source only from L; LAYER_INTERVENTIONS),
    bank-silent schedules (silent below L, source only from L; ATTENUATION_SCHEDULES schedule B), and the membership probes
    (response positions; clean prompt mean) where the cell lacks them. Qwen3-4B also needs its own ICL / native /
    normalized rows on the selection suite (``selection-base-qwen4-*``, K in {1, 8}); Gemma's come from the
    WRITE_SITE_READERS selection run (SQuAD) and confirmation run (DROP/ROPES). Two serial stages, one per model, so no worker switches models."""
    config = base_config(supports, ("selection", "secondary"), smoke=smoke, bank_seed=BANK_SEED)
    config.update({"seed": SEED, "variant": "layer_replication", "models": list(LAYER_REPLICATION_MODELS), "layers": dict(LAYER_REPLICATION_LAYERS),
                   "grid": {f"{suite}:{model}": cell for (suite, model), cell in LAYER_REPLICATION_GRID.items()},
                   "silent_ks": list(LAYER_REPLICATION_SILENT_K), "base_ks": list(LAYER_REPLICATION_BASE_K), "oracle_ks": [8]})
    units: list[dict[str, Any]] = []
    for model in LAYER_REPLICATION_MODELS:
        stage = model_stage(model)
        for suite in ("selection", "secondary"):
            cell = LAYER_REPLICATION_GRID.get((suite, model))
            if cell is None:
                continue
            banks = config["banks"][suite]
            groups = list(enumerate(chunks(config["sources"][suite], config["sources_per_unit"])))
            if cell["base"]:
                base_ks = [1, 4] if smoke else list(LAYER_REPLICATION_BASE_K)
                for index, group in groups:
                    units.append({"id": f"selection-base-{model}-{index:03d}", "stage": stage, "kind": "eval", "model": model,
                                  "suite": suite, "sources": group,
                                  "banks": {s: [b for b in banks[s] if b["condition"] == "random" and b["k"] in base_ks]
                                            for s in group},
                                  "readers": {}, "betas": [config["beta"]], "include_icl": True,
                                  "include_native": True, "cost": 1.0 * len(group) * len(base_ks)})
            oracle_layers = cell["oracle"][:2] if smoke else list(cell["oracle"])
            oracle_ks = [4] if smoke else list(cell["oracle_ks"])
            for index, group in groups:
                units.append({"id": f"oracle-{suite}-{model}-{index:03d}", "stage": stage, "kind": "oracle", "model": model,
                              "suite": suite, "sources": group, "banks": {s: banks[s] for s in group},
                              "layer_from": oracle_layers, "ks": oracle_ks,
                              "cost": 0.5 * len(group) * len(oracle_layers) * len(oracle_ks)})
            silent_ks = [4] if smoke else list(LAYER_REPLICATION_SILENT_K)
            for layer in (cell["silent"][:1] if smoke else cell["silent"]):
                for index, group in groups:
                    units.append({"id": f"schedule-{suite}-{model}-L{layer}-{index:03d}", "stage": stage, "kind": "schedule",
                                  "model": model, "suite": suite, "sources": group, "banks": {s: banks[s] for s in group},
                                  "schedules": ["B"], "layer_from": layer, "ks": silent_ks,
                                  "cost": 0.5 * len(group) * len(silent_ks)})
            for kind in (cell["probes"][:1] if smoke else cell["probes"]):
                units.append(layer_replication_probe_unit(config, suite, model, kind, smoke=smoke))
    stages = [{"name": model_stage(model)} for model in LAYER_REPLICATION_MODELS]
    plan = plan_envelope(config, stages, units, suites, "layer_replication")
    plan["experiment"] = "LAYER_REPLICATION"
    return plan


def serial_order(plan: dict[str, Any]) -> str:
    """The ``--serial`` argument matching the plan's stage order (for a serial stage launcher)."""
    return ",".join(stage["name"] for stage in plan["stages"])


def trained(plan: dict[str, Any], prior: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for entry in plan["config"]["arms"]:
        stage = f"train-{entry['arm']}-lam{lam_tag(entry['lam'])}"
        results = [record.get("result") or {} for record in prior.get(stage, {}).values()]
        if any(result.get("checkpoints") for result in results):
            out.append(entry)
    return out


def expand_gated_reader(plan: dict[str, Any], prior: dict[str, Any]) -> list[dict[str, Any]]:
    """Best and last checkpoint of every trained gated arm on the selection suite and DISTRACTOR_CURVES dev at ``eval_ks``."""
    readers = {f"{entry['name']}_{checkpoint}": reader_path(entry["arm"], entry["lam"], checkpoint, plan["config"]["seed"])
               for entry in trained(plan, prior) for checkpoint in ("best", "last")}
    if not readers:
        return []
    ks = set(plan["config"].get("eval_ks", plan["config"]["k_values"]))
    units = []
    for suite in ("selection", "distractor_curves-dev"):
        for unit in eval_units(plan["config"], suite, tag="gated-reader", readers=readers, betas=[],
                               include_icl=False, include_native=False):
            unit["banks"] = {s: [b for b in banks if b["k"] in ks] for s, banks in unit["banks"].items()}
            unit["cost"] = unit["cost"] * len(ks) / len(plan["config"]["k_values"])
            units.append(unit)
    return units


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true", help="the preregistered five-arm ablation instead of the fast plan")
    parser.add_argument("--probe-only", action="store_true", help="only the membership probe (one diagnostics stage)")
    parser.add_argument("--response_only_oracle", action="store_true", help="RESPONSE_ONLY_ORACLE: response-only oracles and prompt-summary probes")
    parser.add_argument("--attenuation_schedules", action="store_true", help="ATTENUATION_SCHEDULES: attenuation-schedule ceilings and clean-prefill probes")
    parser.add_argument("--predicted-gates", action="store_true", help="schedules with gates predicted by clean-prefill probes")
    parser.add_argument("--schedules", default="A,C", help="predicted gates: comma-separated schedules")
    parser.add_argument("--scorers", default="b_rows:8:last_prompt", help="predicted gates: comma-separated key_type:layer:query_mode")
    parser.add_argument("--modes", default="soft,hard")
    parser.add_argument("--pools", default="rank")
    parser.add_argument("--b-layers", default="8", help="predicted gates: gate layers (layer_from) for schedule B")
    parser.add_argument("--layer-six-ceiling", action="store_true", help="schedule-B true-mask ceilings (Table 16, silent below 6)")
    parser.add_argument("--layer_replication", action="store_true", help="LAYER_REPLICATION: layer curves on Qwen3-4B and DROP/ROPES, Gemma's full sweep")
    args = parser.parse_args()
    from d2l_scale import load_manifest
    from qa_protocol_protocol import load_frozen_manifest

    manifest = load_frozen_manifest(MANIFEST)
    supports = collect_supports(manifest)
    reader_documents = None
    if args.smoke:
        reader_documents = smoke_reader_documents(load_manifest(READER_MANIFEST), json.loads(SELECTION_SUITE.read_text()))
    if args.predicted_gates:
        plan = build_predicted_gates(supports, suite_records(), smoke=args.smoke, schedules=tuple(args.schedules.split(",")),
                                     scorers=tuple(args.scorers.split(",")), modes=tuple(args.modes.split(",")),
                                     pools=tuple(args.pools.split(",")), b_layers=tuple(int(v) for v in args.b_layers.split(",")))
    elif args.layer_six_ceiling:
        plan = build_layer_six_ceiling(supports, suite_records(), smoke=args.smoke)
    elif args.layer_replication:
        plan = build_layer_replication(supports, suite_records(), smoke=args.smoke)
    elif args.attenuation_schedules:
        plan = build_attenuation_schedules(supports, suite_records(), smoke=args.smoke)
    elif args.response_only_oracle:
        plan = build_response_only_oracle(supports, suite_records(), smoke=args.smoke)
    else:
        plan = build(supports, suite_records(), smoke=args.smoke, reader_documents=reader_documents, full=args.full,
                     probe_only=args.probe_only)
        plan["experiment"] = EXPERIMENT
    write_plan(plan, args.output)
    print("serial:", serial_order(plan))


if __name__ == "__main__":
    main()
