"""Reduce the LAYER_INTERVENTIONS/GATED_READERS runs: oracle table, probe AUC tables, gated-reader tables and immunity contrasts.

Subcommands (CPU, over a experiment state directory):

* ``oracle --state <state> --base-state <WRITE_SITE_READERS selection state>``: F1 of the layer-prefix oracles by K
  and starting layer on the selection suite, beside ICL, native, normalized (beta 0.5) and native
  K=1 from the WRITE_SITE_READERS selection run's ``selection-base`` rows (same banks). Checks that ``oracle_from0`` at
  every K answers exactly as native K=1 (the mechanism's correctness gate) and reports the
  paired source-document contrast of each oracle against native and normalized.
* ``probe --state <state>``: per key type and K, the best layer's document-level AUC per token
  and per question, the AUC by layer, and the slot-level AUC.
* ``schedule --state <state> --base-state <WRITE_SITE_READERS selection state>``: ATTENUATION_SCHEDULES's attenuation-schedule ceilings
  (``schedule_*`` responses) in the oracle table's shape with the same contrasts (``vs_native_k1`` at
  K=8 is the ceiling's distance from native K=1, the preregistered pass criterion).
* ``gated --state <state> --base-state <WRITE_SITE_READERS selection state>``: per gated arm and checkpoint, F1 by K
  on the selection suite and DISTRACTOR_CURVES dev pooled, paired contrasts against native (K=1) and
  normalized (K>=2) and against native K=1 at every K (the immunity contrast: the largest evaluated K minus the same
  reader's K=1 is the preregistered immunity number), plus the training log's gate BCE.

The statistics are ``distractor_curves_report``'s (``contrast``, ``cell``, ``mean``), as in WRITE_SITE_READERS.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from distractor_curves_core import K_VALUES
from distractor_curves_report import cell, contrast, mean
from write_site_readers_report import load_units, merge_rows, table, write_json

NORMALIZED = "normalized_b0.5"
BASE_STATE = Path("artifacts/write_site_readers/selection/state")
ORACLE_PREFIX = "oracle_"  # oracle_from<L> (LAYER_INTERVENTIONS layer prefixes), oracle_decode / oracle_decode_last (RESPONSE_ONLY_ORACLE)
FROM0 = "oracle_from0"  # must equal native K=1


BASE_PATTERN = "selection-base-*.json"


def base_rows(base_state: Path, suites: tuple[str, ...] = ("selection",), pattern: str = BASE_PATTERN) -> list[dict]:
    rows, _, _ = merge_rows(load_units(base_state, pattern))
    return [r for r in rows if r["suite"] in suites]


def merged(state: Path, pattern: str, base_state: Path, suites: tuple[str, ...],
           base_pattern: str = BASE_PATTERN) -> tuple[list[dict], list[dict]]:
    rows, _, units = merge_rows(load_units(state, pattern))
    own = {(r["query_id"], r["bank_id"]): r for r in rows}
    for row in base_rows(base_state, suites, base_pattern):
        key = (row["query_id"], row["bank_id"])
        if key in own:
            own[key]["responses"].update({k: v for k, v in row["responses"].items() if k not in own[key]["responses"]})
            own[key]["scores"].update({k: v for k, v in row["scores"].items() if k not in own[key]["scores"]})
        else:
            own[key] = {**row, "responses": dict(row["responses"]), "scores": dict(row["scores"])}
    return list(own.values()), units


def source_of(row: dict) -> str:
    return row["source_id"]


def k1_native_by_question(rows: list[dict]) -> dict[str, dict[str, Any]]:
    """The native K=1 response and score of every question (bank ``<source>:random:1``)."""
    return {r["query_id"]: {"response": r["responses"]["native"], "f1": r["scores"]["native"]["f1"]}
            for r in cell(rows, "random", 1) if "native" in r["responses"]}


def with_k1_native(rows: list[dict]) -> list[dict]:
    """Add ``native_k1`` (the same question's native K=1 score) to every row so contrasts can pair on it."""
    k1 = k1_native_by_question(rows)
    for row in rows:
        if row["query_id"] in k1:
            row["scores"]["native_k1"] = {"f1": k1[row["query_id"]]["f1"], "em": 0.0}
            row["responses"]["native_k1"] = k1[row["query_id"]]["response"]
    return rows


# -- oracle ------------------------------------------------------------------------------------
SCHEDULE_PREFIX = "schedule_"  # ATTENUATION_SCHEDULES: schedule_none / schedule_A / schedule_B<L> / schedule_C


def oracle_arms(rows: list[dict], prefix: str = ORACLE_PREFIX) -> list[str]:
    def order(name: str) -> tuple[int, int, str]:
        tail = name[len(prefix):]
        return (0, int(tail), "") if tail.isdigit() else (1, 0, name)

    return sorted({a for r in rows for a in r["scores"] if a.startswith(prefix)}, key=order)


def oracle_report(state: Path, base_state: Path, pattern: str = "oracle-selection-*.json",
                  prefix: str = ORACLE_PREFIX, suites: tuple[str, ...] = ("selection",),
                  base_pattern: str = BASE_PATTERN) -> dict[str, Any]:
    """Also the ATTENUATION_SCHEDULES schedule report (``pattern="schedule-selection-*.json"``, ``prefix=SCHEDULE_PREFIX``): the same
    table and paired contrasts; the ``from0`` check then has no rows. LAYER_REPLICATION passes the suite and the base-row
    pattern of its cell (``secondary-base-*.json`` from the WRITE_SITE_READERS confirmation run; ``selection-base-qwen4-*.json`` from its own state)."""
    rows, units = merged(state, pattern, base_state, suites, base_pattern)
    rows = with_k1_native(rows)
    arms = oracle_arms(rows, prefix)
    ks = sorted({r["k"] for r in rows if any(a in r["scores"] for a in arms)})
    cells = [("random", k) for k in ks]
    f1 = table(rows, cells)
    mismatches = {}
    for k in ks:
        group = [r for r in cell(rows, "random", k) if FROM0 in r["responses"] and "native_k1" in r["responses"]]
        mismatches[str(k)] = {"rows": len(group),
                              "mismatching": sum(r["responses"][FROM0] != r["responses"]["native_k1"] for r in group)}
    contrasts = {}
    for k in ks:
        group = cell(rows, "random", k)
        contrasts[str(k)] = {arm: {"vs_native": contrast(group, arm, "native", source_of),
                                   "vs_normalized": contrast(group, arm, NORMALIZED, source_of),
                                   "vs_native_k1": contrast(group, arm, "native_k1", source_of)} for arm in arms}
    return {"arms": arms, "prefix": prefix, "f1": f1, "from0_equals_native_k1": mismatches, "contrasts": contrasts,
            "units": len(units), "questions": len({r["query_id"] for r in rows})}


def schedule_report(state: Path, base_state: Path) -> dict[str, Any]:
    return oracle_report(state, base_state, pattern="schedule-selection-*.json", prefix=SCHEDULE_PREFIX)


PREDICTED_PREFIX = "predicted_"


def predicted_report(state: Path, base_state: Path) -> dict[str, Any]:
    """Predicted gates: the oracle table over ``predicted_*`` responses plus the mean gate on source and distractor
    ranks per arm and K, and the immunity number (K=8 minus the arm's own K=1)."""
    out = oracle_report(state, base_state, pattern="predicted-selection-*.json", prefix=PREDICTED_PREFIX)
    rows, _ = merged(state, "predicted-selection-*.json", base_state, ("selection",))
    gates: dict[str, dict[str, dict[str, float]]] = {}
    for arm in out["arms"]:
        gates[arm] = {}
        for k in sorted({r["k"] for r in rows}):
            group = [r["gates"][arm] for r in rows if r["k"] == k and arm in r.get("gates", {})]
            if group:
                finite = [g["distractor_gate"] for g in group if g["distractor_gate"] == g["distractor_gate"]]
                gates[arm][str(k)] = {"source_gate": sum(g["source_gate"] for g in group) / len(group),
                                      "distractor_gate": (sum(finite) / len(finite)) if finite else None, "rows": len(group)}
    out["gates"] = gates
    out["immunity"] = {arm: immunity(rows, arm) for arm in out["arms"]}
    return out


def oracle_table(report: dict[str, Any]) -> str:
    arms = report["arms"]
    prefix = report.get("prefix", ORACLE_PREFIX)
    label = lambda a: a[len(prefix):].replace("from", "from ", 1) if a[len(prefix):].startswith("from") else a[len(prefix):]  # noqa: E731
    header = "| K | ICL | native | normalized | native K=1 | " + " | ".join(label(a) for a in arms) + " |"
    lines = [header, "|" + "---|" * (5 + len(arms))]
    for key, entry in report["f1"].items():
        f = entry["f1"]
        fmt = lambda a: f"{f[a]:.3f}" if a in f and f[a] is not None else "-"  # noqa: E731
        lines.append(f"| {key.split(':')[1]} | {fmt('icl')} | {fmt('native')} | {fmt(NORMALIZED)} | {fmt('native_k1')} | "
                     + " | ".join(fmt(a) for a in arms) + " |")
    return "\n".join(lines)


# -- probe -------------------------------------------------------------------------------------
def probe_report(state: Path, pattern: str = "probe-*.json") -> dict[str, Any]:
    """One entry per probe unit (LAYER_INTERVENTIONS: one unit, query mode ``response``; RESPONSE_ONLY_ORACLE: one per query mode)."""
    units = load_units(state, pattern)
    if not units:
        raise FileNotFoundError("no probe unit result")
    out: dict[str, Any] = {"units": {}}
    for index, unit in enumerate(units):
        result = unit["result"]
        out["units"][unit.get("unit", {}).get("id", f"probe-{index}")] = {
            "query_mode": result.get("query_mode", "response"), "summary": result["summary"],
            "fit_bce_last_quarter": result["fit_bce_last_quarter"], "train_plans": result["train_plans"],
            "test_plans": result["test_plans"], "key_types": result["key_types"]}
    first = next(iter(out["units"].values()))
    out.update(first)  # backwards-compatible top level: the first unit
    return out


def probe_table(report: dict[str, Any], level: str = "token") -> str:
    lines = [f"| unit (query mode) | key type | K | best layer | AUC ({level}) | AUC by layer (0..L-1) |", "|---|---|---|---|---|---|"]
    for unit_id, entry_unit in report.get("units", {"probe": report}).items():
        label = f"{unit_id} ({entry_unit.get('query_mode', 'response')})"
        for key_type, per_k in entry_unit["summary"][level].items():
            for k, entry in sorted(per_k.items(), key=lambda item: int(item[0])):
                by_layer = " ".join("-" if v is None else f"{v:.2f}" for v in entry["by_layer"])
                lines.append(f"| {label} | {key_type} | {k} | {entry['best_layer']} | {entry['best_auc']:.3f} | {by_layer} |")
    return "\n".join(lines)


# -- gated readers -----------------------------------------------------------------------------
def gated_arms(rows: list[dict]) -> list[str]:
    return sorted({a for r in rows for a in r["scores"] if "-lam" in a and (a.endswith("_best") or a.endswith("_last"))})


def immunity(rows: list[dict], arm: str, k_high: int | None = None) -> dict[str, Any]:
    """The largest evaluated K (``k_high``) minus K=1 of the same arm (paired by question), and the ratio of the means."""
    ks = sorted({r["k"] for r in rows if arm in r["scores"]})
    if k_high is None:
        k_high = max((k for k in ks if k > 1), default=None)
    if k_high is None:
        return {}
    k1 = {r["query_id"]: r["scores"][arm]["f1"] for r in cell(rows, "random", 1) if arm in r["scores"]}
    high = [(r["source_id"], r["scores"][arm]["f1"] - k1[r["query_id"]]) for r in cell(rows, "random", k_high)
            if arm in r["scores"] and r["query_id"] in k1]
    if not high:
        return {}
    pseudo = [{"source_id": s, "scores": {"d": {"f1": d}, "z": {"f1": 0.0}}} for s, d in high]
    out = contrast(pseudo, "d", "z", source_of) or {}
    m1, m_high = mean(cell(rows, "random", 1), arm), mean(cell(rows, "random", k_high), arm)
    return {"k_high": k_high, "k_high_minus_k1": out, "k1": m1, "k_high_f1": m_high, "ratio": (m_high / m1) if m1 else None}


def gated_report(state: Path, base_state: Path) -> dict[str, Any]:
    rows, units = merged(state, "gated-reader-*.json", base_state, ("selection", "distractor_curves-dev"))
    rows = with_k1_native(rows)
    arms = gated_arms(rows)
    ks = sorted({r["k"] for r in rows if any(a in r["scores"] for a in arms)}) or list(K_VALUES)  # the fast plan evaluates K in {1, 8}
    cells = [("random", k) for k in ks]
    out: dict[str, Any] = {"arms": arms, "ks": ks, "f1": table(rows, cells), "contrasts": {}, "immunity": {}, "training": {}}
    for arm in arms:
        out["contrasts"][arm] = {}
        for k in ks:
            group = cell(rows, "random", k)
            reference = "native" if k == 1 else NORMALIZED
            out["contrasts"][arm][str(k)] = {"reference": reference, "vs_reference": contrast(group, arm, reference, source_of),
                                             "vs_native": contrast(group, arm, "native", source_of),
                                             "vs_native_k1": contrast(group, arm, "native_k1", source_of)}
        out["immunity"][arm] = immunity(rows, arm)
    for entry in load_units(state, "train-*.json"):
        result = entry.get("result") or {}
        if not result:
            continue
        tail = result.get("loss_tail", [])
        out["training"][entry["unit"]["id"]] = {
            "steps": result.get("steps"), "best_step": result.get("best_step"),
            "validation_history": [{"step": h["step"], "answer_ce": h["answer_ce"], "fresh_f1_per_k": h["fresh_f1_per_k"]}
                                   for h in result.get("validation_history", [])],
            "gate_bce_tail": [row.get("gate_bce") for row in tail if "gate_bce" in row][-10:],
            "ce_tail": [row.get("ce") for row in tail if "ce" in row][-10:],
            "accounting": result.get("accounting", {})}
    return out


def gated_table(report: dict[str, Any]) -> str:
    arms = report["arms"]
    lines = ["| K | ICL | native | normalized | " + " | ".join(arms) + " |", "|" + "---|" * (4 + len(arms))]
    for key, entry in report["f1"].items():
        f = entry["f1"]
        fmt = lambda a: f"{f[a]:.3f}" if a in f and f[a] is not None else "-"  # noqa: E731
        lines.append(f"| {key.split(':')[1]} | {fmt('icl')} | {fmt('native')} | {fmt(NORMALIZED)} | " + " | ".join(fmt(a) for a in arms) + " |")
    lines.append("")
    lines.append("| arm | K=1 | K=high | K=high - K=1 [95% CI] | ratio |")
    lines.append("|---|---|---|---|---|")
    for arm, entry in report["immunity"].items():
        if not entry:
            continue
        d = entry["k_high_minus_k1"]
        lines.append(f"| {arm} | {entry['k1']:.3f} | {entry['k_high_f1']:.3f} (K={entry['k_high']}) | {d.get('difference', float('nan')):+.3f} "
                     f"[{d.get('low', float('nan')):+.3f}, {d.get('high', float('nan')):+.3f}] | {entry['ratio']:.2f} |")
    return "\n".join(lines)


# -- LAYER_REPLICATION generality -----------------------------------------------------------------------------
LAYER_REPLICATION_CELLS = (
    # (model, suite, base state, base pattern): where each cell's ICL / native / normalized / native-K=1 rows live
    ("gemma2", "selection", Path("artifacts/write_site_readers/selection/state"), "selection-base-*.json"),
    ("gemma2", "secondary", Path("artifacts/write_site_readers/confirmation/state"), "secondary-base-*.json"),
    ("qwen4", "selection", None, "selection-base-qwen4-*.json"),  # None: the LAYER_REPLICATION state itself
)


def generality_report(state: Path, cells=LAYER_REPLICATION_CELLS) -> dict[str, Any]:
    """Per (model, suite) cell: the oracle (native-below) and schedule (silent-below) tables with paired
    contrasts, and the probe summaries. Cells without results are skipped and listed under ``missing``."""
    out: dict[str, Any] = {"cells": {}, "missing": []}
    for model, suite, base, base_pattern in cells:
        base_state = state if base is None else base
        entry: dict[str, Any] = {"model": model, "suite": suite, "base_state": str(base_state)}
        units = state / "units"
        for kind, prefix in (("oracle", ORACLE_PREFIX), ("schedule", SCHEDULE_PREFIX)):
            pattern = f"{kind}-{suite}-{model}-*.json"
            if not any(units.glob(pattern)):
                entry[kind] = None
                continue
            entry[kind] = oracle_report(state, base_state, pattern=pattern, prefix=prefix, suites=(suite,),
                                        base_pattern=base_pattern)
        probe_pattern = f"probe-{suite}-{model}-*.json"
        entry["probes"] = probe_report(state, probe_pattern)["units"] if any(units.glob(probe_pattern)) else {}
        if entry["oracle"] is None and entry["schedule"] is None and not entry["probes"]:
            out["missing"].append(f"{suite}:{model}")
            continue
        out["cells"][f"{suite}:{model}"] = entry
    return out


def generality_tables(report: dict[str, Any]) -> str:
    parts = []
    for name, entry in report["cells"].items():
        for kind in ("oracle", "schedule"):
            if entry.get(kind) and entry[kind]["arms"]:
                parts.append(f"### {name} {kind}\n" + oracle_table(entry[kind]))
        if entry.get("probes"):
            parts.append(f"### {name} probes\n" + probe_table({"units": entry["probes"]}, "question"))
    return "\n\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("oracle", "probe", "gated", "schedule", "predicted", "generality"):
        command = sub.add_parser(name)
        command.add_argument("--state", type=Path, required=True)
        if name == "generality":
            command.add_argument("--gemma-squad-base-state", type=Path, default=LAYER_REPLICATION_CELLS[0][2])
            command.add_argument("--gemma-drop-ropes-base-state", type=Path, default=LAYER_REPLICATION_CELLS[1][2])
        elif name != "probe":
            command.add_argument("--base-state", type=Path, default=BASE_STATE)
        command.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command in ("oracle", "schedule"):
        report = oracle_report(args.state, args.base_state) if args.command == "oracle" else schedule_report(args.state, args.base_state)
        text = oracle_table(report)
    elif args.command == "predicted":
        report = predicted_report(args.state, args.base_state)
        text = oracle_table(report) + "\n\n" + "\n".join(
            f"| {arm} | K={k} | source gate {v['source_gate']:.2f} | distractor gate {v['distractor_gate'] if v['distractor_gate'] is None else round(v['distractor_gate'], 2)} |"
            for arm, by_k in report["gates"].items() for k, v in by_k.items())
    elif args.command == "probe":
        report = probe_report(args.state)
        text = "\n\n".join(probe_table(report, level) for level in ("token", "question"))
    elif args.command == "generality":
        baselines = (args.gemma_squad_base_state, args.gemma_drop_ropes_base_state, None)
        cells = tuple((model, suite, base, pattern)
                      for (model, suite, _, pattern), base in zip(LAYER_REPLICATION_CELLS, baselines, strict=True))
        report = generality_report(args.state, cells)
        text = generality_tables(report)
    else:
        report = gated_report(args.state, args.base_state)
        text = gated_table(report)
    if args.output:
        write_json(args.output, report)
    print(text)


if __name__ == "__main__":
    main()
