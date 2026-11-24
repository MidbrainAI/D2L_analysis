"""Regenerate selected paper reports on CPU from the published evidence bundles."""

import argparse
import json
from pathlib import Path
import string
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory

from passage_fitting_report import summarize
from d2l_scale_core import qa_f1
from distractor_curves_report import report
from gated_readers_test_report import score_cells as score_gemma_gates
from gated_readers_qwen_report import score_cells as score_qwen_gates
from layer_interventions_report import generality_report, predicted_report, schedule_report
from answer_position_mask_report import report_answer_position_mask
from personamem_report import report_from_state
from question_conditioned_writes_report import report as write_report

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SCRIPTS = ROOT / "experiments/scripts"
WRITE_VARIANTS = ("mixed-k-kl", "single-document-kl", "contrastive-kl")
REDUCTIONS = (
    "passage-fitting", "answer-position-mask", "distractor-curves", "layer-replication", "layer-six-ceiling", "predicted-gates",
    "personamem", "personamem-baseline", "gated-readers-gemma", "gated-readers-qwen",
    "answer-attribution", *WRITE_VARIANTS,
)
DISTRACTOR = "composition/distractor-curves"


def reference_answers():
    return json.loads((RESULTS / DISTRACTOR / "reference-answers.json").read_text())


def source_articles():
    path = ROOT / "experiments/inputs/distractor_curves/suite/14b15feae84cd72b/suite.json"
    return {d["document_id"]: d["article"] for d in json.loads(path.read_text())["documents"]}


def attribution_audit():
    root = RESULTS / "single-document/answer-attribution"
    rows = json.loads((root / "probe-and-generation-rows.json").read_text())["rows"]
    refs = json.loads((root / "reference-passages-and-answers.json").read_text())
    counts = {}
    flags = []
    for arm in ("native_d2l", "rank", "base"):
        counts[arm] = dict(errors=0, literal_span=0, other_answer_token_f1=0)
        for row in rows:
            scores = row["outcomes"]
            if scores["teacher"]["qa_f1"] < .5 or scores[arm]["qa_f1"] >= .5:
                continue
            doc = refs[row["document_id"]]
            qid = row["query_id"]
            response = row["responses"][arm]
            stripped = response.strip().strip(string.punctuation).strip().casefold()
            span = bool(stripped) and stripped in doc["passage"].casefold()
            siblings = [a for q, answers in doc["answers_by_question"].items() if q != qid for a in answers]
            other = qa_f1(response, siblings) >= .5
            flags.append(dict(question_id=qid, arm=arm, literal_span=span, other_answer_token_f1=other))
            for key, value in (("errors", 1), ("literal_span", span), ("other_answer_token_f1", other)):
                counts[arm][key] += value
    return {"counts": counts, "flags": flags}


def unpack(results, key, output):
    archive = results / key
    with tarfile.open(archive, "r:gz") as tar:
        seen = set()
        for member in tar:
            name = Path(member.name)
            if (not member.isfile() or len(name.parts) != 2 or
                    name.parts[0] not in {"units", "expansions"} or name.suffix != ".json"
                    or name in seen):
                raise ValueError(f"Unexpected archive member: {member.name}")
            seen.add(name)
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as handle:
                target.write_bytes(handle.read())
        if not seen:
            raise ValueError(f"Empty archive: {key}")
    return output


def reduce(name, state, output):
    def read(key):
        return unpack(RESULTS, f"{key}.tar.gz", state / key)

    if name == "passage-fitting":
        result = summarize(read("single-document/passage-fitting/scored-units"), set())
    elif name == "answer-attribution":
        result = attribution_audit()
    elif name == "answer-position-mask":
        result = report_answer_position_mask(read("single-document/read-interventions/answer-position-mask-units"))
    elif name == "distractor-curves":
        articles = source_articles()
        answers = reference_answers()
        bank_state = read(f"{DISTRACTOR}/matched-bank-units")
        result = {model: report(bank_state, model, articles, answers) for model in ("gemma2", "qwen4")}
    elif name in WRITE_VARIANTS:
        result = write_report(
            read(f"single-document/question-conditioned-writes/{name}-units"),
            read(f"{DISTRACTOR}/matched-bank-units"), source_articles(), reference_answers(),
        )
    elif name in {"gated-readers-gemma", "gated-readers-qwen"}:
        variant = name.rsplit("-", 1)[-1]
        model = {"gemma": "gemma2", "qwen": "qwen4"}[variant]
        bank_state = read(f"{DISTRACTOR}/matched-bank-units")
        baseline = sorted((bank_state / "units").glob(f"{model}-test-base-*.json"))
        gated_state = read(f"composition/gated-readers/{variant}-test-units")
        pattern = "*.json" if variant == "gemma" else "*gated-test*.json"
        gated = sorted((gated_state / "units").glob(pattern))
        result = (score_gemma_gates(gated, baseline) if variant == "gemma"
                  else score_qwen_gates(gated, baseline, reference_answers()))
    elif name == "layer-replication":
        base = "composition/reader-controls"
        layer = "composition/layer-interventions/cross-model-and-dataset-units"
        cells = (
            ("gemma2", "selection", read(f"{base}/reader-selection-units"), "selection-base-*.json"),
            ("gemma2", "secondary", read(f"{base}/reader-confirmation-and-transfer-units"), "secondary-base-*.json"),
            ("qwen4", "selection", None, "selection-base-qwen4-*.json"),
        )
        layer_state = read(layer)
        result = generality_report(layer_state, cells)
        for entry in result["cells"].values():
            entry["base_state"] = str(Path(entry["base_state"]).relative_to(state)) + ".tar.gz"
    elif name == "layer-six-ceiling":
        result = schedule_report(read("composition/layer-interventions/layer-six-ceiling-units"),
                                 read("composition/reader-controls/reader-selection-units"))
    elif name == "predicted-gates":
        result = predicted_report(read("composition/layer-interventions/predicted-gate-units"),
                                  read("composition/reader-controls/reader-selection-units"))
    elif name == "personamem-baseline":
        result = report_from_state(read("deployment/personamem/memory-representation-scores"))
    else:
        assert name == "personamem"
        base = "deployment/personamem"
        baseline = read(f"{base}/memory-representation-scores")
        script = SCRIPTS / "personamem/personamem_episodic_composition_report.py"
        subprocess.run([
            sys.executable, str(script), "--baseline-state", str(baseline),
            "--followup-state", str(read(f"{base}/episodic-composition-scores")), "--output", str(output),
        ], check=True)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", choices=REDUCTIONS)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with TemporaryDirectory(prefix="d2l-evidence-") as tmp:
        reduce(args.report, Path(tmp), args.output)


if __name__ == "__main__":
    main()
