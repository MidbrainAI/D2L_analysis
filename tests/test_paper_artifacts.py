"""Check the committed paper result that can be regenerated without models."""

from fnmatch import fnmatchcase
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_evidence_bundles_declare_dataset_terms():
    with (ROOT / "REUSE.toml").open("rb") as handle:
        annotations = tomllib.load(handle)["annotations"]
    for bundle in (ROOT / "results").rglob("*.tar.gz"):
        relative = bundle.relative_to(ROOT).as_posix()
        expected = "CC-BY-4.0" if relative.startswith("results/deployment/personamem/") else "CC-BY-SA-4.0"
        licenses = {annotation["SPDX-License-Identifier"] for annotation in annotations
                    if any(fnmatchcase(relative, pattern)
                           for pattern in ([annotation["path"]] if isinstance(annotation["path"], str)
                                           else annotation["path"]))}
        assert licenses == {expected}, relative


def test_personamem_release_has_only_score_shards():
    directory = ROOT / "results/deployment/personamem"
    for name in ("memory-representation-scores", "episodic-composition-scores"):
        with tarfile.open(directory / f"{name}.tar.gz", "r:gz") as bundle:
            members = bundle.getmembers()
        assert members
        assert all(member.isfile() and member.name == f"units/score-shard-{i:03d}.json"
                   for i, member in enumerate(members))


def test_serving_accounting_matches_published_report(tmp_path):
    script = ROOT / "experiments/scripts/paper_claims/serving_cost.py"
    output = tmp_path / "accounting.json"
    subprocess.run([sys.executable, str(script), "--output", str(output)], check=True, cwd=ROOT)
    expected = ROOT / "results/deployment/serving-cost/compute-and-state.json"
    assert json.loads(output.read_text()) == json.loads(expected.read_text())


def assert_report_matches(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_report_matches(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_report_matches(left, right)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12, nan_ok=True)
    else:
        assert actual == expected


@pytest.mark.parametrize(("reduction", "report"), [
    ("passage-fitting", "single-document/passage-fitting/all-160-passages.json"),
    ("answer-position-mask", "single-document/read-interventions/answer-position-mask.json"),
    ("distractor-curves", "composition/distractor-curves/gemma-and-qwen-report.json"),
    ("layer-replication", "composition/layer-interventions/cross-model-and-dataset-replication.json"),
    ("layer-six-ceiling", "composition/layer-interventions/layer-six-ceiling.json"),
    ("predicted-gates", "composition/layer-interventions/predicted-gates.json"),
    ("personamem", "deployment/personamem/episodic-composition.json"),
    ("personamem-baseline", "deployment/personamem/memory-representation-comparison.json"),
    ("mixed-k-kl", "single-document/question-conditioned-writes/mixed-k-kl.json"),
    ("single-document-kl", "single-document/question-conditioned-writes/single-document-kl.json"),
    ("contrastive-kl", "single-document/question-conditioned-writes/contrastive-kl.json"),
])
def test_paper_report_reductions(tmp_path, reduction, report):
    output = tmp_path / "report.json"
    subprocess.run([
        "bash", "experiments/run-paper", "reduce-paper-results.py", reduction, "--output", str(output),
    ], check=True, cwd=ROOT)
    assert_report_matches(json.loads(output.read_text()), json.loads((ROOT / "results" / report).read_text()))


@pytest.mark.parametrize("model", ["gemma", "qwen"])
def test_gated_reader_numerical_cells(tmp_path, model):
    output = tmp_path / "gated.json"
    subprocess.run([
        "bash", "experiments/run-paper", "reduce-paper-results.py", f"gated-readers-{model}",
        "--output", str(output),
    ], check=True, cwd=ROOT)
    report = ROOT / "results/composition/gated-readers" / f"{model}-test-report.json"
    original = json.loads(report.read_text())
    assert_report_matches(json.loads(output.read_text()), {k: original[k] for k in ("cells", "sources", "rows")})


def test_answer_attribution_audit(tmp_path):
    output = tmp_path / "attribution.json"
    subprocess.run([
        "bash", "experiments/run-paper", "reduce-paper-results.py", "answer-attribution",
        "--output", str(output),
    ], check=True, cwd=ROOT)
    audit = json.loads(output.read_text())
    counts = audit["counts"]
    assert counts == {
        "native_d2l": {"errors": 65, "literal_span": 37, "other_answer_token_f1": 25},
        "rank": {"errors": 59, "literal_span": 37, "other_answer_token_f1": 32},
        "base": {"errors": 291, "literal_span": 32, "other_answer_token_f1": 18},
    }
    assert len({(row["arm"], row["question_id"]) for row in audit["flags"]}) == 65 + 59 + 291
