[![MidBrain](assets/midbrain-logo.svg)](https://www.midbrain.ai/)

# Doc-to-LoRA Analysis

[Read the paper on arXiv](https://arxiv.org/abs/2611.30825) · [MidBrain](https://www.midbrain.ai/) · [Create a MidBrain Memory account](https://memory.midbrain.ai)

Companion repository for [**Doc-to-LoRA Analysis: Query Conditioning, Wrong Facts, and Distractor Interference**](https://arxiv.org/abs/2611.30825). The paper studies single-document factual errors, distractor interference during composition, and serving tradeoffs when compiled state replaces text memory.

If you want to try the memory workflow studied in the PersonaMem-v2 experiments, start at [memory.midbrain.ai](https://memory.midbrain.ai). An account is not needed to verify the published results and does not give access to the frozen memory export used in the paper.

- [`paper/`](paper/) contains the manuscript. Fetch the official styles with `make -C paper styles`, then run `make -C paper` (PDFLaTeX, TeX Live 2025).
- [`results/`](results/) maps each paper figure and table to its evaluation set, baselines, and evidence, including offline verification commands where available.
- [`experiments/`](experiments/) contains the runners, frozen evaluation selections, and experiment plans. Shared model loading, configuration, and scoring are in `experiments/scripts/d2l_scale/`; [Reproduction](REPRODUCING.md) maps each table to its plan and runner.

| Paper section | Experiment code under `experiments/scripts/` | Reports under `results/` |
| --- | --- | --- |
| RQ1: wrong facts, single-document controls and read/write interventions | `d2l_readers_k1/` (reader controls), `k1_write_capacity/` (QA splits, transplants, masks, attribution, passage fitting), `paper_claims/question_conditioned_writes_*` | [`single-document/`](results/single-document/) |
| RQ2: distractor curves, layer localization and reader controls | `paper_claims/distractor_curves_*`, `paper_claims/layer_interventions_*`, `paper_claims/attenuation_schedules_*`, `paper_claims/write_site_readers_*`, `paper_claims/gated_readers_*` | [`composition/`](results/composition/) |
| RQ3: PersonaMem-v2 and serving cost | `personamem/` (score reducers only), `paper_claims/serving_cost.py` | [`deployment/`](results/deployment/) |

## Setup and quality checks

On Linux x86-64, [uv](https://docs.astral.sh/uv/) uses Python 3.11 and installs the
locked CPU environment by default. No model or dataset downloads are needed for
these checks.

```bash
uv sync --locked
uv run --locked ruff check .
uv run --locked reuse lint
uv run --locked pytest -q
make -C paper styles
make -C paper check
```

The test suite recomputes reports and gated-reader intervals from the shipped
evidence and checks the serving-cost calculation against the committed report.
To generate that report separately:

```bash
uv run --locked experiments/run-paper scripts/paper_claims/serving_cost.py \
  --output artifacts/serving_cost/accounting-v1.json
```

`experiments/run-paper` starts Python from `experiments/`; paths in plans and runner arguments are relative to that directory. The accounting command runs without weights or benchmark data. See [Reproduction](REPRODUCING.md) for GPU runs and required external inputs.

## External assets

The repo includes frozen selections, benchmark excerpts, and scored evidence.
Full benchmarks, model weights, upstream Doc-to-LoRA code, and LaTeX styles
are obtained separately. [Third-party notices](THIRD_PARTY.md) list exactly
which material is included, its source, and its terms.

## Citation and licensing

Original software is MIT; the manuscript and original documentation use
CC BY 4.0, matching arXiv. Benchmark-bearing evidence follows the terms
listed in [`LICENSE`](LICENSE), [`REUSE.toml`](REUSE.toml), and the
[third-party notices](THIRD_PARTY.md). The MidBrain wordmark is excluded
from these licenses. See [`CITATION.cff`](CITATION.cff) for the paper citation.
