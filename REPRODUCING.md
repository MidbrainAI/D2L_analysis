# Reproduction

The CPU tests cover the runners' logic on small fixtures. The paper's GPU runs also require external checkpoint and dataset files; the repository does not contain training checkpoints, caches, raw datasets, or the frozen production PersonaMem memory export.

## Verify the results without GPUs

The [`results/`](results/) directories follow the paper's three research
questions. The `.tar.gz` files contain JSON unit results and, where present in
the finished run, expanded plans with the exact bank IDs, document IDs, and
source positions. They contain no checkpoints. Archive members use the same
descriptive names as the bundled reducers.

```bash
uv run --locked experiments/run-paper reduce-paper-results.py distractor-curves \
  --output artifacts/checks/distractor-curves.json
```

Replace `distractor-curves` with a command from the [figure and table index](results/).
The offline entry point reads the shipped evidence without fetching models or
data. For gated readers it recomputes the numerical cells and intervals, not
the original runtime or checkpoint provenance. The experiment-specific
scripts under `experiments/scripts/` are not clean-checkout recipes; they may
expect checkpoint files or predecessor states. Timing results are original
observations, not measurements from the offline checks.

## Released Doc-to-LoRA assets

From the repository root, set up the locked GPU environment with
`uv sync --locked --no-group cpu --group cuda` (Linux x86-64, CUDA 13.0).
Use those same group flags with `uv run` for GPU commands.

```bash
root="$PWD"
assets="$root/experiments/artifacts/third_party/doc-to-lora"
git clone https://github.com/SakanaAI/doc-to-lora.git "$assets/source"
git -C "$assets/source" checkout baa85db4d5df9b29d618af432d6ebf28b3ad5a29
uv run --locked --no-group cpu --group cuda huggingface-cli download SakanaAI/doc-to-lora \
  --revision 7514da3cf18439265aef805f2a00170199e44456 \
  --include 'gemma_2b_d2l/checkpoint-20000/pytorch_model.bin' \
  --include 'qwen_4b_d2l/checkpoint-20000/pytorch_model.bin' \
  --local-dir "$assets/checkpoints"
```

The GPU group locks PyTorch and the companion repository dependencies. The
released compiler uses FlashAttention 2; install an upstream-compatible build
and use a GPU it supports. The backend imports upstream code directly from
`source/src`. Obtain Hugging Face access to
Gemma before loading it. Model revisions and checkpoint hashes are recorded in
[`experiments/inputs/configs/d2l_scaling/`](experiments/inputs/configs/d2l_scaling/).
The original checkpoints use Python pickle. Verify their hashes before loading
and do not substitute untrusted checkpoint files.

## QA data and layer oracles

The source dataset revisions are pinned in `gemma2_qa_protocol.json`.
The preparation chain is `qa_protocol_prepare_sources.py` (one JSONL per
family), `qa_protocol_tokenizer_audit.py` (one-chunk audit), and
`qa_protocol_manifest_freeze.py` (frozen manifest). All three are in
`experiments/scripts/k1_write_capacity/`. The tokenizer revision is
`299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8`, and the manifest builder's
`--prompt-template-version` is
`2820a5954f302f885beb2d69e61fdc857c63968310c49bfa1d1fa33dfa602f80`.
The [original manifest provenance](experiments/inputs/qa_protocol/manifest/pinned-v1/provenance.json)
records the historical digest; public descriptive identifiers and paths may
change file hashes without changing the split. Do not substitute a different
split or reorder evaluation questions.

To prepare a candidate QA manifest after setting up the upstream source and
tokenizer, run from the repository root:

```bash
run_qa() { uv run --locked --no-group cpu --group cuda experiments/run-paper "$@"; }
scripts=scripts/k1_write_capacity
qa=artifacts/qa-manifest-candidate
sources="$qa/sources/pinned-v1"
audit="$qa/audit/pinned-v1"
config=gemma2_qa_protocol
tokenizer_rev=299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8
template_rev=2820a5954f302f885beb2d69e61fdc857c63968310c49bfa1d1fa33dfa602f80

run_qa "$scripts/qa_protocol_prepare_sources.py" --config "$config" --output "$sources"
run_qa "$scripts/qa_protocol_tokenizer_audit.py" --config "$config" \
  --source-dir "$sources" --output "$audit" --tokenizer-revision "$tokenizer_rev"
run_qa "$scripts/qa_protocol_manifest_freeze.py" \
  --source "squad=$sources/squad.jsonl" \
  --source "drop=$sources/drop.jsonl" \
  --source "ropes=$sources/ropes.jsonl" \
  --source-revision squad=7b6d24c440a36b6815f21b70d25016731768db1f \
  --source-revision drop=95cda593fae71b60b5b19f82de3fcf3298c1239c \
  --source-revision ropes=d59f1e2ee2b423d7c6ba71edd47fceb4158b07dd \
  --chunk-audit "$audit/chunk-audit.json" --tokenizer-revision "$tokenizer_rev" \
  --prompt-template-version "$template_rev" --output "$qa/manifest/pinned-v1"
```

The commands write to scratch rather than overwrite the tracked provenance.
This preparation chain is not a validated recreation of the original audit
bytes. Verify the candidate's row order, document and question IDs, and split
against the frozen selections before staging its `manifest.json` at the runner
path below.

**Conditional GPU rerun: cross-model layer replication.** This is not a
clean-checkout command. Before running it, stage:

| Input | Expected location |
| --- | --- |
| Full frozen QA manifest | `experiments/inputs/qa_protocol/manifest/pinned-v1/manifest.json` |
| Reader-training document manifest (used by the Gemma and Qwen runner configs) | `experiments/artifacts/d2l_scaling/gemma2-squad-matched-distill-scale60k-seed141/manifest.json` |
| Pre-head latent shards from the distractor-curve latent stage, for both models | `experiments/artifacts/layer_replication/state/side/latents/{gemma2,qwen4}/` |
| Matched Gemma/SQuAD selection baseline units | Extract `results/composition/reader-controls/reader-selection-units.tar.gz` into `experiments/artifacts/write_site_readers/selection/state/` |
| Matched Gemma/DROP+ROPES baseline units | Extract `results/composition/reader-controls/reader-confirmation-and-transfer-units.tar.gz` into `experiments/artifacts/write_site_readers/confirmation/state/` |

The Qwen baseline is produced by the replication plan itself. The two
manifests and latent shards are not bundled. The latent requirement is for the
probe units; running only the layer interventions does not reproduce the
probe results. For the paper's numbers without these dependencies, use the
`layer-replication` CPU reduction.

Once inputs are staged, list the frozen units and run any unit by ID.
`run-unit.py` runs exactly one unit in the current process and writes a JSON
result under `--state-dir/units/`. It does not choose devices, launch workers,
schedule prerequisites, or retry failures. Use your own shell or scheduler
to run units and select GPUs. The example below lists all units and runs one
Gemma oracle shard; run the remaining required units before reducing a full run.

```bash
uv run --locked experiments/run-paper run-unit.py \
  --plan inputs/layer_interventions/plans/layer_replication-v1.json \
  --state-dir artifacts/layer_replication/state --list
uv run --locked --no-group cpu --group cuda experiments/run-paper run-unit.py \
  --plan inputs/layer_interventions/plans/layer_replication-v1.json \
  --state-dir artifacts/layer_replication/state --runner layer_interventions_runner \
  --unit oracle-selection-gemma2-000
uv run --locked --no-group cpu --group cuda experiments/run-paper scripts/paper_claims/layer_interventions_report.py generality \
  --state artifacts/layer_replication/state \
  --gemma-squad-base-state artifacts/write_site_readers/selection/state \
  --gemma-drop-ropes-base-state artifacts/write_site_readers/confirmation/state \
  --output artifacts/layer_replication/generality-report.json
```

For other frozen plans, choose the runner by experiment:

| Paper item | Frozen plan under `experiments/inputs/` | Runner |
| --- | --- | --- |
| Table 3, mask panel | `paper-runs/answer-position-mask.plan.json` | `answer_position_mask_runner` |
| Table 8 | `passage_fitting/plans/a2-v1.json` | `passage_fitting_format_ceiling_full` |
| Table 9 | `paper-runs/question-conditioned-writes-*.plan.json` | `question_conditioned_writes_runner` |
| Figure 2, Tables 10, 11, 19 | `distractor_curves/plans/full-v2.json` | `distractor_curves_runner` (its `timing` unit is the throughput benchmark) |
| Table 12 | `write_site_readers/plans/{selection,confirmation}-v1.json` | `write_site_readers_runner` |
| Tables 13 and 14 | `gated_readers/{test-confirmation,qwen-gated}-v1.json` | `write_site_readers_runner` (evaluation); training as below |
| Table 4, Table 16, Figures 3 and 4, Table 15, probe-derived gates | `layer_interventions/plans/*.json` | `layer_interventions_runner` |

Follow each plan's
`stages[].after` and `requires` entries. For a result-dependent stage, call
`run-unit.py` with `--expand STAGE` after its prerequisite results are present,
then list or execute the new units. Do not run the same unit into the same
state directory concurrently.

For distributed training, export the selected unit with `--export-unit FILE`
instead of `--runner`. For example, to run a gated-reader training unit on
two selected GPUs:

```bash
uv run --locked --no-group cpu --group cuda experiments/run-paper \
  -m torch.distributed.run --standalone --nproc-per-node=2 --module gated_readers_ddp_train \
  --unit "$UNIT_FILE" --side "$STATE/side" --output "$RAW_RESULT"
```

Call `run-unit.py` for the same plan and unit with
`--import-result RAW_RESULT` to write the reducer-compatible result.

The single-document reader controls (Tables 2 and 6) predate the frozen-plan
runner and are command-line scripts under `experiments/scripts/d2l_readers_k1/`:
`official_d2l_k1_representation_{training,evaluation}.py` for the readout
representations (Table 2), `official_d2l_online_batch_distillation.py` and
`official_d2l_online_reader_evaluation.py` for reader width and layer sharing,
and `official_d2l_{compiler,joint}_finetuning_*.py` for compiler and joint
fine-tuning (Table 6). Each report in
[`results/single-document/reader-controls/`](results/single-document/reader-controls/)
records its contract: the configuration name (in
`experiments/inputs/configs/d2l_scaling/`), the evaluated question IDs, and
the settings it was produced with.

The report scripts accept the local unit state from the corresponding runs.
Results in `results/` can be inspected without that state. Identifiers in the
public evidence are descriptive aliases; document membership, response text,
and scores are unchanged. Sources of earlier runs in the frozen exclusion
and scan files are labeled `earlier-run-NNN`; the documents they excluded are
unchanged.

For the answer-position mask (Table 3), the frozen plan and the recorded
transplant reference it checks against are in
[`experiments/inputs/paper-runs/`](experiments/inputs/paper-runs/). The plan
builder, runner, and reducer are `answer_position_mask_plan.py`,
`answer_position_mask_runner.py`, and `answer_position_mask_report.py` under
`experiments/scripts/k1_write_capacity/`.

## PersonaMem-v2

The public repo verifies Tables 5 and 17 from the released test scores; it
does not rerun the production MidBrain evaluation. Run the `personamem-baseline`
and `personamem` CPU reductions as shown in [Verify the results without GPUs](#verify-the-results-without-gpus). The [score bundles](results/deployment/personamem/)
contain only benchmark question and persona IDs, option labels, and prompt-token
counts for the reported conditions, plus the no-memory token baseline. They
exclude development scores, production prompts, retrieval results, memory
exports, and provider integration code.
