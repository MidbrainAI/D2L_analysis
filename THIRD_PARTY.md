# Third-party materials

See [LICENSE](LICENSE) for the repository's licenses and [REUSE.toml](REUSE.toml)
for path-specific notices. No full benchmark distribution, model weights,
training checkpoint, upstream Doc-to-LoRA checkout, or TeX style bundle is
shipped.

## Included benchmark material

| Upstream work and attribution | Source and terms | What is shipped |
| --- | --- | --- |
| **SQuAD**, Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang; source passages from Wikipedia contributors | [Pinned SQuAD dataset and card](https://huggingface.co/datasets/rajpurkar/squad/tree/7b6d24c440a36b6815f21b70d25016731768db1f), [CC BY-SA 4.0](LICENSES/CC-BY-SA-4.0.txt), [paper](https://aclanthology.org/D16-1264/) | Selected passages, titles, questions, and reference answers in frozen QA inputs, reference and attribution files, and some scored results. `experiments/inputs/single_document_writes/rows-v1.json` is a frozen training/evaluation selection containing both benchmark questions and experiment-generated questions, not a raw dataset dump. |
| **DROP**, Dheeru Dua et al.; source passages from Wikipedia contributors | [Pinned DROP dataset and card](https://huggingface.co/datasets/ucinlp/drop/tree/95cda593fae71b60b5b19f82de3fcf3298c1239c), [CC BY-SA 4.0](LICENSES/CC-BY-SA-4.0.txt), [paper](https://aclanthology.org/N19-1246/) | Selected QA rows, references, and identifiers in the read-intervention inputs and evidence and the cross-dataset frozen selections and scores. |
| **ROPES**, Kevin Lin et al. | [Pinned ROPES dataset and card](https://huggingface.co/datasets/allenai/ropes/tree/d59f1e2ee2b423d7c6ba71edd47fceb4158b07dd), [CC BY 4.0](LICENSES/CC-BY-4.0.txt), [paper](https://aclanthology.org/D19-5808/) | Selected QA rows, references, and identifiers in the read-intervention inputs and evidence and the cross-dataset frozen selections and scores. ROPES material retains CC BY 4.0 when collected with ShareAlike material. |
| **PersonaMem-v2**, Bowen Jiang et al. | [PersonaMem-v2 dataset and card](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2), [CC BY 4.0](LICENSES/CC-BY-4.0.txt), [paper](https://arxiv.org/abs/2512.06688). The original dataset revision was not recorded in the public scores. | Test persona/question IDs and option labels, model predictions, prompt-token counts, and aggregate statistics. No benchmark passages, chat histories, prompts, or memory/retrieval exports. |

The QA material is distributed as a **CC BY-SA 4.0 evidence collection**:
frozen inputs under `experiments/inputs/` other than runner configs, and the
QA evidence under `results/single-document/` and `results/composition/`
as marked in `REUSE.toml`. This also covers archives that may contain
generated responses quoting benchmark text, even when an adjacent aggregate
report contains only statistics. Changes include selecting splits and banks,
reformatting rows, assigning experiment aliases, and adding generated
responses, references, and scores. These are not upstream dataset releases.
See the upstream dataset cards for dataset attribution and source provenance;
for Wikipedia text, retain attribution to its contributors. Keep this notice
and the applicable license with redistributed extracts or archives.

## Downloaded separately

- [SakanaAI Doc-to-LoRA code](https://github.com/SakanaAI/doc-to-lora/tree/baa85db4d5df9b29d618af432d6ebf28b3ad5a29)
  (MIT). [Released compiler checkpoints](https://huggingface.co/SakanaAI/doc-to-lora)
  are separate from that code license; follow their release terms.
- [Gemma-2-2B-IT](https://huggingface.co/google/gemma-2-2b-it)
  ([Gemma terms](https://ai.google.dev/gemma/terms)) and
  [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
  (Apache-2.0). The repo includes evidence from model outputs, not model
  files or model derivatives. Dataset-bearing output is covered by the
  evidence notices above.
- [Official ICML 2026 styles](https://icml.cc/Conferences/2026/AuthorInstructions),
  including the conference style and bibliography style and the supplied
  `algorithm`, `algorithmic`, and `fancyhdr` packages. `make -C paper styles`
  fetches the official bundle into an ignored directory. Its files are not
  relicensed by this repo. The paper builds using the saved bibliography.
- Python dependencies are downloaded by `uv` under their own terms; see
  `uv.lock`. The full benchmarks used for GPU reruns are also obtained
  separately. See [Reproduction](REPRODUCING.md).

## Branding

`assets/midbrain-logo.svg` incorporates the wordmark from the
[MidBrain site](https://www.midbrain.ai/logos/midbrain.svg), with an added
background. It is included for attribution, not licensed for reuse under
MIT or Creative Commons. No trademark license is granted.
