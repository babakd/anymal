# AnyMAL: Image Multimodal LLM Experiments

An educational, research-oriented implementation inspired by the AnyMAL paper
([arXiv:2309.16058](https://arxiv.org/abs/2309.16058)).

The paper covers any-modality inputs such as image, video, audio, and IMU. This
repository currently focuses on the image path: turning vision encoder features
into image tokens that an 8B-class language model can consume.

## Current Status

This is an experimental codebase, not a polished model release. It is useful for
learning, reproducing pieces of the AnyMAL-style training pipeline, and running
architecture/backbone ablations. Public pretrained checkpoints are not included.

The canonical current state lives in [docs/STATUS.md](docs/STATUS.md). As of
2026-05-12, the best candidate is the V9 Qwen/Qwen3-8B path using the
`anymal_v3` 128-token connector interface with a materialized connector output
scale of `1.05`.

For the latest experiment index, read
[experiments/LATEST.md](experiments/LATEST.md). The short version:

| Campaign | Status |
| --- | --- |
| V9 Qwen scale-1.05 | Current viable replacement candidate; all recorded gates passed |
| V8 Qwen decoder swap | Historical integration and compute-matched Qwen groundwork |
| V7/V6 controls | Historical LLaMA-side controls and causal falsification context |
| V4/V5 spatial/recipe work | Historical; useful provenance, not the current default |

Implemented model variants:

| Variant | Vision encoder | Connector | Notes |
| --- | --- | --- | --- |
| `anymal_v1` | CLIP ViT-L/14 at 224 px | Perceiver resampler | Original baseline path |
| `anymal_v2` | SigLIP2 at 384 px | MLP bottleneck plus token compressor | Adds stricter image-token handling |
| `anymal_v3` | SigLIP2 at 384 px | Perceiver-style connector | Current best interface for the Qwen candidate |
| `anymal_v4` | SigLIP2 at 384 px | Spatial global/local Perceiver | Historical spatial-connector branch |

Older V1/V2/V4 configs remain in the repo for comparison and provenance. Check
`docs/STATUS.md` before launching new expensive work.

## Architecture

At a high level:

```text
image -> frozen vision encoder -> trainable connector -> image placeholder tokens -> decoder -> text
```

The current V9 candidate uses:

- LLM: `Qwen/Qwen3-8B`
- Vision encoder: `google/siglip2-so400m-patch14-384`
- Connector: V3 Perceiver-style connector with 128 image tokens
- Training: Stage 1/1B connector alignment; the promoted V9 checkpoint is an
  eval/inference checkpoint, not an optimizer-resume checkpoint

## Installation

```bash
git clone https://github.com/babakd/anymal.git
cd anymal

python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Optional CUDA performance package:

```bash
python3 -m pip install flash-attn --no-build-isolation
```

For Modal-based remote training, also install Modal locally:

```bash
python3 -m pip install modal
modal setup
```

## Model Access

The Llama weights require a Hugging Face account with access to
`meta-llama/Meta-Llama-3-8B-Instruct`.

```bash
huggingface-cli login
python3 scripts/download_checkpoints.py --llama
```

The v1 CLIP path can pre-cache CLIP weights:

```bash
python3 scripts/download_checkpoints.py --clip
```

SigLIP2 weights used by v2-v4 and the V9 Qwen candidate are downloaded by
Transformers/Hugging Face when the model initializes.

## Data

Small local smoke data:

```bash
python3 scripts/create_dummy_data.py
```

Public data helpers:

```bash
python3 scripts/download_data.py --llava
python3 scripts/download_data.py --coco
python3 scripts/download_data.py --laion --samples 1000000
```

Notes:

- `--llava` downloads LLaVA-Instruct-150K annotations. You still need the
  referenced COCO images.
- `--laion` stores LAION metadata and captions; image download is a separate
  step.
- Some v3/v4 calibration configs expect locally prepared VQA/COCO-derived JSON
  files. Those generated experiment artifacts are not bundled as a public
  dataset release.

## Local Training

### v1 Baseline

Stage 1 alignment:

```bash
torchrun --nproc_per_node=8 scripts/train_pretrain.py \
  --config configs/pretrain_image.yaml
```

Stage 2 instruction fine-tuning:

```bash
torchrun --nproc_per_node=8 scripts/train_finetune.py \
  --config configs/finetune.yaml \
  --pretrain_checkpoint ./outputs/pretrain/checkpoint-100000
```

Single-process debug runs are possible after creating dummy data, but they still
initialize the language and vision models:

```bash
python3 scripts/train_pretrain.py \
  --config configs/pretrain_image.yaml \
  --train_data_path ./data/laion_subset \
  --debug \
  --no_flash_attention

python3 scripts/train_finetune.py \
  --config configs/finetune.yaml \
  --train_data_path ./data/llava_instruct_sample.json \
  --image_dir ./data/dummy_images \
  --debug
```

### Current Qwen Candidate

The promoted V9 checkpoint and full result ledger are documented in
[experiments/v9_qwen/results.md](experiments/v9_qwen/results.md). The historical
V8/Qwen setup work is under [experiments/v8_qwen/](experiments/v8_qwen/).

For new long runs, start from [docs/STATUS.md](docs/STATUS.md) and verify the
checkpoint/backbone metadata before spending GPU time.

### v4 Historical Path

Stage 1 caption alignment:

```bash
torchrun --nproc_per_node=8 scripts/train_pretrain.py \
  --config configs/pretrain_v4_alignment.yaml
```

Optional Stage 1B grounding alignment:

```bash
torchrun --nproc_per_node=8 scripts/train_pretrain.py \
  --config configs/pretrain_v4_grounding.yaml
```

Stage 2 semantic calibration:

```bash
torchrun --nproc_per_node=8 scripts/train_finetune.py \
  --config configs/finetune_v4_semantic_calibration.yaml
```

Check the config files before launching a long run. They encode dataset paths,
batch sizes, token counts, connector shapes, and checkpoint locations.

## Modal Training

`modal_train.py` contains the maintained remote-training entrypoint. After Modal
setup and a Hugging Face secret:

```bash
modal secret create huggingface HF_TOKEN=hf_xxx
```

Example commands:

```bash
modal run modal_train.py --use-dummy-data --max-steps 50
modal run modal_train.py \
  --stage pretrain \
  --architecture anymal_v3 \
  --llm-backbone Qwen/Qwen3-8B \
  --gpu-type h100 \
  --max-steps 20
modal run modal_train.py \
  --stage finetune \
  --architecture anymal_v3 \
  --llm-backbone Qwen/Qwen3-8B \
  --dataset v9_qwen_controlaware_stage2 \
  --max-steps 100
modal run --detach modal_train.py \
  --stage finetune \
  --architecture anymal_v3 \
  --llm-backbone Qwen/Qwen3-8B \
  --dataset v9_qwen_controlaware_stage2 \
  --max-steps 800
```

For a fuller walkthrough, see [docs/MODAL_SETUP.md](docs/MODAL_SETUP.md).

## Evaluation

The repo includes VQA and captioning evaluation utilities. The Modal VQA
checkpoint evaluator is the most used path for comparing checkpoints:

```bash
modal run vqa_checkpoint_eval.py \
  --candidate-checkpoint /checkpoints/finetune-output/run-0001/checkpoint-100 \
  --candidate-architecture v4 \
  --max-samples 1000 \
  --output results/vqa_checkpoint_eval.json
```

Eval JSON artifacts live under `results/` only when they are curated and small
enough to track. New large prediction/eval dumps should stay in Modal volumes,
W&B, external storage, or ignored `outputs/`; do not put new artifacts in the
repo root.

There are also local analysis helpers under `scripts/` for inspecting prediction
JSON files and promotion criteria.

## Project Structure

```text
configs/       Training configs for v1-v4 experiments
data/          Dataset loaders and collators
docs/          Current status, runbooks, and historical handoffs
evaluation/    Captioning and VQA evaluation code
experiments/   Versioned experiment plans and result ledgers
models/        AnyMAL model variants, encoders, connectors, LLM wrappers
notebooks/     Educational architecture notebook
results/       Small tracked eval artifacts
scripts/       Local training, data download, and analysis entrypoints
tests/         Unit tests for model, training, evaluation, and monitoring code
training/      Trainers, distributed helpers, health/throughput monitoring
```

Root-level Python files are intentionally limited to compatibility wrappers and
small shared modules. Bulky script implementations live under `scripts/`.

## Requirements

- Python 3.10 or newer recommended
- PyTorch 2.0 or newer
- CUDA GPU for practical training
- Hugging Face access to Llama 3 for LLM-backed runs
- Large disk budget for COCO/LLaVA/LAION-style data and checkpoints

The default research configs target A100/H100-class GPUs. Smaller GPUs require
reducing batch size, sequence length, image tokens, or using Modal.

## Development

```bash
python3 -m pytest tests -q
python3 -m compileall -q models training evaluation data scripts
python3 scripts/repo_health_check.py
modal run scripts/modal_repo_smoke.py
black .
isort .
```

See [AGENTS.md](AGENTS.md) for agent-oriented operating rules and
[CONTRIBUTING.md](CONTRIBUTING.md) for contribution workflow details. If you
remember the old root-heavy layout, see
[docs/REPO_STRUCTURE_MIGRATION.md](docs/REPO_STRUCTURE_MIGRATION.md).

## References

- [AnyMAL paper](https://arxiv.org/abs/2309.16058)
- [Flamingo paper](https://arxiv.org/abs/2204.14198), for the Perceiver
  resampler idea
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [QLoRA paper](https://arxiv.org/abs/2305.14314)

## License

The code in this repository is licensed under the Apache License 2.0. See
[LICENSE](LICENSE).

Users are responsible for complying with third-party model and dataset licenses,
including the Meta Llama 3 license, CLIP/OpenCLIP terms, SigLIP2/Hugging Face
model terms, LAION dataset terms, LLaVA dataset terms, and COCO dataset terms.
