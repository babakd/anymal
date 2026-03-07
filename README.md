# AnyMAL: Multimodal LLM Replication

Educational image-first replication of the AnyMAL paper ([arXiv:2309.16058](https://arxiv.org/abs/2309.16058)) using LLaMA-3-8B as the language model.

This repo currently implements the image path only. Video, audio, and IMU are out of scope here.

## What Is In This Repo

Two model families are implemented and intentionally kept separate:

| Architecture | Status | Vision stack | Visual bridge | Preprocessing |
|-----------|-----------|-----------|-----------|-----------|
| `anymal_v1` | Stable default | CLIP ViT-L/14 | Perceiver Resampler | CLIP transform (`224` or `336`) |
| `anymal_v2` | Alternate architecture | SigLIP2 So400m | Learned token compressor + MLP bottleneck projector | Official SigLIP2 processor (`384` in current configs) |

`v1` is the baseline the repo grew around. `v2` is now wired through training, evaluation, checkpoint metadata, and Modal inference without reusing `v1` assumptions.

## Architecture Overview

### `anymal_v1`

```text
Image -> CLIP ViT-L/14 -> [257, 1024] -> Perceiver Resampler -> [64, 4096] -> LLaMA-3
```

- Frozen vision encoder
- Frozen LLM base
- Stage 1 trains the projector only
- Stage 2 trains projector + LoRA adapters
- Supports placeholder splice or prepend fallback

### `anymal_v2`

```text
Image -> SigLIP2 -> visual tokens -> learned token compressor -> MLP projector -> LLaMA-3
```

- Frozen SigLIP2 encoder
- Frozen LLM base
- Stage 1 trains token compressor + projector
- Stage 2 trains token compressor + projector + LoRA adapters
- Requires explicit image placeholder tokens for strict splice behavior

## Why The Split Matters

The code paths for `v1` and `v2` now differ at the three places that were previously easiest to accidentally couple:

- Preprocessing is architecture-driven instead of hard-coded to CLIP.
- Trainer warmup and optimizer grouping operate on model-declared visual bridge modules.
- Evaluation and Modal inference load checkpoints by architecture metadata and keep `v1`/`v2` runs separate.

This keeps the repo educational: `v1` remains a compact reference implementation, while `v2` shows how to evolve the design without hiding differences behind conditionals scattered across the codebase.

## Installation

```bash
git clone https://github.com/babakd/anymal.git
cd anymal

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Optional
pip install flash-attn --no-build-isolation
```

## Quick Start

### 1. Download weights

```bash
# Vision backbones are pulled from their respective libraries / Hugging Face caches.

# LLaMA requires Meta license acceptance on Hugging Face.
python scripts/download_checkpoints.py --llama
```

### 2. Download data

```bash
python scripts/download_data.py --sample
python scripts/download_data.py --laion --samples 1000000
python scripts/download_data.py --llava
python scripts/download_data.py --coco
```

### 3. Train locally

`v1`:

```bash
torchrun --nproc_per_node=8 scripts/train_pretrain.py \
  --config configs/pretrain_image.yaml

torchrun --nproc_per_node=8 scripts/train_finetune.py \
  --config configs/finetune.yaml \
  --pretrain_checkpoint ./outputs/pretrain/checkpoint-100000
```

`v2`:

```bash
torchrun --nproc_per_node=8 scripts/train_pretrain.py \
  --config configs/pretrain_v2_alignment.yaml

torchrun --nproc_per_node=8 scripts/train_finetune.py \
  --config configs/finetune_v2.yaml \
  --pretrain_checkpoint ./outputs/pretrain_v2/checkpoint-100000
```

For Modal workflows, see [MODAL_SETUP.md](MODAL_SETUP.md).

## Preprocessing Contracts

The model defines the preprocessing family it expects:

- `anymal_v1` uses CLIP-style resize/crop/normalize.
- `anymal_v2` uses the Hugging Face SigLIP2 image processor at the encoder’s native size.

Do not reuse `v1` transforms for `v2`. The training, eval, and inference entry points now derive transforms from the instantiated model.

## Project Structure

```text
anymal/
├── configs/
│   ├── pretrain_image.yaml
│   ├── finetune.yaml
│   ├── pretrain_v2_alignment.yaml
│   └── finetune_v2.yaml
├── data/
│   ├── data_utils.py
│   ├── instruction_dataset.py
│   ├── laion_dataset.py
│   └── multimodal_inputs.py
├── models/
│   ├── anymal.py
│   ├── anymal_v2.py
│   ├── factory.py
│   ├── encoders/
│   ├── llm/
│   └── projectors/
├── evaluation/
├── training/
├── scripts/
├── modal_train.py
├── modal_inference.py
└── notebooks/
```

## Training Notes

### Stage 1

- `v1`: trains the Perceiver Resampler only.
- `v2`: trains the token compressor and projector only.
- Modal Stage 1 currently uses COCO-backed caption data extracted from cached LLaVA assets.

### Stage 2

- `v1`: trains projector + LoRA.
- `v2`: trains token compressor + projector + LoRA.
- Warmup is defined over the full visual bridge, not only a module named `projector`.

### Evaluation and Inference

- Shared prompt helpers now build placeholder-aware multimodal prompts.
- `modal_inference.py` reads checkpoint metadata, loads the matching architecture, and compares checkpoints only within the same architecture family.

## Educational Notes

### Why keep `v1`?

- It is the shortest path from paper idea to working code.
- The Perceiver bridge makes the modality translation problem explicit.
- It is still the easiest place to understand placeholder splice vs prepend behavior.

### Why add `v2`?

- Modern open multimodal models usually pair a stronger vision encoder with a lighter connector.
- The `v2` bridge is much smaller and cheaper to tune than the `v1` Perceiver.
- The repo now shows two different multimodal design choices without collapsing them into one ambiguous implementation.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Enough GPU memory for LLaMA-3-8B + vision encoder + connector
- Significant local disk or Modal volume storage for datasets and checkpoints

## References

- [AnyMAL Paper](https://arxiv.org/abs/2309.16058)
- [Flamingo Paper](https://arxiv.org/abs/2204.14198) (Perceiver Resampler)
- [LLaVA](https://github.com/haotian-liu/LLaVA) (Reference implementation)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)

## License

The code in this repository is licensed under the Apache License 2.0 - see [LICENSE](LICENSE) for details.

**Important:** This project depends on components with their own licenses. Users must comply with all applicable licenses, particularly:

- **Meta Llama 3**: Using the LLM weights requires acceptance of the [Llama 3 Community License Agreement](https://llama.meta.com/llama3/license/), which has restrictions on commercial use and other terms not covered by Apache 2.0.

### Third-Party Licenses & Attributions

This project uses the following resources:

- **LLaMA-3**: [Meta Llama 3 Community License](https://llama.meta.com/llama3/license/)
- **CLIP**: MIT License - OpenAI
- **LAION**: CC-BY-4.0
- **LLaVA-Instruct**: Apache 2.0

#### COCO Dataset Attribution

This project uses images from the [COCO (Common Objects in Context)](https://cocodataset.org/) dataset.

> COCO is a large-scale object detection, segmentation, and captioning dataset.
> - Created by Microsoft COCO team
> - Lin, T.Y., et al. (2014). Microsoft COCO: Common Objects in Context. In: Fleet, D., Pajdla, T., Schiele, B., Tuytelaars, T. (eds) ECCV 2014. [arXiv:1405.0312](https://arxiv.org/abs/1405.0312)

The COCO dataset is licensed under [Creative Commons Attribution 4.0 License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
