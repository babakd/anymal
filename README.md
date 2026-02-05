# AnyMAL: Multimodal LLM Replication

An educational implementation of the AnyMAL paper (arXiv:2309.16058), focusing on the **image** modality using LLaMA-3-8B and CLIP.

Note: The paper covers *any-modality* inputs (image/video/audio/IMU). This repo currently implements the image path end-to-end; other modalities are out of scope here.

## Overview

This project replicates the AnyMAL architecture with a focus on understanding how to convert visual inputs into tokens that a language model can understand.

**Key Components:**
- **Vision Encoder**: CLIP ViT-L/14 (frozen)
- **Projector**: Perceiver Resampler with cross-attention
- **LLM**: LLaMA-3-8B-Instruct with QLoRA

## Architecture

```
Image (224×224) → CLIP ViT → [257, 1024] → Perceiver Resampler → [64, 4096] → LLaMA-3 → Text
```

The key insight is that we only need to train the projection layer to bridge the modality gap. Both the vision encoder and LLM can remain frozen.

## Installation

```bash
# Clone the repository
git clone https://github.com/babakd/anymal.git
cd anymal

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install flash-attention (optional but recommended)
pip install flash-attn --no-build-isolation
```

## Quick Start

### 1. Download Checkpoints

```bash
# Download CLIP (automatically cached)
python scripts/download_checkpoints.py --clip

# Download LLaMA (requires Meta approval via HuggingFace)
# First: Accept license at https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
# Then: huggingface-cli login
python scripts/download_checkpoints.py --llama
```

### 2. Download Data

```bash
# Create a small sample dataset for testing
python scripts/download_data.py --sample

# For real training, download LAION and LLaVA-Instruct
python scripts/download_data.py --laion --samples 1000000
python scripts/download_data.py --llava
python scripts/download_data.py --coco
```

### 3. Training

**Stage 1: Alignment Pretraining**
```bash
# Train the Perceiver Resampler on image-caption pairs
torchrun --nproc_per_node=8 scripts/train_pretrain.py \
    --config configs/pretrain_image.yaml
```

**Stage 2: Instruction Fine-tuning**
```bash
# Fine-tune with LoRA on instruction data
torchrun --nproc_per_node=8 scripts/train_finetune.py \
    --config configs/finetune.yaml \
    --pretrain_checkpoint ./outputs/pretrain/checkpoint-100000
```

## Project Structure

```
anymal/
├── configs/                    # Training configurations
│   ├── base.yaml              # Base settings
│   ├── pretrain_image.yaml    # Stage 1 config
│   └── finetune.yaml          # Stage 2 config
├── data/                       # Data loading
│   ├── laion_dataset.py       # LAION for pretraining
│   ├── instruction_dataset.py # LLaVA for fine-tuning
│   └── data_utils.py          # Utilities
├── models/                     # Model components
│   ├── anymal.py              # Main model
│   ├── encoders/              # Vision encoders
│   ├── projectors/            # Modality projectors
│   └── llm/                   # LLM wrappers
├── training/                   # Training loops
│   ├── pretrain.py            # Stage 1 trainer
│   ├── finetune.py            # Stage 2 trainer
│   └── distributed.py         # Multi-GPU utilities
├── evaluation/                 # Benchmarks
├── scripts/                    # Entry points
└── notebooks/                  # Educational notebooks
```

## Training Details

### Stage 1: Alignment Pretraining

- **Goal**: Teach projector to convert CLIP features to LLM-compatible tokens
- **Data**: LAION image-caption pairs (~10-200M)
- **Trainable**: Only Perceiver Resampler
- **Hyperparameters**:
  - Batch size: 2048 (64 per GPU × 8 GPUs × 4 accumulation)
  - Learning rate: 2e-4
  - Steps: 100K
  - Time: ~1.5-2 days on 8× A100

### Stage 2: Instruction Fine-tuning

- **Goal**: Learn to follow multimodal instructions
- **Data**: LLaVA-Instruct-150K
- **Trainable**: Projector + LoRA adapters
- **Hyperparameters**:
  - Batch size: 256
  - Learning rate: 1e-5
  - Steps: 3K
  - LoRA rank: 64
  - Time: ~2-3 hours on 8× A100

## Educational Resources

Check out the Jupyter notebook in `notebooks/` for a deep dive into the architecture:

- **01_understanding_architecture.ipynb**: Full architecture walkthrough

## Key Concepts

### Why Freeze the Encoders?

- CLIP has learned excellent visual representations from 400M image-text pairs
- LLaMA has strong language understanding from 15T tokens
- We just need to learn the "translation" between them

### Why Perceiver Resampler?

- Compresses 257 tokens to 64 tokens (4× reduction)
- Cross-attention learns task-relevant compression
- Fixed output size regardless of input resolution

### Why QLoRA?

- 8B model needs ~32GB in fp32, ~16GB in fp16
- 4-bit quantization reduces to ~4GB
- LoRA adds only ~0.1% trainable parameters
- Enables training on consumer GPUs

## Model Specifications

| Component | Specification |
|-----------|--------------|
| LLM | LLaMA-3-8B-Instruct |
| Hidden size | 4,096 |
| Vision encoder | CLIP ViT-L/14 |
| Vision dim | 1,024 |
| Image tokens | 64 |
| Projector layers | 6 |
| LoRA rank | 64 |

## Requirements

- Python 3.10+
- PyTorch 2.0+
- 8× A100 80GB (recommended) or 1× A100 (reduced batch size)
- ~500GB disk for data and checkpoints

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
