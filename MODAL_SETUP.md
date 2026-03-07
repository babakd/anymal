# Training AnyMAL on Modal

Modal is a serverless GPU platform - you only pay for actual compute seconds.

This repo supports two architectures on Modal:

- `anymal_v1`: CLIP + Perceiver Resampler. Stable baseline.
- `anymal_v2`: SigLIP2 + learned token compressor + MLP projector. Alternate path with architecture-specific preprocessing.

## One-Time Setup (5 minutes)

### 1. Create Modal Account
Go to https://modal.com and sign up (GitHub login works)

### 2. Install Modal CLI
```bash
pip install modal
```

### 3. Authenticate
```bash
modal setup
```
This opens a browser to authenticate.

### 4. Add HuggingFace Token
You need a HuggingFace token with access to LLaMA-3:
1. Go to https://huggingface.co/settings/tokens
2. Create a token with "read" access
3. Accept the LLaMA license at https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

Then add it as a Modal secret:
```bash
modal secret create huggingface HF_TOKEN=hf_xxxxxxxxxxxxx
```

### 5. (Optional) Get Weights & Biases API Key
For live training dashboards:
1. Sign up at https://wandb.ai
2. Go to https://wandb.ai/settings → API keys
3. Copy your API key

## Running Training

### Quick Test (100 steps, ~$0.25)
```bash
modal run modal_train.py
```

### With Weights & Biases Dashboard

First, add your W&B API key as a Modal secret (recommended - keeps key out of shell history):
```bash
modal secret create wandb WANDB_API_KEY=YOUR_API_KEY
```

Then run with W&B enabled:
```bash
modal run modal_train.py --use-wandb
```

### Longer Run (500 steps, ~$1)
```bash
modal run modal_train.py --max-steps 500
```

### Stage 1 Pretraining (`anymal_v1`)
```bash
modal run modal_train.py --stage pretrain --max-steps 1000
```

### Stage 1 Pretraining on H100 (`anymal_v1`)
```bash
modal run modal_train.py --stage pretrain --gpu-type h100 --max-steps 1000
```

### Stage 1 Pretraining on H100 (`anymal_v2`)
```bash
modal run modal_train.py --stage pretrain --architecture anymal_v2 --gpu-type h100 --max-steps 1000
```

### Stage 2 Finetune on H100 (`anymal_v1`)
```bash
modal run modal_train.py --stage finetune --gpu-type h100 --max-steps 500
```

### Stage 2 Finetune on H100 (`anymal_v2`)
```bash
modal run modal_train.py --stage finetune --architecture anymal_v2 --gpu-type h100 --max-steps 500
```

### Inference Across Saved Checkpoints
```bash
modal run modal_inference.py --num-examples 20
```

`modal_inference.py` now groups checkpoints by saved architecture metadata and compares only within the same family. A run may therefore emit separate `anymal_v1` and `anymal_v2` progressions in the same output JSON.

### About `--use-dummy-data`
```bash
modal run modal_train.py --use-dummy-data --max-steps 50
```

The flag is accepted for compatibility, but current Modal training paths always load real cached images and print a warning when dummy data is requested.

### All Options
```bash
modal run modal_train.py --help
```

Options:
- `--max-steps`: Number of training steps (default: 100)
- `--stage`: "finetune" or "pretrain" (default: finetune)
- `--architecture`: `anymal_v1` or `anymal_v2` (default: `anymal_v1`)
- `--gpu-type`: GPU family for Modal workers: `a100` or `h100` (default: a100)
- `--learning-rate`: Learning rate (default: 1e-5 for finetune, 2e-4 for pretrain)
- `--lora-learning-rate`: Separate LoRA LR for finetune runs
- `--batch-size`: Per-device batch size (default: 4)
- `--use-wandb`: Enable W&B logging
- `--wandb-api-key`: Your W&B API key (get from wandb.ai/settings)
- `--dataset`: `instruct_150k` or `mix_665k` for finetune
- `--use-dummy-data`: Accepted for compatibility, ignored by real-image Modal pipelines

## Architecture Notes

### `anymal_v1`

- Vision preprocessing uses the CLIP transform path.
- Stage 1 saves `projector.pt`.
- Stage 2 loads the latest matching pretrain checkpoint automatically if available.

### `anymal_v2`

- Vision preprocessing uses the official SigLIP2 image processor at the model’s native size.
- The trainable visual bridge is `token_compressor + projector`.
- Stage 1 and Stage 2 checkpoint metadata records the architecture so inference and loading stay strict.

## Cost Estimates

| Run Type | GPU | Steps | Time | Cost |
|----------|-----|-------|------|------|
| Quick test | A100 | 100 | ~5 min | ~$0.25 |
| Short run | A100 | 500 | ~20 min | ~$1.00 |
| Medium run | A100 | 2000 | ~1.5 hr | ~$4.00 |

Modal bills per second, so you only pay for actual usage.

## Viewing Logs

While training:
```bash
modal app logs anymal-training
```

## Downloading Checkpoints

Checkpoints are saved to a Modal Volume. To download:

```python
# download_checkpoint.py
import modal

volume = modal.Volume.from_name("anymal-checkpoints")

@modal.function(volumes={"/checkpoints": volume})
def list_checkpoints():
    import os
    for root, dirs, files in os.walk("/checkpoints"):
        for f in files:
            print(os.path.join(root, f))

# Run: modal run download_checkpoint.py
```

## Troubleshooting

### "Secret not found: huggingface"
Run: `modal secret create huggingface HF_TOKEN=hf_xxx`

### "Access denied" for LLaMA
Accept the license at https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

### Out of memory
Reduce `--batch-size` to 2 or 1

### Timeout
Increase timeout in `modal_train.py` (line 47) or reduce `--max-steps`

### Wrong architecture loaded
If you switch between `v1` and `v2`, do not reuse checkpoints across architectures. The loaders now validate checkpoint metadata and reject cross-architecture loads.
