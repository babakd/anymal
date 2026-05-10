# Training AnyMAL on Modal

Modal is the maintained path for remote AnyMAL experiments when local GPUs are
not enough. The entrypoint is [modal_train.py](modal_train.py).

## One-Time Setup

### 1. Install and authenticate

```bash
python3 -m pip install modal
modal setup
```

### 2. Add a Hugging Face token

You need a Hugging Face token with access to
`meta-llama/Meta-Llama-3-8B-Instruct`.

1. Accept the model terms on Hugging Face.
2. Create a read token at <https://huggingface.co/settings/tokens>.
3. Store it as a Modal secret:

```bash
modal secret create huggingface HF_TOKEN=hf_xxx
```

### 3. Optional: add Weights & Biases

```bash
modal secret create wandb WANDB_API_KEY=your_wandb_key
```

You can also pass `--wandb-api-key`, but a Modal secret keeps the key out of
shell history.

## Common Runs

Fast pipeline check with synthetic data:

```bash
modal run modal_train.py --use-dummy-data --max-steps 50
```

Baseline v1 Stage 1 pretraining:

```bash
modal run modal_train.py \
  --stage pretrain \
  --architecture anymal_v1 \
  --max-steps 1000
```

Current v4 Stage 1 alignment on H100:

```bash
modal run modal_train.py \
  --stage pretrain \
  --architecture anymal_v4 \
  --gpu-type h100 \
  --max-steps 20000
```

Current v4 Stage 2 semantic calibration:

```bash
modal run modal_train.py \
  --stage finetune \
  --architecture anymal_v4 \
  --dataset v4_semantic_calibration \
  --max-steps 100
```

Longer runs should usually be detached:

```bash
modal run --detach modal_train.py \
  --stage finetune \
  --architecture anymal_v4 \
  --dataset v4_semantic_calibration \
  --max-steps 3000
```

Resume from a checkpoint:

```bash
modal run modal_train.py \
  --stage pretrain \
  --architecture anymal_v4 \
  --resume-checkpoint /checkpoints/pretrain-output/run-0001/checkpoint-250
```

## Useful Options

Run `modal run modal_train.py --help` for the full generated option list.

Common flags:

- `--stage`: `pretrain` or `finetune`
- `--architecture`: `anymal_v1`, `anymal_v2`, `anymal_v3`, or `anymal_v4`
- `--gpu-type`: `a100` or `h100`
- `--max-steps`: training steps for the run
- `--batch-size`: per-device batch size
- `--learning-rate`: main optimizer learning rate
- `--lora-learning-rate`: optional separate LoRA learning rate
- `--dataset`: Modal dataset preset, such as `instruct_150k`,
  `v3_grounded`, or `v4_semantic_calibration`
- `--use-dummy-data`: use generated synthetic data for smoke tests
- `--use-wandb`: enable Weights & Biases logging
- `--run-name`: choose the run directory name under the Modal volume
- `--resume-checkpoint`: continue from an existing checkpoint directory
- `--pretrain-checkpoint`: load a Stage 1 checkpoint for Stage 2

v4-specific shape/ablation flags include:

- `--pretrain-image-tokens`
- `--v4-global-image-tokens`
- `--v4-local-image-tokens`
- `--v4-connector-layers`
- `--v4-connector-heads`
- `--v4-connector-ff-mult`
- `--v4-connector-hidden-dim`
- `--v4-connector-type`
- `--v4-use-2d-position-features`

## Logs and Outputs

Training checkpoints are written to the `anymal-checkpoints` Modal volume.

View logs:

```bash
modal app logs anymal-training
```

List files in the checkpoint volume with a small helper:

```python
import os
import modal

app = modal.App("anymal-checkpoint-tools")
volume = modal.Volume.from_name("anymal-checkpoints")

@app.function(volumes={"/checkpoints": volume})
def list_checkpoints():
    for root, _, files in os.walk("/checkpoints"):
        for name in files:
            print(os.path.join(root, name))

@app.local_entrypoint()
def main():
    list_checkpoints.remote()
```

Run it with:

```bash
modal run list_checkpoints.py
```

## Evaluation

VQA checkpoint evaluation also runs on Modal:

```bash
modal run vqa_checkpoint_eval.py \
  --candidate-checkpoint /checkpoints/finetune-output/run-0001/checkpoint-100 \
  --candidate-architecture v4 \
  --max-samples 1000 \
  --output vqa_checkpoint_eval.json
```

## Cost Notes

Modal pricing changes over time, and long multimodal runs can use expensive GPU
classes. Check the current Modal pricing page before launching a long A100/H100
run, start with `--use-dummy-data` or a low `--max-steps`, and prefer
`modal run --detach` for jobs you expect to outlive your terminal session.

## Troubleshooting

### Secret not found: huggingface

Create the secret:

```bash
modal secret create huggingface HF_TOKEN=hf_xxx
```

### Access denied for Llama

Confirm that the Hugging Face account attached to `HF_TOKEN` has accepted access
for `meta-llama/Meta-Llama-3-8B-Instruct`.

### Out of memory

Reduce `--batch-size`, reduce image-token counts, or use `--gpu-type h100`.

### Checkpoint shape mismatch

Make sure the checkpoint architecture and v4 connector settings match the run.
Recent checkpoints include `model_meta.json`; the loaders use it to reject
incompatible architecture or token-shape combinations.

### Detached run did not keep going

Use Modal's detached form:

```bash
modal run --detach modal_train.py ...
```

The `--background` flag exists for older workflows, but `modal run --detach` is
the preferred durable path.
