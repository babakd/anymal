#!/bin/bash
# Stage 2: Instruction Fine-tuning for AnyMAL
# Fine-tunes the model on instruction-following data with LoRA

set -e

# Configuration
NUM_GPUS=${NUM_GPUS:-8}
CONFIG=${CONFIG:-"configs/finetune.yaml"}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/finetune"}

# Stage 1 checkpoint (required)
PRETRAIN_CHECKPOINT=${PRETRAIN_CHECKPOINT:-"./outputs/pretrain/checkpoint-100000"}

# Training hyperparameters (can be overridden)
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-16}
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-2}
MAX_STEPS=${MAX_STEPS:-3000}
LEARNING_RATE=${LEARNING_RATE:-1e-5}

# Logging
USE_WANDB=${USE_WANDB:-false}
WANDB_PROJECT=${WANDB_PROJECT:-"anymal-finetune"}

echo "=============================================="
echo "AnyMAL Stage 2: Instruction Fine-tuning"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  GPUs: ${NUM_GPUS}"
echo "  Config file: ${CONFIG}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Stage 1 checkpoint: ${PRETRAIN_CHECKPOINT}"
echo "  Batch size per GPU: ${PER_DEVICE_BATCH_SIZE}"
echo "  Gradient accumulation: ${GRADIENT_ACCUMULATION}"
echo "  Effective batch size: $((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION * NUM_GPUS))"
echo "  Max steps: ${MAX_STEPS}"
echo "  Learning rate: ${LEARNING_RATE}"
echo ""

# Check if Stage 1 checkpoint exists
if [ ! -d "${PRETRAIN_CHECKPOINT}" ]; then
    echo "ERROR: Stage 1 checkpoint not found at ${PRETRAIN_CHECKPOINT}"
    echo "Please run Stage 1 pretraining first or provide correct path."
    exit 1
fi

# Build command
CMD="torchrun --nproc_per_node=${NUM_GPUS} scripts/train_finetune.py \
    --config ${CONFIG} \
    --pretrain_checkpoint ${PRETRAIN_CHECKPOINT} \
    --per_device_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
    --max_steps ${MAX_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --output_dir ${OUTPUT_DIR}"

# Add wandb if enabled
if [ "${USE_WANDB}" = "true" ]; then
    CMD="${CMD} --use_wandb --wandb_project ${WANDB_PROJECT}"
fi

echo "Running command:"
echo "${CMD}"
echo ""

# Run training
eval ${CMD}

echo ""
echo "=============================================="
echo "Training completed!"
echo "Checkpoints saved to: ${OUTPUT_DIR}"
echo "=============================================="
