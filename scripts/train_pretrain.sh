#!/bin/bash
# Stage 1: Alignment Pretraining for AnyMAL
# Trains the Perceiver Resampler to align image features with LLM space

set -e

# Configuration
NUM_GPUS=${NUM_GPUS:-8}
CONFIG=${CONFIG:-"configs/pretrain_image.yaml"}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/pretrain"}

# Training hyperparameters (can be overridden)
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-64}
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-4}
MAX_STEPS=${MAX_STEPS:-100000}
LEARNING_RATE=${LEARNING_RATE:-2e-4}

# Logging
USE_WANDB=${USE_WANDB:-false}
WANDB_PROJECT=${WANDB_PROJECT:-"anymal-pretrain"}

echo "=============================================="
echo "AnyMAL Stage 1: Alignment Pretraining"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  GPUs: ${NUM_GPUS}"
echo "  Config file: ${CONFIG}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Batch size per GPU: ${PER_DEVICE_BATCH_SIZE}"
echo "  Gradient accumulation: ${GRADIENT_ACCUMULATION}"
echo "  Effective batch size: $((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION * NUM_GPUS))"
echo "  Max steps: ${MAX_STEPS}"
echo "  Learning rate: ${LEARNING_RATE}"
echo ""

# Build command
CMD="torchrun --nproc_per_node=${NUM_GPUS} scripts/train_pretrain.py \
    --config ${CONFIG} \
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
