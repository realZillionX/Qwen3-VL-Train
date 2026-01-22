#!/bin/bash
# SFT Training Script for Qwen3-VL using ms-swift CLI
# Full Fine-Tuning with DeepSpeed ZeRO-3 (8x H200)
# Usage: bash train_sft.sh

# Offline mode
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ========== USER CONFIG ==========
MODEL_PATH="/path/to/Qwen3-VL-32B-Thinking"  # <-- CHANGE THIS
DATASET="train_sft.jsonl"
OUTPUT_DIR="output/sft_qwen3_vl"
NUM_GPUS=8
# ==================================

# DeepSpeed ZeRO-3 Config (inline JSON or use a file)
# ms-swift supports --deepspeed with built-in configs like "zero3" or custom json path
DEEPSPEED_CONFIG="zero3"

# Run distributed training with DeepSpeed
# Using torchrun for multi-GPU
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model "$MODEL_PATH" \
    --dataset "$DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --save_steps 100 \
    --logging_steps 10 \
    --max_length 2048 \
    --bf16 true \
    --sft_type full \
    --report_to tensorboard

echo "SFT Training finished. Output saved to $OUTPUT_DIR"
