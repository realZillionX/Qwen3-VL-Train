#!/bin/bash
# SFT Training Script for Qwen3-VL using ms-swift CLI
# Usage: bash train_sft.sh
# Make sure to modify MODEL_PATH before running.

# Offline mode
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ========== USER CONFIG ==========
MODEL_PATH="/path/to/Qwen3-VL-32B-Thinking"  # <-- CHANGE THIS
DATASET="train_sft.jsonl"
OUTPUT_DIR="output/sft_qwen3_vl"
# ==================================

# Run swift sft command
swift sft \
    --model "$MODEL_PATH" \
    --dataset "$DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_steps 100 \
    --logging_steps 10 \
    --max_length 2048 \
    --bf16 true \
    --sft_type lora \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --lora_rank 16 \
    --lora_alpha 32 \
    --report_to tensorboard

echo "SFT Training finished. Output saved to $OUTPUT_DIR"
