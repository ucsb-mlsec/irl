#!/bin/bash

# Simple SFT Training Script
# This script runs supervised fine-tuning on the IRL dataset

set -e

echo "Starting IRL SFT Training..."

# Default parameters
MODEL_NAME=${1:-"Qwen/Qwen2.5-0.5B-Instruct"}
OUTPUT_DIR=${2:-"/tmp/irl_sft_output"}
BATCH_SIZE=${3:-4}
NUM_EPOCHS=${4:-3}
LEARNING_RATE=${5:-5e-5}

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Batch Size: $BATCH_SIZE"
echo "  Number of Epochs: $NUM_EPOCHS"
echo "  Learning Rate: $LEARNING_RATE"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Set environment variables
export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH="/home/henrygwb/irl:$PYTHONPATH"

# Change to project directory
cd /home/henrygwb/irl

# Check if data files exist
TRAIN_FILE="/home/henrygwb/irl/data/prime_expert_demo.parquet"
VAL_FILE="/home/henrygwb/irl/data/validation.parquet"

if [ ! -f "$TRAIN_FILE" ]; then
    echo "Warning: Training file $TRAIN_FILE not found"
    echo "Available data files:"
    ls -la /home/henrygwb/irl/data/*.parquet 2>/dev/null || echo "No parquet files found in data directory"
    
    # Use alternative data file if available
    ALT_TRAIN_FILE="/home/henrygwb/irl/data/prime_train.parquet"
    if [ -f "$ALT_TRAIN_FILE" ]; then
        echo "Using alternative training file: $ALT_TRAIN_FILE"
        TRAIN_FILE="$ALT_TRAIN_FILE"
    else
        echo "Error: No suitable training file found"
        exit 1
    fi
fi

if [ ! -f "$VAL_FILE" ]; then
    echo "Warning: Validation file $VAL_FILE not found, proceeding without validation"
    VAL_ARGS=""
else
    VAL_ARGS="--val_files $VAL_FILE"
fi

echo "Using training file: $TRAIN_FILE"

# Run the training
python recipe/sft/simple_sft_trainer.py \
    --model_name "$MODEL_NAME" \
    --train_files "$TRAIN_FILE" \
    $VAL_ARGS \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --max_length 1024

echo "SFT Training completed!"
echo "Models saved to: $OUTPUT_DIR"
