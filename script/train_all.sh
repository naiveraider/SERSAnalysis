#!/bin/bash
# Training script for All Tasks
# Trains CNN, TCN, and CNN+Transformer models for all tasks (1-4)

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=========================================="
echo "Training All Tasks - All Models"
echo "=========================================="
echo ""

# Default parameters
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-32}
LEARNING_RATE=${LEARNING_RATE:-0.001}
VALIDATION_SPLIT=${VALIDATION_SPLIT:-0.2}

# Train Task 1
echo ""
echo "=========================================="
echo "Starting Task 1"
echo "=========================================="
bash script/train_task1.sh

# Train Task 2
echo ""
echo "=========================================="
echo "Starting Task 2"
echo "=========================================="
bash script/train_task2.sh

# Train Task 3
echo ""
echo "=========================================="
echo "Starting Task 3"
echo "=========================================="
bash script/train_task3.sh

# Train Task 4
echo ""
echo "=========================================="
echo "Starting Task 4"
echo "=========================================="
bash script/train_task4.sh

echo ""
echo "=========================================="
echo "All Tasks Training Complete!"
echo "=========================================="

