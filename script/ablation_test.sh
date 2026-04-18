#!/bin/bash
# Run specified ablation MeanMax models N times each and average FINAL_TEST_ACC.
#
# Usage:
#   bash script/ablation_test.sh lstm_meanmax cnn_meanmax
#   N_RUNS=5 EPOCHS=100 bash script/ablation_test.sh lstm_meanmax vit_meanmax

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

if [ "$#" -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  echo "Usage: bash script/ablation_test.sh <model1> [model2 ...]"
  echo ""
  echo "Example:"
  echo "  bash script/ablation_test.sh lstm_meanmax cnn_meanmax"
  echo "  N_RUNS=5 EPOCHS=100 bash script/ablation_test.sh lstm_meanmax vit_meanmax tcn_meanmax"
  echo ""
  echo "Supported models:"
  echo "  cnn_meanmax, tcn_meanmax, vit_meanmax, inceptiontime_meanmax,"
  echo "  lstm_meanmax, cnn_lstm_meanmax, cnn_transformer_meanmax, mamba_meanmax"
  exit 0
fi

EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-32}
LEARNING_RATE=${LEARNING_RATE:-0.001}
VALIDATION_SPLIT=${VALIDATION_SPLIT:-0.2}
N_RUNS=${N_RUNS:-10}
DEVICE=${DEVICE:-}

# Build device argument
DEVICE_ARG=""
if [ -n "$DEVICE" ]; then
  DEVICE_ARG="--device $DEVICE"
fi

# Run python script with all arguments
PYTHONPATH=. python script/ablation_test.py \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --learning_rate "$LEARNING_RATE" \
  --validation_split "$VALIDATION_SPLIT" \
  --n_runs "$N_RUNS" \
  $DEVICE_ARG \
  "$@"

