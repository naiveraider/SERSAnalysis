#!/bin/bash
# Run ablation pooling tests for all tasks (1-4)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=========================================="
echo "Ablation Pooling Tests - All Tasks"
echo "=========================================="
echo ""

for task_id in 1 2 3 4; do
  echo "=========================================="
  echo "Starting Ablation Test for Task ${task_id}"
  echo "=========================================="

  TASK_ID="$task_id" bash script/ablation_test.sh

  echo ""
done

echo "=========================================="
echo "All Ablation Pooling Tests Complete!"
echo "=========================================="