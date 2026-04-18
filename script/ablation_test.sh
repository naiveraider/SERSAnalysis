#!/usr/bin/env bash
# Class-style runner for ablation tests (GPU-capable)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
	echo "Usage: DEVICE=cuda:0 N_RUNS=3 SEQ_LEN=128 bash script/ablation_test.sh"
	exit 0
fi

PYTHON=${PYTHON:-python3}
SEQ_LEN=${SEQ_LEN:-512}
DEVICE=${DEVICE:-cpu}
N_RUNS=${N_RUNS:-1}

mkdir -p results
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="results/ablation_test_${TIMESTAMP}.txt"

echo "Ablation MeanMax Test Runner" > "$RESULTS_FILE"
echo "Generated: $(date)" >> "$RESULTS_FILE"
echo "DEVICE=$DEVICE SEQ_LEN=$SEQ_LEN N_RUNS=$N_RUNS" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

run_once() {
	local run_idx="$1"
	local logf
	logf=$(mktemp)
	echo "----- Run $run_idx/$N_RUNS (device=$DEVICE seq_len=$SEQ_LEN) -----" | tee -a "$RESULTS_FILE"
	PYTHONPATH=. "$PYTHON" script/ablation_test.py --seq-len "$SEQ_LEN" --device "$DEVICE" 2>&1 | tee "$logf" | tee -a "$RESULTS_FILE"
	rm -f "$logf"
}

for i in $(seq 1 "$N_RUNS"); do
	run_once "$i"
done

echo "" | tee -a "$RESULTS_FILE"
echo "Logs written to $RESULTS_FILE"

