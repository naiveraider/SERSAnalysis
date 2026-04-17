#!/usr/bin/env bash
"""
Shell wrapper to run the Python ablation test.
"""
set -euo pipefail

PYTHON=${PYTHON:-python3}
SEQ_LEN=${SEQ_LEN:-512}
DEVICE=${DEVICE:-cpu}

$PYTHON script/ablation_test.py --seq-len "$SEQ_LEN" --device "$DEVICE"
