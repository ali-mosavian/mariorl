#!/usr/bin/env bash
#
# Train Mario using Distributed DDQN with optimized stability settings.
#
# Usage:
#   ./scripts/train.sh                     # Run with defaults
#   ./scripts/train.sh --no-ui             # Run without UI
#   ./scripts/train.sh --resume            # Resume from latest checkpoint
#   ./scripts/train.sh --workers 8         # Override worker count
#
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT="$SCRIPT_DIR/.."

(cd "$PROJECT_ROOT" && \
 uv run mario-train-ddqn-dist \
  --workers 16 \
  --accumulate-grads 16 \
  --batch-size 128 \
  --train-steps 8 \
  --tau 0.001 \
  --q-clip 100 \
  --reward-clip 20 \
  --eps-decay-steps 1000000 \
  "$@")