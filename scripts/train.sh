#!/usr/bin/env bash
#
# Train Mario using Distributed DDQN with optimized stability settings.
#
# Key settings for stable training:
# - Soft Q-clipping (tanh) at 50 to prevent saturation while maintaining gradients
# - Lower learning rate (1e-4) for stable updates
# - Slow epsilon decay (2M steps) for thorough exploration
# - Death penalty: -2 (once), Flag bonus: +15
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
  --lr 1e-4 \
  --tau 0.001 \
  --q-scale 1500 \
  --reward-clip 0 \
  --eps-decay-steps 100000 \
  "$@")