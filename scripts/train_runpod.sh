#!/bin/bash
#
# Training script for RunPod or other cloud GPU services
#
# Usage:
#   ./train_runpod.sh [OPTIONS]
#
# Environment variables:
#   LEVEL - Mario level (default: 1,1)
#   WORKERS - Number of worker processes (default: 8)
#   STEPS - Maximum training steps (default: 100000)
#   SAVE_DIR - Save directory (default: checkpoints/runpod_<timestamp>)
#

set -e  # Exit on error

# Configuration
LEVEL="${LEVEL:-1,1}"
WORKERS="${WORKERS:-8}"
STEPS="${STEPS:-100000}"
BUFFER_SIZE="${BUFFER_SIZE:-50000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LATENT_DIM="${LATENT_DIM:-128}"
WM_STEPS="${WM_STEPS:-500}"
Q_STEPS="${Q_STEPS:-500}"
WM_LR="${WM_LR:-0.0001}"
Q_LR="${Q_LR:-0.0001}"

# Auto-generate save dir with timestamp if not set
if [ -z "$SAVE_DIR" ]; then
    TIMESTAMP=$(date +"%Y-%m-%dT%H-%M-%S")
    SAVE_DIR="checkpoints/runpod_${TIMESTAMP}"
fi

echo "========================================"
echo "MadMario World Model Training (RunPod)"
echo "========================================"
echo "Configuration:"
echo "  Level:          $LEVEL"
echo "  Workers:        $WORKERS"
echo "  Max Steps:      $STEPS"
echo "  Buffer Size:    $BUFFER_SIZE"
echo "  Batch Size:     $BATCH_SIZE"
echo "  Latent Dim:     $LATENT_DIM"
echo "  WM Steps/Cycle: $WM_STEPS"
echo "  Q Steps/Cycle:  $Q_STEPS"
echo "  WM LR:          $WM_LR"
echo "  Q LR:           $Q_LR"
echo "  Save Dir:       $SAVE_DIR"
echo "========================================"
echo ""

# Create save directory
mkdir -p "$SAVE_DIR"

# Save config for reference
cat > "$SAVE_DIR/config.txt" <<EOF
Training Configuration
======================
Started: $(date)
Level: $LEVEL
Workers: $WORKERS
Max Steps: $STEPS
Buffer Size: $BUFFER_SIZE
Batch Size: $BATCH_SIZE
Latent Dim: $LATENT_DIM
WM Steps/Cycle: $WM_STEPS
Q Steps/Cycle: $Q_STEPS
WM Learning Rate: $WM_LR
Q Learning Rate: $Q_LR
EOF

# Run training
echo "Starting training..."
echo "Logs will be saved to: $SAVE_DIR/training.log"
echo "Metrics will be saved to: $SAVE_DIR/training.csv"
echo ""

uv run python -m distributed \
    --level "$LEVEL" \
    --workers "$WORKERS" \
    --learner-steps "$STEPS" \
    --buffer-size "$BUFFER_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --world-model \
    --latent-dim "$LATENT_DIM" \
    --wm-steps "$WM_STEPS" \
    --q-steps "$Q_STEPS" \
    --wm-lr "$WM_LR" \
    --q-lr "$Q_LR" \
    --save-dir "$SAVE_DIR" \
    --no-ui \
    2>&1 | tee "$SAVE_DIR/training.log"

echo ""
echo "========================================"
echo "Training completed!"
echo "Results saved to: $SAVE_DIR"
echo "========================================"
echo ""
echo "To watch the trained agent play:"
echo "  uv run python watch.py $SAVE_DIR/weights.pt --world-model"

