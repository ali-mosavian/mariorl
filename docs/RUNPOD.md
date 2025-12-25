# MadMario on RunPod

Guide for running MadMario world model training on RunPod or other cloud GPU services.

## Quick Start

### Option 1: Direct Training (Recommended)

1. **Create RunPod Instance**
   - Choose: PyTorch template or CUDA base image
   - GPU: RTX 3090/4090 or better
   - Storage: 20GB+ persistent volume

2. **Clone and Setup**
   ```bash
   git clone <your-repo-url> /workspace/MadMario
   cd /workspace/MadMario
   curl -LsSf https://astral.sh/uv/install.sh | sh
   export PATH="/root/.cargo/bin:$PATH"
   uv sync
   ```

3. **Run Training**
   ```bash
   chmod +x train_runpod.sh
   ./train_runpod.sh
   ```

4. **Custom Configuration**
   ```bash
   # Train on level 2-1 with 12 workers for 200k steps
   LEVEL=2,1 WORKERS=12 STEPS=200000 ./train_runpod.sh
   
   # Resume from existing checkpoint
   SAVE_DIR=checkpoints/runpod_2024-12-25T12-00-00 ./train_runpod.sh
   ```

### Option 2: Docker Container

1. **Build Container**
   ```bash
   cd /workspace/MadMario
   docker build -t madmario .
   ```

2. **Run Training**
   ```bash
   docker run --gpus all \
     -v /workspace/checkpoints:/workspace/checkpoints \
     -e WORKERS=12 \
     -e STEPS=200000 \
     madmario
   ```

## Environment Variables

Configure training via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LEVEL` | `1,1` | Mario level (world,stage) |
| `WORKERS` | `8` | Number of worker processes |
| `STEPS` | `100000` | Maximum training steps |
| `BUFFER_SIZE` | `50000` | Replay buffer capacity |
| `BATCH_SIZE` | `64` | Training batch size |
| `LATENT_DIM` | `128` | Latent space dimension |
| `WM_STEPS` | `500` | World model steps per cycle |
| `Q_STEPS` | `500` | Q-network steps per cycle |
| `WM_LR` | `0.0001` | World model learning rate |
| `Q_LR` | `0.0001` | Q-network learning rate |
| `SAVE_DIR` | auto | Checkpoint directory |

## Monitoring Training

### Via Training Log
```bash
tail -f checkpoints/<your-run>/training.log
```

### Via CSV Metrics
```bash
# View latest metrics
tail checkpoints/<your-run>/training.csv

# Plot metrics (requires matplotlib)
uv run python -c "
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('checkpoints/<your-run>/training.csv')
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0,0].plot(df['step'], df['wm_recon_mse'])
axes[0,0].set_title('Reconstruction MSE')

axes[0,1].plot(df['step'], df['wm_ssim'])
axes[0,1].set_title('SSIM')

axes[1,0].plot(df['step'], df['q_mean'])
axes[1,0].set_title('Q Mean')

axes[1,1].plot(df['step'], df['q_loss'])
axes[1,1].set_title('Q Loss')

plt.tight_layout()
plt.savefig('checkpoints/<your-run>/metrics.png')
print('Saved to metrics.png')
"
```

## Checkpointing & Resumption

Training automatically:
- Saves weights every 100 steps to `weights.pt`
- Saves full checkpoint every 1000 steps to `checkpoint.pt`
- Resumes from `checkpoint.pt` if it exists

To resume interrupted training:
```bash
SAVE_DIR=checkpoints/runpod_<timestamp> ./train_runpod.sh
```

## Downloading Results

### Via RunPod Web UI
1. Go to "My Pods"
2. Click on your pod
3. Navigate to Files tab
4. Download `checkpoints/<your-run>/weights.pt`

### Via SSH/SCP
```bash
scp -r <pod-ssh>:~/MadMario/checkpoints/<your-run> ./local-checkpoints/
```

### Via RunPod API
```bash
# Use RunPod's file sync feature or GraphQL API
```

## Watching Agent Play

After training completes:

```bash
# Download weights.pt to local machine
# Then run:
uv run python watch.py checkpoints/<your-run>/weights.pt --world-model --episodes 5
```

## Performance Tips

1. **Workers**: Set to 1-2x your CPU cores (8-16 on most RunPod instances)
2. **Buffer Size**: Larger is better but uses more RAM (50k-100k recommended)
3. **Batch Size**: Increase if you have GPU memory (64-256)
4. **Learning Rates**: Start with defaults, reduce if loss explodes

## Expected Training Time

On RTX 4090:
- **10k steps**: ~30 minutes (early learning)
- **50k steps**: ~2.5 hours (decent behavior)
- **100k steps**: ~5 hours (good performance)
- **200k steps**: ~10 hours (strong agent)

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size or buffer size
BATCH_SIZE=32 BUFFER_SIZE=25000 ./train_runpod.sh
```

### Training Stalls
- Check `training.log` for errors
- Verify workers are running: `ps aux | grep python`
- Check buffer size in CSV: should grow steadily at first

### Resume Not Working
- Ensure `SAVE_DIR` points to existing checkpoint directory
- Check `checkpoint.pt` exists in that directory
- Verify architecture matches (same `LATENT_DIM`)

## Cost Estimation

RunPod costs (approximate):
- RTX 3090: ~$0.25/hr → ~$2.50 for 100k steps
- RTX 4090: ~$0.60/hr → ~$3.00 for 100k steps
- A100 40GB: ~$1.50/hr → ~$7.50 for 100k steps

## Support

If you encounter issues:
1. Check `training.log` for errors
2. Verify system has sufficient resources
3. Try reducing workers/batch size
4. Open an issue on GitHub

