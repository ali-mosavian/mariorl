# Mario RL

**Reinforcement Learning for Super Mario Bros with World Models**

A modular deep reinforcement learning framework featuring world models with latent representations, distributed training, and advanced DQN techniques.

## ✨ Features

- 🧠 **World Model Architecture** - Learn abstract latent representations for better generalization
- 🚀 **Distributed Training** - Multi-worker parallel data collection
- 🎯 **Dueling Double DQN** - Advanced Q-learning with target networks
- 🔄 **Prioritized Experience Replay** - Sample important transitions more frequently
- 📊 **Comprehensive Metrics** - Track reconstruction quality, Q-values, and training progress
- 🐳 **Docker Support** - Ready for deployment on RunPod and other cloud services

## 🎮 World Model Overview

The world model learns to:
1. **Encode** raw pixel frames into compact latent representations (z)
2. **Predict** next latent states given current state and action (dynamics model)
3. **Estimate** rewards from latent states
4. **Decode** latent states back to frames (for validation)

The Q-network then operates entirely in latent space, enabling:
- Faster training
- Better generalization across levels
- More abstract reasoning

## 📦 Installation

### Quick Start (with uv)

```bash
# Clone repository
git clone https://github.com/yourusername/mario-rl.git
cd mario-rl

# Install with uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Or install with pip
pip install -e .
```

### For Development

```bash
# Install with dev dependencies
uv sync --extra dev

# Or with pip
pip install -e ".[dev]"
```

## 🚀 Quick Start

### Training

```bash
# Basic training on level 1-1
uv run python scripts/train.py --level 1,1 --workers 4 --world-model

# Train for longer with more workers
uv run python scripts/train.py \
  --level 1,1 \
  --workers 8 \
  --learner-steps 100000 \
  --world-model \
  --wm-steps 500 \
  --q-steps 500

# Train on different level
uv run python scripts/train.py --level 2,1 --workers 6 --world-model
```

### Watch Trained Agent

```bash
# Watch the agent play
uv run python scripts/watch.py checkpoints/<run-name>/weights.pt --world-model

# Watch on a different level
uv run python scripts/watch.py checkpoints/<run-name>/weights.pt --world-model --level 1-2
```

## 📂 Project Structure

```
mario-rl/
├── mario_rl/                    # Main package
│   ├── agent/                   # Neural networks and replay buffers
│   │   ├── world_model.py       # World model architecture
│   │   ├── neural.py            # DuelingDDQN network
│   │   └── replay.py            # Experience replay buffer
│   ├── environment/             # Mario environment and wrappers
│   │   ├── mariogym.py          # Multi-level Mario gym
│   │   └── wrappers.py          # Frame skip, resize, etc.
│   ├── training/                # Distributed training system
│   │   ├── world_model_learner.py  # World model + Q-network training
│   │   ├── learner.py           # Standard DQN learner
│   │   ├── worker.py            # Experience collection worker
│   │   ├── shared_buffer.py     # Multiprocess replay buffer
│   │   └── training_ui.py       # Curses-based training UI
│   └── utils/                   # Utilities and metrics
├── scripts/                     # Command-line scripts
│   ├── train.py                 # Main training script
│   ├── watch.py                 # Watch agent play
│   └── train_runpod.sh          # RunPod training script
├── docker/                      # Docker configuration
│   ├── Dockerfile
│   └── .dockerignore
├── docs/                        # Documentation
│   └── RUNPOD.md                # RunPod deployment guide
└── tests/                       # Unit tests

```

## 🔧 Configuration

### Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--level` | `1,1` | Mario level (world,stage) |
| `--workers` | `4` | Number of worker processes |
| `--learner-steps` | `-1` | Max training steps (infinite if -1) |
| `--buffer-size` | `50000` | Replay buffer capacity |
| `--batch-size` | `64` | Training batch size |
| `--world-model` | `False` | Use world model architecture |
| `--latent-dim` | `128` | Latent space dimension |
| `--wm-steps` | `500` | World model steps per cycle |
| `--q-steps` | `500` | Q-network steps per cycle |
| `--ui` / `--no-ui` | `True` | Show training UI |

### World Model Training

The training alternates between two phases:

1. **World Model Phase** (500 steps):
   - Train encoder/decoder on reconstruction
   - Train dynamics model to predict next latent state
   - Train reward predictor

2. **Q-Network Phase** (500 steps):
   - Freeze world model
   - Train Q-network in latent space
   - Update target network periodically

## 📊 Monitoring Training

### Interactive UI

By default, training shows a curses-based UI with:
- Worker statistics (episodes, rewards, x-position)
- Learner metrics (loss, Q-values, buffer size)
- World model metrics (MSE, SSIM, dynamics loss)

### Log Files

All runs save to `checkpoints/<timestamp>/`:
- `weights.pt` - Latest network weights
- `checkpoint.pt` - Full training state (for resumption)
- `training.csv` - Metrics logged every 100 steps
- `training.log` - Full training log

### Plot Metrics

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('checkpoints/<run>/training.csv')

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
plt.show()
```

## 🐳 Docker & RunPod

### Build Docker Image

```bash
cd docker
docker build -t mario-rl .
```

### Run Locally with Docker

```bash
docker run --gpus all \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  -e WORKERS=8 \
  -e STEPS=100000 \
  mario-rl
```

### Deploy on RunPod

See [docs/RUNPOD.md](docs/RUNPOD.md) for detailed instructions.

Quick start:
```bash
# On RunPod instance
git clone <your-repo> /workspace/mario-rl
cd /workspace/mario-rl
./scripts/train_runpod.sh
```

## 🧪 Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=mario_rl --cov-report=html

# Run specific test
uv run pytest tests/test_world_model.py
```

## 📈 Performance

### Expected Training Time (RTX 4090)

- **10k steps**: ~30 min (early learning)
- **50k steps**: ~2.5 hours (decent behavior)
- **100k steps**: ~5 hours (good performance)
- **200k steps**: ~10 hours (strong agent)

### Signs of Good Training

- **Reconstruction MSE** drops to <0.01
- **SSIM** increases to >0.9
- **Q-values** become positive
- **Worker x-positions** steadily increase
- **Episode rewards** improve over time

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

MIT License - see [LICENSE](LICENSE) for details

## 🙏 Acknowledgments

- Based on [Dueling DQN](https://arxiv.org/abs/1511.06581)
- Inspired by [World Models](https://worldmodels.github.io/)
- Built with [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros)

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{mario_rl_2025,
  author = {Your Name},
  title = {Mario RL: Reinforcement Learning for Super Mario Bros with World Models},
  year = {2025},
  url = {https://github.com/yourusername/mario-rl}
}
```

## 📧 Contact

- Issues: [GitHub Issues](https://github.com/yourusername/mario-rl/issues)
- Email: your.email@example.com

