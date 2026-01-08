# Mario RL

**Reinforcement Learning for Super Mario Bros with Distributed Training**

A modular deep reinforcement learning framework featuring distributed A3C-style gradient sharing, world models, and advanced DQN techniques.

## ✨ Features

- 🚀 **Distributed Training** - A3C-style gradient sharing across multiple workers
- 🧠 **Multiple Model Types** - DDQN and Dreamer (world model) architectures
- 🎯 **Dueling Double DQN** - Advanced Q-learning with target networks
- 🔄 **Prioritized Experience Replay** - Sample important transitions more frequently
- 📊 **Real-time Monitoring** - ncurses-based training dashboard
- 🐳 **Docker Support** - Ready for deployment on cloud services

## 🏗️ Architecture

The distributed training system uses **gradient sharing** (A3C-style) where workers compute gradients locally and send them to a central coordinator.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           MAIN PROCESS                                   │
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │              Shared Memory Gradient Pool                         │   │
│   │              (workers → coordinator, ~2MB per packet)            │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
          │                    │                    │
          ▼                    ▼                    ▼
   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
   │  WORKER 0   │      │  WORKER 1   │      │  WORKER N   │
   │             │      │             │      │             │
   │ 1. Collect  │      │ 1. Collect  │      │ 1. Collect  │
   │ 2. Sample   │      │ 2. Sample   │      │ 2. Sample   │
   │ 3. Backward │      │ 3. Backward │      │ 3. Backward │
   │ 4. Send     │      │ 4. Send     │      │ 4. Send     │
   │    grads    │      │    grads    │      │    grads    │
   └──────┬──────┘      └──────┬──────┘      └──────┬──────┘
          │                    │                    │
          └────────────────────┼────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    COORDINATOR      │
                    │                     │
                    │  1. Poll gradients  │
                    │  2. Aggregate       │
                    │  3. Optimizer step  │
                    │  4. Save weights    │
                    │  5. Update targets  │
                    └─────────────────────┘
```

### Key Components

| Component | Location | Description |
|-----------|----------|-------------|
| **Model Protocol** | `mario_rl/models/base.py` | Interface for all models (forward, select_action, state_dict) |
| **Learner Protocol** | `mario_rl/learners/base.py` | Interface for all learners (compute_loss, update_targets) |
| **DoubleDQN** | `mario_rl/models/ddqn.py` | Dueling Double DQN with softsign activation |
| **DreamerModel** | `mario_rl/models/dreamer.py` | World model with encoder, dynamics, actor-critic |
| **TrainingWorker** | `mario_rl/distributed/training_worker.py` | Full worker with env, buffer, gradient computation |
| **TrainingCoordinator** | `mario_rl/distributed/training_coordinator.py` | Gradient aggregation, LR scheduling, checkpointing |
| **SharedGradientPool** | `mario_rl/distributed/shm_gradient_pool.py` | Lock-free gradient sharing via mmap |
| **SharedHeartbeat** | `mario_rl/distributed/shm_heartbeat.py` | Worker health monitoring |
| **ReplayBuffer** | `mario_rl/core/replay_buffer.py` | N-step returns with optional PER |
| **EnvRunner** | `mario_rl/core/env_runner.py` | Environment step collection |
| **TrainingUI** | `mario_rl/ui/training_ui.py` | ncurses monitoring dashboard |

### Model Types

**DDQN (Double DQN)**
- Dueling architecture with separate value and advantage streams
- Softsign activation to bound Q-values to `[-q_scale, q_scale]`
- Double Q-learning for reduced overestimation

**Dreamer (World Model)**
- VAE-style encoder for latent state representation
- GRU-based dynamics model for state prediction
- Actor-critic heads trained on imagined trajectories
- Lambda-returns (TD(λ)) for value estimation

## 🎮 World Model Overview

The Dreamer world model learns to:
1. **Encode** raw pixel frames into compact latent representations (z)
2. **Predict** next latent states given current state and action (dynamics model)
3. **Estimate** rewards from latent states
4. **Act** via actor-critic trained on imagined rollouts

This enables:
- Faster training through imagination
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

### Training with New Modular System

```bash
# Train DDQN with 4 workers
uv run python scripts/train_distributed.py --model ddqn --workers 4

# Train Dreamer (world model) with 8 workers
uv run python scripts/train_distributed.py --model dreamer --workers 8

# Train without UI (for headless servers)
uv run python scripts/train_distributed.py --model ddqn --workers 4 --no-ui

# Custom configuration
uv run python scripts/train_distributed.py \
  --model ddqn \
  --workers 8 \
  --lr 1e-4 \
  --batch-size 32 \
  --buffer-size 10000
```

### Legacy Training (still supported)

```bash
# Basic training on level 1-1
uv run mario-train-ddqn-dist --workers 4 --level 1,1

# Train for longer with more workers
uv run mario-train-ddqn-dist \
  --workers 8 \
  --level random \
  --accumulate-grads 4
```

### Watch Trained Agent

```bash
# Watch the agent play
uv run python scripts/watch.py checkpoints/<run-name>/weights.pt

# Watch on a different level
uv run python scripts/watch.py checkpoints/<run-name>/weights.pt --level 1-2
```

## 📂 Project Structure

```
mario-rl/
├── mario_rl/                      # Main package
│   ├── models/                    # Model definitions (new modular)
│   │   ├── base.py                # Model protocol
│   │   ├── ddqn.py                # DoubleDQN model
│   │   └── dreamer.py             # Dreamer world model
│   ├── learners/                  # Learning algorithms (new modular)
│   │   ├── base.py                # Learner protocol
│   │   ├── ddqn.py                # DDQN learner (loss, targets)
│   │   └── dreamer.py             # Dreamer learner (world + behavior)
│   ├── distributed/               # Distributed training (new modular)
│   │   ├── worker.py              # Base gradient worker
│   │   ├── coordinator.py         # Base gradient coordinator
│   │   ├── training_worker.py     # Full worker with env + buffer
│   │   ├── training_coordinator.py # Full coordinator with scheduling
│   │   ├── shm_gradient_pool.py   # Shared memory gradient buffers
│   │   └── shm_heartbeat.py       # Worker health monitoring
│   ├── core/                      # Core components
│   │   ├── replay_buffer.py       # Unified buffer (N-step + PER)
│   │   ├── env_runner.py          # Environment step collection
│   │   ├── config.py              # Configuration dataclasses
│   │   └── types.py               # Core data types
│   ├── ui/                        # Monitoring UI
│   │   ├── training_ui.py         # ncurses dashboard
│   │   └── metrics.py             # Metrics aggregation
│   ├── env/                       # Environment wrappers
│   │   └── mario_env.py           # Mario environment creation
│   ├── agent/                     # Legacy networks (being deprecated)
│   │   ├── ddqn_net.py            # Legacy DDQN
│   │   └── world_model.py         # Legacy world model
│   └── training/                  # Legacy training (being deprecated)
│       ├── ddqn_worker.py         # Legacy worker
│       ├── ddqn_learner.py        # Legacy learner
│       └── shared_gradient_tensor.py  # Shared memory implementation
├── scripts/                       # Command-line scripts
│   ├── train_distributed.py       # New modular training script
│   ├── train_ddqn_distributed.py  # Legacy distributed training
│   └── watch.py                   # Watch agent play
├── tests/                         # Comprehensive test suite
│   ├── models/                    # Model tests
│   ├── learners/                  # Learner tests
│   ├── distributed/               # Distributed component tests
│   └── core/                      # Core component tests
└── docker/                        # Docker configuration
```

## 🔧 Configuration

### New Modular Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `ddqn` | Model type: `ddqn` or `dreamer` |
| `--workers` | `4` | Number of worker processes |
| `--lr` | `1e-4` | Learning rate |
| `--lr-min` | `1e-5` | Minimum learning rate (cosine decay) |
| `--gamma` | `0.99` | Discount factor |
| `--n-step` | `3` | N-step returns |
| `--batch-size` | `32` | Training batch size |
| `--buffer-size` | `10000` | Per-worker replay buffer capacity |
| `--collect-steps` | `64` | Steps per collection cycle |
| `--no-ui` | `False` | Disable ncurses UI |

### Training Cycle

Each worker runs this loop:
1. **Collect** 64 steps from environment
2. **Sample** batch from local replay buffer
3. **Compute** gradients via backprop
4. **Send** gradients to coordinator via shared memory
5. **Sync** weights from coordinator's file

The coordinator:
1. **Polls** gradients from all workers
2. **Aggregates** gradients (averaging)
3. **Applies** optimizer step with gradient clipping
4. **Updates** learning rate (cosine annealing)
5. **Saves** weights for workers to sync
6. **Updates** target network periodically

### Dreamer Training

When using `--model dreamer`, training includes:

1. **World Model Phase**:
   - Encode observations to latent states
   - Train dynamics model (GRU) to predict next latent
   - Train reward and terminal predictors
   - Reconstruction loss for encoder validation

2. **Behavior Phase**:
   - Imagine trajectories using learned dynamics
   - Train actor to maximize imagined returns
   - Train critic on lambda-returns (TD(λ))

## 🔄 Shared Memory IPC

The distributed system uses memory-mapped files for zero-copy gradient transfer:

### SharedGradientTensorPool

Each worker has a dedicated gradient buffer:
- **Path**: `/dev/shm/mariorl_grads_<worker_id>.bin`
- **Size**: ~2MB per worker (depends on model)
- **Format**: Header (version, ready flag, metadata) + flattened gradients
- **Lock-free**: Workers write, coordinator reads (no contention)

### SharedHeartbeats

Workers report health via shared memory:
- **Path**: `/dev/shm/mariorl_heartbeats.bin`
- **Format**: Float64 timestamps, one per worker
- **Monitoring**: Coordinator detects stale workers (no heartbeat > timeout)
- **Recovery**: Stale workers are restarted automatically

## 📊 Monitoring Training

### Interactive UI

By default, training shows an ncurses-based dashboard with:
- Worker statistics (episodes, rewards, steps, gradients sent)
- Coordinator metrics (loss, Q-values, learning rate, updates)
- Recent log messages
- Optional: reward/loss graphs

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

The codebase has comprehensive tests (250+ tests) following TDD principles.

```bash
# Run all tests
uv run pytest

# Run modular component tests only
uv run pytest tests/models/ tests/learners/ tests/distributed/ tests/core/ -v

# Run with coverage
uv run pytest --cov=mario_rl --cov-report=html

# Run specific test module
uv run pytest tests/models/test_ddqn.py -v

# Run tests matching a pattern
uv run pytest -k "select_action" -v
```

### Test Categories

| Category | Tests | Description |
|----------|-------|-------------|
| `tests/models/` | ~50 | Model protocols, DDQN, Dreamer |
| `tests/learners/` | ~60 | DDQN learner, Dreamer learner |
| `tests/distributed/` | ~90 | Workers, coordinators, shared memory |
| `tests/core/` | ~45 | Replay buffer, env runner |

## 🔌 Extending the Framework

### Adding a New Model

1. Create a new model in `mario_rl/models/`:

```python
from mario_rl.models.base import Model

class MyModel(nn.Module, Model):
    """Implement the Model protocol."""
    
    def forward(self, x: Tensor, network: str = "online") -> Tensor:
        ...
    
    def select_action(self, x: Tensor, epsilon: float = 0.0) -> Tensor:
        ...
    
    def sync_target(self) -> None:
        ...
```

2. Create a corresponding learner in `mario_rl/learners/`:

```python
from mario_rl.learners.base import Learner

class MyLearner(Learner):
    """Implement the Learner protocol."""
    
    def compute_loss(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        dones: Tensor,
        gamma: float = 0.99,
    ) -> tuple[Tensor, dict[str, Any]]:
        ...
    
    def update_targets(self, tau: float = 0.005) -> None:
        ...
```

3. Add to `scripts/train_distributed.py` factory function.

### Protocol Interfaces

**Model Protocol** (`mario_rl/models/base.py`):
- `forward(x, network)` - Forward pass through specified network
- `select_action(x, epsilon)` - Action selection with ε-greedy
- `state_dict()` / `load_state_dict()` - Serialization
- `sync_target()` - Target network update
- `parameters()` - For optimizer
- `to(device)` - Device transfer

**Learner Protocol** (`mario_rl/learners/base.py`):
- `model` - Access to underlying Model
- `compute_loss(...)` - Compute training loss and metrics
- `update_targets(tau)` - Soft update target networks

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

- [Dueling DQN](https://arxiv.org/abs/1511.06581) - Dueling network architecture
- [Double DQN](https://arxiv.org/abs/1509.06461) - Double Q-learning
- [A3C](https://arxiv.org/abs/1602.01783) - Asynchronous gradient sharing
- [Dreamer](https://arxiv.org/abs/1912.01603) - World model with imagination
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) - PER
- [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros) - Mario environment

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{mario_rl_2025,
  author = {Your Name},
  title = {Mario RL: Distributed Reinforcement Learning for Super Mario Bros},
  year = {2025},
  url = {https://github.com/yourusername/mario-rl}
}
```

## 📧 Contact

- Issues: [GitHub Issues](https://github.com/yourusername/mario-rl/issues)
- Email: your.email@example.com

