# Project Structure

This document describes the organization of the mario-rl project.

## Directory Layout

```
mario-rl/
├── mario_rl/               # Main Python package
│   ├── __init__.py         # Package root with main exports
│   ├── agent/              # Neural networks and replay buffers
│   │   ├── world_model.py  # World model architecture
│   │   ├── neural.py       # DuelingDDQN implementation
│   │   └── replay.py       # Experience replay buffer
│   ├── environment/        # Mario environment
│   │   ├── mariogym.py     # Multi-level Mario gym wrapper
│   │   ├── wrappers.py     # Frame skip, resize, etc.
│   │   └── frame_stack.py  # Frame stacking utilities
│   ├── training/           # Distributed training system
│   │   ├── world_model_learner.py  # World model + Q training
│   │   ├── learner.py      # Standard DQN learner
│   │   ├── worker.py       # Experience collection worker
│   │   ├── shared_buffer.py # Multiprocess replay buffer
│   │   └── training_ui.py  # Training UI
│   └── utils/              # Utilities
│       └── metrics.py      # Metrics and visualization
├── scripts/                # Command-line scripts
│   ├── train.py            # Main training script
│   ├── watch.py            # Watch agent play
│   └── train_runpod.sh     # RunPod training script
├── docker/                 # Docker configuration
│   ├── Dockerfile          # Container definition
│   └── .dockerignore       # Docker ignore patterns
├── docs/                   # Documentation
│   └── RUNPOD.md           # RunPod deployment guide
└── tests/                  # Unit tests
    └── __init__.py
```

## Module Organization

### `mario_rl.agent`
Core RL components:
- **world_model.py**: World model with encoder, decoder, dynamics, and reward prediction
- **neural.py**: Dueling Double DQN architecture
- **replay.py**: Experience replay buffer with prioritization

### `mario_rl.environment`
Mario environment wrappers:
- **mariogym.py**: Multi-level Mario environment
- **wrappers.py**: Frame skip, resize, grayscale wrappers
- **frame_stack.py**: Frame stacking utilities

### `mario_rl.training`
Distributed training system:
- **world_model_learner.py**: Trains world model and Q-network in alternating phases
- **learner.py**: Standard DQN learner (without world model)
- **worker.py**: Worker process for experience collection
- **shared_buffer.py**: Multiprocess-safe replay buffer
- **training_ui.py**: Curses-based training UI

### `mario_rl.utils`
Utilities and helpers:
- **metrics.py**: Training metrics and visualization

## Import Structure

The package is organized for clean imports:

```python
# Main imports
from mario_rl import MarioWorldModel, LatentDDQN, DuelingDDQNNet

# Agent imports
from mario_rl.agent import MarioWorldModel, LatentDDQN, DuelingDDQNNet, ExperienceBatch

# Environment imports
from mario_rl.environment import SuperMarioBrosMultiLevel, SkipFrame, ResizeObservation

# Training imports
from mario_rl.training import (
    WorldModelLearner,
    Worker,
    SharedReplayBuffer,
    run_world_model_learner,
    run_worker
)
```

## Scripts

### `scripts/train.py`
Main training script with CLI:
```bash
uv run python scripts/train.py --level 1,1 --workers 4 --world-model
```

### `scripts/watch.py`
Watch trained agent play:
```bash
uv run python scripts/watch.py checkpoints/<run>/weights.pt --world-model
```

### `scripts/train_runpod.sh`
RunPod deployment script with environment variables:
```bash
WORKERS=8 STEPS=100000 ./scripts/train_runpod.sh
```

## Installation

### Development Installation
```bash
# Clone repository
git clone <repo-url> mario-rl
cd mario-rl

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### With Dependencies
```bash
# Install dev dependencies
uv sync --extra dev

# Or with pip
pip install -e ".[dev]"
```

## Dependencies

### Core Dependencies
- **torch>=2.0.0**: Deep learning framework
- **numpy>=1.24.0**: Numerical computing
- **gymnasium>=0.29.0**: RL environment interface
- **gym-super-mario-bros>=7.4.0**: Mario environment
- **nes-py>=8.2.1**: NES emulator
- **pyglet>=1.4.0,<=1.5.21**: Rendering (limited by nes-py)
- **lz4>=4.0.0**: Compression for replay buffer
- **scikit-image>=0.21.0**: Image processing for wrappers
- **click>=8.1.0**: CLI framework
- **pillow>=10.0.0**: Image handling

### Development Dependencies
- **pytest>=7.4.0**: Testing framework
- **pytest-cov>=4.1.0**: Coverage reporting
- **black>=23.7.0**: Code formatting
- **ruff>=0.0.285**: Linting
- **mypy>=1.5.0**: Type checking

## Testing

Run tests:
```bash
# All tests
uv run pytest

# With coverage
uv run pytest --cov=mario_rl --cov-report=html

# Specific test
uv run pytest tests/test_world_model.py
```

## Git Workflow

The project uses conventional commits:
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Maintenance tasks

## Docker Deployment

Build and run:
```bash
cd docker
docker build -t mario-rl .
docker run --gpus all -v $(pwd)/checkpoints:/workspace/checkpoints mario-rl
```

## Configuration Files

- **pyproject.toml**: Package configuration, dependencies, and tool settings
- **.gitignore**: Git ignore patterns
- **docker/.dockerignore**: Docker ignore patterns
- **README.md**: Main documentation
- **docs/RUNPOD.md**: RunPod deployment guide

