# Mario RL

**Reinforcement Learning for Super Mario Bros with Distributed Training**

A modular deep reinforcement learning framework featuring distributed A3C-style gradient sharing, world models, and advanced DQN techniques.

## ✨ Features

- 🚀 **Distributed Training** - A3C-style gradient sharing across multiple workers
- 🧠 **Multiple Model Types** - DDQN and Dreamer (world model) architectures
- 🎯 **Dueling Double DQN** - Advanced Q-learning with target networks
- 🔄 **Prioritized Experience Replay** - Sample important transitions more frequently
- 📊 **Real-time Monitoring** - ncurses-based training dashboard
- 📈 **Unified Metrics System** - Collectors pattern with ZMQ pub/sub
- 💀 **Death Hotspot Tracking** - Aggregates death positions for curriculum learning
- 💾 **Snapshot State Machine** - Intelligent save/restore for practicing difficult sections
- 🐳 **Docker Support** - Ready for deployment on cloud services

## 🏗️ Architecture Overview

The system is built with a modular, layered architecture following SOLID principles:

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                              TRAINING SYSTEM                                   │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         MAIN PROCESS                                    │   │
│  │                                                                         │   │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────────────┐ │   │
│  │  │ Event Subscriber │  │ MetricAggregator │  │ DeathHotspotAggregate  │ │   │
│  │  │ (ZMQ PULL)       │──│ (combine workers)│  │ (25px buckets/level)   │ │   │
│  │  └──────────────────┘  └──────────────────┘  └────────────────────────┘ │   │
│  │           │                     │                       │               │   │
│  │           ▼                     ▼                       ▼               │   │
│  │  ┌─────────────────────────────────────────────────────────────────────┐│   │
│  │  │                    Training UI (curses)                             ││   │
│  │  │  Workers: steps, rewards, ε, best_x, deaths, grads_sent             ││   │
│  │  │  Learner: loss, q_mean, td_error, lr, updates/sec                   ││   │
│  │  └─────────────────────────────────────────────────────────────────────┘│   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                      Shared Memory Gradient Pool                        │   │
│  │               (lock-free mmap, ~2MB per worker gradient packet)         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│         │                         │                         │                  │
│         ▼                         ▼                         ▼                  │
│  ┌─────────────┐          ┌─────────────┐           ┌─────────────┐            │
│  │  WORKER 0   │          │  WORKER 1   │           │  WORKER N   │            │
│  │             │          │             │           │             │            │
│  │ EnvRunner   │          │ EnvRunner   │           │ EnvRunner   │            │
│  │ ReplayBuffer│          │ ReplayBuffer│           │ ReplayBuffer│            │
│  │ Collectors  │          │ Collectors  │           │ Collectors  │            │
│  │ MetricLogger│──ZMQ────▶│ MetricLogger│──ZMQ─────▶│ MetricLogger│───┐        │
│  └─────────────┘          └─────────────┘           └─────────────┘   │        │
│         │                         │                         │         │        │
│         └─────────────────────────┼─────────────────────────┘         │        │
│                                   ▼                                   │        │
│                      ┌─────────────────────┐                          │        │
│                      │    COORDINATOR      │◀─────────────────────────┘        │
│                      │                     │                                   │
│                      │  GradientPool.poll()│                                   │
│                      │  Optimizer.step()   │                                   │
│                      │  LR Scheduler       │                                   │
│                      │  Target sync (τ)    │                                   │
│                      │  Checkpointing      │                                   │
│                      └─────────────────────┘                                   │
└────────────────────────────────────────────────────────────────────────────────┘
```

## 📊 Metrics System Architecture

The metrics system follows the **Collector Pattern** for clean separation of concerns:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           METRICS DATA FLOW                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   WORKER PROCESS                           MAIN PROCESS                     │
│   ═══════════════                         ═════════════                     │
│                                                                             │
│   ┌─────────────┐     on_step()           ┌──────────────────┐              │
│   │ Environment │────────────────────────▶│ EventSubscriber  │              │
│   │  step(a)    │                         │   (ZMQ PULL)     │              │
│   └─────────────┘                         └────────┬─────────┘              │
│          │                                         │                        │
│          ▼                                         ▼                        │
│   ┌─────────────────────────────────┐     ┌──────────────────┐              │
│   │    CompositeCollector           │     │ MetricAggregator │              │
│   │  ┌──────────────────────────┐   │     │  - per worker    │              │
│   │  │ MarioCollector           │   │     │  - mean/sum/max  │              │
│   │  │  - x_pos, deaths, flags  │   │     │  - rolling stats │              │
│   │  │  - speed, game_time      │   │     └────────┬─────────┘              │
│   │  └──────────────────────────┘   │              │                        │
│   │  ┌──────────────────────────┐   │              ▼                        │
│   │  │ DDQNCollector            │   │     ┌──────────────────┐              │
│   │  │  - loss, q_mean, q_max   │   │     │   Training UI    │              │
│   │  │  - td_error, grad_norm   │   │     │  (real-time)     │              │
│   │  └──────────────────────────┘   │     └──────────────────┘              │
│   │  ┌──────────────────────────┐   │                                       │
│   │  │ SystemCollector          │   │     ┌──────────────────┐              │
│   │  │  - steps, episodes       │   │     │ death_hotspots   │              │
│   │  │  - buffer_size, sps      │   │────▶│    .json         │              │
│   │  └──────────────────────────┘   │     │ (25px buckets)   │              │
│   └──────────────┬──────────────────┘     └──────────────────┘              │
│                  │                                                          │
│                  ▼                                                          │
│   ┌─────────────────────────────────┐                                       │
│   │        MetricLogger             │                                       │
│   │  - Counter: deaths, flags       │     ┌──────────────────┐              │
│   │  - Gauge: x_pos, epsilon        │────▶│ worker_N.csv     │              │
│   │  - Rolling: reward, loss        │     │ (on-the-fly)     │              │
│   │  - publish() → ZMQ PUSH         │     └──────────────────┘              │
│   └─────────────────────────────────┘                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Metric Types

| Type | Description | Example |
|------|-------------|---------|
| **Counter** | Monotonically increasing | `deaths`, `flags`, `grads_sent` |
| **Gauge** | Current value (can go up/down) | `x_pos`, `epsilon`, `buffer_size` |
| **Rolling** | Rolling average over window | `reward`, `loss`, `speed` |

### Collector Protocol

```python
class MetricCollector(Protocol):
    """Worker-side metric collector."""
    
    def on_step(self, info: dict[str, Any]) -> None:
        """Called after each env step."""
        
    def on_episode_end(self, info: dict[str, Any]) -> None:
        """Called at episode end."""
        
    def on_train_step(self, metrics: dict[str, Any]) -> None:
        """Called after each training step."""
        
    def flush(self) -> None:
        """Publish accumulated metrics."""
```

### Available Collectors

| Collector | Metrics | Usage |
|-----------|---------|-------|
| `MarioCollector` | x_pos, deaths, flags, speed | Game-specific |
| `DDQNCollector` | loss, q_mean, q_max, td_error | DDQN training |
| `DreamerCollector` | wm_loss, actor_loss, critic_loss | Dreamer training |
| `SystemCollector` | steps, episodes, buffer_size, sps | System metrics |
| `CompositeCollector` | Combines multiple collectors | Composition |

## 💀 Death Hotspot Aggregation

Tracks where Mario dies to enable curriculum learning via emulator snapshots:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     DEATH HOTSPOT SYSTEM                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   WORKER                                    MAIN PROCESS                    │
│   ══════                                    ════════════                    │
│                                                                             │
│   Mario dies at x=523           ZMQ         ┌────────────────────────────┐  │
│       │                   ─────────────────▶│ DeathHotspotAggregate      │  │
│       ▼                                     │                            │  │
│   death_positions:                          │  Level 1-1:                │  │
│     level_id: "1-1"                         │   bucket[500]: 47 deaths   │  │
│     positions: [523]                        │   bucket[525]: 23 deaths   │  │
│                                             │   bucket[775]: 12 deaths   │  │
│                                             │                            │  │
│                                             │  → Save every 60s          │  │
│   ┌───────────────────────┐                 └────────────┬───────────────┘  │
│   │ Worker loads hotspots │◀─────────────────────────────┘                  │
│   │ every 30 seconds      │       death_hotspots.json                       │
│   └───────────┬───────────┘                                                 │
│               │                                                             │
│               ▼                                                             │
│   ┌───────────────────────────────────────────────────────────────────────┐ │
│   │  Snapshot Decisions                                                   │ │
│   │                                                                       │ │
│   │  suggest_snapshot_positions("1-1", count=3)                           │ │
│   │    → [475, 750]  # Positions BEFORE hotspots                          │ │
│   │                                                                       │ │
│   │  suggest_restore_position("1-1", death_x=530)                         │ │
│   │    → 450  # Position to restore from for practice                     │ │
│   └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Usage

```python
from mario_rl.metrics import DeathHotspotAggregate

# Load or create
agg = DeathHotspotAggregate.load_or_create(Path("death_hotspots.json"))

# Record deaths
agg.record_death("1-1", x_pos=523)
agg.record_deaths_batch("1-1", [525, 530, 520])

# Get hotspots (positions with >= 3 deaths)
hotspots = agg.get_hotspots("1-1", min_deaths=3)
# → [(500, 47), (525, 23), (775, 12)]

# Suggest where to save emulator state
positions = agg.suggest_snapshot_positions("1-1", count=3, min_spacing=100)
# → [475, 750] - positions BEFORE death hotspots

# After dying, suggest where to restore
restore = agg.suggest_restore_position("1-1", death_x=530)
# → 450 - position to restore for practice
```

## 💾 Snapshot State Machine

The snapshot system uses a state machine to intelligently save and restore emulator states based on death hotspots. This allows the agent to practice difficult sections without restarting from the beginning.

### State Machine Diagram

```
                                    ┌─────────────────┐
                                    │                 │
                                    │    RUNNING      │◄──────────────────────┐
                                    │                 │                       │
                                    └────────┬────────┘                       │
                                             │                                │
                    ┌────────────────────────┼────────────────────────┐       │
                    │                        │                        │       │
                    ▼                        ▼                        ▼       │
        ┌───────────────────┐    ┌───────────────────┐    ┌──────────────┐    │
        │                   │    │                   │    │              │    │
        │ APPROACHING_      │    │  CHECKPOINT_DUE   │    │    DEAD      │    │
        │ HOTSPOT           │    │  (time-based)     │    │              │    │
        │                   │    │                   │    └──────┬───────┘    │
        └─────────┬─────────┘    └─────────┬─────────┘           │            │
                  │                        │                     │            │
                  │                        │           ┌─────────┴─────────┐  │
                  ▼                        ▼           ▼                   ▼  │
        ┌───────────────────┐    ┌───────────────────────────┐   ┌─────────────────┐
        │                   │    │                           │   │                 │
        │  SAVE_SNAPSHOT    │───►│      SNAPSHOT_SAVED       │   │  EVALUATE_      │
        │  (near hotspot)   │    │                           │   │  RESTORE        │
        │                   │    └─────────────┬─────────────┘   │                 │
        └───────────────────┘                  │                 └────────┬────────┘
                                               │                          │
                                               │              ┌───────────┴───────────┐
                                               │              │                       │
                                               │              ▼                       ▼
                                               │    ┌─────────────────┐     ┌─────────────────┐
                                               │    │                 │     │                 │
                                               │    │   RESTORING     │     │   GIVE_UP       │
                                               │    │                 │     │                 │
                                               │    └────────┬────────┘     └────────┬────────┘
                                               │             │                       │
                                               │             │ (success)             │
                                               └─────────────┴───────────────────────┘
                                                             │
                                                             ▼
                                                      (back to RUNNING
                                                       or episode ends)
```

### State Descriptions

| State | Description | Transitions To |
|-------|-------------|----------------|
| **RUNNING** | Normal gameplay | APPROACHING_HOTSPOT, CHECKPOINT_DUE, DEAD |
| **APPROACHING_HOTSPOT** | Near a known death hotspot | SAVE_SNAPSHOT, DEAD, RUNNING |
| **CHECKPOINT_DUE** | Time-based checkpoint triggered | SAVE_SNAPSHOT, DEAD |
| **SAVE_SNAPSHOT** | Saving emulator state | SNAPSHOT_SAVED |
| **SNAPSHOT_SAVED** | Save completed | RUNNING |
| **DEAD** | Death detected | EVALUATE_RESTORE |
| **EVALUATE_RESTORE** | Deciding whether to restore | RESTORING, GIVE_UP |
| **RESTORING** | Loading snapshot | RUNNING |
| **GIVE_UP** | Too many failed restores | Episode ends |

### Snapshot Strategy

**When to Save:**
1. **Hotspot-based**: Save when approaching a death hotspot (100 pixels before), at optimal position (50 pixels before the hotspot)
2. **Time-based**: Fallback saves every 500 game ticks in unexplored areas

**When to Restore:**
1. On death, evaluate if restoration is beneficial
2. Use hotspot data to suggest optimal restore position
3. Track progress - if restored 3+ times without progress, give up

**When to Give Up:**
1. Max restores without progress exceeded (default: 3)
2. No snapshots available
3. Restore position too close to death position

### Usage

The snapshot system is integrated as an environment wrapper:

```python
from mario_rl.environment.snapshot_wrapper import create_snapshot_mario_env

# Create environment with automatic snapshot handling
env = create_snapshot_mario_env(
    level=(1, 1),
    hotspot_path=Path("death_hotspots.json"),
    checkpoint_interval=500,
    max_restores_without_progress=3,
)

# Use like a normal Gym environment
obs, info = env.reset()
for _ in range(1000):
    action = agent.act(obs)
    obs, reward, done, truncated, info = env.step(action)
    # Snapshots are handled automatically!
    
    # Check what happened
    if info.get("snapshot_saved"):
        print(f"Saved snapshot at x={info['x_pos']}")
    if info.get("snapshot_restored"):
        print(f"Restored from snapshot!")

# Access statistics
print(f"Total saves: {env.total_saves}")
print(f"Total restores: {env.total_restores}")
```

### Monitoring

Snapshot metrics are displayed in:
- **Text UI**: `💾=12 ⏮=8(2/3)` shows saves, restores, and stuck/max counters
- **Dashboard**: "Saves" and "Restores" columns in worker table

## 🔄 Distributed Training Architecture

The distributed training system uses **gradient sharing** (A3C-style) where workers compute gradients locally and send them to a central coordinator.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           MAIN PROCESS                                  │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │              Shared Memory Gradient Pool                        │   │
│   │              (workers → coordinator, ~2MB per packet)           │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
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
| **EnvRunner** | `mario_rl/core/env_runner.py` | Game-agnostic environment step collection |
| **MetricLogger** | `mario_rl/metrics/logger.py` | Per-worker metrics tracking with ZMQ publish |
| **MetricAggregator** | `mario_rl/metrics/aggregator.py` | Combines metrics from all workers |
| **MetricCollector** | `mario_rl/metrics/collectors/` | Collector pattern for metrics extraction |
| **DeathHotspotAggregate** | `mario_rl/metrics/levels.py` | Death position aggregation for curriculum learning |
| **SnapshotStateMachine** | `mario_rl/training/snapshot_state_machine.py` | State machine for save/restore decisions |
| **SnapshotHandler** | `mario_rl/training/snapshot_handler.py` | High-level snapshot coordination |
| **SnapshotMarioEnvironment** | `mario_rl/environment/snapshot_wrapper.py` | Environment wrapper with auto snapshot |
| **TrainingUI** | `mario_rl/training/training_ui.py` | ncurses monitoring dashboard |
| **EventPublisher** | `mario_rl/distributed/events.py` | ZMQ-based event publishing |

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
│   ├── models/                    # Model definitions
│   │   ├── base.py                # Model protocol
│   │   ├── ddqn.py                # DoubleDQN model
│   │   └── dreamer.py             # Dreamer world model
│   │
│   ├── learners/                  # Learning algorithms
│   │   ├── base.py                # Learner protocol
│   │   ├── ddqn.py                # DDQN learner (loss, targets)
│   │   └── dreamer.py             # Dreamer learner (world + behavior)
│   │
│   ├── distributed/               # Distributed training
│   │   ├── events.py              # ZMQ event pub/sub system
│   │   ├── worker.py              # Base gradient worker
│   │   ├── coordinator.py         # Base gradient coordinator
│   │   ├── training_worker.py     # Full worker with env + buffer
│   │   ├── training_coordinator.py # Full coordinator with scheduling
│   │   ├── shm_gradient_pool.py   # Shared memory gradient buffers
│   │   └── shm_heartbeat.py       # Worker health monitoring
│   │
│   ├── metrics/                   # Unified metrics system
│   │   ├── schema.py              # Metric definitions (MetricType, MetricDef)
│   │   ├── logger.py              # MetricLogger (track, save CSV, publish)
│   │   ├── aggregator.py          # MetricAggregator (combine workers)
│   │   ├── levels.py              # Per-level stats + DeathHotspotAggregate
│   │   └── collectors/            # Collector pattern implementations
│   │       ├── protocol.py        # MetricCollector protocol
│   │       ├── mario.py           # MarioCollector (game metrics)
│   │       ├── ddqn.py            # DDQNCollector (training metrics)
│   │       ├── dreamer.py         # DreamerCollector (world model metrics)
│   │       ├── system.py          # SystemCollector (steps, episodes)
│   │       ├── composite.py       # CompositeCollector (combines collectors)
│   │       └── coordinator.py     # Coordinator-side collectors
│   │
│   ├── core/                      # Core components
│   │   ├── replay_buffer.py       # Unified buffer (N-step + PER)
│   │   ├── env_runner.py          # Game-agnostic env step collection
│   │   └── config.py              # Configuration dataclasses
│   │
│   ├── environment/               # Environment wrappers
│   │   ├── factory.py             # Mario environment creation
│   │   └── snapshot_wrapper.py    # Snapshot-enabled environment wrapper
│   │
│   ├── training/                  # Training utilities
│   │   ├── training_ui.py         # ncurses monitoring dashboard
│   │   ├── shared_gradient_tensor.py  # Shared memory implementation
│   │   ├── snapshot.py            # SnapshotManager for save/restore
│   │   ├── snapshot_state_machine.py  # State machine for snapshot decisions
│   │   └── snapshot_handler.py    # High-level snapshot handler
│   │
│   ├── buffers/                   # Replay buffers
│   │   └── nstep.py               # N-step transition buffer
│   │
│   └── agent/                     # Neural network architectures
│       └── neural.py              # FrameNet, DuelingDQNNet
│
├── scripts/                       # Command-line scripts
│   ├── train_distributed.py       # Main distributed training script
│   ├── train_ddqn_distributed.py  # Legacy distributed training
│   └── watch.py                   # Watch agent play
│
├── tests/                         # Comprehensive test suite (~500 tests)
│   ├── models/                    # Model tests
│   ├── learners/                  # Learner tests
│   ├── distributed/               # Distributed component tests
│   ├── metrics/                   # Metrics system tests
│   │   ├── collectors/            # Collector tests
│   │   ├── test_logger.py         # MetricLogger tests
│   │   ├── test_aggregator.py     # MetricAggregator tests
│   │   └── test_levels.py         # LevelStats + DeathHotspotAggregate tests
│   ├── training/                  # Training component tests
│   │   ├── test_snapshot_state_machine.py  # State machine tests
│   │   └── test_snapshot_handler.py        # Handler tests
│   └── core/                      # Core component tests
│
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
1. **Collect** 64 steps from environment (via EnvRunner)
2. **Extract** metrics via collectors (MarioCollector, DDQNCollector, etc.)
3. **Sample** batch from local replay buffer
4. **Compute** gradients via backprop
5. **Send** gradients to coordinator via shared memory
6. **Publish** metrics to ZMQ (every 5 gradient sends)
7. **Send** death positions for hotspot aggregation
8. **Sync** weights from coordinator's file
9. **Reload** death hotspots periodically (every 30s)

The coordinator:
1. **Polls** gradients from all workers (SharedGradientPool)
2. **Aggregates** gradients (averaging)
3. **Applies** optimizer step with gradient clipping
4. **Updates** learning rate (cosine annealing)
5. **Saves** weights for workers to sync
6. **Updates** target network (soft update with τ)
7. **Publishes** learner metrics to ZMQ

The main process:
1. **Receives** events via ZMQ subscriber
2. **Aggregates** worker metrics (MetricAggregator)
3. **Aggregates** death positions (DeathHotspotAggregate)
4. **Updates** Training UI
5. **Saves** death hotspots to disk (every 60s)

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

## 🔌 Inter-Process Communication

### ZMQ Event System

Workers and coordinator communicate metrics and events via ZeroMQ:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ZMQ PUB/SUB TOPOLOGY                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   WORKER 0          WORKER 1          COORDINATOR                           │
│   ════════          ════════          ═══════════                           │
│                                                                             │
│   EventPublisher    EventPublisher    EventPublisher                        │
│   (ZMQ PUSH)        (ZMQ PUSH)        (ZMQ PUSH)                            │
│        │                 │                 │                                │
│        └─────────────────┼─────────────────┘                                │
│                          │                                                  │
│                          ▼                                                  │
│                  ┌───────────────┐                                          │
│                  │  ZMQ PULL     │                                          │
│                  │  (main proc)  │                                          │
│                  └───────┬───────┘                                          │
│                          │                                                  │
│                          ▼                                                  │
│                  ┌───────────────┐      ┌───────────────┐                   │
│                  │ Aggregator    │─────▶│  Training UI  │                   │
│                  └───────────────┘      └───────────────┘                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Event Types

| Event Type | Source | Data |
|------------|--------|------|
| `metrics` | Workers/Coordinator | Metric snapshot (counters, gauges, rolling) |
| `log` | Any | Log message |
| `death_positions` | Workers | Level ID + death x positions |
| `worker_status` | Workers | Steps, episodes, ε, best_x |
| `learner_status` | Coordinator | Loss, q_mean, lr, updates |

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

### Log Files & Checkpoints

All runs save to `checkpoints/<timestamp>/`:

| File | Description |
|------|-------------|
| `weights.pt` | Latest network weights (for workers to sync) |
| `checkpoint.pt` | Full training state (model, optimizer, step count) |
| `worker_N.csv` | Per-worker metrics (written on-the-fly) |
| `coordinator.csv` | Coordinator metrics |
| `death_hotspots.json` | Aggregated death positions per level |
| `training.log` | Full training log |

### Checkpoint Contents

```python
# checkpoint.pt contains:
{
    "model_state_dict": {...},      # Network weights
    "optimizer_state_dict": {...},   # Optimizer state
    "global_step": 100000,           # Training step
    "weight_version": 1500,          # Weight update count
    "lr_scheduler_state": {...},     # LR scheduler state
}
```

### Resuming Training

```bash
# Resume from checkpoint
uv run python scripts/train_distributed.py \
  --resume checkpoints/2025-01-08_123456/checkpoint.pt
```

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
| `tests/distributed/` | ~90 | Workers, coordinators, shared memory, events |
| `tests/metrics/` | ~200 | Logger, aggregator, collectors, levels, schema |
| `tests/training/` | ~60 | Snapshot state machine, snapshot handler |
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

