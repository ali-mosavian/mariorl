"""
Simple synchronous DDQN implementation with modern stability techniques.

Single-process, no distributed training.
Implements Double DQN with Dueling architecture and prioritized replay.

    ┌──────────────────────────────────────────────────────────────┐
    │                     Training Loop                             │
    │                                                               │
    │   ┌─────────┐   ┌──────────┐   ┌────────┐   ┌────────────┐  │
    │   │ Collect │ → │  Store   │ → │ Sample │ → │   Update   │  │
    │   │ Experience │  │ in Buffer│   │ Batch  │   │   Network  │  │
    │   └─────────┘   └──────────┘   └────────┘   └────────────┘  │
    │       ↑                                           │          │
    │       └───────────────────────────────────────────┘          │
    └──────────────────────────────────────────────────────────────┘

Key differences from PPO:
- Off-policy: learns from replay buffer (can reuse old experiences)
- Value-based: learns Q(s,a) directly, no policy network
- Epsilon-greedy exploration (not entropy bonus)
- More sample efficient but can be less stable

Stability Techniques Applied:
- GELU activation (smoother than ReLU)
- LayerNorm (stabilizes hidden representations)
- Orthogonal initialization
- Dropout (regularization)
- Gradient clipping
- Huber loss (robust to outliers)
- Soft target updates (smoother than hard sync)
- Reward scaling (normalized rewards)

Usage:
    uv run mario-train-ddqn --level 1,1 --total-steps 1000000
"""

import time
import argparse
from typing import Tuple
from collections import deque

import torch
import numpy as np
from torch.optim import AdamW

from mario_rl.agent.ddqn_net import DoubleDQN
from mario_rl.environment.env_factory import LevelType
from mario_rl.environment.env_factory import make_mario_env


def parse_level(level_str: str) -> LevelType:
    """Parse level string like '1,1' or 'random'."""
    if level_str == "random":
        return "random"
    if level_str == "sequential":
        return "sequential"
    parts = level_str.split(",")
    if len(parts) == 2:
        world = int(parts[0])
        stage = int(parts[1])
        if 1 <= world <= 8 and 1 <= stage <= 4:
            return (world, stage)  # type: ignore[return-value]
    raise ValueError(f"Invalid level: {level_str}")


class ReplayBuffer:
    """
    Simple replay buffer with uniform sampling.

    Stores transitions and samples random batches for training.
    Uses numpy arrays for efficiency.
    """

    def __init__(self, capacity: int, obs_shape: Tuple[int, ...]):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.pos = 0
        self.size = 0

        # Pre-allocate storage
        self.states = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition to the buffer."""
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = float(done)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random batch of transitions."""
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self) -> int:
        return self.size


def linear_schedule(start: float, end: float, progress: float) -> float:
    """Linear interpolation between start and end based on progress [0, 1]."""
    return start + (end - start) * progress


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple synchronous DDQN")
    parser.add_argument("--level", "-l", type=str, default="1,1", help="Level (e.g., '1,1' or 'random')")
    parser.add_argument("--total-steps", type=int, default=1_000_000, help="Total training steps")
    parser.add_argument("--buffer-size", type=int, default=100_000, help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update coefficient")
    parser.add_argument(
        "--target-update-freq", type=int, default=1, help="Steps between target updates (1 = soft update every step)"
    )
    parser.add_argument("--train-freq", type=int, default=4, help="Steps between training updates")
    parser.add_argument("--learning-starts", type=int, default=10_000, help="Steps before training starts")
    parser.add_argument("--eps-start", type=float, default=1.0, help="Starting epsilon")
    parser.add_argument("--eps-end", type=float, default=0.05, help="Final epsilon")
    parser.add_argument("--eps-fraction", type=float, default=0.1, help="Fraction of training for epsilon decay")
    parser.add_argument("--max-grad-norm", type=float, default=10.0, help="Max gradient norm")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 regularization")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    args = parser.parse_args()

    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Create environment
    level = parse_level(args.level)
    env = make_mario_env(level=level, render_mode=None)
    obs_shape = env.observation_space.shape  # (4, 64, 64)
    n_actions = env.action_space.n
    print(f"Observation shape: {obs_shape}, Actions: {n_actions}")

    # Create network
    model = DoubleDQN(
        input_shape=obs_shape,
        num_actions=n_actions,
        feature_dim=512,
        hidden_dim=256,
        dropout=args.dropout,
    ).to(device)

    # Use AdamW for better regularization
    optimizer = AdamW(
        model.online.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    print(f"Model parameters: {sum(p.numel() for p in model.online.parameters()):,}")

    # Create replay buffer
    buffer = ReplayBuffer(args.buffer_size, obs_shape)

    # Training state
    obs, _ = env.reset()
    obs = np.array(obs, dtype=np.float32)
    total_steps = 0
    episode_count = 0
    episode_reward = 0.0
    episode_length = 0
    best_x = 0
    current_x = 0
    start_time = time.time()

    # Episode tracking
    recent_rewards: deque[float] = deque(maxlen=100)
    recent_x_at_death: deque[int] = deque(maxlen=20)
    recent_times_to_flag: deque[int] = deque(maxlen=20)
    recent_speed: deque[float] = deque(maxlen=20)
    total_deaths = 0
    total_flags = 0
    episode_start_time = 400

    # Training metrics
    recent_losses: deque[float] = deque(maxlen=100)
    recent_q_values: deque[float] = deque(maxlen=100)

    print("=" * 80)
    print("Starting DDQN training...")
    print(f"  Level: {args.level}")
    print(f"  Device: {device}")
    print(f"  LR: {args.lr}, Buffer: {args.buffer_size:,}, Batch: {args.batch_size}")
    print(f"  Epsilon: {args.eps_start} → {args.eps_end} over {int(args.eps_fraction * args.total_steps):,} steps")
    print(f"  Tau: {args.tau}, Train freq: {args.train_freq}, Learning starts: {args.learning_starts:,}")
    print("=" * 80)

    # Debug: test reward function
    print("Testing reward function (10 random steps):")
    test_obs, _ = env.reset()
    for i in range(10):
        action = env.action_space.sample()
        _, reward, terminated, truncated, info = env.step(action)
        x_pos = info.get("x_pos", 0)
        print(f"  Step {i}: action={action}, reward={reward:.4f}, x_pos={x_pos}")
        if terminated or truncated:
            test_obs, _ = env.reset()
    obs, _ = env.reset()
    obs = np.array(obs, dtype=np.float32)
    print("=" * 80)

    while total_steps < args.total_steps:
        # Calculate epsilon
        epsilon_progress = min(1.0, total_steps / (args.eps_fraction * args.total_steps))
        epsilon = linear_schedule(args.eps_start, args.eps_end, epsilon_progress)

        # Select action
        model.eval()
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
            action = model.get_action(obs_t, epsilon).item()

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_obs = np.array(next_obs, dtype=np.float32)

        # Store transition
        buffer.add(obs, action, reward, next_obs, done)

        # Track episode stats
        episode_reward += reward
        episode_length += 1
        x_pos = info.get("x_pos", 0)
        current_x = x_pos
        if x_pos > best_x:
            best_x = x_pos

        if done:
            recent_rewards.append(episode_reward)

            # Track deaths and flags
            flag_get = info.get("flag_get", False)
            is_dead = info.get("is_dead", False) or info.get("is_dying", False)
            game_time = info.get("time", 0)
            time_elapsed = episode_start_time - game_time

            if time_elapsed > 0:
                speed = x_pos / time_elapsed
                recent_speed.append(speed)

            if flag_get:
                total_flags += 1
                recent_times_to_flag.append(game_time)
            elif is_dead:
                total_deaths += 1
                recent_x_at_death.append(x_pos)

            episode_count += 1
            episode_reward = 0.0
            episode_length = 0
            next_obs, _ = env.reset()
            next_obs = np.array(next_obs, dtype=np.float32)

        obs = next_obs
        total_steps += 1

        # Training
        if total_steps >= args.learning_starts and total_steps % args.train_freq == 0:
            model.train()

            # Sample batch
            states, actions, rewards, next_states, dones = buffer.sample(args.batch_size)

            # Convert to tensors
            states_t = torch.from_numpy(states).to(device)
            actions_t = torch.from_numpy(actions).to(device)
            rewards_t = torch.from_numpy(rewards).to(device)
            next_states_t = torch.from_numpy(next_states).to(device)
            dones_t = torch.from_numpy(dones).to(device)

            # Compute loss
            loss, info = model.compute_loss(
                states_t,
                actions_t,
                rewards_t,
                next_states_t,
                dones_t,
                gamma=args.gamma,
            )

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.online.parameters(), args.max_grad_norm)
            optimizer.step()

            # Track metrics
            recent_losses.append(info["loss"])
            recent_q_values.append(info["q_mean"])

            # Update target network
            if total_steps % args.target_update_freq == 0:
                model.soft_update(args.tau)

        # Logging (every 1000 steps)
        if total_steps % 1000 == 0:
            elapsed = time.time() - start_time
            steps_per_sec = total_steps / elapsed if elapsed > 0 else 0
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
            avg_loss = np.mean(recent_losses) if recent_losses else 0.0
            avg_q = np.mean(recent_q_values) if recent_q_values else 0.0
            avg_x_at_death = np.mean(recent_x_at_death) if recent_x_at_death else 0.0
            avg_time_to_flag = np.mean(recent_times_to_flag) if recent_times_to_flag else 0.0
            avg_speed = np.mean(recent_speed) if recent_speed else 0.0

            print(
                f"Step {total_steps:>7,} | "
                f"Ep: {episode_count:>4} | "
                f"Loss: {avg_loss:>7.4f} | "
                f"Q: {avg_q:>6.2f} | "
                f"ε: {epsilon:>5.3f} | "
                f"Buf: {len(buffer):>6,}"
            )
            print(
                f"         Avg R: {avg_reward:>6.1f} | "
                f"X: {current_x:>4} | "
                f"Best: {best_x:>4} | "
                f"Speed: {avg_speed:>5.2f} x/t | "
                f"SPS: {steps_per_sec:>4.0f}"
            )
            print(
                f"         Deaths: {total_deaths:>4} (X: {avg_x_at_death:>4.0f}) | "
                f"Flags: {total_flags:>3} (T: {avg_time_to_flag:>3.0f})"
            )
            print("-" * 85)

    print("=" * 80)
    print("Training complete!")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Episodes: {episode_count}")
    print(f"  Best X position: {best_x}")
    print(f"  Total deaths: {total_deaths}")
    print(f"  Total flags: {total_flags}")
    if total_flags > 0:
        print(f"  Flag rate: {total_flags / episode_count * 100:.1f}%")
    print("=" * 80)

    try:
        env.close()
    except Exception:
        pass  # Environment might already be closed


if __name__ == "__main__":
    main()
