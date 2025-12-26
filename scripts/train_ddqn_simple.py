"""
Enhanced DDQN implementation with advanced techniques.

Single-process training with:
- Prioritized Experience Replay (PER)
- N-step returns
- Learning rate scheduling
- Dueling architecture
- Double DQN

    ┌──────────────────────────────────────────────────────────────────────────┐
    │                          Training Loop                                    │
    │                                                                           │
    │   ┌─────────┐   ┌──────────┐   ┌────────────┐   ┌────────────────────┐  │
    │   │ Collect │ → │  Store   │ → │  Priority  │ → │      Update        │  │
    │   │ N-steps │   │ in PER   │   │  Sampling  │   │ Network + Priority │  │
    │   └─────────┘   └──────────┘   └────────────┘   └────────────────────┘  │
    │       ↑                                                 │                │
    │       └─────────────────────────────────────────────────┘                │
    └──────────────────────────────────────────────────────────────────────────┘

Improvements over basic DDQN:
1. PER: Focus on surprising transitions (high TD error)
2. N-step: Better credit assignment for delayed rewards
3. LR scheduling: Start high, decay for fine-tuning
4. Lower epsilon: Less random actions after learning basics

Usage:
    uv run mario-train-ddqn --level 1,1 --total-steps 2000000

    # With N-step returns (better for Mario)
    uv run mario-train-ddqn --level 1,1 --n-step 3 --total-steps 2000000

    # More aggressive settings
    uv run mario-train-ddqn --level 1,1 --n-step 5 --eps-end 0.01 --total-steps 3000000
"""

import time
import argparse
from typing import Tuple
from collections import deque

import torch
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

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


class SumTree:
    """
    Binary sum tree for O(log n) priority sampling.

    Each leaf stores a priority. Internal nodes store sum of children.
    Allows efficient sampling proportional to priority and O(log n) updates.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_pointer = 0

    def _propagate(self, idx: int, change: float) -> None:
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Find leaf index for given cumulative sum s."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        """Total priority sum."""
        return float(self.tree[0])

    def add(self, priority: float) -> int:
        """Add priority and return the data index."""
        self.update(self.data_pointer, priority)
        data_idx = self.data_pointer
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        return data_idx

    def update(self, data_idx: int, priority: float) -> None:
        """Update priority at data index."""
        idx = data_idx + self.capacity - 1
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float]:
        """Get data index and priority for cumulative sum s."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return data_idx, self.tree[idx]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.

    Samples transitions proportional to their TD error priority.
    Uses importance sampling weights to correct for bias.

    Priority formula: p_i = (|TD_error| + ε)^α
    Sampling probability: P(i) = p_i / Σp_j
    Importance weight: w_i = (N * P(i))^(-β) / max(w)
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: Tuple[int, ...],
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        eps: float = 1e-6,
    ):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.alpha = alpha  # Priority exponent (0 = uniform, 1 = full priority)
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta = beta_start
        self.eps = eps  # Small constant to prevent zero priority

        self.tree = SumTree(capacity)
        self.max_priority = 1.0

        # Pre-allocate storage
        self.states = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self.size = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add transition with max priority (will be updated after training)."""
        idx = self.tree.add(self.max_priority**self.alpha)

        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)

        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample batch with priority-based sampling.

        Returns:
            states, actions, rewards, next_states, dones, indices, weights
        """
        indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)

        # Divide priority range into segments for stratified sampling
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, priority = self.tree.get(s)
            indices[i] = idx
            priorities[i] = priority

        # Compute importance sampling weights
        sampling_probs = priorities / self.tree.total()
        weights = (self.size * sampling_probs) ** (-self.beta)
        weights = weights / weights.max()  # Normalize

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            indices,
            weights.astype(np.float32),
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities based on TD errors."""
        priorities = (np.abs(td_errors) + self.eps) ** self.alpha
        for idx, priority in zip(indices, priorities, strict=True):
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority ** (1 / self.alpha))

    def update_beta(self, progress: float) -> None:
        """Anneal beta from beta_start to beta_end."""
        self.beta = self.beta_start + (self.beta_end - self.beta_start) * progress

    def __len__(self) -> int:
        return self.size


class NStepBuffer:
    """
    N-step return buffer for computing multi-step TD targets.

    Stores the last n transitions and computes:
    R_n = r_t + γ*r_{t+1} + γ²*r_{t+2} + ... + γⁿ⁻¹*r_{t+n-1}

    This provides better credit assignment for delayed rewards.
    """

    def __init__(self, n_step: int, gamma: float):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer: deque = deque(maxlen=n_step)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Tuple[np.ndarray, int, float, np.ndarray, bool] | None:
        """
        Add transition and return n-step transition if ready.

        Returns None until n transitions are collected, then returns
        the n-step aggregated transition.
        """
        self.buffer.append((state, action, reward, next_state, done))

        # Not enough transitions yet
        if len(self.buffer) < self.n_step:
            return None

        # Compute n-step return
        n_step_reward = 0.0
        for i, (_, _, r, _, d) in enumerate(self.buffer):
            n_step_reward += (self.gamma**i) * r
            if d:
                # Episode ended, use this as the final transition
                return (
                    self.buffer[0][0],  # Initial state
                    self.buffer[0][1],  # Initial action
                    n_step_reward,
                    self.buffer[i][3],  # State after episode end
                    True,
                )

        # Return n-step transition
        return (
            self.buffer[0][0],  # Initial state
            self.buffer[0][1],  # Initial action
            n_step_reward,
            self.buffer[-1][3],  # State after n steps
            self.buffer[-1][4],  # Done flag after n steps
        )

    def flush(self) -> list:
        """Flush remaining transitions at episode end."""
        transitions = []
        while len(self.buffer) > 0:
            n_step_reward = 0.0
            last_idx = len(self.buffer) - 1
            for i, (_, _, r, _, d) in enumerate(self.buffer):
                n_step_reward += (self.gamma**i) * r
                if d:
                    last_idx = i
                    break

            transitions.append(
                (
                    self.buffer[0][0],
                    self.buffer[0][1],
                    n_step_reward,
                    self.buffer[last_idx][3],
                    self.buffer[last_idx][4],
                )
            )
            self.buffer.popleft()

        return transitions

    def reset(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()


def linear_schedule(start: float, end: float, progress: float) -> float:
    """Linear interpolation between start and end based on progress [0, 1]."""
    return start + (end - start) * min(1.0, progress)


def main() -> None:
    parser = argparse.ArgumentParser(description="Enhanced DDQN with PER and N-step")
    parser.add_argument("--level", "-l", type=str, default="1,1", help="Level (e.g., '1,1' or 'random')")
    parser.add_argument("--total-steps", type=int, default=2_000_000, help="Total training steps")
    parser.add_argument("--buffer-size", type=int, default=100_000, help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Initial learning rate")
    parser.add_argument("--lr-end", type=float, default=1e-5, help="Final learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update coefficient")
    parser.add_argument("--target-update-freq", type=int, default=1, help="Steps between target updates")
    parser.add_argument("--train-freq", type=int, default=4, help="Steps between training updates")
    parser.add_argument("--learning-starts", type=int, default=10_000, help="Steps before training starts")
    # Epsilon schedule
    parser.add_argument("--eps-start", type=float, default=1.0, help="Starting epsilon")
    parser.add_argument("--eps-end", type=float, default=0.01, help="Final epsilon (lower for exploitation)")
    parser.add_argument("--eps-fraction", type=float, default=0.3, help="Fraction of training for epsilon decay")
    # N-step returns
    parser.add_argument("--n-step", type=int, default=3, help="N-step returns (1 = standard TD)")
    # PER parameters
    parser.add_argument("--per-alpha", type=float, default=0.6, help="PER priority exponent")
    parser.add_argument("--per-beta-start", type=float, default=0.4, help="PER importance sampling start")
    parser.add_argument("--per-beta-end", type=float, default=1.0, help="PER importance sampling end")
    # Regularization
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

    # Use AdamW with learning rate scheduling
    optimizer = AdamW(
        model.online.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Cosine annealing LR scheduler
    total_training_steps = (args.total_steps - args.learning_starts) // args.train_freq
    scheduler = CosineAnnealingLR(optimizer, T_max=total_training_steps, eta_min=args.lr_end)

    print(f"Model parameters: {sum(p.numel() for p in model.online.parameters()):,}")

    # Create PER buffer
    buffer = PrioritizedReplayBuffer(
        capacity=args.buffer_size,
        obs_shape=obs_shape,
        alpha=args.per_alpha,
        beta_start=args.per_beta_start,
        beta_end=args.per_beta_end,
    )

    # N-step buffer
    n_step_buffer = NStepBuffer(n_step=args.n_step, gamma=args.gamma)
    # Compute n-step gamma for TD target
    n_step_gamma = args.gamma**args.n_step

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
    recent_td_errors: deque[float] = deque(maxlen=100)

    print("=" * 90)
    print("Starting Enhanced DDQN training...")
    print(f"  Level: {args.level}")
    print(f"  Device: {device}")
    print(f"  LR: {args.lr} → {args.lr_end} (cosine), Buffer: {args.buffer_size:,}, Batch: {args.batch_size}")
    print(f"  Epsilon: {args.eps_start} → {args.eps_end} over {int(args.eps_fraction * args.total_steps):,} steps")
    print(f"  N-step: {args.n_step}, Gamma: {args.gamma}, N-step gamma: {n_step_gamma:.4f}")
    print(f"  PER: α={args.per_alpha}, β={args.per_beta_start}→{args.per_beta_end}")
    print(f"  Tau: {args.tau}, Train freq: {args.train_freq}, Learning starts: {args.learning_starts:,}")
    print("=" * 90)

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
    print("=" * 90)

    while total_steps < args.total_steps:
        # Calculate epsilon (longer decay)
        epsilon_progress = total_steps / (args.eps_fraction * args.total_steps)
        epsilon = linear_schedule(args.eps_start, args.eps_end, epsilon_progress)

        # Update PER beta
        training_progress = max(0, (total_steps - args.learning_starts)) / (args.total_steps - args.learning_starts)
        buffer.update_beta(training_progress)

        # Select action
        model.eval()
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
            action = model.get_action(obs_t, epsilon).item()

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_obs = np.array(next_obs, dtype=np.float32)

        # Add to n-step buffer
        n_step_transition = n_step_buffer.add(obs, action, reward, next_obs, done)

        # Store n-step transition in PER buffer
        if n_step_transition is not None:
            buffer.add(*n_step_transition)

        # Track episode stats
        episode_reward += reward
        episode_length += 1
        x_pos = info.get("x_pos", 0)
        current_x = x_pos
        if x_pos > best_x:
            best_x = x_pos

        if done:
            # Flush remaining n-step transitions
            for transition in n_step_buffer.flush():
                buffer.add(*transition)

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

            # Sample batch with priorities
            states, actions, rewards, next_states, dones, indices, weights = buffer.sample(args.batch_size)

            # Convert to tensors
            states_t = torch.from_numpy(states).to(device)
            actions_t = torch.from_numpy(actions).to(device)
            rewards_t = torch.from_numpy(rewards).to(device)
            next_states_t = torch.from_numpy(next_states).to(device)
            dones_t = torch.from_numpy(dones).to(device)
            weights_t = torch.from_numpy(weights).to(device)

            # Compute loss with n-step gamma
            loss, loss_info, td_errors = compute_weighted_loss(
                model,
                states_t,
                actions_t,
                rewards_t,
                next_states_t,
                dones_t,
                weights_t,
                gamma=n_step_gamma,
            )

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.online.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            # Update priorities in buffer
            buffer.update_priorities(indices, td_errors.cpu().numpy())

            # Track metrics
            recent_losses.append(loss_info["loss"])
            recent_q_values.append(loss_info["q_mean"])
            recent_td_errors.append(loss_info["td_error_mean"])

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
            avg_td = np.mean(recent_td_errors) if recent_td_errors else 0.0
            avg_x_at_death = np.mean(recent_x_at_death) if recent_x_at_death else 0.0
            avg_time_to_flag = np.mean(recent_times_to_flag) if recent_times_to_flag else 0.0
            avg_speed = np.mean(recent_speed) if recent_speed else 0.0
            current_lr = scheduler.get_last_lr()[0]

            print(
                f"Step {total_steps:>7,} | "
                f"Ep: {episode_count:>4} | "
                f"Loss: {avg_loss:>6.3f} | "
                f"Q: {avg_q:>6.2f} | "
                f"TD: {avg_td:>5.2f} | "
                f"ε: {epsilon:>5.3f} | "
                f"β: {buffer.beta:>4.2f}"
            )
            print(
                f"         Avg R: {avg_reward:>6.1f} | "
                f"X: {current_x:>4} | "
                f"Best: {best_x:>4} | "
                f"Speed: {avg_speed:>5.2f} x/t | "
                f"LR: {current_lr:.1e}"
            )
            print(
                f"         Deaths: {total_deaths:>4} (X: {avg_x_at_death:>4.0f}) | "
                f"Flags: {total_flags:>3} (T: {avg_time_to_flag:>3.0f}) | "
                f"SPS: {steps_per_sec:>4.0f}"
            )
            print("-" * 95)

    print("=" * 90)
    print("Training complete!")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Episodes: {episode_count}")
    print(f"  Best X position: {best_x}")
    print(f"  Total deaths: {total_deaths}")
    print(f"  Total flags: {total_flags}")
    if total_flags > 0:
        print(f"  Flag rate: {total_flags / episode_count * 100:.1f}%")
    print("=" * 90)

    try:
        env.close()
    except Exception:
        pass


def compute_weighted_loss(
    model: DoubleDQN,
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    weights: torch.Tensor,
    gamma: float,
) -> Tuple[torch.Tensor, dict, torch.Tensor]:
    """
    Compute weighted Double DQN loss with importance sampling.

    Returns:
        loss: Weighted Huber loss
        info: Dict with metrics
        td_errors: Per-sample TD errors for priority update
    """
    # Current Q-values for taken actions
    current_q = model.online(states)
    current_q_selected = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Double DQN target
    with torch.no_grad():
        # Select best actions using online network
        next_q_online = model.online(next_states)
        best_actions = next_q_online.argmax(dim=1)

        # Evaluate using target network
        next_q_target = model.target(next_states)
        next_q_selected = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)

        # TD target
        target_q = rewards + gamma * next_q_selected * (1.0 - dones)

    # Per-sample TD errors
    td_errors = (current_q_selected - target_q).abs().detach()

    # Weighted Huber loss
    element_wise_loss = torch.nn.functional.huber_loss(current_q_selected, target_q, reduction="none", delta=1.0)
    loss = (weights * element_wise_loss).mean()

    info = {
        "loss": loss.item(),
        "q_mean": current_q_selected.mean().item(),
        "q_max": current_q_selected.max().item(),
        "td_error_mean": td_errors.mean().item(),
        "target_q_mean": target_q.mean().item(),
    }

    return loss, info, td_errors


if __name__ == "__main__":
    main()
