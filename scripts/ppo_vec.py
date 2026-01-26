#!/usr/bin/env python
"""Vectorized PPO training script for Super Mario Bros.

Uses gymnasium's AsyncVectorEnv for parallel environment collection.
Implements Proximal Policy Optimization with:
- Actor-Critic network with shared CNN backbone
- GAE (Generalized Advantage Estimation)
- Clipped surrogate objective
- Value function clipping
- Entropy bonus for exploration

Logs metrics compatible with the Streamlit dashboard.

Usage:
    uv run python scripts/ppo_vec.py --envs 8 --steps 1000000
"""

import sys
import time
from pathlib import Path
from datetime import datetime
from collections import deque
from dataclasses import dataclass

import click
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mario_rl.agent.ddqn_net import DDQNBackbone, layer_init, set_skip_weight_init
from mario_rl.environment.factory import create_mario_env
from mario_rl.metrics.logger import MetricLogger
from mario_rl.metrics.schema import CoordinatorMetrics


def log(msg: str) -> None:
    """Print and flush immediately."""
    print(msg)
    sys.stdout.flush()


# =============================================================================
# PPO Actor-Critic Network (using DDQN Attention Backbone)
# =============================================================================


class PPONetwork(nn.Module):
    """Actor-Critic network with shared attention backbone for PPO.

    Uses the same DDQNBackbone as the DQN model:
    - 3 conv layers: 64×64 → 32×32 → 16×16 → 8×8
    - Self-attention over 8×8 grid (64 positions)
    - FC layers with LayerNorm
    - Optional action history input

    Then adds:
    - Policy head (actor): outputs action logits
    - Value head (critic): outputs state value estimate
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        num_actions: int,
        feature_dim: int = 512,
        dropout: float = 0.1,
        action_history_len: int = 0,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.action_history_len = action_history_len

        # Calculate action history dimension (flattened one-hot)
        action_history_dim = action_history_len * num_actions if action_history_len > 0 else 0

        # Shared attention backbone (same as DDQN)
        self.backbone = DDQNBackbone(
            input_shape=input_shape,
            feature_dim=feature_dim,
            dropout=dropout,
            action_history_dim=action_history_dim,
            embed_dim=32,
            num_heads=4,
        )

        # Policy head (actor)
        self.policy = nn.Sequential(
            layer_init(nn.Linear(feature_dim, 256)),
            nn.GELU(),
            layer_init(nn.Linear(256, num_actions), std=0.01),
        )

        # Value head (critic)
        self.value = nn.Sequential(
            layer_init(nn.Linear(feature_dim, 256)),
            nn.GELU(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )

    def forward(
        self, x: torch.Tensor, action_history: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning policy logits and value estimate.

        Args:
            x: Observations (B, C, H, W) as uint8 [0, 255]
            action_history: Optional action history (B, history_len, num_actions) one-hot

        Returns:
            policy_logits: (B, num_actions)
            value: (B,)
        """
        # Backbone handles normalization internally
        features = self.backbone(x, action_history)

        # Heads
        policy_logits = self.policy(features)
        value = self.value(features).squeeze(-1)

        return policy_logits, value

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: torch.Tensor | None = None,
        action_history: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log prob, entropy, and value for PPO.

        Args:
            x: Observations (B, C, H, W)
            action: Optional actions to evaluate (for training)
            action_history: Optional action history (B, history_len, num_actions) one-hot

        Returns:
            action: Sampled or provided action (B,)
            log_prob: Log probability of action (B,)
            entropy: Policy entropy (B,)
            value: Value estimate (B,)
        """
        policy_logits, value = self.forward(x, action_history)
        probs = F.softmax(policy_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value

    def get_attention_map(self) -> torch.Tensor | None:
        """Get attention map from backbone for visualization."""
        return self.backbone.get_attention_map()


# =============================================================================
# Rollout Buffer for PPO
# =============================================================================


@dataclass
class RolloutBuffer:
    """Buffer for storing rollout data for PPO updates.

    Stores complete trajectories for on-policy learning.
    """

    obs: np.ndarray  # (steps, envs, *obs_shape)
    actions: np.ndarray  # (steps, envs)
    log_probs: np.ndarray  # (steps, envs)
    rewards: np.ndarray  # (steps, envs)
    dones: np.ndarray  # (steps, envs)
    values: np.ndarray  # (steps, envs)
    action_history: np.ndarray | None = None  # (steps, envs, history_len, num_actions)

    # Computed after rollout
    advantages: np.ndarray | None = None  # (steps, envs)
    returns: np.ndarray | None = None  # (steps, envs)

    @classmethod
    def create(
        cls,
        num_steps: int,
        num_envs: int,
        obs_shape: tuple[int, ...],
        action_history_len: int = 0,
        num_actions: int = 0,
    ) -> "RolloutBuffer":
        """Create an empty rollout buffer."""
        action_history = None
        if action_history_len > 0:
            action_history = np.zeros(
                (num_steps, num_envs, action_history_len, num_actions), dtype=np.float32
            )
        return cls(
            obs=np.zeros((num_steps, num_envs, *obs_shape), dtype=np.uint8),
            actions=np.zeros((num_steps, num_envs), dtype=np.int64),
            log_probs=np.zeros((num_steps, num_envs), dtype=np.float32),
            rewards=np.zeros((num_steps, num_envs), dtype=np.float32),
            dones=np.zeros((num_steps, num_envs), dtype=np.float32),
            values=np.zeros((num_steps, num_envs), dtype=np.float32),
            action_history=action_history,
        )

    def compute_gae(self, next_value: np.ndarray, gamma: float, gae_lambda: float) -> None:
        """Compute Generalized Advantage Estimation.

        Args:
            next_value: Value estimate of final next state (num_envs,)
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        num_steps = self.rewards.shape[0]
        self.advantages = np.zeros_like(self.rewards)
        self.returns = np.zeros_like(self.rewards)

        last_gae = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_values = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            self.advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

        self.returns = self.advantages + self.values

    def get_batches(self, batch_size: int, device: str) -> list[dict[str, torch.Tensor]]:
        """Get shuffled minibatches for PPO update.

        Args:
            batch_size: Size of each minibatch
            device: Device to move tensors to

        Returns:
            List of batches, each containing obs, actions, log_probs, advantages, returns, values, action_history
        """
        num_steps, num_envs = self.rewards.shape
        total_size = num_steps * num_envs

        # Flatten all data
        obs_flat = self.obs.reshape(total_size, *self.obs.shape[2:])
        actions_flat = self.actions.reshape(total_size)
        log_probs_flat = self.log_probs.reshape(total_size)
        advantages_flat = self.advantages.reshape(total_size)
        returns_flat = self.returns.reshape(total_size)
        values_flat = self.values.reshape(total_size)

        # Flatten action history if present
        action_history_flat = None
        if self.action_history is not None:
            action_history_flat = self.action_history.reshape(total_size, *self.action_history.shape[2:])

        # Normalize advantages
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)

        # Shuffle indices
        indices = np.random.permutation(total_size)

        # Create batches
        batches = []
        for start in range(0, total_size, batch_size):
            end = min(start + batch_size, total_size)
            batch_indices = indices[start:end]

            batch = {
                "obs": torch.from_numpy(obs_flat[batch_indices]).to(device),
                "actions": torch.from_numpy(actions_flat[batch_indices]).to(device),
                "old_log_probs": torch.from_numpy(log_probs_flat[batch_indices]).to(device),
                "advantages": torch.from_numpy(advantages_flat[batch_indices]).to(device),
                "returns": torch.from_numpy(returns_flat[batch_indices]).to(device),
                "old_values": torch.from_numpy(values_flat[batch_indices]).to(device),
                "action_history": None,
            }
            if action_history_flat is not None:
                batch["action_history"] = torch.from_numpy(action_history_flat[batch_indices]).to(device)
            batches.append(batch)

        return batches


# =============================================================================
# PPO Metrics Schema
# =============================================================================


class PPOMetrics:
    """PPO-specific metrics for logging."""

    from mario_rl.metrics.schema import MetricDef, MetricType

    EPISODES = MetricDef("episodes", MetricType.COUNTER)
    STEPS = MetricDef("steps", MetricType.COUNTER)
    REWARD = MetricDef("reward", MetricType.ROLLING)
    EPISODE_LENGTH = MetricDef("episode_length", MetricType.ROLLING)
    STEPS_PER_SEC = MetricDef("steps_per_sec", MetricType.GAUGE)

    X_POS = MetricDef("x_pos", MetricType.GAUGE)
    BEST_X = MetricDef("best_x", MetricType.GAUGE)
    BEST_X_EVER = MetricDef("best_x_ever", MetricType.GAUGE)
    WORLD = MetricDef("world", MetricType.GAUGE)
    STAGE = MetricDef("stage", MetricType.GAUGE)
    DEATHS = MetricDef("deaths", MetricType.COUNTER)
    FLAGS = MetricDef("flags", MetricType.COUNTER)
    SPEED = MetricDef("speed", MetricType.ROLLING)

    # PPO-specific
    POLICY_LOSS = MetricDef("policy_loss", MetricType.ROLLING)
    VALUE_LOSS = MetricDef("value_loss", MetricType.ROLLING)
    ENTROPY = MetricDef("entropy", MetricType.ROLLING)
    LOSS = MetricDef("loss", MetricType.ROLLING)
    GRAD_NORM = MetricDef("grad_norm", MetricType.ROLLING)
    CLIP_FRACTION = MetricDef("clip_fraction", MetricType.GAUGE)
    EXPLAINED_VAR = MetricDef("explained_var", MetricType.GAUGE)
    ACTION_ENTROPY = MetricDef("action_entropy", MetricType.GAUGE)
    ACTION_DIST = MetricDef("action_dist", MetricType.TEXT)

    @classmethod
    def definitions(cls) -> list:
        return [
            cls.EPISODES,
            cls.STEPS,
            cls.REWARD,
            cls.EPISODE_LENGTH,
            cls.STEPS_PER_SEC,
            cls.X_POS,
            cls.BEST_X,
            cls.BEST_X_EVER,
            cls.WORLD,
            cls.STAGE,
            cls.DEATHS,
            cls.FLAGS,
            cls.SPEED,
            cls.POLICY_LOSS,
            cls.VALUE_LOSS,
            cls.ENTROPY,
            cls.LOSS,
            cls.GRAD_NORM,
            cls.CLIP_FRACTION,
            cls.EXPLAINED_VAR,
            cls.ACTION_ENTROPY,
            cls.ACTION_DIST,
        ]


# =============================================================================
# Main Training Loop
# =============================================================================


def make_env(level: tuple, action_history_len: int):
    """Factory function for creating Mario env."""

    def _init():
        env = create_mario_env(level=level, action_history_len=action_history_len)
        return env.env  # Unwrap to get gym env

    return _init


@click.command()
@click.option("--envs", "-n", default=8, help="Number of parallel environments")
@click.option("--steps", default=1_000_000, help="Total training steps")
@click.option("--lr", default=2.5e-4, help="Learning rate")
@click.option("--gamma", default=0.99, help="Discount factor")
@click.option("--gae-lambda", default=0.95, help="GAE lambda")
@click.option("--clip-eps", default=0.2, help="PPO clip epsilon")
@click.option("--clip-vloss", is_flag=True, default=True, help="Clip value loss")
@click.option("--ent-coef", default=0.01, help="Entropy coefficient")
@click.option("--vf-coef", default=0.5, help="Value function coefficient")
@click.option("--max-grad-norm", default=0.5, help="Max gradient norm for clipping")
@click.option("--rollout-steps", default=128, help="Steps per rollout")
@click.option("--minibatch-size", default=256, help="Minibatch size for updates")
@click.option("--update-epochs", default=4, help="Number of epochs per update")
@click.option("--anneal-lr", is_flag=True, default=True, help="Anneal learning rate")
@click.option("--action-history-len", default=12, help="Action history length (0 to disable)")
@click.option("--level", default="1,1", help="Level to train on (e.g. '1,1')")
@click.option("--save-dir", default="checkpoints", help="Save directory")
def main(
    envs: int,
    steps: int,
    lr: float,
    gamma: float,
    gae_lambda: float,
    clip_eps: float,
    clip_vloss: bool,
    ent_coef: float,
    vf_coef: float,
    max_grad_norm: float,
    rollout_steps: int,
    minibatch_size: int,
    update_epochs: int,
    anneal_lr: bool,
    action_history_len: int,
    level: str,
    save_dir: str,
) -> None:
    """Train PPO with vectorized environments."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")
    log(f"Environments: {envs}")
    log(f"Total steps: {steps:,}")
    log(f"Rollout steps: {rollout_steps}")
    log(f"Batch size: {rollout_steps * envs}")
    log(f"Minibatch size: {minibatch_size}")
    log(f"Update epochs: {update_epochs}")

    # Parse level
    try:
        world, stage = map(int, level.split(","))
        level_tuple = (world, stage)
    except ValueError:
        level_tuple = (1, 1)
    log(f"Level: {level_tuple}")

    # Setup save dir
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = Path(save_dir) / f"vec_ppo_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Initialize metric loggers
    coord_logger = MetricLogger(
        source_id="coordinator",
        schema=CoordinatorMetrics,
        csv_path=run_dir / "coordinator.csv",
    )
    worker_logger = MetricLogger(
        source_id="worker_0",
        schema=PPOMetrics,
        csv_path=run_dir / "worker_0.csv",
    )

    # Create vectorized environment
    log("Creating async vectorized environment...")
    from gymnasium.vector import AsyncVectorEnv

    env_fns = [make_env(level_tuple, action_history_len=action_history_len) for _ in range(envs)]
    vec_env = AsyncVectorEnv(env_fns)
    log(f"Observation space: {vec_env.single_observation_space}")
    log(f"Action space: {vec_env.single_action_space}")
    log(f"Action history length: {action_history_len}")

    obs_shape = vec_env.single_observation_space.shape
    num_actions = vec_env.single_action_space.n

    # Create model with attention backbone
    log("Creating PPO model with attention backbone...")
    set_skip_weight_init(False)  # Enable proper initialization
    model = PPONetwork(
        input_shape=obs_shape,
        num_actions=num_actions,
        feature_dim=512,
        dropout=0.1,
        action_history_len=action_history_len,
    ).to(device)
    log(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    log(f"Architecture: CNN(3 layers) → Self-Attention(8×8) → Policy/Value heads")
    if action_history_len > 0:
        log(f"Action history: {action_history_len} steps × {num_actions} actions = {action_history_len * num_actions} features")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    # Rollout buffer
    rollout = RolloutBuffer.create(
        rollout_steps, envs, obs_shape,
        action_history_len=action_history_len,
        num_actions=num_actions,
    )

    # Action history tracking (one-hot encoded, per environment)
    # Shape: (envs, history_len, num_actions)
    current_action_history = np.zeros((envs, action_history_len, num_actions), dtype=np.float32)

    # Training state
    obs, _ = vec_env.reset()
    total_steps = 0
    num_updates = 0
    episode_count = 0
    episode_rewards = np.zeros(envs)
    deaths = 0
    flags = 0

    # Rolling stats
    reward_history: deque[float] = deque(maxlen=100)
    x_pos_history: deque[float] = deque(maxlen=100)
    policy_loss_history: deque[float] = deque(maxlen=100)
    value_loss_history: deque[float] = deque(maxlen=100)
    entropy_history: deque[float] = deque(maxlen=100)
    loss_history: deque[float] = deque(maxlen=100)
    grad_norm_history: deque[float] = deque(maxlen=100)
    speed_history: deque[float] = deque(maxlen=100)

    best_x_ever = 0
    action_counts = np.zeros(num_actions)

    log(f"\nSaving to: {run_dir}")
    log("=" * 60)
    log("Starting PPO training...")
    start_time = time.time()

    while total_steps < steps:
        # Anneal learning rate
        if anneal_lr:
            frac = 1.0 - total_steps / steps
            current_lr = lr * frac
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

        # Collect rollout
        for step in range(rollout_steps):
            rollout.obs[step] = obs

            # Prepare action history tensor
            action_history_t = None
            if action_history_len > 0:
                action_history_t = torch.from_numpy(current_action_history).to(device)
                rollout.action_history[step] = current_action_history.copy()

            with torch.no_grad():
                obs_t = torch.from_numpy(obs).to(device)
                action, log_prob, _, value = model.get_action_and_value(
                    obs_t, action_history=action_history_t
                )
                action = action.cpu().numpy()
                log_prob = log_prob.cpu().numpy()
                value = value.cpu().numpy()

            rollout.actions[step] = action
            rollout.log_probs[step] = log_prob
            rollout.values[step] = value

            # Track action distribution
            for a in action:
                action_counts[a] += 1

            # Step environment
            next_obs, rewards, terminateds, truncateds, infos = vec_env.step(action)
            dones = np.logical_or(terminateds, truncateds)

            rollout.rewards[step] = rewards
            rollout.dones[step] = dones

            # Update action history (shift and add new action as one-hot)
            if action_history_len > 0:
                # Shift history: drop oldest, add newest
                current_action_history[:, :-1, :] = current_action_history[:, 1:, :]
                # Add new action as one-hot
                current_action_history[:, -1, :] = 0
                for i, a in enumerate(action):
                    current_action_history[i, -1, a] = 1.0

            # Track episode stats
            episode_rewards += rewards
            for i, done in enumerate(dones):
                if done:
                    episode_count += 1
                    ep_reward = episode_rewards[i]
                    reward_history.append(ep_reward)

                    x_pos = int(infos["x_pos"][i]) if "x_pos" in infos else 0
                    game_time = int(infos["time"][i]) if "time" in infos else 400
                    flag_get = bool(infos["flag_get"][i]) if "flag_get" in infos else False

                    x_pos_history.append(x_pos)
                    best_x_ever = max(best_x_ever, x_pos)

                    time_spent = max(1, 400 - game_time)
                    speed = x_pos / time_spent
                    speed_history.append(speed)

                    if ep_reward < -10:
                        deaths += 1
                    if flag_get:
                        flags += 1

                    episode_rewards[i] = 0

                    # Reset action history for this env on episode end
                    if action_history_len > 0:
                        current_action_history[i] = 0

            obs = next_obs

        total_steps += rollout_steps * envs

        # Compute advantages with GAE
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).to(device)
            action_history_t = None
            if action_history_len > 0:
                action_history_t = torch.from_numpy(current_action_history).to(device)
            _, _, _, next_value = model.get_action_and_value(obs_t, action_history=action_history_t)
            next_value = next_value.cpu().numpy()

        rollout.compute_gae(next_value, gamma, gae_lambda)

        # PPO update
        clip_fractions = []

        for _epoch in range(update_epochs):
            batches = rollout.get_batches(minibatch_size, device)

            for batch in batches:
                # Get new action probs and values
                _, new_log_prob, entropy, new_value = model.get_action_and_value(
                    batch["obs"], batch["actions"], action_history=batch["action_history"]
                )

                # Policy loss (clipped surrogate)
                log_ratio = new_log_prob - batch["old_log_probs"]
                ratio = torch.exp(log_ratio)
                with torch.no_grad():
                    clip_fraction = ((ratio - 1.0).abs() > clip_eps).float().mean().item()
                    clip_fractions.append(clip_fraction)

                pg_loss1 = -batch["advantages"] * ratio
                pg_loss2 = -batch["advantages"] * torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                if clip_vloss:
                    v_loss_unclipped = (new_value - batch["returns"]) ** 2
                    v_clipped = batch["old_values"] + torch.clamp(
                        new_value - batch["old_values"], -clip_eps, clip_eps
                    )
                    v_loss_clipped = (v_clipped - batch["returns"]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((new_value - batch["returns"]) ** 2).mean()

                # Entropy loss
                entropy_loss = entropy.mean()

                # Total loss
                loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                # Track metrics
                policy_loss_history.append(pg_loss.item())
                value_loss_history.append(v_loss.item())
                entropy_history.append(entropy_loss.item())
                loss_history.append(loss.item())
                grad_norm_history.append(grad_norm.item())

        num_updates += 1

        # Compute explained variance
        y_pred = rollout.values.flatten()
        y_true = rollout.returns.flatten()
        var_y = np.var(y_true)
        explained_var = 1 - np.var(y_true - y_pred) / (var_y + 1e-8) if var_y > 0 else 0

        # Log metrics every update
        elapsed = time.time() - start_time
        sps = total_steps / elapsed

        roll_reward = float(np.mean(reward_history)) if reward_history else 0.0
        roll_x = float(np.mean(x_pos_history)) if x_pos_history else 0.0
        roll_policy_loss = float(np.mean(policy_loss_history)) if policy_loss_history else 0.0
        roll_value_loss = float(np.mean(value_loss_history)) if value_loss_history else 0.0
        roll_entropy = float(np.mean(entropy_history)) if entropy_history else 0.0
        roll_loss = float(np.mean(loss_history)) if loss_history else 0.0
        roll_grad_norm = float(np.mean(grad_norm_history)) if grad_norm_history else 0.0
        roll_speed = float(np.mean(speed_history)) if speed_history else 0.0

        action_probs = action_counts / max(1, action_counts.sum())
        action_entropy = float(-np.sum(action_probs * np.log(action_probs + 1e-10)) / np.log(num_actions))
        action_dist_str = ",".join(f"{p*100:.1f}" for p in action_probs)

        # Worker metrics
        worker_logger._counters["episodes"] = episode_count
        worker_logger._counters["steps"] = total_steps
        worker_logger._counters["deaths"] = deaths
        worker_logger._counters["flags"] = flags

        worker_logger.gauge("steps_per_sec", sps)
        worker_logger.gauge("x_pos", roll_x)
        worker_logger.gauge("best_x", roll_x)
        worker_logger.gauge("best_x_ever", best_x_ever)
        worker_logger.gauge("world", level_tuple[0])
        worker_logger.gauge("stage", level_tuple[1])
        worker_logger.gauge("clip_fraction", float(np.mean(clip_fractions)))
        worker_logger.gauge("explained_var", explained_var)
        worker_logger.gauge("action_entropy", action_entropy)
        worker_logger.text("action_dist", action_dist_str)

        worker_logger.observe("reward", roll_reward)
        worker_logger.observe("policy_loss", roll_policy_loss)
        worker_logger.observe("value_loss", roll_value_loss)
        worker_logger.observe("entropy", roll_entropy)
        worker_logger.observe("loss", roll_loss)
        worker_logger.observe("grad_norm", roll_grad_norm)
        worker_logger.observe("speed", roll_speed)

        worker_logger.flush()

        # Coordinator metrics
        coord_logger._counters["update_count"] = num_updates
        coord_logger._counters["total_steps"] = total_steps
        coord_logger._counters["total_episodes"] = episode_count

        coord_logger.gauge("grads_per_sec", num_updates / elapsed if elapsed > 0 else 0)
        coord_logger.gauge("learning_rate", current_lr if anneal_lr else lr)
        coord_logger.gauge("avg_reward", roll_reward)
        coord_logger.gauge("avg_speed", roll_speed)
        coord_logger.gauge("avg_loss", roll_loss)

        coord_logger.observe("loss", roll_loss)
        coord_logger.observe("grad_norm", roll_grad_norm)

        coord_logger.flush()

        # Console logging
        if num_updates % 10 == 0:
            log(
                f"Steps: {total_steps:8,} | Episodes: {episode_count:5} | "
                f"Reward: {roll_reward:7.1f} | X: {roll_x:6.1f} | "
                f"SPS: {sps:.0f} | Flags: {flags}"
            )

        # Save checkpoint
        if num_updates % 100 == 0:
            torch.save(model.state_dict(), run_dir / "weights.pt")

    # Final save
    torch.save(model.state_dict(), run_dir / "weights_final.pt")

    coord_logger.close()
    worker_logger.close()

    log("=" * 60)
    log("Training complete!")
    log(f"Final rolling reward: {np.mean(reward_history):.1f}")
    log(f"Final rolling x_pos: {np.mean(x_pos_history):.1f}")
    log(f"Total flags captured: {flags}")
    log(f"Saved to: {run_dir}")
    log(f"View metrics with: uv run streamlit run mario_rl/dashboard/app.py -- --checkpoint {run_dir}")

    vec_env.close()


if __name__ == "__main__":
    main()
