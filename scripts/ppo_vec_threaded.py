#!/usr/bin/env python
"""Vectorized PPO training script using C++ threaded emulation.

Uses VectorSuperMarioBrosEnv for parallel environment collection with
GPU-based frame preprocessing. This avoids Python multiprocessing overhead
and leverages C++ threads for true parallelism.

Implements Proximal Policy Optimization with:
- Actor-Critic network with shared CNN backbone
- GAE (Generalized Advantage Estimation)
- Clipped surrogate objective
- Value function clipping
- Entropy bonus for exploration
- GPU-based frame preprocessing (grayscale, resize, stack)

Logs metrics compatible with the Streamlit dashboard.

Usage:
    uv run python scripts/ppo_vec_threaded.py --envs 16 --steps 1000000
"""

import sys
import time
from pathlib import Path
from collections import deque
from datetime import datetime
from dataclasses import dataclass

import click
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from gymnasium.spaces import Box
from gymnasium.spaces import Discrete
from gym_super_mario_bros.smb_game import LevelMode
from gym_super_mario_bros import VectorSuperMarioBrosEnv

from mario_rl.agent.ddqn_net import layer_init
from mario_rl.agent.ddqn_net import DDQNBackbone
from mario_rl.metrics.logger import MetricLogger
from mario_rl.metrics.schema import CoordinatorMetrics
from mario_rl.agent.ddqn_net import set_skip_weight_init


def log(msg: str) -> None:
    """Print and flush immediately."""
    print(msg)
    sys.stdout.flush()


# =============================================================================
# GPU Frame Preprocessor (Compilable)
# =============================================================================


class FramePreprocessor(nn.Module):
    """Compilable frame preprocessing module.

    Converts RGB uint8 frames to grayscale uint8, resized.
    This is a pure function with no state, so it can be compiled.
    """

    def __init__(self, size: int = 64):
        super().__init__()
        self.size = size
        # Register grayscale weights as buffer (ITU-R BT.601)
        self.register_buffer("rgb_weights", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert RGB uint8 tensor to grayscale resized uint8 tensor.

        Args:
            x: (B, 3, H, W) uint8 tensor [0, 255]

        Returns:
            (B, 64, 64) uint8 tensor [0, 255]
        """
        # Convert to float for interpolation
        x = x.float()

        # Grayscale via weighted sum
        x = (x * self.rgb_weights).sum(dim=1, keepdim=True)  # (B, 1, H, W)

        # Resize to target size
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)

        # Back to uint8
        return x.squeeze(1).to(torch.uint8)  # (B, 64, 64)


class GPUPreprocessor:
    """GPU-based frame preprocessing and stacking.

    Converts raw RGB frames (B, 240, 256, 3) uint8 to grayscale, resized,
    stacked tensors (B, num_stack, size, size) uint8 on GPU.

    Uses a compiled FramePreprocessor module for the core transforms.
    """

    def __init__(
        self,
        num_envs: int,
        num_stack: int = 4,
        size: int = 64,
        device: str = "cuda",
        compile_preprocessor: bool = True,
    ):
        self.device = device
        self.num_envs = num_envs
        self.num_stack = num_stack
        self.size = size

        # Frame stack buffer on GPU: (num_envs, num_stack, H, W) as uint8
        self.frame_buffer = torch.zeros(
            (num_envs, num_stack, size, size),
            dtype=torch.uint8,
            device=device,
        )

        # Compilable preprocessing module
        self._preprocessor = FramePreprocessor(size).to(device)
        if compile_preprocessor:
            try:
                self._preprocessor = torch.compile(self._preprocessor, mode="reduce-overhead")
                log("  ✓ Frame preprocessor compiled")
            except Exception as e:
                log(f"  ⚠ Could not compile preprocessor: {e}")

    def _process_frame(self, obs: np.ndarray) -> torch.Tensor:
        """Convert raw RGB uint8 to grayscale uint8 tensor.

        Args:
            obs: (B, 240, 256, 3) uint8 numpy array

        Returns:
            (B, 64, 64) uint8 tensor on GPU
        """
        # Transfer and permute: (B, H, W, C) -> (B, C, H, W)
        x = torch.from_numpy(np.ascontiguousarray(obs)).to(self.device, non_blocking=True)
        x = x.permute(0, 3, 1, 2)  # (B, 3, 240, 256) uint8

        # Use compiled preprocessor (handles uint8 -> grayscale uint8)
        return self._preprocessor(x)

    def process(self, obs: np.ndarray) -> torch.Tensor:
        """Process frame and update stack.

        Args:
            obs: (B, 240, 256, 3) uint8 numpy array

        Returns:
            (B, num_stack, 64, 64) uint8 tensor on GPU
        """
        frame = self._process_frame(obs)

        # Roll buffer and insert new frame
        self.frame_buffer = torch.roll(self.frame_buffer, -1, dims=1)
        self.frame_buffer[:, -1] = frame

        return self.frame_buffer

    def reset(self, obs: np.ndarray) -> torch.Tensor:
        """Reset all frame stacks with initial observation.

        Args:
            obs: (B, 240, 256, 3) uint8 numpy array

        Returns:
            (B, num_stack, 64, 64) uint8 tensor on GPU
        """
        frame = self._process_frame(obs)
        # Fill entire stack with same frame
        self.frame_buffer[:] = frame.unsqueeze(1).expand(-1, self.num_stack, -1, -1)
        return self.frame_buffer

    def reset_envs(self, obs: np.ndarray, mask: np.ndarray) -> None:
        """Reset frame stacks for specific terminated environments.

        Args:
            obs: (B, 240, 256, 3) uint8 - observations (reset obs for terminated)
            mask: (B,) bool - which envs terminated
        """
        if not mask.any():
            return

        # Process only terminated envs' observations
        frame = self._process_frame(obs[mask])

        # Reset those envs' stacks
        mask_t = torch.from_numpy(mask).to(self.device)
        self.frame_buffer[mask_t] = frame.unsqueeze(1).expand(-1, self.num_stack, -1, -1)


# =============================================================================
# Vectorized Mario Environment Wrapper
# =============================================================================


class VectorMarioEnv:
    """Vectorized Mario environment with GPU preprocessing.

    Wraps VectorSuperMarioBrosEnv and provides:
    - Action space mapping (7 SIMPLE_MOVEMENT -> 256 NES buttons)
    - Frame skipping (4 frames per step, still parallel)
    - Reward shaping (death penalty, flag bonus)
    - GPU-based preprocessing (grayscale, resize, normalize, stack)

    Args:
        num_envs: Number of parallel environments
        level_mode: Level selection mode (SINGLE, SEQUENTIAL, RANDOM)
        target: Target level for SINGLE mode, e.g. (1, 1)
        device: PyTorch device for GPU preprocessing
        frame_skip: Number of frames to skip per step
        frame_stack: Number of frames to stack
        obs_size: Output observation size (obs_size x obs_size)
        death_penalty: Penalty added on death
        flag_bonus: Bonus added on flag capture
        action_history_len: Length of action history to track
    """

    # SIMPLE_MOVEMENT action mapping to NES controller buttons
    # [NOOP, right, right+A, right+B, right+A+B, A, left]
    # NES buttons: A=1, B=2, select=32, start=16, up=8, down=4, left=64, right=128
    ACTION_MAP = np.array([0, 128, 129, 130, 131, 1, 64], dtype=np.uint8)

    def __init__(
        self,
        num_envs: int,
        level_mode: LevelMode = LevelMode.SEQUENTIAL,
        target: tuple[int, int] | None = None,
        device: str = "cuda",
        frame_skip: int = 4,
        frame_stack: int = 4,
        obs_size: int = 64,
        death_penalty: float = -475.0,
        flag_bonus: float = 500.0,
        action_history_len: int = 0,
    ):
        self._num_envs = num_envs
        self._frame_skip = frame_skip
        self._death_penalty = death_penalty
        self._flag_bonus = flag_bonus
        self._device = device
        self._action_history_len = action_history_len

        # Raw vectorized emulator
        self._vec = VectorSuperMarioBrosEnv(
            num_envs,
            level_mode=level_mode,
            target=target,
            copy_obs=True,  # Need stable obs for GPU transfer
            auto_reset=True,
        )

        # GPU preprocessor
        self._gpu = GPUPreprocessor(num_envs, frame_stack, obs_size, device)

        # Action history tracking (one-hot encoded)
        self._num_actions = 7
        if action_history_len > 0:
            self._action_history = np.zeros(
                (num_envs, action_history_len, self._num_actions),
                dtype=np.float32,
            )
        else:
            self._action_history = None

        # Gymnasium-compatible spaces (uint8 grayscale stacked frames)
        self.single_observation_space = Box(
            low=0,
            high=255,
            shape=(frame_stack, obs_size, obs_size),
            dtype=np.uint8,
        )
        self.single_action_space = Discrete(self._num_actions)
        self.num_envs = num_envs

        # Pre-allocated buffers
        self._rewards = np.zeros(num_envs, dtype=np.float32)
        self._terminated = np.zeros(num_envs, dtype=bool)
        self._truncated = np.zeros(num_envs, dtype=bool)

    @property
    def observation_space(self):
        return self.single_observation_space

    @property
    def action_space(self):
        return self.single_action_space

    def reset(self, **kwargs) -> tuple[torch.Tensor, dict]:
        """Reset all environments.

        Returns:
            observations: (num_envs, num_stack, H, W) tensor on GPU
            info: Dictionary with per-env info arrays
        """
        obs, info = self._vec.reset(**kwargs)
        stacked = self._gpu.reset(obs)

        # Reset action history
        if self._action_history is not None:
            self._action_history[:] = 0
            info["action_history"] = self._action_history.copy()

        return stacked, info

    def step(self, actions: np.ndarray) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray, dict]:
        """Step all environments with frame skipping.

        Args:
            actions: (num_envs,) array of actions in [0, 6]

        Returns:
            observations: (num_envs, num_stack, H, W) tensor on GPU
            rewards: (num_envs,) float32 array
            terminated: (num_envs,) bool array
            truncated: (num_envs,) bool array (always False)
            info: Dictionary with per-env info arrays
        """
        # Map 7-action space to NES controller buttons
        raw_actions = self.ACTION_MAP[actions]

        # Frame skip with reward accumulation
        self._rewards[:] = 0
        self._terminated[:] = False

        for _ in range(self._frame_skip):
            obs, rewards, term, trunc, info = self._vec.step(raw_actions)
            self._rewards += rewards

            # Track terminations (auto-reset gives new obs)
            newly_terminated = term & ~self._terminated
            self._terminated |= term

            # Clear actions for terminated envs (they reset mid-skip)
            raw_actions = np.where(newly_terminated, 0, raw_actions)

            if self._terminated.all():
                break

        # Reward shaping
        death_mask = self._terminated & ~info["flag_get"]
        self._rewards[death_mask] += self._death_penalty
        self._rewards[info["flag_get"]] += self._flag_bonus

        # GPU preprocessing
        stacked = self._gpu.process(obs)

        # Reset frame stacks for terminated envs
        self._gpu.reset_envs(obs, self._terminated)

        # Update action history
        if self._action_history is not None:
            # Shift history
            self._action_history[:, :-1, :] = self._action_history[:, 1:, :]
            # Add new action as one-hot
            self._action_history[:, -1, :] = 0
            for i, a in enumerate(actions):
                self._action_history[i, -1, a] = 1.0
            # Reset history for terminated envs
            self._action_history[self._terminated] = 0
            info["action_history"] = self._action_history.copy()

        return stacked, self._rewards.copy(), self._terminated.copy(), self._truncated, info

    def close(self):
        """Close the environment."""
        self._vec.close()


# =============================================================================
# PPO Actor-Critic Network
# =============================================================================


class PPONetwork(nn.Module):
    """Actor-Critic network with shared attention backbone for PPO.

    Includes built-in preprocessing (grayscale, resize, normalize) so the entire
    pipeline can be compiled with torch.compile.

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

        # Shared attention backbone
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

    def forward(self, x: torch.Tensor, action_history: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning policy logits and value estimate.

        Args:
            x: Observations (B, C, H, W) uint8 [0, 255]
            action_history: Optional action history (B, history_len, num_actions) one-hot

        Returns:
            policy_logits: (B, num_actions)
            value: (B,)
        """
        features = self.backbone(x, action_history)

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
            x: Observations (B, C, H, W) uint8 [0, 255]
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


# =============================================================================
# Rollout Buffer for PPO
# =============================================================================


@dataclass
class RolloutBuffer:
    """Buffer for storing rollout data for PPO updates.

    Stores complete trajectories for on-policy learning.
    All tensors are stored on GPU for faster processing.
    """

    obs: torch.Tensor  # (steps, envs, C, H, W)
    actions: torch.Tensor  # (steps, envs)
    log_probs: torch.Tensor  # (steps, envs)
    rewards: torch.Tensor  # (steps, envs)
    dones: torch.Tensor  # (steps, envs)
    values: torch.Tensor  # (steps, envs)
    action_history: torch.Tensor | None = None  # (steps, envs, history_len, num_actions)

    # Computed after rollout
    advantages: torch.Tensor | None = None
    returns: torch.Tensor | None = None

    @classmethod
    def create(
        cls,
        num_steps: int,
        num_envs: int,
        obs_shape: tuple[int, ...],
        device: str,
        action_history_len: int = 0,
        num_actions: int = 0,
    ) -> "RolloutBuffer":
        """Create an empty rollout buffer on GPU."""
        action_history = None
        if action_history_len > 0:
            action_history = torch.zeros(
                (num_steps, num_envs, action_history_len, num_actions),
                dtype=torch.float32,
                device=device,
            )
        return cls(
            obs=torch.zeros((num_steps, num_envs, *obs_shape), dtype=torch.uint8, device=device),
            actions=torch.zeros((num_steps, num_envs), dtype=torch.int64, device=device),
            log_probs=torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device),
            rewards=torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device),
            dones=torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device),
            values=torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device),
            action_history=action_history,
        )

    def compute_gae(self, next_value: torch.Tensor, gamma: float, gae_lambda: float) -> None:
        """Compute Generalized Advantage Estimation on GPU.

        Args:
            next_value: Value estimate of final next state (num_envs,)
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        num_steps = self.rewards.shape[0]
        device = self.rewards.device

        self.advantages = torch.zeros_like(self.rewards)
        self.returns = torch.zeros_like(self.rewards)

        last_gae = torch.zeros(self.rewards.shape[1], device=device)

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_values = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    def get_batches(self, batch_size: int) -> list[dict[str, torch.Tensor]]:
        """Get shuffled minibatches for PPO update.

        Args:
            batch_size: Size of each minibatch

        Returns:
            List of batches on GPU
        """
        num_steps, num_envs = self.rewards.shape
        total_size = num_steps * num_envs
        device = self.rewards.device

        # Flatten all data
        assert self.advantages is not None and self.returns is not None
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
        indices = torch.randperm(total_size, device=device)

        # Create batches
        batches = []
        for start in range(0, total_size, batch_size):
            end = min(start + batch_size, total_size)
            batch_indices = indices[start:end]

            batch = {
                "obs": obs_flat[batch_indices],
                "actions": actions_flat[batch_indices],
                "old_log_probs": log_probs_flat[batch_indices],
                "advantages": advantages_flat[batch_indices],
                "returns": returns_flat[batch_indices],
                "old_values": values_flat[batch_indices],
                "action_history": None,
            }
            if action_history_flat is not None:
                batch["action_history"] = action_history_flat[batch_indices]
            batches.append(batch)

        return batches


# =============================================================================
# PPO Metrics Schema
# =============================================================================


class PPOMetrics:
    """PPO-specific metrics for logging."""

    from mario_rl.metrics.schema import MetricDef
    from mario_rl.metrics.schema import MetricType

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


def get_device() -> str:
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


@click.command()
@click.option("--envs", "-n", default=16, help="Number of parallel environments")
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
@click.option("--level", default="sequential", help="Level mode: 'X,Y' for single, 'sequential', or 'random'")
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
    """Train PPO with C++ threaded vectorized environments."""
    device = get_device()
    log(f"Device: {device}")
    log(f"Environments: {envs}")
    log(f"Total steps: {steps:,}")
    log(f"Rollout steps: {rollout_steps}")
    log(f"Batch size: {rollout_steps * envs}")
    log(f"Minibatch size: {minibatch_size}")
    log(f"Update epochs: {update_epochs}")

    # Parse level mode
    level_mode = LevelMode.SEQUENTIAL
    target = None
    if level.lower() == "sequential":
        level_mode = LevelMode.SEQUENTIAL
        log("Level mode: SEQUENTIAL (staggered start)")
    elif level.lower() == "random":
        level_mode = LevelMode.RANDOM
        log("Level mode: RANDOM")
    else:
        try:
            world, stage = map(int, level.split(","))
            level_mode = LevelMode.SINGLE
            target = (world, stage)
            log(f"Level mode: SINGLE ({world}-{stage})")
        except ValueError:
            log(f"Invalid level '{level}', using SEQUENTIAL")
            level_mode = LevelMode.SEQUENTIAL

    # Setup save dir
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = Path(save_dir) / f"vec_ppo_threaded_{timestamp}"
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

    # Create vectorized environment with GPU preprocessing
    log("Creating C++ threaded vectorized environment...")
    vec_env = VectorMarioEnv(
        num_envs=envs,
        level_mode=level_mode,
        target=target,
        device=device,
        frame_skip=4,
        frame_stack=4,
        obs_size=64,
        death_penalty=-475.0,
        flag_bonus=500.0,
        action_history_len=action_history_len,
    )
    log(f"Observation space: {vec_env.single_observation_space}")
    log(f"Action space: {vec_env.single_action_space}")
    log(f"Action history length: {action_history_len}")

    obs_shape = vec_env.single_observation_space.shape
    num_actions = vec_env.single_action_space.n

    # Create model
    log("Creating PPO model with attention backbone...")
    set_skip_weight_init(False)
    model = PPONetwork(
        input_shape=obs_shape,
        num_actions=num_actions,
        feature_dim=512,
        dropout=0.1,
        action_history_len=action_history_len,
    ).to(device)
    log(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    log("Architecture: CNN(3 layers) → Self-Attention(8×8) → Policy/Value heads")
    if action_history_len > 0:
        log(f"Action history: {action_history_len} steps × {num_actions} actions")

    # Compile model for better performance
    try:
        model = torch.compile(model, mode="reduce-overhead")
        log("  ✓ Model compiled with torch.compile (reduce-overhead mode)")
    except Exception as e:
        log(f"  ⚠ Could not compile model: {e}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    # Rollout buffer (on GPU)
    rollout = RolloutBuffer.create(
        rollout_steps,
        envs,
        obs_shape,
        device,
        action_history_len=action_history_len,
        num_actions=num_actions,
    )

    # Training state
    obs, info = vec_env.reset()  # obs is already on GPU
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
    current_lr = lr

    # Timing tracking
    rollout_time = 0.0
    update_time = 0.0

    log(f"\nSaving to: {run_dir}")
    log("=" * 60)
    log("Starting PPO training with C++ threaded emulation...")
    log("")
    start_time = time.time()

    # Calculate steps per update for progress tracking
    steps_per_update = rollout_steps * envs

    # Create progress bar
    pbar = tqdm(
        total=steps,
        desc="Training",
        unit="steps",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )

    while total_steps < steps:
        rollout_start = time.time()
        # Anneal learning rate
        if anneal_lr:
            frac = 1.0 - total_steps / steps
            current_lr = lr * frac
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

        # Collect rollout with sub-progress bar
        # saved_frames = steps * envs (frames we store after frame skip)
        # actual_frames = saved_frames * frame_skip (raw NES emulator frames)
        frame_skip = 4
        total_saved_frames = rollout_steps * envs
        saved_fps_str = ""
        actual_fps_str = ""
        rollout_pbar = tqdm(
            total=total_saved_frames,
            leave=False,
            dynamic_ncols=True,
            unit="f",
            bar_format="  Rollout: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
        )
        rollout_loop_start = time.time()
        for step in range(rollout_steps):
            # Update progress with rates
            elapsed = time.time() - rollout_loop_start
            if elapsed > 0 and step > 0:
                saved_collected = step * envs
                actual_collected = saved_collected * frame_skip
                saved_fps = saved_collected / elapsed
                actual_fps = actual_collected / elapsed
                saved_fps_str = f"{saved_fps:.0f}"
                actual_fps_str = f"{actual_fps:.0f}"
                rollout_pbar.bar_format = f"  Rollout: {{percentage:3.0f}}%|{{bar}}| {{n_fmt}}/{{total_fmt}} [{saved_fps_str} saved/s, {actual_fps_str} actual/s]"
            rollout_pbar.update(envs)
            rollout.obs[step] = obs

            # Prepare action history tensor
            action_history_t = None
            if action_history_len > 0 and "action_history" in info and rollout.action_history is not None:
                action_history_t = torch.from_numpy(info["action_history"]).to(device)
                rollout.action_history[step] = action_history_t

            with torch.no_grad():
                action, log_prob, _, value = model.get_action_and_value(obs, action_history=action_history_t)

            rollout.actions[step] = action
            rollout.log_probs[step] = log_prob
            rollout.values[step] = value

            # Track action distribution
            actions_np = action.cpu().numpy()
            for a in actions_np:
                action_counts[a] += 1

            # Step environment (obs is already on GPU)
            next_obs, rewards, terminateds, truncateds, info = vec_env.step(actions_np)
            dones = terminateds | truncateds

            rollout.rewards[step] = torch.from_numpy(rewards).to(device)
            rollout.dones[step] = torch.from_numpy(dones.astype(np.float32)).to(device)

            # Track episode stats
            episode_rewards += rewards
            for i, done in enumerate(dones):
                if done:
                    episode_count += 1
                    ep_reward = episode_rewards[i]
                    reward_history.append(ep_reward)

                    x_pos = int(info["x_pos"][i])
                    game_time = int(info["time"][i])
                    flag_get = bool(info["flag_get"][i])
                    world = int(info["world"][i])
                    stage = int(info["stage"][i])

                    x_pos_history.append(x_pos)
                    was_best = x_pos > best_x_ever
                    best_x_ever = max(best_x_ever, x_pos)

                    time_spent = max(1, 400 - game_time)
                    speed = x_pos / time_spent
                    speed_history.append(speed)

                    if ep_reward < -10:
                        deaths += 1
                    if flag_get:
                        flags += 1
                        tqdm.write(
                            f"  🚩 FLAG! Env {i} completed {world}-{stage} | X: {x_pos} | Reward: {ep_reward:.0f} | Total flags: {flags}"
                        )
                    elif was_best and x_pos > 500:  # Only log significant new records
                        tqdm.write(f"  ⭐ NEW BEST! Env {i} on {world}-{stage} | X: {x_pos} | Reward: {ep_reward:.0f}")

                    episode_rewards[i] = 0

            obs = next_obs

        rollout_pbar.close()
        total_steps += rollout_steps * envs
        rollout_time += time.time() - rollout_start
        update_start = time.time()

        # Compute advantages with GAE
        with torch.no_grad():
            action_history_t = None
            if action_history_len > 0 and "action_history" in info:
                action_history_t = torch.from_numpy(info["action_history"]).to(device)
            _, _, _, next_value = model.get_action_and_value(obs, action_history=action_history_t)

        rollout.compute_gae(next_value, gamma, gae_lambda)

        # PPO update with sub-progress bar
        clip_fractions = []
        num_batches = (rollout_steps * envs + minibatch_size - 1) // minibatch_size
        total_minibatches = update_epochs * num_batches

        update_pbar = tqdm(
            total=total_minibatches,
            desc="  Updating",
            leave=False,
            dynamic_ncols=True,
            unit="batch",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} batches [{rate_fmt}]",
        )

        for _epoch in range(update_epochs):
            batches = rollout.get_batches(minibatch_size)

            for batch in batches:
                update_pbar.update(1)
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
                    v_clipped = batch["old_values"] + torch.clamp(new_value - batch["old_values"], -clip_eps, clip_eps)
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

        update_pbar.close()
        num_updates += 1
        update_time += time.time() - update_start

        # Compute explained variance
        with torch.no_grad():
            y_pred = rollout.values.flatten()
            assert rollout.returns is not None
            y_true = rollout.returns.flatten()
            var_y = y_true.var()
            explained_var = (1 - (y_true - y_pred).var() / (var_y + 1e-8)).item() if var_y > 0 else 0

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
        worker_logger.gauge("world", 1)  # Multi-level mode
        worker_logger.gauge("stage", 1)
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
        coord_logger.gauge("learning_rate", current_lr)
        coord_logger.gauge("avg_reward", roll_reward)
        coord_logger.gauge("avg_speed", roll_speed)
        coord_logger.gauge("avg_loss", roll_loss)

        coord_logger.observe("loss", roll_loss)
        coord_logger.observe("grad_norm", roll_grad_norm)

        coord_logger.flush()

        # Update progress bar
        pbar.update(steps_per_update)
        pbar.set_postfix(
            {
                "loss": f"{roll_loss:.3f}",
                "reward": f"{roll_reward:.1f}",
                "x": f"{roll_x:.0f}",
            }
        )

        # Detailed stats every 10 updates
        if num_updates % 10 == 0:
            # Timing breakdown
            total_compute = rollout_time + update_time
            rollout_pct = rollout_time / total_compute * 100 if total_compute > 0 else 0
            update_pct = update_time / total_compute * 100 if total_compute > 0 else 0

            tqdm.write("")
            tqdm.write(f"  📊 Update {num_updates} | Episodes: {episode_count} | Deaths: {deaths} | Flags: {flags}")
            tqdm.write(
                f"     Reward: {roll_reward:7.1f} | X: {roll_x:6.0f} | BestX: {best_x_ever:5.0f} | Speed: {roll_speed:.1f}"
            )
            tqdm.write(
                f"     Loss: {roll_loss:.4f} (π:{roll_policy_loss:.4f} v:{roll_value_loss:.4f}) | "
                f"Entropy: {roll_entropy:.3f} | Clip: {float(np.mean(clip_fractions)):.2f} | ExplVar: {explained_var:.2f}"
            )
            tqdm.write(
                f"     LR: {current_lr:.2e} | GradNorm: {roll_grad_norm:.2f} | "
                f"SPS: {sps:.0f} | Time split: {rollout_pct:.0f}% rollout / {update_pct:.0f}% update"
            )
            # Action distribution
            action_names = ["NOOP", "→", "→A", "→B", "→AB", "A", "←"]
            action_str = " ".join(f"{action_names[i]}:{action_probs[i]*100:.0f}%" for i in range(num_actions))
            tqdm.write(f"     Actions: {action_str}")

        # Save checkpoint
        if num_updates % 100 == 0:
            torch.save(model.state_dict(), run_dir / "weights.pt")

    # Final save
    torch.save(model.state_dict(), run_dir / "weights_final.pt")

    pbar.close()
    coord_logger.close()
    worker_logger.close()

    log("")
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
