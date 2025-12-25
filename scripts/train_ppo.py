#!/usr/bin/env python3
"""
PPO training script using stable-baselines3.

Uses SubprocVecEnv for parallel environment rollouts.

Usage:
    uv run python scripts/train_ppo.py --num-envs 8 --level random
    uv run mario-train-ppo --num-envs 8 --level random --resume
"""

import os
import sys
import time
import signal
from typing import Dict
from typing import List
from pathlib import Path
from typing import Optional
from datetime import datetime
from multiprocessing import Queue

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import click
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback

sys.path.insert(0, str(Path(__file__).parent.parent))

from mario_rl.training.training_ui import UIMessage
from mario_rl.training.training_ui import TrainingUI
from mario_rl.training.training_ui import MessageType
from mario_rl.environment.env_factory import LevelType
from mario_rl.environment.env_factory import make_env_fn


def run_training_ui(num_workers: int, ui_queue: Queue) -> None:
    """Run the training UI in a separate process."""
    ui = TrainingUI(num_workers=num_workers, ui_queue=ui_queue, use_ppo=True)
    ui.run()


class UICallback(BaseCallback):
    """
    Callback that sends training metrics to the ncurses UI.
    """

    def __init__(
        self,
        ui_queue: Queue,
        num_envs: int,
        level_str: str = "random",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.ui_queue = ui_queue
        self.num_envs = num_envs
        self.level_str = level_str
        self.episode_rewards: Dict[int, float] = dict.fromkeys(range(num_envs), 0.0)
        self.episode_lengths: Dict[int, int] = dict.fromkeys(range(num_envs), 0)
        self.episode_counts: Dict[int, int] = dict.fromkeys(range(num_envs), 0)
        self.total_flags: Dict[int, int] = dict.fromkeys(range(num_envs), 0)
        self.best_x: Dict[int, int] = dict.fromkeys(range(num_envs), 0)
        self.reward_history: Dict[int, List[float]] = {i: [] for i in range(num_envs)}
        self.start_time = time.time()
        self._last_learner_update = 0.0

    def _on_step(self) -> bool:
        """Called after each step in the environment."""
        # Get info from all environments
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", np.zeros(self.num_envs))
        dones = self.locals.get("dones", np.zeros(self.num_envs, dtype=bool))

        for i, (info, reward, done) in enumerate(zip(infos, rewards, dones, strict=True)):
            self.episode_rewards[i] += reward
            self.episode_lengths[i] += 1

            # Track x position and flags
            x_pos = info.get("x_pos", 0)
            if x_pos > self.best_x[i]:
                self.best_x[i] = x_pos

            if info.get("flag_get", False):
                self.total_flags[i] += 1

            if done:
                # Episode finished - update stats
                self.episode_counts[i] += 1
                self.reward_history[i].append(self.episode_rewards[i])
                if len(self.reward_history[i]) > 100:
                    self.reward_history[i].pop(0)

                # Send worker status to UI
                rolling_avg = np.mean(self.reward_history[i]) if self.reward_history[i] else 0.0
                self._send_worker_status(i, x_pos, rolling_avg)

                # Reset episode tracking
                self.episode_rewards[i] = 0.0
                self.episode_lengths[i] = 0

        # Send learner status periodically (every 0.5 seconds)
        now = time.time()
        if now - self._last_learner_update > 0.5:
            self._send_learner_status()
            self._last_learner_update = now

        return True

    def _send_worker_status(self, env_id: int, x_pos: int, rolling_avg: float) -> None:
        """Send worker status to UI."""
        msg = UIMessage(
            msg_type=MessageType.WORKER_STATUS,
            source_id=env_id,
            data={
                "episode": self.episode_counts[env_id],
                "step": self.episode_lengths[env_id],
                "reward": self.episode_rewards[env_id],
                "x_pos": x_pos,
                "best_x": self.best_x[env_id],
                "best_x_ever": self.best_x[env_id],
                "deaths": 0,
                "flags": self.total_flags[env_id],
                "epsilon": 0.0,  # PPO doesn't use epsilon
                "experiences": self.num_timesteps,
                "q_mean": 0.0,
                "q_max": 0.0,
                "weight_sync_count": 0,
                "steps_per_sec": self.num_timesteps / max(1, time.time() - self.start_time),
                "snapshot_restores": 0,
                "current_level": self.level_str,
                "rolling_avg_reward": rolling_avg,
                "first_flag_time": 0.0,
            },
        )
        try:
            self.ui_queue.put_nowait(msg)
        except Exception:
            pass

    def _send_learner_status(self) -> None:
        """Send learner/PPO status to UI."""
        # Get training metrics from logger if available
        policy_loss = 0.0
        value_loss = 0.0
        entropy = 0.0
        clip_fraction = 0.0

        if hasattr(self.model, "logger") and self.model.logger is not None:
            # Try to get recent logged values
            try:
                name_to_value = getattr(self.model.logger, "name_to_value", {})
                policy_loss = name_to_value.get("train/policy_gradient_loss", 0.0)
                value_loss = name_to_value.get("train/value_loss", 0.0)
                entropy = name_to_value.get("train/entropy_loss", 0.0)
                clip_fraction = name_to_value.get("train/clip_fraction", 0.0)
            except Exception:
                pass

        elapsed = time.time() - self.start_time
        steps_per_sec = self.num_timesteps / max(1, elapsed)

        msg = UIMessage(
            msg_type=MessageType.PPO_STATUS,
            source_id=0,
            data={
                "step": self.num_timesteps,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "clip_fraction": clip_fraction,
                "steps_per_sec": steps_per_sec,
                "elapsed_time": elapsed,
            },
        )
        try:
            self.ui_queue.put_nowait(msg)
        except Exception:
            pass


def parse_level(level_str: str) -> LevelType:
    """Parse level string into LevelType."""
    if level_str == "random":
        return "random"
    elif level_str == "sequential":
        return "sequential"
    else:
        # Parse "W,S" format
        parts = level_str.split(",")
        if len(parts) == 2:
            world = int(parts[0])
            stage = int(parts[1])
            return (world, stage)  # type: ignore
        raise ValueError(f"Invalid level format: {level_str}")


@click.command()
@click.option("--num-envs", "-n", default=8, help="Number of parallel environments")
@click.option("--level", "-l", default="random", help="Level: 'random', 'sequential', or 'W,S' (e.g., '1,1')")
@click.option("--total-timesteps", "-t", default=10_000_000, help="Total training timesteps")
@click.option("--save-dir", default="checkpoints", help="Directory for saving checkpoints")
@click.option("--resume", is_flag=True, help="Resume from latest checkpoint")
@click.option("--checkpoint", default=None, help="Specific checkpoint to resume from")
@click.option("--no-ui", is_flag=True, help="Disable ncurses UI (text output only)")
@click.option("--learning-rate", "-lr", default=2.5e-4, help="Learning rate")
@click.option("--n-steps", default=128, help="Steps per env before update")
@click.option("--batch-size", default=256, help="Minibatch size")
@click.option("--n-epochs", default=4, help="Epochs per update")
def main(
    num_envs: int,
    level: str,
    total_timesteps: int,
    save_dir: str,
    resume: bool,
    checkpoint: Optional[str],
    no_ui: bool,
    learning_rate: float,
    n_steps: int,
    batch_size: int,
    n_epochs: int,
) -> None:
    """Train Mario agent using PPO with stable-baselines3."""
    # Parse level
    level_type = parse_level(level)

    # Create save directory
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = Path(save_dir) / f"ppo_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("PPO Training")
    print(f"  Environments: {num_envs}")
    print(f"  Level: {level}")
    print(f"  Save dir: {run_dir}")
    print(f"  Total timesteps: {total_timesteps:,}")

    # Create vectorized environment
    print("Creating environments...")
    env_fns = [make_env_fn(level=level_type, seed=i) for i in range(num_envs)]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)

    # Load or create model
    model_path = None
    if checkpoint:
        model_path = Path(checkpoint)
    elif resume:
        # Find latest checkpoint
        checkpoints = list(Path(save_dir).glob("ppo_*/best_model.zip"))
        if checkpoints:
            model_path = max(checkpoints, key=lambda p: p.stat().st_mtime)

    if model_path and model_path.exists():
        print(f"Loading model from {model_path}")
        model = PPO.load(model_path, env=vec_env)
    else:
        print("Creating new PPO model...")
        model = PPO(
            "CnnPolicy",
            vec_env,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            clip_range=0.1,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=0 if not no_ui else 1,
            tensorboard_log=str(run_dir / "tensorboard"),
            policy_kwargs={"normalize_images": False},  # Images already normalized to [0,1]
        )

    # Setup callbacks
    callbacks = [
        CheckpointCallback(
            save_freq=50000 // num_envs,
            save_path=str(run_dir),
            name_prefix="ppo_mario",
        ),
    ]

    # Setup UI
    ui_queue: Queue = Queue()
    ui_process = None

    if not no_ui:
        # Add UI callback - format level for display
        if isinstance(level_type, tuple):
            level_display = f"{level_type[0]}-{level_type[1]}"
        else:
            level_display = str(level_type)
        callbacks.append(UICallback(ui_queue, num_envs, level_str=level_display))

        # Start UI in separate process (must use module-level function for pickling)
        from multiprocessing import Process

        ui_process = Process(target=run_training_ui, args=(num_envs, ui_queue), daemon=True)
        ui_process.start()

    # Handle Ctrl+C gracefully
    def signal_handler(signum, frame):
        print("\nSaving model before exit...")
        model.save(run_dir / "final_model")
        vec_env.close()
        if ui_process:
            ui_process.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Train
    try:
        if no_ui:
            print("Starting training...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=no_ui,
        )

        # Save final model
        model.save(run_dir / "best_model")
        print(f"\nTraining complete! Model saved to {run_dir / 'best_model'}")

    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        model.save(run_dir / "interrupted_model")

    finally:
        vec_env.close()
        if ui_process:
            ui_process.terminate()


if __name__ == "__main__":
    main()
