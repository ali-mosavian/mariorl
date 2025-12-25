"""
Worker process for distributed training.

Runs environment with snapshots, collects experiences, pushes to shared buffer.
Periodically loads latest network weights from disk.
"""

import os
import sys
import time
import multiprocessing as mp

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Optional
from pathlib import Path
from dataclasses import field
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fix missing 'key' import in nes_py._image_viewer
import nes_py._image_viewer as _iv
from pyglet.window import key

_iv.key = key

import numpy as np
import torch

from nes_py.wrappers import JoypadSpace
from gymnasium.spaces import Box
from gymnasium.wrappers import GrayscaleObservation
from gymnasium.wrappers import TransformObservation
from gymnasium.wrappers import FrameStackObservation
from gym_super_mario_bros import actions as smb_actions

from mario_rl.agent.neural import DuelingDDQNNet
from mario_rl.environment.mariogym import SuperMarioBrosMultiLevel
from mario_rl.training.shared_buffer import SharedReplayBuffer
from mario_rl.agent.world_model import MarioWorldModel
from mario_rl.agent.world_model import LatentDDQN


def create_env(level=(1, 1), render_frames=False):
    """Create wrapped Mario environment."""
    from mario_rl.environment.wrappers import SkipFrame
    from mario_rl.environment.wrappers import ResizeObservation

    base_env = SuperMarioBrosMultiLevel(level=level)
    env = JoypadSpace(base_env, actions=smb_actions.COMPLEX_MOVEMENT)
    env = SkipFrame(env, skip=4, render_frames=render_frames)
    env = GrayscaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, shape=64)
    env = TransformObservation(
        env,
        func=lambda x: x / 255.0,
        observation_space=Box(low=0.0, high=1.0, shape=(64, 64, 1), dtype=np.float32),
    )
    fstack = FrameStackObservation(env, stack_size=4)
    return fstack, base_env


@dataclass
class Worker:
    """
    Worker that collects experiences and pushes to shared buffer.
    Does not learn - only runs inference with periodically updated weights.
    """

    # Required fields (no defaults)
    worker_id: int
    shared_buffer: SharedReplayBuffer
    weights_path: Path

    # Configuration fields with defaults
    level: tuple = (1, 1)
    render_frames: bool = False
    exploration_rate: float = 0.25
    exploration_rate_min: float = 0.1
    exploration_rate_decay: float = 0.99999
    weight_sync_interval: int = 1000
    device: Optional[str] = None
    ui_queue: Optional[mp.Queue] = None
    
    # World model configuration
    use_world_model: bool = False
    latent_dim: int = 128

    # Private fields initialized in __post_init__
    env: Any = field(init=False, repr=False)
    base_env: Any = field(init=False, repr=False)
    fstack: Any = field(init=False, repr=False)
    action_dim: int = field(init=False)
    net: Any = field(init=False, repr=False)
    world_model: Any = field(init=False, repr=False, default=None)
    latent_q_net: Any = field(init=False, repr=False, default=None)
    curr_step: int = field(init=False, default=0)
    episodes_completed: int = field(init=False, default=0)
    experiences_pushed: int = field(init=False, default=0)
    last_q_mean: float = field(init=False, default=0.0)
    last_q_max: float = field(init=False, default=0.0)
    action_counts: np.ndarray = field(init=False, repr=False)
    steps_per_sec: float = field(init=False, default=0.0)
    _step_times: list = field(init=False, repr=False, default_factory=list)
    _last_time: float = field(init=False, repr=False, default=0.0)
    last_weight_sync: float = field(init=False, default=0.0)
    weight_sync_count: int = field(init=False, default=0)
    snapshot_restores: int = field(init=False, default=0)

    def __post_init__(self):
        """Initialize environment and network after dataclass fields are set."""
        # Create environment
        self.env, self.base_env = create_env(
            level=self.level, render_frames=self.render_frames
        )
        self.fstack = self.env
        self.action_dim = self.env.action_space.n

        # Set device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create network(s) for inference
        state_dim = (4, 64, 64, 1)
        
        if self.use_world_model:
            # World model mode: encoder + latent Q-network
            self.world_model = MarioWorldModel(
                frame_shape=state_dim,
                num_actions=self.action_dim,
                latent_dim=self.latent_dim,
                hidden_dim=256,
            ).to(self.device)
            self.latent_q_net = LatentDDQN(
                latent_dim=self.latent_dim,
                num_actions=self.action_dim,
                hidden_dim=256,
            ).to(self.device)
            self.world_model.eval()
            self.latent_q_net.eval()
            self.net = None  # Not used in world model mode
        else:
            # Standard DQN mode
            self.net = DuelingDDQNNet(
                input_shape=state_dim, num_actions=self.action_dim, hidden_dim=512
            )
            self.net = self.net.to(self.device)
            self.net.eval()
            self.world_model = None
            self.latent_q_net = None

        # Initialize stats
        self.curr_step = 0
        self.episodes_completed = 0
        self.experiences_pushed = 0

        # Initialize real-time metrics
        self.last_q_mean = 0.0
        self.last_q_max = 0.0
        self.action_counts = np.zeros(self.action_dim, dtype=np.int32)
        self.steps_per_sec = 0.0
        self._step_times = []
        self._last_time = time.time()
        self.last_weight_sync = 0.0
        self.weight_sync_count = 0

    def load_weights(self) -> bool:
        """Load latest weights from disk. Returns True if successful."""
        if not self.weights_path.exists():
            return False
        try:
            checkpoint = torch.load(self.weights_path, map_location=self.device, weights_only=True)
            
            if self.use_world_model:
                # World model mode: load encoder and latent Q-network
                if isinstance(checkpoint, dict) and "world_model" in checkpoint:
                    self.world_model.load_state_dict(checkpoint["world_model"])
                    self.latent_q_net.load_state_dict(checkpoint["q_network"])
                else:
                    # Fallback: might be old format
                    return False
            else:
                # Standard DQN mode
                if isinstance(checkpoint, dict) and "world_model" in checkpoint:
                    # World model weights but worker in DQN mode - skip
                    return False
                # Learner saves online network state, load into online network
                self.net.online.load_state_dict(checkpoint)
            
            self.last_weight_sync = time.time()
            self.weight_sync_count += 1
            return True
        except Exception as e:
            # Log the error for debugging
            if self.curr_step % 5000 == 0:  # Only log occasionally to avoid spam
                self._log(f"Failed to load weights: {e}")
            return False

    @torch.no_grad()
    def act(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy."""
        state_tensor = torch.from_numpy(np.expand_dims(state, 0)).to(self.device)
        
        if self.use_world_model:
            # World model mode: encode state to latent, then get Q-values
            z = self.world_model.encode(state_tensor, deterministic=True)
            q_values = self.latent_q_net.online(z)
        else:
            # Standard DQN mode
            q_values = self.net.online(state_tensor)

        # Track Q-value stats
        self.last_q_mean = float(q_values.mean())
        self.last_q_max = float(q_values.max())

        if np.random.rand() < self.exploration_rate:
            action = np.random.randint(0, self.action_dim)
        else:
            action = q_values.argmax(dim=-1).item()

        # Track action distribution
        self.action_counts[action] += 1

        # Decay exploration
        self.exploration_rate = max(
            self.exploration_rate_min,
            self.exploration_rate * self.exploration_rate_decay,
        )
        self.curr_step += 1

        # Calculate steps per second
        now = time.time()
        self._step_times.append(now - self._last_time)
        self._last_time = now
        if len(self._step_times) > 100:
            self._step_times.pop(0)
        if self._step_times:
            avg_step_time = sum(self._step_times) / len(self._step_times)
            self.steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0

        return action

    def run_episode(
        self, episode: int = 0, best_x: int = 0, total_flags: int = 0
    ) -> dict:
        """Run one episode, pushing experiences to shared buffer."""
        state, _ = self.env.reset()

        # Snapshot tracking
        slot_to_time: Dict[int, int] = {}
        time_to_slot: Dict[int, int] = {}
        slot_to_state: Dict[int, Tuple[Any, List[Any], Any]] = {}
        num_deaths = 0
        total_reward = 0.0
        max_x_pos = 0

        # Stuck detection disabled
        # stuck_check_interval = 200
        # stuck_threshold = 10
        # last_stuck_check_x = 0
        # last_stuck_check_step = 0
        # stuck_restore_count = 0
        # max_stuck_restores = 3

        for step in range(2500):
            action = self.act(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Check death
            is_dead = info.get("is_dying", False) or info.get("is_dead", False)
            if is_dead:
                reward = -100

            total_reward += reward
            max_x_pos = max(max_x_pos, info.get("x_pos", 0))

            # Push experience to shared buffer
            success = self.shared_buffer.push(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                actions=list(range(self.action_dim)),
            )
            if success:
                self.experiences_pushed += 1

            # Send real-time status every 50 steps
            if step % 50 == 0:
                self._send_status(
                    episode=episode,
                    reward=total_reward,
                    x_pos=max_x_pos,
                    best_x=best_x,
                    deaths=num_deaths,
                    flags=total_flags,
                    step=step,
                )

            # Snapshot logic
            world_time = info.get("time", 0) // 2
            restore_time = world_time + 2

            # RESTORE on death
            if is_dead and (restore_time in time_to_slot) and world_time > 15:
                slot_id = time_to_slot[restore_time]
                saved_state, saved_frames, nes_snapshot = slot_to_state[slot_id]

                self.fstack.obs_queue.clear()
                for f in saved_frames:
                    self.fstack.obs_queue.append(f)

                self.base_env.env.load_state(nes_snapshot)
                state = saved_state
                num_deaths += 1
                self.snapshot_restores += 1
                continue

            # SAVE snapshot at new time points
            if world_time not in time_to_slot and world_time > 0:
                slot_id = world_time % 10
                slot_to_time[slot_id] = world_time
                time_to_slot = {v: k for k, v in slot_to_time.items()}
                nes_snapshot = self.base_env.env.dump_state()
                slot_to_state[slot_id] = (
                    state,
                    list(self.fstack.obs_queue),
                    nes_snapshot,
                )

            state = next_state

            # Stuck detection disabled
            # if step - last_stuck_check_step >= stuck_check_interval:
            #     current_x = info.get("x_pos", 0)
            #     x_progress = current_x - last_stuck_check_x
            #     if x_progress < stuck_threshold and world_time > 100:
            #         # Handle stuck condition...
            #         pass

            # Sync weights periodically
            if self.curr_step % self.weight_sync_interval == 0 and self.curr_step > 0:
                if self.load_weights():
                    pass  # Success - weight_sync_count already incremented
                else:
                    # Log failure occasionally to avoid spam
                    if self.curr_step % 5000 == 0:
                        self._log(f"⚠️ Failed to sync weights at step {self.curr_step}")

            if done or info.get("flag_get", False):
                break

        self.episodes_completed += 1
        return {
            "reward": total_reward,
            "deaths": num_deaths,
            "x_pos": max_x_pos,
            "steps": step + 1,
            "flag_get": info.get("flag_get", False),
        }

    def _log(self, text: str):
        """Log message to UI or stdout."""
        if self.ui_queue is not None:
            from distributed.training_ui import send_worker_log

            send_worker_log(self.ui_queue, self.worker_id, text)
        else:
            print(f"[W{self.worker_id}] {text}", flush=True)

    def _send_status(
        self,
        episode: int,
        reward: float,
        x_pos: int,
        best_x: int,
        deaths: int,
        flags: int,
        step: int = 0,
        is_end: bool = False,
    ):
        """Send status update to UI."""
        if self.ui_queue is not None:
            from distributed.training_ui import send_worker_status

            send_worker_status(
                self.ui_queue,
                self.worker_id,
                episode=episode,
                reward=reward,
                x_pos=x_pos,
                best_x=best_x,
                deaths=deaths,
                flags=flags,
                epsilon=self.exploration_rate,
                experiences=self.experiences_pushed,
                q_mean=self.last_q_mean,
                q_max=self.last_q_max,
                steps_per_sec=self.steps_per_sec,
                step=step,
                curr_step=self.curr_step,
                last_weight_sync=self.last_weight_sync,
                weight_sync_count=self.weight_sync_count,
                snapshot_restores=self.snapshot_restores,
            )
        elif step > 0 and step % 200 == 0:
            # Print real-time status during episode (every 200 steps)
            print(
                f"[W{self.worker_id}] Ep {episode} step {step:4d} | "
                f"x={x_pos:4d} | "
                f"Q={self.last_q_mean:6.1f}/{self.last_q_max:6.1f} | "
                f"r={reward:8.0f} | "
                f"💀={deaths} | "
                f"{self.steps_per_sec:.0f} sps",
                flush=True,
            )

        if is_end and self.ui_queue is None:
            # Print episode end summary in text mode
            print(
                f"[W{self.worker_id}] ══ Ep {episode:4d} DONE | "
                f"x={x_pos:4d} best={best_x:4d} | "
                f"r={reward:8.0f} | "
                f"💀={deaths:2d} 🏁={flags} | "
                f"steps={step:4d} | "
                f"ε={self.exploration_rate:.3f} | "
                f"exp={self.experiences_pushed:,}",
                flush=True,
            )

    def run(self, num_episodes: int = -1):
        """
        Run worker continuously or for specified number of episodes.
        num_episodes=-1 means run forever.
        """
        # Wait for initial weights from learner
        import time as time_module
        max_wait = 30  # Wait up to 30 seconds
        wait_start = time_module.time()
        self._log(f"Waiting for weights file: {self.weights_path}")
        while not self.weights_path.exists() and (time_module.time() - wait_start) < max_wait:
            time_module.sleep(0.5)
        
        if self.weights_path.exists():
            self._log(f"Weights file found, loading...")
            # Try to load initial weights
            if self.load_weights():
                self._log(f"✅ Loaded initial weights (sync count={self.weight_sync_count})")
            else:
                self._log(f"⚠️ Failed to load weights file that exists at {self.weights_path}")
        else:
            self._log(f"⚠️ Weights file not found after {max_wait}s: {self.weights_path}")

        episode = 0
        total_flags = 0
        best_x_pos = 0
        last_weight_load = 0

        # Send initial status
        self._send_status(episode, 0, 0, 0, 0, 0)

        while num_episodes < 0 or episode < num_episodes:
            stats = self.run_episode(
                episode=episode + 1,  # Pass current episode (1-indexed for display)
                best_x=best_x_pos,
                total_flags=total_flags,
            )
            episode += 1

            # Track best progress
            if stats["x_pos"] > best_x_pos:
                best_x_pos = stats["x_pos"]

            if stats["flag_get"]:
                total_flags += 1
                self._log(
                    f"🏁 FLAG GET! Episode {episode}, x={stats['x_pos']}, flags={total_flags}"
                )

            # Send end-of-episode status update
            self._send_status(
                episode=episode,
                reward=stats["reward"],
                x_pos=stats["x_pos"],
                best_x=best_x_pos,
                deaths=stats["deaths"],
                flags=total_flags,
                step=stats["steps"],
                is_end=True,
            )

            # Note when weights were synced
            if self.curr_step - last_weight_load >= self.weight_sync_interval:
                if self.load_weights():
                    self._log(f"🔄 Synced weights at step {self.curr_step}")
                    last_weight_load = self.curr_step

        self._log(
            f"✅ Finished {episode} episodes. Best x={best_x_pos}, Flags={total_flags}"
        )


def run_worker(
    worker_id: int,
    shared_buffer: SharedReplayBuffer,
    weights_path: Path,
    num_episodes: int = -1,
    ui_queue: Optional[mp.Queue] = None,
    use_world_model: bool = False,
    latent_dim: int = 128,
    **kwargs,
):
    """Entry point for worker process."""
    worker = Worker(
        worker_id=worker_id,
        shared_buffer=shared_buffer,
        weights_path=weights_path,
        ui_queue=ui_queue,
        use_world_model=use_world_model,
        latent_dim=latent_dim,
        **kwargs,
    )
    worker.run(num_episodes=num_episodes)


if __name__ == "__main__":
    # Test worker standalone
    buffer = SharedReplayBuffer(max_len=10000)
    weights_path = Path("checkpoints/test_weights.pt")

    worker = Worker(
        worker_id=0,
        shared_buffer=buffer,
        weights_path=weights_path,
        render_frames=True,
    )
    worker.run(num_episodes=5)
