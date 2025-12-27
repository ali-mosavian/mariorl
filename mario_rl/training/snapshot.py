"""
Game state snapshot manager for practicing from checkpoints.

Allows the agent to restore to earlier points in the game after death,
enabling more efficient learning of difficult sections.
"""

from typing import Any
from typing import Dict
from typing import Tuple
from typing import Protocol
from dataclasses import field
from dataclasses import dataclass

import numpy as np

from mario_rl.core.types import GameSnapshot
from mario_rl.core.config import SnapshotConfig


class FrameStackWrapper(Protocol):
    """Protocol for frame stack wrappers."""

    obs_queue: Any  # deque of frames


class BaseEnvironment(Protocol):
    """Protocol for base Mario environment."""

    env: Any  # Inner environment with dump_state/load_state


@dataclass
class SnapshotManager:
    """
    Manages game state snapshots for practicing from checkpoints.

    Saves checkpoints periodically during gameplay and can restore
    to earlier points after death. Tracks progress to avoid getting
    stuck in death loops.
    """

    config: SnapshotConfig

    # References to environment components (injected)
    base_env: Any = field(repr=False)
    fstack: Any = field(repr=False)

    # Internal state
    _slot_to_time: Dict[int, int] = field(init=False, default_factory=dict)
    _slot_to_state: Dict[int, GameSnapshot] = field(init=False, default_factory=dict)

    # Progress tracking
    restore_count: int = field(init=False, default=0)
    _best_x_at_restore: int = field(init=False, default=0)
    _restores_without_progress: int = field(init=False, default=0)

    def reset(self) -> None:
        """Clear all snapshots and progress tracking for new episode."""
        self._slot_to_time.clear()
        self._slot_to_state.clear()
        self._best_x_at_restore = 0
        self._restores_without_progress = 0

    def maybe_save(self, state: np.ndarray, info: dict) -> bool:
        """
        Save snapshot if at a new checkpoint time.

        Args:
            state: Current observation
            info: Environment info dict with 'time' key

        Returns:
            True if snapshot was saved
        """
        if not self.config.enabled:
            return False

        game_time = info.get("time", 0)
        checkpoint_time = game_time // self.config.interval

        # Only save at new checkpoint times (not already saved)
        time_to_slot = {v: k for k, v in self._slot_to_time.items()}
        if checkpoint_time in time_to_slot or checkpoint_time <= 0:
            return False

        # Use rotating slots
        slot_id = checkpoint_time % self.config.slots
        self._slot_to_time[slot_id] = checkpoint_time

        try:
            # Get NES state directly as numpy array
            nes_state = self.base_env.env.dump_state()

            # Save COPIES of frame stack queue as immutable tuple
            frame_queue: tuple = ()
            if hasattr(self.fstack, "obs_queue"):
                frame_queue = tuple(np.array(f, copy=True) for f in self.fstack.obs_queue)

            self._slot_to_state[slot_id] = GameSnapshot(
                observation=np.array(state, copy=True),
                frame_queue=frame_queue,
                nes_state=nes_state,
            )
            return True
        except Exception:
            return False

    def try_restore(self, info: dict, current_best_x: int) -> Tuple[np.ndarray, bool]:
        """
        Try to restore from a recent snapshot after death.

        Only restores if:
        1. Snapshots are enabled
        2. We have a checkpoint from 1 interval earlier
        3. We haven't exceeded max restores without progress

        Args:
            info: Environment info dict with 'time' key
            current_best_x: Current best x position in this episode

        Returns:
            Tuple of (restored_state, success)
        """
        if not self.config.enabled or not self._slot_to_state:
            return np.array([]), False

        # Already at max restores without progress - let episode end
        if self._restores_without_progress >= self.config.max_restores_without_progress:
            return np.array([]), False

        game_time = info.get("time", 0)
        checkpoint_time = game_time // self.config.interval
        restore_time = checkpoint_time + 1  # Checkpoint from 1 interval earlier

        # Only restore if we have the EXACT checkpoint
        time_to_slot = {v: k for k, v in self._slot_to_time.items()}
        if restore_time not in time_to_slot:
            return np.array([]), False

        slot_id = time_to_slot[restore_time]
        if slot_id not in self._slot_to_state:
            return np.array([]), False

        snapshot = self._slot_to_state[slot_id]

        try:
            # Restore frame stack queue FIRST (like MadMario does)
            if snapshot.frame_queue and hasattr(self.fstack, "obs_queue"):
                self.fstack.obs_queue.clear()
                for frame in snapshot.frame_queue:
                    self.fstack.obs_queue.append(frame)

            # Then restore NES emulator state
            self.base_env.env.load_state(snapshot.nes_state)

            # CRITICAL: Reset the NESEnv's _done flag
            # load_state() doesn't reset this, causing "cannot step in done env" error
            self.base_env.env._done = False

            # Verify observation is valid
            if snapshot.observation.size == 0:
                return np.array([]), False

            self.restore_count += 1

            # Track progress ONLY on successful restore
            if current_best_x > self._best_x_at_restore:
                self._restores_without_progress = 0
                self._best_x_at_restore = current_best_x
            else:
                self._restores_without_progress += 1

            return snapshot.observation, True
        except Exception:
            # Restore failed - return failure
            return np.array([]), False

    @property
    def restores_without_progress(self) -> int:
        """Number of consecutive restores without x progress."""
        return self._restores_without_progress

    @property
    def max_restores(self) -> int:
        """Maximum restores allowed without progress."""
        return self.config.max_restores_without_progress
