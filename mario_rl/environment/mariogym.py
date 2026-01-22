"""
Mario environment wrappers.

Uses the reward shaping from our forked gym-super-mario-bros:
- time_penalty:   -0.1        Every step costs, no camping!
- progress:       +0.5/pixel  Bonus for new forward progress
- momentum:       scales      Bonus for sustained speed
- powerup_loss:   -15.0       Penalty for losing powerup
- death:          -500.0      Heavy penalty (increased from -25 to improve death learning)
- flag:           +100.0      Ultimate goal
"""

from typing import List
from typing import Tuple
from typing import Literal
from typing import Optional

import numpy as np
import gymnasium as gym
from nes_py import NESEnv
from nes_py._image_viewer import ImageViewer
from gym_super_mario_bros import SuperMarioBrosEnv

RomModes = Literal["vanilla", "pixel", "downsample"]
LevelModes = Literal["sequential", "random"] | Tuple[Literal[1, 2, 3, 4, 5, 6, 7, 8], Literal[1, 2, 3, 4]]


class MarioBrosLevel(SuperMarioBrosEnv):
    """Single level Mario environment using original gym-super-mario-bros rewards."""

    def __init__(
        self,
        rom_mode: RomModes = "vanilla",
        lost_levels: bool = False,
        target: Tuple[int, int] | None = None,
    ):
        super(MarioBrosLevel, self).__init__(rom_mode, lost_levels, target)
        self.render_mode = "rgb_array"


class SuperMarioBrosMultiLevel(gym.Env):
    """Multi-level Mario environment with random or sequential level selection.

    Uses lazy initialization for random/sequential modes to avoid creating
    all 32 NES emulators upfront. Levels are created on-demand when first selected.
    """

    rand: np.random.RandomState
    level: LevelModes
    envs: List[List[Optional[MarioBrosLevel]]]
    env: Optional[MarioBrosLevel] = None
    viewer: Optional[ImageViewer] = None
    _rom_mode: RomModes  # Store for lazy creation
    _current_level_idx: int  # For sequential mode

    metadata = {
        **SuperMarioBrosEnv.metadata,
    }
    reward_range = SuperMarioBrosEnv.reward_range
    action_space = SuperMarioBrosEnv.action_space
    observation_space = SuperMarioBrosEnv.observation_space

    # All valid level targets for random/sequential modes
    _ALL_LEVELS: List[Tuple[int, int]] = [
        (world, stage) for world in range(1, 9) for stage in range(1, 5)
    ]

    def __init__(self, rom_mode: RomModes = "vanilla", level: LevelModes = "random"):
        self.level = level
        self._rom_mode = rom_mode
        self.rand = np.random.RandomState()
        self.render_mode = "rgb_array"
        self._current_level_idx = -1  # Start before first level for sequential

        # Initialize empty grid - levels created lazily on first use
        self.envs = [[None for _ in range(4)] for _ in range(8)]

        if isinstance(level, tuple):
            # Single level mode: create only that level now
            world, stage = level
            self.envs[world - 1][stage - 1] = MarioBrosLevel(rom_mode=rom_mode, target=(world, stage))
        # For "random"/"sequential": envs remain empty, created lazily in _get_or_create_level

        self.reset()

    def _get_or_create_level(self, world: int, stage: int) -> MarioBrosLevel:
        """Get existing level or create it lazily if not yet initialized."""
        env = self.envs[world - 1][stage - 1]
        if env is None:
            env = MarioBrosLevel(rom_mode=self._rom_mode, target=(world, stage))
            self.envs[world - 1][stage - 1] = env
        return env

    def _next_level(self) -> MarioBrosLevel:
        if isinstance(self.level, str):
            if self.level == "random":
                # Pick random level target, create lazily if needed
                idx = self.rand.randint(0, len(self._ALL_LEVELS))
                world, stage = self._ALL_LEVELS[idx]
                return self._get_or_create_level(world, stage)
            elif self.level == "sequential":
                # Move to next level in sequence
                self._current_level_idx = (self._current_level_idx + 1) % len(self._ALL_LEVELS)
                world, stage = self._ALL_LEVELS[self._current_level_idx]
                return self._get_or_create_level(world, stage)

        elif isinstance(self.level, tuple):
            w, stage = self.level
            return self._get_or_create_level(w, stage)

        raise RuntimeError(f"Invalid level_mode: {repr(self.level)}")

    @property
    def current_level(self) -> str:
        """Get current level as string (e.g., '1-1', 'random', 'sequential')."""
        if self.env is None:
            if isinstance(self.level, str):
                return self.level
            else:
                return f"{self.level[0]}-{self.level[1]}"

        # Get the target from the current env
        for attr in ("_target", "target"):
            if hasattr(self.env, attr):
                world, stage = getattr(self.env, attr)
                return f"{world}-{stage}"

        # Try reading from ROM state
        if hasattr(self.env, "ram") and len(self.env.ram) > 0:
            try:
                world = self.env.ram[0x075F] + 1
                stage = self.env.ram[0x075C] + 1
                return f"{world}-{stage}"
            except Exception:
                pass

        # Fallback
        if isinstance(self.level, str):
            return self.level
        else:
            return f"{self.level[0]}-{self.level[1]}"

    def seed(self, seed=None):
        if seed is None:
            return []
        self.rand.seed(seed)
        return [seed]

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[gym.core.ObsType, dict]:
        """Reset the environment and return initial observation."""
        env = self.env = self._next_level()

        self.snapshot = env._backup
        self.restore_snapshot = env._restore
        self.get_keys_to_action = env.get_keys_to_action
        self.get_action_meanings = env.get_action_meanings

        return env.reset()  # type: ignore[no-any-return]

    def step(self, action: gym.core.ActType) -> Tuple[gym.core.ObsType, float, bool, bool, dict]:
        if self.env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self.env.step(action)  # type: ignore[no-any-return]

    def close(self):
        if self.env is None:
            raise ValueError("env has already been closed.")

        for stages in self.envs:
            for stage in stages:
                if stage is not None:
                    stage.close()

        self.env = None

        if self.viewer is not None:
            self.viewer.close()

    def render(self, mode="human"):
        if mode == "human":
            if self.viewer is None:
                self.viewer = ImageViewer(
                    caption=self.__class__.__name__,
                    height=NESEnv.height * 3,
                    width=NESEnv.width * 3,
                    monitor_keyboard=True,
                )
            self.viewer.show(self.env.screen)
        elif mode == "rgb_array":
            return self.env.screen
        else:
            render_modes = [repr(x) for x in self.metadata["render.modes"]]
            msg = "valid render modes are: {}".format(", ".join(render_modes))
            raise NotImplementedError(msg)
