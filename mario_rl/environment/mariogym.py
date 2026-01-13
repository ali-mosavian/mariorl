"""
Mario environment wrappers.

Uses the reward shaping from our forked gym-super-mario-bros:
- time_penalty:   -0.1        Every step costs, no camping!
- progress:       +0.5/pixel  Bonus for new forward progress
- momentum:       scales      Bonus for sustained speed
- powerup_loss:   -15.0       Penalty for losing powerup
- death:          -50.0       Heavy penalty
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
    """Multi-level Mario environment with random or sequential level selection."""
    
    rand: np.random.RandomState
    level: LevelModes
    envs: List[List[Optional[MarioBrosLevel]]]
    env: Optional[MarioBrosLevel] = None
    viewer: Optional[ImageViewer] = None

    metadata = {
        **SuperMarioBrosEnv.metadata,
    }
    reward_range = SuperMarioBrosEnv.reward_range
    action_space = SuperMarioBrosEnv.action_space
    observation_space = SuperMarioBrosEnv.observation_space

    def __init__(self, rom_mode: RomModes = "vanilla", level: LevelModes = "random"):
        self.level = level
        self.rand = np.random.RandomState()
        self.render_mode = "rgb_array"

        if isinstance(level, tuple):
            world, stage = level
            self.envs = [[None for _ in range(4)] for _ in range(8)]
            self.envs[world - 1][stage - 1] = MarioBrosLevel(rom_mode=rom_mode, target=(world, stage))
        else:
            self.envs = [
                [MarioBrosLevel(rom_mode=rom_mode, target=(world, stage)) for stage in range(1, 5)]
                for world in range(1, 9)
            ]

        self.reset()

    def _next_level(self) -> MarioBrosLevel:
        if isinstance(self.level, str):
            levels: list[MarioBrosLevel] = [e for row in self.envs for e in row if e is not None]
            if self.level == "random":
                return self.rand.choice(levels, 1)[0]  # type: ignore[no-any-return]
            elif self.level == "sequential":
                env = self.env or levels[-1]
                return levels[(levels.index(env) + 1) % len(levels)]

        elif isinstance(self.level, tuple):
            w, level = self.level
            result = self.envs[w - 1][level - 1]
            if result is None:
                raise RuntimeError(f"Level {w}-{level} not initialized")
            return result

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
