import dataclasses
from typing import List
from typing import Tuple
from typing import Literal
from typing import Optional
from dataclasses import dataclass

import numpy as np
import gymnasium as gym
from nes_py import NESEnv
from nes_py._image_viewer import ImageViewer
from gym_super_mario_bros import SuperMarioBrosEnv

RomModes = Literal["vanilla", "pixel", "downsample"]
LevelModes = Literal["sequential", "random"] | Tuple[Literal[1, 2, 3, 4, 5, 6, 7, 8], Literal[1, 2, 3, 4]]

# Timeout threshold: if game time <= this value when dying, it's a timeout (not skill death)
TIMEOUT_THRESHOLD = 10


@dataclass(frozen=True)
class State:
    time: int = 0
    score: int = 0
    x_pos: int = 0
    x_pos_max: int = 0
    coins: int = 0
    powerup_state: int = 0
    is_alive: bool = True
    got_flag: bool = False


# +


@dataclass(frozen=True)
class Reward:
    x_reward: int = 0
    time_penalty: int = 0
    death_penalty: int = 0
    coin_reward: int = 0
    score_reward: int = 0
    powerup_reward: int = 0
    finish_reward: int = 0

    @staticmethod
    def calc(c: State, last: State) -> "Reward":
        # Detect death transition (was alive, now dead)
        just_died = last.is_alive and not c.is_alive
        
        # When dying or dead, ignore x_delta to avoid spurious negative rewards
        # from position resetting during death animation
        if just_died or not c.is_alive:
            x_delta = 0
        else:
            # Use position DELTA (current - previous), not (current - max)
            # This avoids huge negative rewards after death when x resets
            x_delta = c.x_pos - last.x_pos
        
        return Reward(
            x_reward=min(100, max(-15, x_delta)),
            time_penalty=min(0, c.time - last.time),
            death_penalty=int(just_died) * -1000,  # Only penalize on death transition
            coin_reward=min(3, max(0, c.coins - last.coins)),
            score_reward=min(3, max(0, c.score - last.score)),
            powerup_reward=(c.powerup_state - last.powerup_state) * 100,
            finish_reward=c.got_flag * 1000,
        )

    def total_reward(self) -> float:
        """
        Compute normalized reward for stable RL training.
        
        Design principles:
        - Forward progress is the PRIMARY signal (+0.1 to +1 per frame)
        - Flag completion is a SIGNIFICANT bonus (+15) to incentivize finishing
        - Death is a moderate penalty (-2) - enough to discourage but not overwhelming
        - Per-frame range: ~-0.15 to ~+1.0 (normal movement)
        - Terminal events: death=-2, flag=+15
        
        With restarts/resume, the agent needs to relearn early, so keeping
        death penalty low allows more exploration without crushing Q-values.
        """
        # Forward progress: main learning signal
        # x_reward ranges -15 to +100, scale to -0.15 to +1.0
        progress = self.x_reward / 100.0
        
        # Death penalty: moderate, not overwhelming
        # -2 is meaningful but allows recovery during exploration
        # Only applied ONCE on death transition, not every frame while dead
        death = self.death_penalty / 500.0  # -2 when transitioning to dead
        
        # Flag bonus: significant reward to incentivize level completion  
        # +15 makes completing the level worth finishing
        flag = self.finish_reward / 66.67  # +15 when flag
        
        # Powerup: small bonus/penalty
        powerup = self.powerup_reward / 200.0  # -0.5 to +1
        
        return progress + death + flag + powerup


# -


class MarioBrosLevel(SuperMarioBrosEnv):
    reward_range = (-2.5, 16)  # Normal: -0.15 to +1.0, Terminal: death=-2, flag=+15
    _last_state: Optional[State] = None

    def __init__(
        self,
        rom_mode: RomModes = "vanilla",
        lost_levels: bool = False,
        target: Tuple[int, int] | None = None,
    ):
        super(MarioBrosLevel, self).__init__(rom_mode, lost_levels, target)
        self.render_mode = "rgb_array"

    @property
    def state(self) -> State:
        return State(
            time=self._time,
            score=self._score,
            x_pos=self._x_position,
            x_pos_max=(
                max(self._x_position, self._last_state.x_pos_max) if self._last_state is not None else self._x_position
            ),
            coins=self._coins,
            powerup_state=self._powerup_state,
            got_flag=self._flag_get,
            is_alive=not (self._is_dying or self._is_dead),
        )

    @property
    def _powerup_state(self):
        return max(0, min(2, self.ram[0x0756]))

    @property
    def _reward_state(self) -> Reward:
        return Reward.calc(self.state, self._last_state or self.state)

    def _get_reward(self) -> float:
        return self._reward_state.total_reward()

    def _did_reset(self):
        SuperMarioBrosEnv._did_reset(self)
        self._last_state = None

    def step(self, action):
        self._last_state = self.state
        return SuperMarioBrosEnv.step(self, action)

    def _get_info(self):
        # Timeout = died because timer ran out (not a skill-based death)
        is_dying_or_dead = self._is_dying or self._is_dead
        is_timeout = is_dying_or_dead and self._time <= TIMEOUT_THRESHOLD and not self._flag_get
        
        return {
            "reward": dataclasses.asdict(self._reward_state),
            **SuperMarioBrosEnv._get_info(self),
            "is_dying": self._is_dying,
            "is_dead": self._is_dead,
            "is_timeout": is_timeout,
            "state": dataclasses.asdict(self.state),
            "last_state": dataclasses.asdict(self._last_state) if self._last_state else None,
        }


class SuperMarioBrosMultiLevel(gym.Env):
    rand: np.random.RandomState
    level: LevelModes
    envs: List[List[Optional[MarioBrosLevel]]]
    env: Optional[MarioBrosLevel] = None
    viewer: Optional[ImageViewer] = None

    metadata = {
        **SuperMarioBrosEnv.metadata,
    }
    reward_range = MarioBrosLevel.reward_range
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

        # Get the target from the current env (check all possible attributes)
        for attr in ("_target", "target", "_world_number", "_stage_number"):
            if hasattr(self.env, attr):
                if attr in ("_target", "target"):
                    world, stage = getattr(self.env, attr)
                    return f"{world}-{stage}"

        # Try reading from ROM state
        if hasattr(self.env, "ram") and len(self.env.ram) > 0:
            # These are memory addresses in Super Mario Bros NES ROM
            try:
                world = self.env.ram[0x075F] + 1  # World number (0-indexed in RAM)
                stage = self.env.ram[0x075C] + 1  # Stage number (0-indexed in RAM)
                return f"{world}-{stage}"
            except Exception:
                pass

        # Fallback to level mode
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
        """
        Reset the state of the environment and returns an initial observation.

        Returns:
            state (np.ndarray): next frame as a result of the given action

        """
        # select a new level
        env = self.env = self._next_level()

        self.snapshot = env._backup
        self.restore_snapshot = env._restore
        self.get_keys_to_action = env.get_keys_to_action
        self.get_action_meanings = env.get_action_meanings

        # reset the environment
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
