import dataclasses

from typing import List
from typing import Tuple
from typing import Literal
from typing import Optional
from dataclasses import dataclass

import numpy as np
import gymnasium as gym

from nes_py import NESEnv
from gym_super_mario_bros import SuperMarioBrosEnv
from nes_py._image_viewer import ImageViewer


RomModes = Literal["vanilla", "pixel", "downsample"]
LevelModes = (
    Literal["sequential", "random"]
    | Tuple[Literal[1, 2, 3, 4, 5, 6, 7, 8], Literal[1, 2, 3, 4]]
)


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
    def calc(c: State, l: State) -> "Reward":
        return Reward(
            x_reward=min(100, max(-15, c.x_pos - l.x_pos_max)),
            time_penalty=min(0, c.time - l.time),
            death_penalty=(not c.is_alive) * -1000,
            coin_reward=min(3, max(0, c.coins - l.coins)),
            score_reward=min(3, max(0, c.score - l.score)),
            powerup_reward=(c.powerup_state - l.powerup_state) * 100,
            finish_reward=c.got_flag * 1000,
        )

    def total_reward(self) -> float:
        return (
            self.x_reward
            + self.powerup_reward
            +
            # self.coin_reward +
            # self.score_reward +
            # self.time_penalty +
            self.death_penalty
            + self.finish_reward
            + -0.1
        )


# -


class MarioBrosLevel(SuperMarioBrosEnv):
    reward_range = (-25, 25)
    _last_state: Optional[State] = None

    def __init__(
        self,
        rom_mode: RomModes = "vanilla",
        lost_levels: bool = False,
        target: Tuple[int, int] = None,
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
                max(self._x_position, self._last_state.x_pos_max)
                if self._last_state is not None
                else self._x_position
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
        return {
            "reward": dataclasses.asdict(self._reward_state),
            **SuperMarioBrosEnv._get_info(self),
            "is_dying": self._is_dying,
            "is_dead": self._is_dead,
            "state": dataclasses.asdict(self.state),
            "last_state": dataclasses.asdict(self._last_state)
            if self._last_state
            else None,
        }


class SuperMarioBrosMultiLevel(gym.Env):
    rand: np.random
    level: LevelModes
    envs: List[List[MarioBrosLevel]]
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
            self.envs = [[None] * 4] * 8
            self.envs[world - 1][stage - 1] = MarioBrosLevel(
                rom_mode=rom_mode, target=(world, stage)
            )
        else:
            self.envs = [
                [
                    MarioBrosLevel(rom_mode=rom_mode, target=(world, stage))
                    for stage in range(1, 5)
                ]
                for world in range(1, 9)
            ]

        self.reset()

    def _next_level(self) -> MarioBrosLevel:
        if isinstance(self.level, str):
            levels = sum(self.envs, [])
            if self.level == "random":
                return self.rand.choice(levels, 1)[0]
            elif self.level == "sequential":
                env = self.env or levels[-1]
                return levels[(levels.index(env) + 1) % len(levels)]

        elif isinstance(self.level, tuple):
            w, l = self.level
            return self.envs[w - 1][l - 1]

        raise RuntimeError(f"Invalid level_mode: {repr(self.level)}")

    def seed(self, seed=None):
        if seed is None:
            return []

        self.rand.seed(seed)
        return [seed]

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[gym.core.ObsType, dict]:
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
        return env.reset()

    def step(
        self, action: gym.core.ActType
    ) -> Tuple[gym.core.ObsType, float, bool, bool, dict]:
        return self.env.step(action)

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
