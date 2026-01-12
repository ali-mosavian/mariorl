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

# Progressive milestones: closer together early, spread out later
# Designed to guide struggling agents while still rewarding late-game progress
MILESTONES: tuple[int, ...] = (100, 200, 350, 500, 750, 1000, 1500, 2000, 2500, 3000)


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
    milestones_reached: frozenset[int] = frozenset()  # Track which milestones have been reached


@dataclass(frozen=True)
class Reward:
    """
    Reward components for Mario RL training.
    
    Design principles:
    - alive_bonus: Small per-step reward for survival
    - x_reward: Main signal proportional to forward movement
    - speed_bonus: Rewards running over walking (skill-based)
    - milestone_bonus: Significant bonuses at progressive X positions
    - death_penalty: One-time penalty on death transition
    - finish_reward: Bonus for flag capture
    - powerup_reward: State change bonus/penalty
    
    Removed (to avoid reward farming):
    - coin_reward, score_reward, time_penalty, exploration_bonus
    """

    x_reward: int = 0
    speed_bonus: int = 0
    alive_bonus: int = 0  # +1 per step alive
    milestone_bonus: int = 0  # Number of NEW milestones reached this step
    death_penalty: int = 0
    finish_reward: int = 0
    powerup_reward: int = 0

    @staticmethod
    def calc(c: State, last: State) -> "Reward":
        """Calculate reward from state transition."""
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

        # Speed bonus: reward for high forward velocity
        # Only applies when moving forward (x_delta > 0)
        speed = 0
        if x_delta > 0:
            # Bonus scales with speed, capped to prevent exploitation
            # Normal walking ~1-2 pixels/frame, running ~3-4 pixels/frame
            speed = min(10, x_delta)  # Cap at 10 for very fast movement

        # Alive bonus: small constant reward for survival
        alive = 1 if c.is_alive and not just_died else 0

        # Milestone bonus: count NEW milestones reached this step
        new_milestones = 0
        for m in MILESTONES:
            if c.x_pos >= m and m not in last.milestones_reached:
                new_milestones += 1

        return Reward(
            x_reward=min(100, max(-15, x_delta)),
            speed_bonus=speed,
            alive_bonus=alive,
            milestone_bonus=new_milestones,
            death_penalty=int(just_died) * -1000,  # Only penalize on death transition
            finish_reward=int(c.got_flag) * 1000,
            powerup_reward=(c.powerup_state - last.powerup_state) * 100,
        )

    def total_reward(self) -> float:
        """
        Compute normalized reward for stable RL training.

        Components:
        - alive: +0.01 per step (survival incentive)
        - progress: -0.15 to +1.0 (main signal from x movement)
        - speed: 0 to +0.1 (rewards running over walking)
        - milestone: +2.0 each (at X=100,200,350,500,750,1000,1500,2000,2500,3000)
        - death: -1.0 (one-time on death transition)
        - flag: +5.0 (level completion)
        - powerup: -0.5 to +1.0 (state changes)

        Per-step range: ~-0.15 to ~3.1 (if hitting milestone while running)
        Terminal range: -1.0 to +5.0
        Episode range: ~-1 to ~110 (full level with flag)
        """
        # Survival incentive: +0.01 per step alive
        alive = self.alive_bonus * 0.01

        # Forward progress: main learning signal
        # x_reward ranges -15 to +100, scale to -0.15 to +1.0
        progress = self.x_reward / 100.0

        # Speed bonus: rewards running over walking
        # speed_bonus ranges 0 to 10, scale to 0 to +0.1
        speed = self.speed_bonus / 100.0

        # Milestone bonuses: significant guide posts
        # +2.0 per milestone reached (can hit multiple in one step if teleporting)
        milestone = self.milestone_bonus * 2.0

        # Death penalty: moderate, allows exploration
        # -1.0 (reduced from -2.0) when transitioning to dead
        death = self.death_penalty / 1000.0

        # Flag bonus: reduced from +15 to +5 (milestones provide intermediate goals)
        flag = self.finish_reward / 200.0

        # Powerup: small bonus/penalty for state changes
        powerup = self.powerup_reward / 200.0  # -0.5 to +1.0

        return alive + progress + speed + milestone + death + flag + powerup


class MarioBrosLevel(SuperMarioBrosEnv):
    # Reward range updated for new system:
    # Per-step: -0.15 to +3.1, Terminal: death=-1, flag=+5
    reward_range = (-1.5, 120)
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
        # Track which milestones have been reached based on current and previous max X
        x_pos = self._x_position
        x_pos_max = max(x_pos, self._last_state.x_pos_max) if self._last_state is not None else x_pos

        # Compute milestones reached (all milestones <= x_pos_max)
        milestones_reached = frozenset(m for m in MILESTONES if x_pos_max >= m)

        return State(
            time=self._time,
            score=self._score,
            x_pos=x_pos,
            x_pos_max=x_pos_max,
            coins=self._coins,
            powerup_state=self._powerup_state,
            got_flag=self._flag_get,
            is_alive=not (self._is_dying or self._is_dead),
            milestones_reached=milestones_reached,
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
