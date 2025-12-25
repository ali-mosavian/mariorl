"""
Environment factory for stable-baselines3 compatibility.

Creates Mario environments with channels-first observation format
required by SB3's CNN policies.
"""

from typing import Literal
from typing import Callable

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box
from nes_py.wrappers import JoypadSpace
from gymnasium.wrappers import GrayscaleObservation
from gymnasium.wrappers import TransformObservation
from gymnasium.wrappers import FrameStackObservation
from gym_super_mario_bros import actions as smb_actions

from mario_rl.environment.wrappers import SkipFrame
from mario_rl.environment.wrappers import ResizeObservation
from mario_rl.environment.mariogym import SuperMarioBrosMultiLevel

LevelType = Literal["sequential", "random"] | tuple[Literal[1, 2, 3, 4, 5, 6, 7, 8], Literal[1, 2, 3, 4]]


class ChannelsFirstWrapper(TransformObservation):
    """
    Convert observation from (H, W, C) or (F, H, W, C) to channels-first format.

    For frame-stacked observations (F, H, W, C) -> (F, H, W) by squeezing C=1.
    This gives SB3 CNN a (4, 64, 64) input which it treats as 4 channels.
    """

    def __init__(self, env: Env):
        # Get original shape from Box space
        obs_space = env.observation_space
        assert isinstance(obs_space, Box), f"Expected Box space, got {type(obs_space)}"
        old_shape = obs_space.shape
        old_low = float(obs_space.low.flat[0])
        old_high = float(obs_space.high.flat[0])

        # (F, H, W, C) -> (F, H, W) where C=1
        assert old_shape is not None, "Observation space shape is None"
        if len(old_shape) == 4 and old_shape[-1] == 1:
            new_shape = old_shape[:3]  # (F, H, W)
        else:
            raise ValueError(f"Expected shape (F, H, W, 1), got {old_shape}")

        new_obs_space = Box(
            low=old_low,
            high=old_high,
            shape=new_shape,
            dtype=np.float32,
        )

        super().__init__(
            env,
            func=lambda obs: obs.squeeze(-1),  # Remove last dim (C=1)
            observation_space=new_obs_space,
        )


def make_mario_env(
    level: LevelType = "random",
    render_mode: str | None = None,
) -> Env:
    """
    Create a Mario environment compatible with stable-baselines3.

    Args:
        level: Level selection mode ("random", "sequential", or (world, stage) tuple)
        render_mode: "human" for rendering, None for headless

    Returns:
        Gymnasium environment with observation shape (4, 64, 64)
    """
    render_frames = render_mode == "human"

    # Fix pyglet key import for nes_py viewer (only needed when rendering)
    if render_frames:
        try:
            from pyglet.window import key
            import nes_py._image_viewer as _iv

            _iv.key = key
        except Exception:
            pass

    # Create base environment
    base_env = SuperMarioBrosMultiLevel(level=level)
    env = JoypadSpace(base_env, actions=smb_actions.COMPLEX_MOVEMENT)

    # Apply wrappers
    env = SkipFrame(env, skip=4, render_frames=render_frames)
    env = GrayscaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, shape=64)
    env = TransformObservation(
        env,
        func=lambda x: x.astype(np.float32) / 255.0,
        observation_space=Box(low=0.0, high=1.0, shape=(64, 64, 1), dtype=np.float32),
    )
    env = FrameStackObservation(env, stack_size=4)

    # Convert to channels-first for SB3 CNN: (4, 64, 64, 1) -> (4, 64, 64)
    env = ChannelsFirstWrapper(env)

    return env


def make_env_fn(
    level: LevelType = "random",
    seed: int = 0,
) -> Callable[[], Env]:
    """
    Create an environment factory function for SubprocVecEnv.

    Args:
        level: Level selection mode
        seed: Random seed for this environment

    Returns:
        Callable that creates a new environment instance
    """

    def _init() -> Env:
        env = make_mario_env(level=level, render_mode=None)
        env.reset(seed=seed)
        return env  # type: ignore[return-value]

    return _init
