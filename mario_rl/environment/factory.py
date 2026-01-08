"""
Environment factory for creating wrapped Mario environments.

This is the single source of truth for environment creation.
All other modules should use this factory.
"""

from typing import Any
from typing import Tuple

import numpy as np
from nes_py.wrappers import JoypadSpace
from gymnasium.wrappers import FrameStackObservation
from gym_super_mario_bros import actions as smb_actions

from mario_rl.core.config import LevelType
from mario_rl.environment.wrappers import SkipFrame
from mario_rl.environment.wrappers import ResizeObservation
from mario_rl.environment.wrappers import GrayScaleObservation
from mario_rl.environment.mariogym import SuperMarioBrosMultiLevel


class MarioEnvironment:
    """
    Container for a wrapped Mario environment and its components.

    Provides access to the full wrapped environment, the base environment
    (for save/restore), and the frame stack wrapper.
    """

    def __init__(
        self,
        env: Any,
        base_env: SuperMarioBrosMultiLevel,
        fstack: FrameStackObservation,
    ):
        self.env = env
        self.base_env = base_env
        self.fstack = fstack

    @property
    def action_space(self) -> Any:
        return self.env.action_space

    @property
    def observation_space(self) -> Any:
        return self.env.observation_space

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        result = self.env.reset(**kwargs)
        return (np.asarray(result[0]), dict(result[1]))

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        result = self.env.step(action)
        return (np.asarray(result[0]), float(result[1]), bool(result[2]), bool(result[3]), dict(result[4]))

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass


def create_mario_env(
    level: LevelType = (1, 1),
    render_frames: bool = False,
) -> MarioEnvironment:
    """
    Create a wrapped Mario environment.

    Args:
        level: Level specification - tuple (world, stage) or "random"/"sequential"
        render_frames: Whether to render frames for visualization

    Returns:
        MarioEnvironment containing the wrapped env and components
    """
    if render_frames:
        try:
            from pyglet.window import key
            import nes_py._image_viewer as _iv

            _iv.key = key
        except Exception:
            pass

    base_env = SuperMarioBrosMultiLevel(level=level)
    env = JoypadSpace(base_env, actions=smb_actions.COMPLEX_MOVEMENT)
    env = SkipFrame(env, skip=4, render_frames=render_frames)
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=64)
    # Note: Normalization (x/255) is done in the neural network on GPU
    # This keeps observations as uint8 (smaller memory, faster transfer)
    fstack = FrameStackObservation(env, stack_size=4)

    return MarioEnvironment(env=fstack, base_env=base_env, fstack=fstack)


# Backwards compatibility - returns tuple like old create_env
def create_env(
    level: LevelType = (1, 1),
    render_frames: bool = False,
) -> Tuple[Any, SuperMarioBrosMultiLevel, FrameStackObservation]:
    """
    Create a wrapped Mario environment (legacy interface).

    Args:
        level: Level specification
        render_frames: Whether to render frames

    Returns:
        Tuple of (env, base_env, fstack)
    """
    mario_env = create_mario_env(level, render_frames)
    return mario_env.env, mario_env.base_env, mario_env.fstack
