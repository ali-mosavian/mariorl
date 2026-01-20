"""
Environment factory for creating wrapped Mario environments.

This is the single source of truth for environment creation.
All other modules should use this factory.
"""

from typing import Any
from typing import Tuple

import numpy as np
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros import actions as smb_actions

from mario_rl.core.config import LevelType
from mario_rl.environment.wrappers import SkipFrame
from mario_rl.environment.frame_stack import FrameStack
from mario_rl.environment.wrappers import FlagBonusWrapper
from mario_rl.environment.wrappers import ResizeObservation
from mario_rl.environment.wrappers import DeathPenaltyWrapper
from mario_rl.environment.wrappers import ActionHistoryWrapper
from mario_rl.environment.wrappers import GrayScaleObservation
from mario_rl.environment.wrappers import RAMObservationWrapper
from mario_rl.environment.mariogym import SuperMarioBrosMultiLevel


class MarioEnvironment:
    """
    Container for a wrapped Mario environment and its components.

    Provides access to the full wrapped environment, the base environment
    (for save/restore), and the frame stack wrapper (if using pixel obs).
    """

    def __init__(
        self,
        env: Any,
        base_env: SuperMarioBrosMultiLevel,
        fstack: FrameStack | None,
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

    def render(self, mode: str = "rgb_array") -> np.ndarray | None:
        """Render the current frame.
        
        Args:
            mode: "rgb_array" returns RGB numpy array, "human" displays window
            
        Returns:
            RGB frame as numpy array if mode="rgb_array", None otherwise
        """
        return self.base_env.render(mode=mode)

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass


def create_mario_env(
    level: LevelType = (1, 1),
    render_frames: bool = False,
    lz4_compress: bool = True,
    sum_rewards: bool = False,
    action_history_len: int = 4,
) -> MarioEnvironment:
    """
    Create a wrapped Mario environment.

    Args:
        level: Level specification - tuple (world, stage) or "random"/"sequential"
        render_frames: Whether to render frames for visualization
        lz4_compress: Whether to use LZ4 compression for frame stacking (saves memory)
        sum_rewards: If True, sum rewards across frame skips (old behavior).
                    If False (default), use only the last reward for cleaner credit assignment.
        action_history_len: If > 0, track this many previous actions and include in info dict.
                           Helps the network understand action effects (e.g., jump timing).

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
    env = JoypadSpace(base_env, actions=smb_actions.SIMPLE_MOVEMENT)
    # Add strong death penalty (-475 to bring total death cost from ~-25 to ~-500)
    env = DeathPenaltyWrapper(env, penalty=-475.0)
    # Add flag capture bonus (+500 to make success strongly positive, symmetric with death)
    env = FlagBonusWrapper(env, bonus=500.0)
    env = SkipFrame(env, skip=4, render_frames=render_frames, sum_rewards=sum_rewards)
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=64)

    # Add action history tracking if requested
    if action_history_len > 0:
        env = ActionHistoryWrapper(env, history_len=action_history_len, num_actions=7)

    # Note: Normalization (x/255) is done in the neural network on GPU
    # This keeps observations as uint8 (smaller memory, faster transfer)
    # LZ4 compression further reduces memory for replay buffer storage
    fstack = FrameStack(env, num_stack=4, lz4_compress=lz4_compress)

    return MarioEnvironment(env=fstack, base_env=base_env, fstack=fstack)


def create_ram_env(
    level: LevelType = (1, 1),
    render_frames: bool = False,
) -> MarioEnvironment:
    """
    Create a RAM-based Mario environment.

    Returns NES RAM (2048 bytes) as observation instead of pixel frames.
    This is a simpler, lower-dimensional representation that contains the
    full game state.

    Args:
        level: Level specification - tuple (world, stage) or "random"/"sequential"
        render_frames: Whether to render frames for visualization

    Returns:
        MarioEnvironment with RAM observations of shape (2048,)
    """
    if render_frames:
        try:
            from pyglet.window import key
            import nes_py._image_viewer as _iv

            _iv.key = key
        except Exception:
            pass

    base_env = SuperMarioBrosMultiLevel(level=level)
    env = JoypadSpace(base_env, actions=smb_actions.SIMPLE_MOVEMENT)
    # Add strong death penalty (-475 to bring total death cost from ~-25 to ~-500)
    env = DeathPenaltyWrapper(env, penalty=-475.0)
    # Add flag capture bonus (+500 to make success strongly positive, symmetric with death)
    env = FlagBonusWrapper(env, bonus=500.0)
    env = SkipFrame(env, skip=4, render_frames=render_frames, sum_rewards=False)
    # Return RAM instead of pixels - no frame stacking needed
    env = RAMObservationWrapper(env)

    return MarioEnvironment(env=env, base_env=base_env, fstack=None)


# Backwards compatibility - returns tuple like old create_env
def create_env(
    level: LevelType = (1, 1),
    render_frames: bool = False,
    lz4_compress: bool = True,
) -> Tuple[Any, SuperMarioBrosMultiLevel, FrameStack | None]:
    """
    Create a wrapped Mario environment (legacy interface).

    Args:
        level: Level specification
        render_frames: Whether to render frames
        lz4_compress: Whether to use LZ4 compression for frame stacking

    Returns:
        Tuple of (env, base_env, fstack)
    """
    mario_env = create_mario_env(level, render_frames, lz4_compress)
    return mario_env.env, mario_env.base_env, mario_env.fstack
