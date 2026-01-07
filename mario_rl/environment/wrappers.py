from typing import Tuple

import cv2
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box


class ResizeObservation(gym.ObservationWrapper):
    """Resize observations using OpenCV (much faster than skimage)."""
    
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        # cv2.resize is ~10x faster than skimage.transform.resize
        # INTER_AREA is best for downscaling (avoids aliasing)
        # Note: cv2.resize takes (width, height), but self.shape is (height, width)
        resized = cv2.resize(observation, (self.shape[1], self.shape[0]), interpolation=cv2.INTER_AREA)
        # cv2.resize strips single channel dim, restore it if needed
        if len(observation.shape) == 3 and observation.shape[2] == 1 and len(resized.shape) == 2:
            resized = resized[:, :, np.newaxis]
        return resized


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip, render_frames: bool = False, playback_speed: float = 1.0):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip
        self.render_frames = render_frames
        self.next_render_time = None
        self.playback_speed = playback_speed

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for _i in range(self._skip):
            # Accumulate reward and repeat the same action
            s_, r, done, truncated, info = self.env.step(action)
            total_reward += r

            if self.render_frames:
                import time

                # Calculate when the next frame should be rendered
                current_time = time.time()
                if self.next_render_time is None or current_time >= self.next_render_time:
                    self.env.render()
                    # Schedule next render time based on playback speed
                    # At speed 1.0: render every 1/60 sec (60 fps)
                    frame_time = (1.0 / 60.0) / self.playback_speed
                    self.next_render_time = current_time + frame_time

            if done or truncated:
                break
        return s_, total_reward, done, truncated, info

    def reset(self, **kwargs):
        self.next_render_time = None  # Reset timing when env resets
        return self.env.reset(**kwargs)


class GrayScaleObservation(gym.ObservationWrapper):
    """Convert observation to grayscale."""

    def __init__(self, env, keep_dim=False):
        super().__init__(env)
        self.keep_dim = keep_dim

        # Get the original shape from the observation space
        original_shape = env.observation_space.shape
        assert len(original_shape) == 3 and original_shape[2] == 3, "Expected shape (H, W, 3)"

        # New shape either (H, W, 1) or (H, W)
        if self.keep_dim:
            new_shape = original_shape[:2] + (1,)
        else:
            new_shape = original_shape[:2]

        # Update observation space
        self.observation_space = Box(low=0, high=255, shape=new_shape, dtype=np.uint8)

    def observation(self, obs):
        # RGB to grayscale conversion using standard weights
        obs = np.dot(obs[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        if self.keep_dim:
            obs = np.expand_dims(obs, axis=-1)
        return obs


class FrameStack(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._stack = []

    def reset(self, **kwargs) -> Tuple[gym.core.ObsType, dict]:
        return self.env.reset(**kwargs)  # type: ignore[no-any-return]

    def step(self, action) -> Tuple[gym.core.ObsType, float, bool, bool, dict]:
        return self.env.step(action)  # type: ignore[no-any-return]
