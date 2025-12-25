import time

from typing import Tuple
from collections import deque

import numpy as np
import gymnasium as gym

from skimage import transform
from gymnasium.spaces import Box
from gym_super_mario_bros import SuperMarioBrosEnv


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        resize_obs = transform.resize(observation, self.shape)
        # cast float back to uint8
        resize_obs *= 255
        resize_obs = resize_obs.astype(np.uint8)
        return resize_obs


class SkipFrame(gym.Wrapper):
    def __init__(
        self, env, skip, render_frames: bool = False, playback_speed: float = 1.0
    ):
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
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            s_, r, done, truncated, info = self.env.step(action)
            total_reward += r
            if done or truncated:
                break

            if self.render_frames:
                self.env.render()

        return s_, total_reward, done, truncated, info


class SwapAxes(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    @staticmethod
    def _swap(x):
        return x.swapaxes(-1, -2).swapaxes(-2, -3)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return SwapAxes._swap(obs), info

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        return (SwapAxes._swap(state), reward, terminated, truncated, info)


class NESCompatabilityLayer(gym.core.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def _get_info(self):
        return {
            **SuperMarioBrosEnv._get_info(self.env.unwrapped),
            "is_dying": self.env.unwrapped._is_dying,
            "is_dead": self.env.unwrapped._is_dead,
        }

    def reset(self, **kwargs) -> Tuple[gym.core.ObsType, dict]:
        return self.env.reset(**kwargs), self._get_info()

    def step(
        self, act: gym.core.ActType
    ) -> Tuple[gym.core.ObsType, float, bool, bool, dict]:
        return (
            *self.env.step(act)[:-1],
            self._get_info(),
        )


class NESRamObservation(gym.core.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=env.unwrapped.ram.shape, dtype="u1"
        )

    def observation(self, _) -> np.ndarray:
        return self.env.unwrapped.ram


class NESSaveRestore(gym.core.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._stack = list()

    def reset(self, **kwargs) -> Tuple[gym.core.ObsType, dict]:
        self._stack.clear()
        return self.env.reset(**kwargs)

    def push(self):
        self._stack.append(self.env.unwrapped.env.ram.copy())

    def pop(self):
        if len(self._stack) == 0:
            return

        state = self._stack.pop()
        self.env.unwrapped.env.ram[:] = state[:]
