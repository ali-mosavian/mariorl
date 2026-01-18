from typing import Tuple
from collections import deque

import cv2
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict


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
    def __init__(
        self,
        env,
        skip,
        render_frames: bool = False,
        playback_speed: float = 1.0,
        sum_rewards: bool = False,
    ):
        """Return only every `skip`-th frame.
        
        Args:
            env: The environment to wrap
            skip: Number of frames to skip (repeat action)
            render_frames: Whether to render frames
            playback_speed: Playback speed multiplier
            sum_rewards: If True, sum rewards across skipped frames (old behavior).
                        If False, return only the last frame's reward (new default).
                        Using last reward gives cleaner credit assignment - the reward
                        reflects the outcome state, not intermediate transitions.
        """
        super().__init__(env)
        self._skip = skip
        self.render_frames = render_frames
        self.next_render_time = None
        self.playback_speed = playback_speed
        self.sum_rewards = sum_rewards

    def step(self, action):
        """Repeat action for skip frames, return last reward by default."""
        total_reward = 0.0
        last_reward = 0.0
        done = False
        for _i in range(self._skip):
            # Repeat the same action
            s_, r, done, truncated, info = self.env.step(action)
            total_reward += r
            last_reward = r

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
        
        # Return summed or last reward based on configuration
        reward = total_reward if self.sum_rewards else last_reward
        return s_, reward, done, truncated, info

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
        # cv2.cvtColor is much faster than np.dot for grayscale conversion
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        if self.keep_dim:
            gray = gray[:, :, np.newaxis]
        return gray


class ActionHistoryWrapper(gym.Wrapper):
    """
    Wrapper that tracks previous actions and includes them in the info dict.
    
    This allows the agent to know what actions it took recently, which helps
    with understanding action effects (e.g., "I pressed jump last frame, so
    I'm probably mid-air now").
    
    The action history is stored as a one-hot encoded array in the info dict
    under the key 'action_history'. The network can use this to correlate
    visual observations with recent actions.
    """

    def __init__(self, env, history_len: int = 4, num_actions: int = 7):
        """
        Args:
            env: The environment to wrap
            history_len: Number of previous actions to track
            num_actions: Size of action space (for one-hot encoding)
        """
        super().__init__(env)
        self.history_len = history_len
        self.num_actions = num_actions
        self.action_history: deque = deque(maxlen=history_len)
        
        # Initialize with "no action" (all zeros, or could use a special token)
        self._reset_history()
    
    def _reset_history(self):
        """Reset action history to initial state (no actions taken)."""
        self.action_history.clear()
        for _ in range(self.history_len):
            self.action_history.append(-1)  # -1 indicates no action yet
    
    def _get_action_history_array(self) -> np.ndarray:
        """Get action history as one-hot encoded array.
        
        Returns:
            Array of shape (history_len, num_actions) with one-hot encoding.
            Actions that haven't been taken yet (-1) are all zeros.
        """
        history = np.zeros((self.history_len, self.num_actions), dtype=np.float32)
        for i, action in enumerate(self.action_history):
            if action >= 0:
                history[i, action] = 1.0
        return history
    
    def reset(self, **kwargs):
        """Reset environment and action history."""
        self._reset_history()
        obs, info = self.env.reset(**kwargs)
        info['action_history'] = self._get_action_history_array()
        return obs, info
    
    def step(self, action):
        """Take action and update history."""
        # Record this action in history
        self.action_history.append(action)
        
        # Step environment
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Add action history to info
        info['action_history'] = self._get_action_history_array()
        
        return obs, reward, done, truncated, info

