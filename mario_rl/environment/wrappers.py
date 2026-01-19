from typing import Tuple
from collections import deque

import cv2
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict


class DeathPenaltyWrapper(gym.Wrapper):
    """
    Add a strong death penalty to encourage survival.
    
    The base gym-super-mario-bros only gives ~-25 for death, which is too weak
    for the agent to learn that death is catastrophic. This wrapper adds an
    additional penalty to make death strongly negative.
    
    Args:
        env: The environment to wrap
        penalty: Additional penalty to add on death (default -475 to bring total to ~-500)
    """
    
    def __init__(self, env: gym.Env, penalty: float = -475.0):
        super().__init__(env)
        self.penalty = penalty
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Add death penalty if terminated but NOT from flag capture
        if terminated and not info.get("flag_get", False):
            reward += self.penalty
        
        return obs, reward, terminated, truncated, info


class FlagBonusWrapper(gym.Wrapper):
    """
    Add a large bonus for capturing the flag to reinforce success.
    
    The base gym-super-mario-bros gives no special bonus for reaching the flag,
    just the normal forward movement reward (~2-5). This wrapper adds a large
    bonus to make flag capture strongly positive, symmetric with the death penalty.
    
    Args:
        env: The environment to wrap
        bonus: Bonus to add on flag capture (default +500)
    """
    
    def __init__(self, env: gym.Env, bonus: float = 500.0):
        super().__init__(env)
        self.bonus = bonus
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Add flag bonus if flag was captured
        if info.get("flag_get", False):
            reward += self.bonus
        
        return obs, reward, terminated, truncated, info


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


class RelativeDeathMemory(gym.Wrapper):
    """
    Wrapper that tracks death positions and provides them as relative distances.
    
    Instead of encoding absolute positions like "X=690 is dangerous", this encodes
    relative positions like "50 pixels ahead is dangerous". This helps the network
    generalize - it learns "danger 50px ahead = jump" which works for ALL pits.
    
    The death memory is stored as a vector in the info dict under 'death_memory'.
    Each bin represents a distance range from Mario's current position.
    """

    def __init__(
        self,
        env,
        num_bins: int = 16,
        max_lookahead: int = 256,
        max_lookbehind: int = 64,
        memory_size: int = 500,
        merge_threshold: int = 20,
    ):
        """
        Args:
            env: The environment to wrap
            num_bins: Number of distance bins (25% behind, 75% ahead)
            max_lookahead: Maximum distance ahead to track (pixels)
            max_lookbehind: Maximum distance behind to track (pixels)
            memory_size: Maximum deaths to remember per level
            merge_threshold: Merge deaths within this distance (pixels)
        """
        super().__init__(env)
        self.num_bins = num_bins
        self.max_lookahead = max_lookahead
        self.max_lookbehind = max_lookbehind
        self.memory_size = memory_size
        self.merge_threshold = merge_threshold
        
        # Store deaths per level: {(world, stage): [(x, count), ...]}
        self.death_memory: dict[tuple, list] = {}
        self.current_level = (1, 1)
        
        # Bin layout: 25% behind, 75% ahead
        self.behind_bins = num_bins // 4
        self.ahead_bins = num_bins - self.behind_bins
        self.behind_bin_size = max_lookbehind / max(self.behind_bins, 1)
        self.ahead_bin_size = max_lookahead / max(self.ahead_bins, 1)
    
    def _get_relative_death_vector(self, current_x: int) -> np.ndarray:
        """
        Create vector of death densities relative to current position.
        
        Bins are organized as: [behind_bins..., ahead_bins...]
        - Behind bins: distances from -max_lookbehind to 0
        - Ahead bins: distances from 0 to +max_lookahead
        """
        deaths = self.death_memory.get(self.current_level, [])
        death_vector = np.zeros(self.num_bins, dtype=np.float32)
        
        for death_x, count in deaths:
            relative_x = death_x - current_x
            
            if -self.max_lookbehind <= relative_x < 0:
                # Death is behind Mario
                bin_idx = int((-relative_x) / self.behind_bin_size)
                bin_idx = min(bin_idx, self.behind_bins - 1)
                # Reverse so closer deaths are in higher bins
                death_vector[self.behind_bins - 1 - bin_idx] += count
                
            elif 0 <= relative_x < self.max_lookahead:
                # Death is ahead of Mario
                bin_idx = int(relative_x / self.ahead_bin_size)
                bin_idx = min(bin_idx, self.ahead_bins - 1)
                death_vector[self.behind_bins + bin_idx] += count
        
        # Normalize to [0, 1]
        max_val = death_vector.max()
        if max_val > 0:
            death_vector = death_vector / max_val
        
        return death_vector
    
    def _record_death(self, x: int):
        """Record a death, merging nearby deaths to avoid duplicates."""
        if self.current_level not in self.death_memory:
            self.death_memory[self.current_level] = []
        
        deaths = self.death_memory[self.current_level]
        
        # Check if there's already a death nearby - if so, increment count
        for i, (dx, count) in enumerate(deaths):
            if abs(dx - x) < self.merge_threshold:
                deaths[i] = (dx, count + 1)
                return
        
        # New death location
        deaths.append((x, 1))
        
        # Limit memory size - keep most frequent deaths
        if len(deaths) > self.memory_size:
            deaths.sort(key=lambda d: d[1], reverse=True)
            self.death_memory[self.current_level] = deaths[:self.memory_size]
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Update current level
        self.current_level = (info.get("world", 1), info.get("stage", 1))
        
        # Record death position
        if done and reward < 0:
            self._record_death(info.get("x_pos", 0))
        
        # Add relative death vector to info
        info["death_memory"] = self._get_relative_death_vector(info.get("x_pos", 0))
        
        return obs, reward, done, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_level = (info.get("world", 1), info.get("stage", 1))
        info["death_memory"] = self._get_relative_death_vector(info.get("x_pos", 40))
        return obs, info
    
    def get_death_count(self) -> int:
        """Get total number of recorded deaths across all levels."""
        return sum(len(deaths) for deaths in self.death_memory.values())
    
    def get_death_positions(self, level: tuple | None = None) -> list[tuple[int, int]]:
        """Get death positions for a level (or current level if None)."""
        level = level or self.current_level
        return self.death_memory.get(level, [])

