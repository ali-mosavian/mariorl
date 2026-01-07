"""
Running reward normalization for stable RL training.

Normalizes rewards using running mean and standard deviation,
similar to stable-baselines3's VecNormalize.
"""

from dataclasses import field
from dataclasses import dataclass

import numpy as np


@dataclass
class RunningMeanStd:
    """
    Tracks running mean and variance using Welford's algorithm.
    
    Numerically stable online algorithm for computing mean and variance.
    """
    
    mean: float = 0.0
    var: float = 1.0
    count: float = 1e-4  # Small epsilon to avoid division by zero
    
    def update(self, x: float) -> None:
        """Update running statistics with a new value."""
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.var += (delta * delta2 - self.var) / self.count
    
    def update_batch(self, batch: np.ndarray) -> None:
        """Update running statistics with a batch of values."""
        for x in batch.flat:
            self.update(x)
    
    @property
    def std(self) -> float:
        """Standard deviation (with minimum for stability)."""
        return max(np.sqrt(self.var), 1e-8)


@dataclass
class RewardNormalizer:
    """
    Normalizes rewards using running statistics.
    
    Maintains running mean/std of rewards and normalizes new rewards.
    Optionally clips normalized rewards to prevent outliers.
    
    Args:
        clip: Maximum absolute value for normalized rewards (0 to disable)
        gamma: Discount factor for return normalization (not used for per-step)
        epsilon: Small value added to std for numerical stability
    """
    
    clip: float = 10.0  # Clip normalized rewards to [-10, 10]
    epsilon: float = 1e-8
    
    # Running statistics
    _stats: RunningMeanStd = field(default_factory=RunningMeanStd)
    
    def normalize(self, reward: float, update: bool = True) -> float:
        """
        Normalize a reward using running statistics.
        
        Args:
            reward: Raw reward value
            update: Whether to update running statistics
            
        Returns:
            Normalized reward
        """
        if update:
            self._stats.update(reward)
        
        # Normalize: (reward - mean) / std
        normalized = (reward - self._stats.mean) / (self._stats.std + self.epsilon)
        
        # Optionally clip
        if self.clip > 0:
            normalized = float(np.clip(normalized, -self.clip, self.clip))
        
        return normalized
    
    def normalize_batch(self, rewards: np.ndarray, update: bool = True) -> np.ndarray:
        """Normalize a batch of rewards."""
        if update:
            self._stats.update_batch(rewards)
        
        normalized = (rewards - self._stats.mean) / (self._stats.std + self.epsilon)
        
        if self.clip > 0:
            normalized = np.clip(normalized, -self.clip, self.clip)
        
        return normalized
    
    @property
    def mean(self) -> float:
        """Current running mean."""
        return self._stats.mean
    
    @property
    def std(self) -> float:
        """Current running std."""
        return self._stats.std
    
    @property
    def count(self) -> int:
        """Number of samples seen."""
        return int(self._stats.count)

