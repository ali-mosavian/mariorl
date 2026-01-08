"""
Per-level metrics tracking.

Tracks statistics per game level (e.g., 1-1, 4-2) for analyzing
which levels are hardest and where the agent struggles.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LevelStats:
    """Statistics for a single game level.
    
    Tracks attempts, completions, deaths, and their positions/times.
    Provides derived metrics like completion rate and death hotspots.
    """
    
    level_id: str
    
    # Counters
    attempts: int = 0
    completions: int = 0
    deaths: int = 0
    
    # Best values
    best_x: int = 0
    best_completion_time: int = 999999
    
    # Rolling histories (capped to avoid unbounded memory)
    completion_times: deque[int] = field(default_factory=lambda: deque(maxlen=50))
    rewards: deque[float] = field(default_factory=lambda: deque(maxlen=50))
    speeds: deque[float] = field(default_factory=lambda: deque(maxlen=50))
    death_positions: deque[int] = field(default_factory=lambda: deque(maxlen=100))
    
    def start_attempt(self) -> None:
        """Record starting an attempt on this level."""
        self.attempts += 1
    
    def record_death(self, x_pos: int) -> None:
        """Record a death at the given x position.
        
        Args:
            x_pos: X position where death occurred
        """
        self.deaths += 1
        self.death_positions.append(x_pos)
        if x_pos > self.best_x:
            self.best_x = x_pos
    
    def record_completion(self, game_time: int, reward: float) -> None:
        """Record successfully completing the level.
        
        Args:
            game_time: Game ticks to complete
            reward: Total reward for the episode
        """
        self.completions += 1
        self.completion_times.append(game_time)
        self.rewards.append(reward)
        if game_time < self.best_completion_time:
            self.best_completion_time = game_time
    
    @property
    def completion_rate(self) -> float:
        """Completion rate (completions / attempts)."""
        return self.completions / self.attempts if self.attempts > 0 else 0.0
    
    @property
    def avg_death_x(self) -> float:
        """Average x position of deaths."""
        if not self.death_positions:
            return 0.0
        return sum(self.death_positions) / len(self.death_positions)
    
    @property
    def avg_completion_time(self) -> float:
        """Average completion time."""
        if not self.completion_times:
            return 0.0
        return sum(self.completion_times) / len(self.completion_times)
    
    @property
    def avg_reward(self) -> float:
        """Average reward on this level."""
        if not self.rewards:
            return 0.0
        return sum(self.rewards) / len(self.rewards)
    
    @property
    def death_histogram(self) -> dict[int, int]:
        """Death positions bucketed by 100 pixels.
        
        Returns:
            Dict mapping bucket start (0, 100, 200, ...) to count.
        """
        buckets: dict[int, int] = {}
        for x in self.death_positions:
            bucket = (x // 100) * 100
            buckets[bucket] = buckets.get(bucket, 0) + 1
        return buckets
    
    def snapshot(self) -> dict[str, Any]:
        """Get snapshot of all stats for serialization.
        
        Returns:
            Dict with all current statistics.
        """
        return {
            "level_id": self.level_id,
            "attempts": self.attempts,
            "completions": self.completions,
            "deaths": self.deaths,
            "completion_rate": self.completion_rate,
            "avg_completion_time": self.avg_completion_time,
            "best_completion_time": self.best_completion_time if self.best_completion_time < 999999 else 0,
            "avg_death_x": self.avg_death_x,
            "best_x": self.best_x,
            "avg_reward": self.avg_reward,
        }
    
    def save_state(self) -> dict[str, Any]:
        """Serialize full state for checkpointing.
        
        Returns:
            Dict that can be saved with torch.save().
        """
        return {
            "level_id": self.level_id,
            "attempts": self.attempts,
            "completions": self.completions,
            "deaths": self.deaths,
            "best_x": self.best_x,
            "best_completion_time": self.best_completion_time,
            "completion_times": list(self.completion_times),
            "rewards": list(self.rewards),
            "speeds": list(self.speeds),
            "death_positions": list(self.death_positions),
        }
    
    def load_state(self, state: dict[str, Any]) -> None:
        """Restore state from checkpoint.
        
        Args:
            state: Dict from save_state() or loaded from checkpoint.
        """
        self.attempts = state.get("attempts", 0)
        self.completions = state.get("completions", 0)
        self.deaths = state.get("deaths", 0)
        self.best_x = state.get("best_x", 0)
        self.best_completion_time = state.get("best_completion_time", 999999)
        
        self.completion_times.clear()
        self.completion_times.extend(state.get("completion_times", []))
        
        self.rewards.clear()
        self.rewards.extend(state.get("rewards", []))
        
        self.speeds.clear()
        self.speeds.extend(state.get("speeds", []))
        
        self.death_positions.clear()
        self.death_positions.extend(state.get("death_positions", []))


@dataclass
class LevelTracker:
    """Tracks statistics across all levels.
    
    Provides access to per-level stats and aggregation methods.
    """
    
    _levels: dict[str, LevelStats] = field(default_factory=dict)
    
    def get(self, level_id: str) -> LevelStats:
        """Get or create stats for a level.
        
        Args:
            level_id: Level identifier (e.g., "1-1")
            
        Returns:
            LevelStats instance for that level.
        """
        if level_id not in self._levels:
            self._levels[level_id] = LevelStats(level_id=level_id)
        return self._levels[level_id]
    
    @property
    def all_levels(self) -> dict[str, LevelStats]:
        """Get dict of all tracked levels."""
        return self._levels
    
    def snapshot_all(self) -> dict[str, dict[str, Any]]:
        """Get snapshots of all levels.
        
        Returns:
            Dict mapping level_id to level snapshot.
        """
        return {lid: stats.snapshot() for lid, stats in self._levels.items()}
    
    def hardest_levels(self, n: int = 5) -> list[tuple[str, float]]:
        """Get levels with lowest completion rate.
        
        Args:
            n: Number of levels to return
            
        Returns:
            List of (level_id, completion_rate) tuples, sorted by rate ascending.
        """
        rates = [
            (lid, stats.completion_rate)
            for lid, stats in self._levels.items()
            if stats.attempts > 0
        ]
        return sorted(rates, key=lambda x: x[1])[:n]
    
    def death_hotspots(self, level_id: str) -> dict[int, int]:
        """Get death position histogram for a level.
        
        Args:
            level_id: Level to get hotspots for
            
        Returns:
            Dict mapping bucket start to death count.
        """
        return self.get(level_id).death_histogram
    
    def save_state(self) -> dict[str, dict[str, Any]]:
        """Serialize all levels for checkpointing.
        
        Returns:
            Dict mapping level_id to level state.
        """
        return {lid: stats.save_state() for lid, stats in self._levels.items()}
    
    def load_state(self, state: dict[str, dict[str, Any]]) -> None:
        """Restore all levels from checkpoint.
        
        Args:
            state: Dict from save_state() or loaded from checkpoint.
        """
        for level_id, level_state in state.items():
            self.get(level_id).load_state(level_state)
