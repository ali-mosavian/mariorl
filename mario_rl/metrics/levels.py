"""
Per-level metrics tracking.

Tracks statistics per game level (e.g., 1-1, 4-2) for analyzing
which levels are hardest and where the agent struggles.

Includes death hotspot aggregation for intelligent snapshot/restore decisions.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
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


# =============================================================================
# Death Hotspot Aggregation for Snapshot/Restore
# =============================================================================


@dataclass
class DeathHotspotAggregate:
    """Aggregates death positions from all workers for snapshot/restore decisions.
    
    Workers report death positions, coordinator aggregates and saves to disk.
    Workers can load the aggregate to decide where to create checkpoints
    and where to restore from when Mario dies.
    
    Data is stored in 25-pixel buckets per level:
        {level_id: {bucket_start: death_count, ...}, ...}
    
    Usage:
        # Coordinator side - aggregate and save
        agg = DeathHotspotAggregate(save_path=Path("death_hotspots.json"))
        agg.record_death("1-1", x_pos=150)
        agg.save()
        
        # Worker side - load and use for snapshot decisions
        agg = DeathHotspotAggregate.load(Path("death_hotspots.json"))
        positions = agg.suggest_snapshot_positions("1-1", count=3)
    """
    
    bucket_size: int = 25
    save_path: Path | None = None
    
    # Internal data: level_id -> {bucket_start -> death_count}
    _data: dict[str, dict[int, int]] = field(default_factory=dict, init=False)
    
    # Track if modified since last save
    _dirty: bool = field(default=False, init=False)
    
    def _bucket_for(self, x_pos: int) -> int:
        """Get bucket start position for an x position."""
        return (x_pos // self.bucket_size) * self.bucket_size
    
    def record_death(self, level_id: str, x_pos: int) -> None:
        """Record a death at the given position.
        
        Args:
            level_id: Level identifier (e.g., "1-1")
            x_pos: X position where death occurred
        """
        if level_id not in self._data:
            self._data[level_id] = {}
        
        bucket = self._bucket_for(x_pos)
        self._data[level_id][bucket] = self._data[level_id].get(bucket, 0) + 1
        self._dirty = True
    
    def record_deaths_batch(self, level_id: str, positions: list[int]) -> None:
        """Record multiple deaths efficiently.
        
        Args:
            level_id: Level identifier
            positions: List of x positions where deaths occurred
        """
        if level_id not in self._data:
            self._data[level_id] = {}
        
        for x_pos in positions:
            bucket = self._bucket_for(x_pos)
            self._data[level_id][bucket] = self._data[level_id].get(bucket, 0) + 1
        
        if positions:
            self._dirty = True
    
    def get_histogram(self, level_id: str) -> dict[int, int]:
        """Get death histogram for a level.
        
        Args:
            level_id: Level identifier
            
        Returns:
            Dict mapping bucket_start -> death_count
        """
        return dict(self._data.get(level_id, {}))
    
    def get_hotspots(
        self, 
        level_id: str, 
        min_deaths: int = 3,
        top_n: int | None = None,
    ) -> list[tuple[int, int]]:
        """Get bucket positions with significant deaths.
        
        Args:
            level_id: Level identifier
            min_deaths: Minimum deaths to be considered a hotspot
            top_n: If set, return only top N hotspots
            
        Returns:
            List of (bucket_start, count) tuples sorted by count descending.
        """
        level_data = self._data.get(level_id, {})
        hotspots = [
            (bucket, count) 
            for bucket, count in level_data.items() 
            if count >= min_deaths
        ]
        hotspots.sort(key=lambda x: x[1], reverse=True)
        
        if top_n is not None:
            hotspots = hotspots[:top_n]
        
        return hotspots
    
    def suggest_snapshot_positions(
        self, 
        level_id: str, 
        count: int = 3,
        min_spacing: int = 100,
        min_deaths: int = 2,
    ) -> list[int]:
        """Suggest x positions for creating save states based on death hotspots.
        
        Positions are chosen BEFORE death hotspots (offset back by bucket_size)
        so restoring there gives the agent a chance to practice that section.
        
        Args:
            level_id: Level identifier
            count: Number of positions to suggest
            min_spacing: Minimum pixels between suggested positions
            min_deaths: Minimum deaths in a bucket to consider it
            
        Returns:
            List of x positions (sorted ascending) for creating snapshots.
            Positions are offset to be BEFORE the death hotspot.
        """
        hotspots = self.get_hotspots(level_id, min_deaths=min_deaths)
        if not hotspots:
            return []
        
        # Convert hotspot bucket starts to snapshot positions
        # Snapshot position = bucket_start - bucket_size (before the danger zone)
        candidates = [
            max(0, bucket - self.bucket_size)
            for bucket, _ in hotspots
        ]
        
        # Select positions with minimum spacing
        selected: list[int] = []
        for pos in candidates:
            # Check spacing against all selected positions
            if all(abs(pos - s) >= min_spacing for s in selected):
                selected.append(pos)
                if len(selected) >= count:
                    break
        
        return sorted(selected)
    
    def suggest_restore_position(
        self,
        level_id: str,
        death_x: int,
        lookback: int = 200,
    ) -> int | None:
        """Suggest a position to restore from after dying.
        
        Looks for the nearest hotspot before the death position and suggests
        restoring to a position before that hotspot.
        
        Args:
            level_id: Level identifier
            death_x: X position where death occurred
            lookback: Maximum distance to look back for a hotspot
            
        Returns:
            Suggested restore position, or None if no suitable position found.
        """
        hotspots = self.get_hotspots(level_id, min_deaths=2)
        if not hotspots:
            return None
        
        # Find hotspots within lookback range before death
        death_bucket = self._bucket_for(death_x)
        relevant = [
            (bucket, count)
            for bucket, count in hotspots
            if death_bucket - lookback <= bucket <= death_bucket
        ]
        
        if not relevant:
            return None
        
        # Return position before the highest-death hotspot in range
        highest = max(relevant, key=lambda x: x[1])
        return max(0, highest[0] - self.bucket_size * 2)
    
    def merge(self, other: "DeathHotspotAggregate") -> None:
        """Merge data from another aggregate (e.g., from worker).
        
        Args:
            other: Another DeathHotspotAggregate to merge in
        """
        for level_id, level_data in other._data.items():
            if level_id not in self._data:
                self._data[level_id] = {}
            
            for bucket, count in level_data.items():
                self._data[level_id][bucket] = (
                    self._data[level_id].get(bucket, 0) + count
                )
        
        if other._data:
            self._dirty = True
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for serialization.
        
        Returns:
            Dict suitable for JSON serialization.
        """
        return {
            "bucket_size": self.bucket_size,
            "levels": {
                level_id: {str(k): v for k, v in buckets.items()}
                for level_id, buckets in self._data.items()
            },
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any], save_path: Path | None = None) -> "DeathHotspotAggregate":
        """Create from dict.
        
        Args:
            data: Dict from to_dict()
            save_path: Optional path for future saves
            
        Returns:
            New DeathHotspotAggregate instance.
        """
        agg = cls(
            bucket_size=data.get("bucket_size", 25),
            save_path=save_path,
        )
        
        for level_id, buckets in data.get("levels", {}).items():
            agg._data[level_id] = {int(k): v for k, v in buckets.items()}
        
        return agg
    
    def save(self, path: Path | None = None) -> None:
        """Save aggregate to disk.
        
        Args:
            path: Optional path override (defaults to self.save_path)
        """
        target = path or self.save_path
        if target is None:
            raise ValueError("No save path specified")
        
        target.parent.mkdir(parents=True, exist_ok=True)
        
        with open(target, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        
        self._dirty = False
    
    def save_if_dirty(self, path: Path | None = None) -> bool:
        """Save only if data has changed since last save.
        
        Args:
            path: Optional path override
            
        Returns:
            True if saved, False if no changes.
        """
        if self._dirty:
            self.save(path)
            return True
        return False
    
    @classmethod
    def load(cls, path: Path) -> "DeathHotspotAggregate":
        """Load aggregate from disk.
        
        Args:
            path: Path to JSON file
            
        Returns:
            Loaded DeathHotspotAggregate instance.
            
        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        with open(path) as f:
            data = json.load(f)
        
        return cls.from_dict(data, save_path=path)
    
    @classmethod
    def load_or_create(cls, path: Path, bucket_size: int = 25) -> "DeathHotspotAggregate":
        """Load from disk or create new if file doesn't exist.
        
        Args:
            path: Path to JSON file
            bucket_size: Bucket size for new instance
            
        Returns:
            DeathHotspotAggregate instance (loaded or new).
        """
        if path.exists():
            return cls.load(path)
        return cls(bucket_size=bucket_size, save_path=path)
    
    def summary(self) -> dict[str, Any]:
        """Get summary of all levels.
        
        Returns:
            Dict with per-level statistics.
        """
        return {
            level_id: {
                "total_deaths": sum(buckets.values()),
                "hotspot_count": len([c for c in buckets.values() if c >= 3]),
                "worst_bucket": max(buckets.items(), key=lambda x: x[1]) if buckets else None,
            }
            for level_id, buckets in self._data.items()
        }
