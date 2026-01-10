"""Data aggregation functions for the training dashboard."""

from dataclasses import dataclass
from dataclasses import field

import pandas as pd


@dataclass
class LevelStats:
    """Aggregated statistics for a single level."""
    level: str
    episodes: int = 0
    deaths: int = 0
    flags: int = 0
    best_x: int = 0
    rewards: list[float] = field(default_factory=list)
    speeds: list[float] = field(default_factory=list)
    x_positions: list[int] = field(default_factory=list)
    
    @property
    def avg_reward(self) -> float:
        return sum(self.rewards) / len(self.rewards) if self.rewards else 0.0
    
    @property
    def avg_speed(self) -> float:
        return sum(self.speeds) / len(self.speeds) if self.speeds else 0.0


@dataclass
class ActionDistPoint:
    """Action distribution at a single time point."""
    steps: int
    percentages: list[float]  # 12 action percentages


@dataclass
class RatePoint:
    """Death/completion rate at a single time point."""
    steps: int
    deaths: int
    flags: int
    episodes: int
    
    @property
    def deaths_per_episode(self) -> float:
        return self.deaths / self.episodes if self.episodes > 0 else 0
    
    @property
    def completion_rate(self) -> float:
        return (self.flags / self.episodes * 100) if self.episodes > 0 else 0


def level_sort_key(level: str) -> tuple[int, int]:
    """Sort key for level names like '1-1', '2-3', etc."""
    try:
        parts = level.split("-")
        return (int(parts[0]), int(parts[1]))
    except (ValueError, IndexError):
        return (999, 999)


def aggregate_level_stats(workers: dict[int, pd.DataFrame]) -> dict[str, LevelStats]:
    """Aggregate statistics for each level from all workers."""
    stats: dict[str, LevelStats] = {}
    
    for wid, df in workers.items():
        if len(df) == 0:
            continue
        
        for _, row in df.iterrows():
            level = _get_level_from_row(row)
            if level is None:
                continue
            
            if level not in stats:
                stats[level] = LevelStats(level=level)
            
            s = stats[level]
            s.episodes += 1
            s.deaths += int(row.get("deaths", 0))
            s.flags += int(row.get("flags", 0))
            
            x_pos = row.get("x_pos", 0)
            if pd.notna(x_pos):
                s.best_x = max(s.best_x, int(x_pos))
                s.x_positions.append(int(x_pos))
            
            reward = row.get("reward")
            if pd.notna(reward):
                s.rewards.append(float(reward))
            
            speed = row.get("speed")
            if pd.notna(speed):
                s.speeds.append(float(speed))
    
    return stats


def aggregate_action_distribution(
    workers: dict[int, pd.DataFrame],
) -> dict[str, list[ActionDistPoint]]:
    """
    Aggregate action distribution data per level.
    
    Returns: {level: [ActionDistPoint, ...]} sorted by steps
    """
    result: dict[str, list[ActionDistPoint]] = {}
    
    for wid, df in workers.items():
        if len(df) == 0 or "action_dist" not in df.columns:
            continue
        
        for _, row in df.iterrows():
            level = _get_level_from_row(row)
            if level is None:
                continue
            
            dist_str = row.get("action_dist", "")
            steps = int(row.get("steps", 0))
            
            if dist_str and isinstance(dist_str, str):
                try:
                    pcts = [float(p) for p in dist_str.split(",")]
                    if len(pcts) == 12:
                        if level not in result:
                            result[level] = []
                        result[level].append(ActionDistPoint(steps=steps, percentages=pcts))
                except (ValueError, TypeError):
                    pass
    
    # Sort each level's data by steps
    for level in result:
        result[level].sort(key=lambda x: x.steps)
    
    return result


def aggregate_rate_data(
    workers: dict[int, pd.DataFrame],
) -> dict[str, list[RatePoint]]:
    """
    Aggregate death/flag/episode rate data per level.
    
    Returns: {level: [RatePoint, ...]} sorted by steps
    """
    result: dict[str, list[RatePoint]] = {}
    
    for wid, df in workers.items():
        if len(df) == 0:
            continue
        
        for _, row in df.iterrows():
            level = _get_level_from_row(row)
            if level is None:
                continue
            
            steps = int(row.get("steps", 0))
            deaths = int(row.get("deaths", 0))
            flags = int(row.get("flags", 0))
            episodes = int(row.get("episodes", 0))
            
            if level not in result:
                result[level] = []
            result[level].append(RatePoint(
                steps=steps, deaths=deaths, flags=flags, episodes=episodes
            ))
    
    # Sort each level's data by steps
    for level in result:
        result[level].sort(key=lambda x: x.steps)
    
    return result


def aggregate_death_hotspots_from_csv(
    workers: dict[int, pd.DataFrame],
) -> dict[str, dict[int, int]]:
    """
    Aggregate death hotspots from worker CSV death_positions column.
    
    Returns: {level: {position_bucket: count}}
    """
    hotspots: dict[str, dict[int, int]] = {}
    
    for wid, df in workers.items():
        if "death_positions" not in df.columns:
            continue
        
        for _, row in df.iterrows():
            death_str = row.get("death_positions", "")
            if not death_str or not isinstance(death_str, str) or ":" not in death_str:
                continue
            
            try:
                level, positions_str = death_str.split(":", 1)
                if not positions_str:
                    continue
                
                positions = [int(p.strip()) for p in positions_str.split(",") if p.strip()]
                
                if level not in hotspots:
                    hotspots[level] = {}
                
                for pos in positions:
                    bucket = (pos // 25) * 25
                    hotspots[level][bucket] = hotspots[level].get(bucket, 0) + 1
            except (ValueError, TypeError):
                continue
    
    return hotspots


def sample_data(data: list, max_points: int = 30) -> list:
    """Sample data points evenly for visualization."""
    if len(data) <= max_points:
        return data
    
    indices = [int(i * len(data) / max_points) for i in range(max_points)]
    return [data[i] for i in indices]


def _get_level_from_row(row: pd.Series) -> str | None:
    """Extract level identifier from a DataFrame row."""
    level = row.get("current_level", "?")
    
    if level == "?" and "world" in row and "stage" in row:
        try:
            level = f"{int(row['world'])}-{int(row['stage'])}"
        except (ValueError, TypeError):
            return None
    
    if level == "?" or pd.isna(level):
        return None
    
    return str(level)
