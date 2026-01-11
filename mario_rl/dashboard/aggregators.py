"""Data aggregation functions for the training dashboard using DuckDB."""

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import duckdb
import pandas as pd


@dataclass
class LevelStats:
    """Aggregated statistics for a single level."""
    level: str
    episodes: int = 0
    deaths: int = 0
    flags: int = 0
    best_x: int = 0
    avg_reward: float = 0.0
    min_reward: float = 0.0
    max_reward: float = 0.0
    avg_speed: float = 0.0
    min_speed: float = 0.0
    max_speed: float = 0.0


@dataclass
class ActionDistPoint:
    """Action distribution at a single time point."""
    steps: int
    percentages: list[float]  # 12 action percentages


@dataclass
class RatePoint:
    """Death/completion/timeout rate at a single time point."""
    steps: int
    deaths: int
    flags: int
    episodes: int
    timeouts: int = 0
    
    @property
    def deaths_per_episode(self) -> float:
        return self.deaths / self.episodes if self.episodes > 0 else 0
    
    @property
    def timeouts_per_episode(self) -> float:
        return self.timeouts / self.episodes if self.episodes > 0 else 0
    
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


def _get_csv_glob(checkpoint_dir: str) -> str:
    """Get the glob pattern for worker CSV files."""
    metrics_path = Path(checkpoint_dir) / "metrics"
    if metrics_path.exists():
        return str(metrics_path / "worker_*.csv")
    return str(Path(checkpoint_dir) / "worker_*.csv")


def _get_level_expr(columns: list[str]) -> str:
    """Get SQL expression for level based on available columns."""
    if "current_level" in columns:
        if "world" in columns and "stage" in columns:
            return "COALESCE(current_level, CAST(world AS VARCHAR) || '-' || CAST(stage AS VARCHAR))"
        return "current_level"
    elif "world" in columns and "stage" in columns:
        return "CAST(world AS VARCHAR) || '-' || CAST(stage AS VARCHAR)"
    return "'1-1'"  # Default level if no level info


def aggregate_level_stats(workers: dict[int, pd.DataFrame]) -> dict[str, LevelStats]:
    """Aggregate statistics for each level from all workers using DuckDB."""
    if not workers:
        return {}
    
    # Combine all worker dataframes
    all_dfs = [df.assign(worker_id=wid) for wid, df in workers.items() if len(df) > 0]
    if not all_dfs:
        return {}
    
    combined = pd.concat(all_dfs, ignore_index=True)
    columns = list(combined.columns)
    level_expr = _get_level_expr(columns)
    
    # Build column expressions based on what's available
    deaths_expr = "SUM(COALESCE(deaths, 0))" if "deaths" in columns else "0"
    flags_expr = "SUM(COALESCE(flags, 0))" if "flags" in columns else "0"
    speed_expr = "speed" if "speed" in columns else "0"
    
    # Build best_x expression dynamically from available columns
    best_x_cols = [c for c in ["best_x_ever", "best_x", "x_pos"] if c in columns]
    if best_x_cols:
        best_x_expr = f"MAX(COALESCE({', '.join(best_x_cols)}, 0))"
    else:
        best_x_expr = "0"
    
    result = duckdb.sql(f"""
        SELECT 
            {level_expr} AS level,
            COUNT(*) AS episodes,
            {deaths_expr} AS deaths,
            {flags_expr} AS flags,
            {best_x_expr} AS best_x,
            AVG(COALESCE(reward, 0)) AS avg_reward,
            MIN(COALESCE(reward, 0)) AS min_reward,
            MAX(COALESCE(reward, 0)) AS max_reward,
            AVG(COALESCE({speed_expr}, 0)) AS avg_speed,
            MIN(COALESCE({speed_expr}, 0)) AS min_speed,
            MAX(COALESCE({speed_expr}, 0)) AS max_speed
        FROM combined
        WHERE {level_expr} IS NOT NULL AND {level_expr} != '?'
        GROUP BY level
        ORDER BY level
    """).df()
    
    stats: dict[str, LevelStats] = {}
    for _, row in result.iterrows():
        level = str(row["level"])
        stats[level] = LevelStats(
            level=level,
            episodes=int(row["episodes"]),
            deaths=int(row["deaths"]),
            flags=int(row["flags"]),
            best_x=int(row["best_x"]),
            avg_reward=float(row["avg_reward"]),
            min_reward=float(row["min_reward"]),
            max_reward=float(row["max_reward"]),
            avg_speed=float(row["avg_speed"]),
            min_speed=float(row["min_speed"]),
            max_speed=float(row["max_speed"]),
        )
    
    return stats


def aggregate_action_distribution(
    workers: dict[int, pd.DataFrame],
) -> dict[str, list[ActionDistPoint]]:
    """Aggregate action distribution data per level using DuckDB."""
    if not workers:
        return {}
    
    # Combine all worker dataframes
    all_dfs = [df.assign(worker_id=wid) for wid, df in workers.items() if len(df) > 0]
    if not all_dfs:
        return {}
    
    combined = pd.concat(all_dfs, ignore_index=True)
    columns = list(combined.columns)
    
    if "action_dist" not in columns:
        return {}
    
    level_expr = _get_level_expr(columns)
    
    result = duckdb.sql(f"""
        SELECT 
            {level_expr} AS level,
            steps,
            action_dist
        FROM combined
        WHERE action_dist IS NOT NULL 
          AND action_dist != ''
          AND {level_expr} IS NOT NULL
          AND {level_expr} != '?'
        ORDER BY level, steps
    """).df()
    
    action_data: dict[str, list[ActionDistPoint]] = {}
    for _, row in result.iterrows():
        level = str(row["level"])
        dist_str = row["action_dist"]
        steps = int(row["steps"])
        
        if dist_str and isinstance(dist_str, str):
            try:
                pcts = [float(p) for p in dist_str.split(",")]
                if len(pcts) == 12:
                    if level not in action_data:
                        action_data[level] = []
                    action_data[level].append(ActionDistPoint(steps=steps, percentages=pcts))
            except (ValueError, TypeError):
                pass
    
    return action_data


def aggregate_rate_data(
    workers: dict[int, pd.DataFrame],
    step_bucket_size: int = 10000,
) -> dict[str, list[RatePoint]]:
    """Aggregate death/flag/episode/timeout rate data per level using DuckDB."""
    if not workers:
        return {}
    
    # Combine all worker dataframes with worker IDs
    all_dfs = [df.assign(worker_id=wid) for wid, df in workers.items() if len(df) > 0]
    if not all_dfs:
        return {}
    
    combined = pd.concat(all_dfs, ignore_index=True)
    columns = list(combined.columns)
    level_expr = _get_level_expr(columns)
    
    # Build column expressions based on what's available
    deaths_expr = "COALESCE(deaths, 0)" if "deaths" in columns else "0"
    flags_expr = "COALESCE(flags, 0)" if "flags" in columns else "0"
    timeouts_expr = "COALESCE(timeouts, 0)" if "timeouts" in columns else "0"
    
    # Use DuckDB for bucketing and aggregation
    result = duckdb.sql(f"""
        WITH bucketed AS (
            SELECT 
                {level_expr} AS level,
                worker_id,
                (steps // {step_bucket_size}) * {step_bucket_size} AS step_bucket,
                {deaths_expr} AS deaths,
                {flags_expr} AS flags,
                COALESCE(episodes, 0) AS episodes,
                {timeouts_expr} AS timeouts,
                ROW_NUMBER() OVER (PARTITION BY worker_id, {level_expr}, (steps // {step_bucket_size}) ORDER BY steps DESC) AS rn
            FROM combined
            WHERE {level_expr} IS NOT NULL AND {level_expr} != '?'
        ),
        latest_per_worker AS (
            SELECT level, step_bucket, deaths, flags, episodes, timeouts
            FROM bucketed
            WHERE rn = 1
        )
        SELECT 
            level,
            step_bucket,
            SUM(deaths) AS total_deaths,
            SUM(flags) AS total_flags,
            SUM(episodes) AS total_episodes,
            SUM(timeouts) AS total_timeouts
        FROM latest_per_worker
        GROUP BY level, step_bucket
        ORDER BY level, step_bucket
    """).df()
    
    rate_data: dict[str, list[RatePoint]] = {}
    for _, row in result.iterrows():
        level = str(row["level"])
        if level not in rate_data:
            rate_data[level] = []
        rate_data[level].append(RatePoint(
            steps=int(row["step_bucket"]),
            deaths=int(row["total_deaths"]),
            flags=int(row["total_flags"]),
            episodes=int(row["total_episodes"]),
            timeouts=int(row["total_timeouts"]),
        ))
    
    return rate_data


def aggregate_death_hotspots_from_csv(
    workers: dict[int, pd.DataFrame],
) -> dict[str, dict[int, int]]:
    """Aggregate death hotspots from worker CSV death_positions column using DuckDB."""
    if not workers:
        return {}
    
    # Combine all worker dataframes
    all_dfs = [df for df in workers.values() if len(df) > 0 and "death_positions" in df.columns]
    if not all_dfs:
        return {}
    
    combined = pd.concat(all_dfs, ignore_index=True)
    
    if "death_positions" not in combined.columns:
        return {}
    
    # Filter and extract death positions using DuckDB
    result = duckdb.sql("""
        WITH parsed AS (
            SELECT 
                split_part(CAST(death_positions AS VARCHAR), ':', 1) AS level,
                split_part(CAST(death_positions AS VARCHAR), ':', 2) AS positions_str
            FROM combined
            WHERE death_positions IS NOT NULL 
              AND CAST(death_positions AS VARCHAR) != ''
              AND CAST(death_positions AS VARCHAR) LIKE '%:%'
        ),
        exploded AS (
            SELECT 
                level,
                CAST(TRIM(unnest(string_split(positions_str, ','))) AS INTEGER) AS pos
            FROM parsed
            WHERE positions_str IS NOT NULL AND positions_str != ''
        )
        SELECT 
            level,
            (pos // 25) * 25 AS bucket,
            COUNT(*) AS count
        FROM exploded
        WHERE pos IS NOT NULL
        GROUP BY level, bucket
        ORDER BY level, bucket
    """).df()
    
    hotspots: dict[str, dict[int, int]] = {}
    for _, row in result.iterrows():
        level = str(row["level"])
        bucket = int(row["bucket"])
        count = int(row["count"])
        
        if level not in hotspots:
            hotspots[level] = {}
        hotspots[level][bucket] = count
    
    return hotspots


def sample_data(data: list, max_points: int = 30) -> list:
    """Sample data points evenly for visualization."""
    if len(data) <= max_points:
        return data
    
    indices = [int(i * len(data) / max_points) for i in range(max_points)]
    return [data[i] for i in indices]
