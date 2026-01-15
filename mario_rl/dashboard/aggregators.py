"""Data aggregation functions for the training dashboard using DuckDB.

Queries CSV + Parquet files directly via DuckDB for optimal performance.
"""

from dataclasses import dataclass

import duckdb
import pandas as pd

from mario_rl.dashboard.query import get_metrics_dir
from mario_rl.dashboard.query import query_workers

# Max valid x position in Super Mario Bros (levels are ~3000-3500 px long)
MAX_VALID_X_POS = 5000


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
    percentages: list[float]  # 7 or 12 action percentages


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


def _get_level_expr(columns: list[str]) -> str:
    """Get SQL expression for level based on available columns."""
    if "current_level" in columns:
        if "world" in columns and "stage" in columns:
            return "COALESCE(current_level, CAST(world AS VARCHAR) || '-' || CAST(stage AS VARCHAR))"
        return "current_level"
    elif "world" in columns and "stage" in columns:
        return "CAST(world AS VARCHAR) || '-' || CAST(stage AS VARCHAR)"
    return "'1-1'"  # Default level if no level info


def aggregate_level_stats_direct(checkpoint_dir: str) -> dict[str, LevelStats]:
    """Aggregate level statistics directly from files using DuckDB."""
    try:
        # Get columns first to build dynamic query
        columns = query_workers(checkpoint_dir, "SELECT * FROM workers LIMIT 0").columns
    except Exception:
        return {}

    if not columns:
        return {}

    level_expr = _get_level_expr(columns)

    # Build column expressions based on what's available
    deaths_expr = "SUM(COALESCE(deaths, 0))" if "deaths" in columns else "0"
    flags_expr = "SUM(COALESCE(flags, 0))" if "flags" in columns else "0"
    speed_expr = "speed" if "speed" in columns else "0"

    # Build best_x expression dynamically from available columns
    best_x_cols = [c for c in ["best_x_ever", "best_x", "x_pos"] if c in columns]
    if best_x_cols:
        coalesce_expr = f"COALESCE({', '.join(best_x_cols)}, 0)"
        best_x_expr = f"MAX(CASE WHEN {coalesce_expr} <= {MAX_VALID_X_POS} THEN {coalesce_expr} ELSE 0 END)"
    else:
        best_x_expr = "0"

    try:
        result = query_workers(
            checkpoint_dir,
            f"""
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
            FROM workers
            WHERE {level_expr} IS NOT NULL AND {level_expr} != '?'
            GROUP BY level
            ORDER BY level
        """,
        ).df()
    except Exception:
        return {}

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


def aggregate_action_distribution_direct(
    checkpoint_dir: str,
    max_points_per_level: int = 100,
) -> dict[str, list[ActionDistPoint]]:
    """Aggregate action distribution data per level directly from files."""
    try:
        columns = query_workers(checkpoint_dir, "SELECT * FROM workers LIMIT 0").columns
    except Exception:
        return {}

    if "action_dist" not in columns:
        return {}

    level_expr = _get_level_expr(columns)

    try:
        result = query_workers(
            checkpoint_dir,
            f"""
            WITH ranked AS (
                SELECT 
                    {level_expr} AS level,
                    steps,
                    action_dist,
                    NTILE({max_points_per_level}) OVER (PARTITION BY {level_expr} ORDER BY steps) as bucket
                FROM workers
                WHERE action_dist IS NOT NULL 
                  AND action_dist != ''
                  AND {level_expr} IS NOT NULL
                  AND {level_expr} != '?'
            ),
            first_per_bucket AS (
                SELECT level, steps, action_dist, bucket,
                    ROW_NUMBER() OVER (PARTITION BY level, bucket ORDER BY steps) as bucket_rn
                FROM ranked
            )
            SELECT level, steps, action_dist
            FROM first_per_bucket
            WHERE bucket_rn = 1
            ORDER BY level, steps
        """,
        ).df()
    except Exception:
        return {}

    # Use zip over numpy arrays instead of iterrows()
    action_data: dict[str, list[ActionDistPoint]] = {}
    levels = result["level"].values
    steps_arr = result["steps"].values
    dists = result["action_dist"].values

    for level, steps, dist_str in zip(levels, steps_arr, dists):
        level = str(level)
        if dist_str and isinstance(dist_str, str):
            try:
                pcts = [float(p) for p in dist_str.split(",")]
                # Accept both 7 (SIMPLE_MOVEMENT) and 12 (COMPLEX_MOVEMENT) actions
                if len(pcts) == 7 or len(pcts) == 12:
                    if level not in action_data:
                        action_data[level] = []
                    action_data[level].append(
                        ActionDistPoint(steps=int(steps), percentages=pcts)
                    )
            except (ValueError, TypeError):
                pass

    return action_data


def aggregate_rate_data_direct(
    checkpoint_dir: str,
    step_bucket_size: int = 10000,
) -> dict[str, list[RatePoint]]:
    """Aggregate death/flag/episode/timeout rate data per level directly from files."""
    try:
        columns = query_workers(checkpoint_dir, "SELECT * FROM workers LIMIT 0").columns
    except Exception:
        return {}

    if not columns:
        return {}

    level_expr = _get_level_expr(columns)

    deaths_expr = "COALESCE(deaths, 0)" if "deaths" in columns else "0"
    flags_expr = "COALESCE(flags, 0)" if "flags" in columns else "0"
    timeouts_expr = "COALESCE(timeouts, 0)" if "timeouts" in columns else "0"

    try:
        result = query_workers(
            checkpoint_dir,
            f"""
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
                FROM workers
                WHERE {level_expr} IS NOT NULL AND {level_expr} != '?'
            ),
            latest_per_worker AS (
                SELECT 
                    level, 
                    worker_id,
                    step_bucket, 
                    deaths, 
                    flags, 
                    episodes, 
                    timeouts
                FROM bucketed
                WHERE rn = 1
            ),
            with_deltas AS (
                SELECT 
                    level,
                    worker_id,
                    step_bucket,
                    deaths - COALESCE(LAG(deaths) OVER (PARTITION BY worker_id, level ORDER BY step_bucket), 0) AS delta_deaths,
                    flags - COALESCE(LAG(flags) OVER (PARTITION BY worker_id, level ORDER BY step_bucket), 0) AS delta_flags,
                    episodes - COALESCE(LAG(episodes) OVER (PARTITION BY worker_id, level ORDER BY step_bucket), 0) AS delta_episodes,
                    timeouts - COALESCE(LAG(timeouts) OVER (PARTITION BY worker_id, level ORDER BY step_bucket), 0) AS delta_timeouts
                FROM latest_per_worker
            )
            SELECT 
                level,
                step_bucket,
                SUM(CASE WHEN delta_deaths > 0 THEN delta_deaths ELSE 0 END) AS total_deaths,
                SUM(CASE WHEN delta_flags > 0 THEN delta_flags ELSE 0 END) AS total_flags,
                SUM(CASE WHEN delta_episodes > 0 THEN delta_episodes ELSE 0 END) AS total_episodes,
                SUM(CASE WHEN delta_timeouts > 0 THEN delta_timeouts ELSE 0 END) AS total_timeouts
            FROM with_deltas
            GROUP BY level, step_bucket
            ORDER BY level, step_bucket
        """,
        ).df()
    except Exception:
        return {}

    rate_data: dict[str, list[RatePoint]] = {}
    levels = result["level"].values
    buckets = result["step_bucket"].values
    deaths = result["total_deaths"].values
    flags = result["total_flags"].values
    episodes = result["total_episodes"].values
    timeouts = result["total_timeouts"].values

    for level, bucket, d, f, e, t in zip(levels, buckets, deaths, flags, episodes, timeouts):
        level = str(level)
        if level not in rate_data:
            rate_data[level] = []
        rate_data[level].append(
            RatePoint(
                steps=int(bucket),
                deaths=int(d),
                flags=int(f),
                episodes=int(e),
                timeouts=int(t),
            )
        )

    return rate_data


def aggregate_death_hotspots_direct(checkpoint_dir: str) -> dict[str, dict[int, int]]:
    """Aggregate death hotspots directly from files using DuckDB."""
    try:
        columns = query_workers(checkpoint_dir, "SELECT * FROM workers LIMIT 0").columns
    except Exception:
        return {}

    if "death_positions" not in columns:
        return {}

    try:
        result = query_workers(
            checkpoint_dir,
            f"""
            WITH parsed AS (
                SELECT 
                    split_part(CAST(death_positions AS VARCHAR), ':', 1) AS level,
                    split_part(CAST(death_positions AS VARCHAR), ':', 2) AS positions_str
                FROM workers
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
              AND pos > 0 
              AND pos <= {MAX_VALID_X_POS}
            GROUP BY level, bucket
            ORDER BY level, bucket
        """,
        ).df()
    except Exception:
        return {}

    hotspots: dict[str, dict[int, int]] = {}
    levels = result["level"].values
    buckets = result["bucket"].values
    counts = result["count"].values

    for level, bucket, count in zip(levels, buckets, counts):
        level = str(level)
        if level not in hotspots:
            hotspots[level] = {}
        hotspots[level][int(bucket)] = int(count)

    return hotspots


def sample_data(data: list, max_points: int = 30) -> list:
    """Sample data points evenly for visualization."""
    if len(data) <= max_points:
        return data

    indices = [int(i * len(data) / max_points) for i in range(max_points)]
    return [data[i] for i in indices]


# =============================================================================
# Legacy functions that work on DataFrames (for compatibility with tabs)
# =============================================================================


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
        coalesce_expr = f"COALESCE({', '.join(best_x_cols)}, 0)"
        best_x_expr = f"MAX(CASE WHEN {coalesce_expr} <= {MAX_VALID_X_POS} THEN {coalesce_expr} ELSE 0 END)"
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
    max_points_per_level: int = 100,
) -> dict[str, list[ActionDistPoint]]:
    """Aggregate action distribution data per level using DuckDB."""
    if not workers:
        return {}

    all_dfs = [df.assign(worker_id=wid) for wid, df in workers.items() if len(df) > 0]
    if not all_dfs:
        return {}

    combined = pd.concat(all_dfs, ignore_index=True)
    columns = list(combined.columns)

    if "action_dist" not in columns:
        return {}

    level_expr = _get_level_expr(columns)

    result = duckdb.sql(f"""
        WITH ranked AS (
            SELECT 
                {level_expr} AS level,
                steps,
                action_dist,
                NTILE({max_points_per_level}) OVER (PARTITION BY {level_expr} ORDER BY steps) as bucket
            FROM combined
            WHERE action_dist IS NOT NULL 
              AND action_dist != ''
              AND {level_expr} IS NOT NULL
              AND {level_expr} != '?'
        ),
        first_per_bucket AS (
            SELECT level, steps, action_dist, bucket,
                ROW_NUMBER() OVER (PARTITION BY level, bucket ORDER BY steps) as bucket_rn
            FROM ranked
        )
        SELECT level, steps, action_dist
        FROM first_per_bucket
        WHERE bucket_rn = 1
        ORDER BY level, steps
    """).df()

    action_data: dict[str, list[ActionDistPoint]] = {}
    levels = result["level"].values
    steps_arr = result["steps"].values
    dists = result["action_dist"].values

    for level, steps, dist_str in zip(levels, steps_arr, dists):
        level = str(level)
        if dist_str and isinstance(dist_str, str):
            try:
                pcts = [float(p) for p in dist_str.split(",")]
                if len(pcts) == 7 or len(pcts) == 12:
                    if level not in action_data:
                        action_data[level] = []
                    action_data[level].append(
                        ActionDistPoint(steps=int(steps), percentages=pcts)
                    )
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

    all_dfs = [df.assign(worker_id=wid) for wid, df in workers.items() if len(df) > 0]
    if not all_dfs:
        return {}

    combined = pd.concat(all_dfs, ignore_index=True)
    columns = list(combined.columns)
    level_expr = _get_level_expr(columns)

    deaths_expr = "COALESCE(deaths, 0)" if "deaths" in columns else "0"
    flags_expr = "COALESCE(flags, 0)" if "flags" in columns else "0"
    timeouts_expr = "COALESCE(timeouts, 0)" if "timeouts" in columns else "0"

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
            SELECT 
                level, 
                worker_id,
                step_bucket, 
                deaths, 
                flags, 
                episodes, 
                timeouts
            FROM bucketed
            WHERE rn = 1
        ),
        with_deltas AS (
            SELECT 
                level,
                worker_id,
                step_bucket,
                deaths - COALESCE(LAG(deaths) OVER (PARTITION BY worker_id, level ORDER BY step_bucket), 0) AS delta_deaths,
                flags - COALESCE(LAG(flags) OVER (PARTITION BY worker_id, level ORDER BY step_bucket), 0) AS delta_flags,
                episodes - COALESCE(LAG(episodes) OVER (PARTITION BY worker_id, level ORDER BY step_bucket), 0) AS delta_episodes,
                timeouts - COALESCE(LAG(timeouts) OVER (PARTITION BY worker_id, level ORDER BY step_bucket), 0) AS delta_timeouts
            FROM latest_per_worker
        )
        SELECT 
            level,
            step_bucket,
            SUM(CASE WHEN delta_deaths > 0 THEN delta_deaths ELSE 0 END) AS total_deaths,
            SUM(CASE WHEN delta_flags > 0 THEN delta_flags ELSE 0 END) AS total_flags,
            SUM(CASE WHEN delta_episodes > 0 THEN delta_episodes ELSE 0 END) AS total_episodes,
            SUM(CASE WHEN delta_timeouts > 0 THEN delta_timeouts ELSE 0 END) AS total_timeouts
        FROM with_deltas
        GROUP BY level, step_bucket
        ORDER BY level, step_bucket
    """).df()

    rate_data: dict[str, list[RatePoint]] = {}
    levels = result["level"].values
    buckets = result["step_bucket"].values
    deaths = result["total_deaths"].values
    flags = result["total_flags"].values
    episodes = result["total_episodes"].values
    timeouts = result["total_timeouts"].values

    for level, bucket, d, f, e, t in zip(levels, buckets, deaths, flags, episodes, timeouts):
        level = str(level)
        if level not in rate_data:
            rate_data[level] = []
        rate_data[level].append(
            RatePoint(
                steps=int(bucket),
                deaths=int(d),
                flags=int(f),
                episodes=int(e),
                timeouts=int(t),
            )
        )

    return rate_data


def aggregate_death_hotspots_from_csv(
    workers: dict[int, pd.DataFrame],
) -> dict[str, dict[int, int]]:
    """Aggregate death hotspots from worker CSV death_positions column using DuckDB."""
    if not workers:
        return {}

    all_dfs = [df for df in workers.values() if len(df) > 0 and "death_positions" in df.columns]
    if not all_dfs:
        return {}

    combined = pd.concat(all_dfs, ignore_index=True)

    if "death_positions" not in combined.columns:
        return {}

    result = duckdb.sql(f"""
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
          AND pos > 0 
          AND pos <= {MAX_VALID_X_POS}
        GROUP BY level, bucket
        ORDER BY level, bucket
    """).df()

    hotspots: dict[str, dict[int, int]] = {}
    levels = result["level"].values
    buckets = result["bucket"].values
    counts = result["count"].values

    for level, bucket, count in zip(levels, buckets, counts):
        level = str(level)
        if level not in hotspots:
            hotspots[level] = {}
        hotspots[level][int(bucket)] = int(count)

    return hotspots
