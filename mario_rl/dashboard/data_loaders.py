"""Data loading functions for the training dashboard.

Uses DuckDB to read CSV + Parquet files directly for optimal performance.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from mario_rl.dashboard.query import query_workers
from mario_rl.dashboard.query import query_coordinator

# Max valid x position in Super Mario Bros (levels are ~3000-3500 px long)
MAX_VALID_X_POS = 5000


def _is_checkpoint_dir(path: Path) -> bool:
    """Check if a directory is a valid checkpoint directory.

    Valid checkpoint directories either:
    - Have a known naming pattern (_dist_, vec_dqn_, etc.)
    - Contain coordinator.csv or worker_*.csv files
    """
    if not path.is_dir():
        return False

    # Check naming patterns (distributed training, vectorized training)
    name = path.name
    if "_dist_" in name or name.startswith("vec_dqn_"):
        return True

    # Check for metrics files directly in the directory
    if (path / "coordinator.csv").exists():
        return True
    if list(path.glob("worker_*.csv")):
        return True

    # Check for metrics subdirectory
    metrics_dir = path / "metrics"
    if metrics_dir.exists():
        if (metrics_dir / "coordinator.csv").exists():
            return True
        if list(metrics_dir.glob("worker_*.csv")):
            return True

    return False


@st.cache_data(ttl=2)
def find_latest_checkpoint(base_dir: str = "checkpoints") -> str | None:
    """Find the most recent checkpoint directory."""
    base = Path(base_dir)
    if not base.exists():
        return None
    dirs = sorted(
        [d for d in base.iterdir() if _is_checkpoint_dir(d)],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    return str(dirs[0]) if dirs else None


@st.cache_data(ttl=5)
def list_checkpoints(base_dir: str = "checkpoints") -> list[str]:
    """List all checkpoint directories, sorted by modification time (newest first)."""
    base = Path(base_dir)
    if not base.exists():
        return []
    dirs = sorted(
        [d for d in base.iterdir() if _is_checkpoint_dir(d)],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    return [str(d) for d in dirs]


@st.cache_data(ttl=2)
def load_coordinator_metrics(checkpoint_dir: str) -> pd.DataFrame | None:
    """Load coordinator metrics directly from files via DuckDB."""
    try:
        df = query_coordinator(checkpoint_dir, "SELECT * FROM coordinator ORDER BY timestamp").df()
        if len(df) == 0:
            return None

        # Compute elapsed_min from timestamp if not present
        if "elapsed_min" not in df.columns and "timestamp" in df.columns:
            start_time = df["timestamp"].iloc[0]
            df["elapsed_min"] = (df["timestamp"] - start_time) / 60.0
        return df
    except Exception as e:
        st.error(f"Error loading coordinator metrics: {e}")
        return None


@st.cache_data(ttl=2)
def load_death_hotspots(checkpoint_dir: str) -> dict[str, dict[int, int]] | None:
    """Load death hotspots from JSON file."""
    json_path = Path(checkpoint_dir) / "death_hotspots.json"
    if not json_path.exists():
        return None
    try:
        with open(json_path) as f:
            data = json.load(f)
        # Convert string keys back to int for position buckets
        # Filter outliers like 65535 (max uint16) which are invalid
        return {
            level: {int(pos): count for pos, count in buckets.items() if 0 < int(pos) <= MAX_VALID_X_POS}
            for level, buckets in data.items()
        }
    except Exception:
        return None


@st.cache_data(ttl=2)
def load_worker_metrics(checkpoint_dir: str) -> dict[int, pd.DataFrame]:
    """Load all worker metrics directly from files via DuckDB."""
    try:
        # Query all workers at once with worker_id column
        combined_df = query_workers(
            checkpoint_dir,
            "SELECT * FROM workers ORDER BY worker_id, timestamp",
        ).df()

        if len(combined_df) == 0:
            return {}

        # Split into dict by worker_id for compatibility with tabs
        workers = {}
        for worker_id in combined_df["worker_id"].unique():
            worker_df = combined_df[combined_df["worker_id"] == worker_id].copy()
            # Drop the worker_id column since it's now the dict key
            if "worker_id" in worker_df.columns:
                worker_df = worker_df.drop(columns=["worker_id"])
            # Drop filename column if present
            if "filename" in worker_df.columns:
                worker_df = worker_df.drop(columns=["filename"])
            if len(worker_df) > 0:
                workers[int(worker_id)] = worker_df

        return workers
    except Exception as e:
        st.error(f"Error loading worker metrics: {e}")
        return {}


@st.cache_data(ttl=2)
def load_worker_latest(checkpoint_dir: str) -> pd.DataFrame | None:
    """Load only the latest row per worker (for summary tables)."""
    try:
        df = query_workers(
            checkpoint_dir,
            """
            SELECT * FROM (
                SELECT *,
                    ROW_NUMBER() OVER (PARTITION BY worker_id ORDER BY timestamp DESC) as rn
                FROM workers
            )
            WHERE rn = 1
            ORDER BY worker_id
            """,
        ).df()

        if len(df) == 0:
            return None

        # Drop helper columns
        df = df.drop(columns=["rn"], errors="ignore")
        if "filename" in df.columns:
            df = df.drop(columns=["filename"])

        return df
    except Exception:
        return None
