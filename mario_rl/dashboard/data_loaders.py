"""Data loading functions for the training dashboard."""

import json
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

# Max valid x position in Super Mario Bros (levels are ~3000-3500 px long)
MAX_VALID_X_POS = 5000


@st.cache_data(ttl=2)
def find_latest_checkpoint(base_dir: str = "checkpoints") -> str | None:
    """Find the most recent checkpoint directory."""
    base = Path(base_dir)
    if not base.exists():
        return None
    # Look for any dist_ directories (ddqn_dist_ or dreamer_dist_)
    dirs = sorted(
        [d for d in base.iterdir() if d.is_dir() and "_dist_" in d.name],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    return str(dirs[0]) if dirs else None


@st.cache_data(ttl=2)
def load_coordinator_metrics(checkpoint_dir: str) -> pd.DataFrame | None:
    """Load coordinator metrics CSV."""
    # Try metrics subdirectory first, then root (for compatibility)
    csv_path = Path(checkpoint_dir) / "metrics" / "coordinator.csv"
    if not csv_path.exists():
        csv_path = Path(checkpoint_dir) / "coordinator.csv"
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
        # Compute elapsed_min from timestamp if not present
        if "elapsed_min" not in df.columns and "timestamp" in df.columns:
            start_time = df["timestamp"].iloc[0]
            df["elapsed_min"] = (df["timestamp"] - start_time) / 60.0
        return df
    except Exception:
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
            level: {
                int(pos): count 
                for pos, count in buckets.items() 
                if 0 < int(pos) <= MAX_VALID_X_POS
            }
            for level, buckets in data.items()
        }
    except Exception:
        return None


@st.cache_data(ttl=2)
def load_worker_metrics(checkpoint_dir: str) -> dict[int, pd.DataFrame]:
    """Load all worker metrics CSVs using DuckDB for 2-4x faster loading."""
    checkpoint_path = Path(checkpoint_dir)
    
    # Try metrics subdirectory first, then root (for compatibility)
    metrics_dir = checkpoint_path / "metrics"
    if not metrics_dir.exists():
        metrics_dir = checkpoint_path
    
    glob_pattern = str(metrics_dir / "worker_*.csv")
    
    # Check if any CSV files exist
    csv_files = list(metrics_dir.glob("worker_*.csv"))
    if not csv_files:
        return {}
    
    try:
        # Use DuckDB to read all CSVs at once (2-4x faster than pandas per-file)
        # Extract worker_id from filename using regex
        combined_df = duckdb.sql(f"""
            SELECT 
                *,
                CAST(regexp_extract(filename, 'worker_(\\d+)', 1) AS INTEGER) as worker_id 
            FROM read_csv_auto('{glob_pattern}', filename=true, union_by_name=true)
        """).df()
        
        # Split into dict by worker_id for compatibility with aggregators
        workers = {}
        for worker_id in combined_df["worker_id"].unique():
            worker_df = combined_df[combined_df["worker_id"] == worker_id].drop(
                columns=["filename", "worker_id"]
            )
            if len(worker_df) > 0:
                workers[int(worker_id)] = worker_df
        
        return workers
    except Exception:
        # Fallback to pandas if DuckDB fails
        workers = {}
        for csv_file in sorted(csv_files):
            try:
                worker_id = int(csv_file.stem.split("_")[1])
                df = pd.read_csv(csv_file)
                if len(df) > 0:
                    workers[worker_id] = df
            except Exception:
                continue
        return workers
