"""Data loading functions for the training dashboard."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st


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
    csv_path = Path(checkpoint_dir) / "metrics" / "coordinator.csv"
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
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
        return {
            level: {int(pos): count for pos, count in buckets.items()}
            for level, buckets in data.items()
        }
    except Exception:
        return None


@st.cache_data(ttl=2)
def load_worker_metrics(checkpoint_dir: str) -> dict[int, pd.DataFrame]:
    """Load all worker metrics CSVs."""
    metrics_dir = Path(checkpoint_dir) / "metrics"
    if not metrics_dir.exists():
        return {}
    
    workers = {}
    for csv_file in sorted(metrics_dir.glob("worker_*.csv")):
        try:
            worker_id = int(csv_file.stem.split("_")[1])
            df = pd.read_csv(csv_file)
            if len(df) > 0:
                workers[worker_id] = df
        except Exception:
            continue
    
    return workers
