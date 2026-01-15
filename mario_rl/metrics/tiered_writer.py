"""
TieredMetricWriter - CSV hot tier with Parquet cold tier.

Appends to CSV for fast writes, rotates to Parquet when threshold reached.
Combines benefits of:
- Fast appends (CSV)
- Efficient storage and reads (Parquet columnar + compression)
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import TextIO

import pandas as pd


def read_source(base_path: Path, source_id: str) -> pd.DataFrame:
    """Read all data for a source (Parquet shards + CSV).
    
    Args:
        base_path: Directory containing metric files.
        source_id: Source identifier (e.g., "worker_0").
    
    Returns:
        Combined DataFrame with all metrics for this source.
    """
    base_path = Path(base_path)
    dfs: list[pd.DataFrame] = []
    
    # Find and read all Parquet shards for this source
    parquet_pattern = f"{source_id}_*.parquet"
    for parquet_path in sorted(base_path.glob(parquet_pattern)):
        dfs.append(pd.read_parquet(parquet_path))
    
    # Read CSV if exists
    csv_path = base_path / f"{source_id}.csv"
    if csv_path.exists():
        try:
            csv_df = pd.read_csv(csv_path)
            if len(csv_df) > 0:
                dfs.append(csv_df)
        except pd.errors.EmptyDataError:
            pass
    
    if not dfs:
        return pd.DataFrame()
    
    combined = pd.concat(dfs, ignore_index=True)
    if "timestamp" in combined.columns:
        combined = combined.sort_values("timestamp").reset_index(drop=True)
    
    return combined


def read_all_sources(base_path: Path, pattern: str = "worker_*") -> pd.DataFrame:
    """Read all data from all sources matching pattern.
    
    Args:
        base_path: Directory containing metric files.
        pattern: Glob pattern for source IDs (e.g., "worker_*").
    
    Returns:
        Combined DataFrame with source_id column identifying each source.
    """
    base_path = Path(base_path)
    
    # Find all unique source IDs from both CSV and Parquet files
    source_ids: set[str] = set()
    
    # From CSV files
    for csv_path in base_path.glob(f"{pattern}.csv"):
        source_ids.add(csv_path.stem)
    
    # From Parquet files (extract source_id from name like "worker_0_000001.parquet")
    for parquet_path in base_path.glob(f"{pattern}_*.parquet"):
        # Extract source_id: "worker_0_000001" -> "worker_0"
        match = re.match(r"(.+)_\d{6}\.parquet$", parquet_path.name)
        if match:
            source_ids.add(match.group(1))
    
    if not source_ids:
        return pd.DataFrame()
    
    # Read each source and add source_id column
    dfs: list[pd.DataFrame] = []
    for source_id in sorted(source_ids):
        df = read_source(base_path, source_id)
        if len(df) > 0:
            df["source_id"] = source_id
            dfs.append(df)
    
    if not dfs:
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True)


@dataclass
class TieredMetricWriter:
    """Append to CSV, rotate to Parquet when threshold reached.
    
    Usage:
        writer = TieredMetricWriter(
            base_path=Path("metrics"),
            source_id="worker_0",
            max_csv_rows=5000,
        )
        
        # During training
        writer.write({"timestamp": 1.0, "steps": 100, "loss": 0.5})
        
        # When done
        writer.close()
    
    Files created:
        metrics/worker_0.csv           # Hot: recent data
        metrics/worker_0_000000.parquet  # Cold: archived shard
        metrics/worker_0_000001.parquet  # Cold: archived shard
    """
    
    base_path: Path
    source_id: str
    max_csv_rows: int = 5000
    
    # Internal state
    _csv_file: TextIO | None = field(default=None, init=False, repr=False)
    _csv_writer: csv.DictWriter | None = field(default=None, init=False, repr=False)
    _fieldnames: list[str] = field(default_factory=list, init=False, repr=False)
    _row_count: int = field(default=0, init=False, repr=False)
    _shard_index: int = field(default=0, init=False, repr=False)
    
    def __post_init__(self) -> None:
        """Initialize base_path and discover existing shards."""
        self.base_path = Path(self.base_path)
        self._discover_existing_shards()
    
    def _discover_existing_shards(self) -> None:
        """Find existing Parquet shards and set shard index accordingly."""
        pattern = f"{self.source_id}_*.parquet"
        existing = list(self.base_path.glob(pattern))
        if existing:
            indices = []
            for p in existing:
                match = re.match(rf"{re.escape(self.source_id)}_(\d{{6}})\.parquet$", p.name)
                if match:
                    indices.append(int(match.group(1)))
            if indices:
                self._shard_index = max(indices) + 1
    
    def _csv_path(self) -> Path:
        """Get path to current CSV file."""
        return self.base_path / f"{self.source_id}.csv"
    
    def _parquet_path(self, shard: int) -> Path:
        """Get path to a Parquet shard."""
        return self.base_path / f"{self.source_id}_{shard:06d}.parquet"
    
    def _open_csv(self, fieldnames: list[str]) -> None:
        """Open CSV file for writing."""
        self._fieldnames = fieldnames
        self._csv_file = open(self._csv_path(), "w", newline="")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
        self._csv_writer.writeheader()
    
    def _reopen_csv_with_new_columns(self, new_fieldnames: list[str]) -> None:
        """Reopen CSV with expanded schema (new columns added)."""
        # Read existing data
        self._csv_file.close()
        existing_df = pd.read_csv(self._csv_path())
        
        # Rewrite with new schema
        self._fieldnames = new_fieldnames
        self._csv_file = open(self._csv_path(), "w", newline="")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=new_fieldnames)
        self._csv_writer.writeheader()
        
        # Rewrite existing rows
        for _, row in existing_df.iterrows():
            self._csv_writer.writerow(row.to_dict())
    
    def write(self, record: dict[str, Any]) -> None:
        """Append record to CSV, rotate to Parquet if threshold reached.
        
        Args:
            record: Dict of metric values to write.
        """
        # Check if we need to handle new columns
        record_keys = list(record.keys())
        
        if self._csv_file is None:
            # First write - create CSV with this record's schema
            self._open_csv(record_keys)
        elif set(record_keys) - set(self._fieldnames):
            # New columns appeared - need to rewrite CSV with expanded schema
            new_fieldnames = self._fieldnames + [
                k for k in record_keys if k not in self._fieldnames
            ]
            self._reopen_csv_with_new_columns(new_fieldnames)
        
        # Write record (filling missing columns with empty string)
        row = {k: record.get(k, "") for k in self._fieldnames}
        self._csv_writer.writerow(row)
        self._csv_file.flush()
        self._row_count += 1
        
        # Rotate if threshold reached
        if self._row_count >= self.max_csv_rows:
            self._rotate_to_parquet()
    
    def _rotate_to_parquet(self) -> None:
        """Convert CSV to Parquet shard and start fresh CSV."""
        # Close current CSV
        self._csv_file.close()
        self._csv_file = None
        
        # Read CSV and write as Parquet
        csv_path = self._csv_path()
        df = pd.read_csv(csv_path)
        
        parquet_path = self._parquet_path(self._shard_index)
        df.to_parquet(parquet_path, compression="zstd", index=False)
        
        # Increment shard index and reset row count
        self._shard_index += 1
        self._row_count = 0
        
        # Delete old CSV (will be recreated on next write)
        csv_path.unlink()
    
    def read_all(self) -> pd.DataFrame:
        """Read all data from Parquet shards and CSV.
        
        Returns:
            Combined DataFrame with all metrics, ordered by timestamp.
        """
        dfs: list[pd.DataFrame] = []
        
        # Read all Parquet shards
        for shard in range(self._shard_index):
            parquet_path = self._parquet_path(shard)
            if parquet_path.exists():
                dfs.append(pd.read_parquet(parquet_path))
        
        # Read current CSV if it exists and has data
        csv_path = self._csv_path()
        if csv_path.exists():
            try:
                csv_df = pd.read_csv(csv_path)
                if len(csv_df) > 0:
                    dfs.append(csv_df)
            except pd.errors.EmptyDataError:
                pass
        
        if not dfs:
            return pd.DataFrame()
        
        # Combine and sort by timestamp if present
        combined = pd.concat(dfs, ignore_index=True)
        if "timestamp" in combined.columns:
            combined = combined.sort_values("timestamp").reset_index(drop=True)
        
        return combined
    
    def close(self) -> None:
        """Close the CSV file."""
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
    
    def __enter__(self) -> TieredMetricWriter:
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - close file."""
        self.close()
