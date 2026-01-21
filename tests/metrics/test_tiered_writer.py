"""Tests for TieredMetricWriter - CSV hot tier with Parquet cold tier."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import pandas as pd

from mario_rl.metrics.tiered_writer import TieredMetricWriter


@pytest.fixture
def metrics_dir(tmp_path: Path) -> Path:
    """Create a temporary metrics directory."""
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    return metrics


@pytest.fixture
def writer(metrics_dir: Path) -> TieredMetricWriter:
    """Create a TieredMetricWriter for testing."""
    return TieredMetricWriter(
        base_path=metrics_dir,
        source_id="worker_0",
        max_csv_rows=5,  # Low threshold for testing
    )


class TestBasicWrites:
    """Test basic CSV writing functionality."""

    def test_write_creates_csv_file(self, writer: TieredMetricWriter, metrics_dir: Path) -> None:
        """First write should create the CSV file."""
        writer.write({"timestamp": 1.0, "steps": 100, "loss": 0.5})

        csv_path = metrics_dir / "worker_0.csv"
        assert csv_path.exists()

    def test_write_includes_header(self, writer: TieredMetricWriter, metrics_dir: Path) -> None:
        """CSV should have header row."""
        writer.write({"timestamp": 1.0, "steps": 100, "loss": 0.5})

        csv_path = metrics_dir / "worker_0.csv"
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["timestamp"] == "1.0"
        assert rows[0]["steps"] == "100"
        assert rows[0]["loss"] == "0.5"

    def test_multiple_writes_append(self, writer: TieredMetricWriter, metrics_dir: Path) -> None:
        """Multiple writes should append to CSV."""
        writer.write({"timestamp": 1.0, "steps": 100})
        writer.write({"timestamp": 2.0, "steps": 200})
        writer.write({"timestamp": 3.0, "steps": 300})

        csv_path = metrics_dir / "worker_0.csv"
        df = pd.read_csv(csv_path)

        assert len(df) == 3
        assert list(df["steps"]) == [100, 200, 300]


class TestRotation:
    """Test rotation from CSV to Parquet."""

    def test_rotation_creates_parquet(self, writer: TieredMetricWriter, metrics_dir: Path) -> None:
        """When max_csv_rows reached, should create Parquet shard."""
        # Write exactly max_csv_rows (5) to trigger rotation
        for i in range(5):
            writer.write({"timestamp": float(i), "steps": i * 100})

        parquet_path = metrics_dir / "worker_0_000000.parquet"
        assert parquet_path.exists()

    def test_rotation_truncates_csv(self, writer: TieredMetricWriter, metrics_dir: Path) -> None:
        """After rotation, CSV should be empty/reset."""
        # Write 5 rows to trigger rotation
        for i in range(5):
            writer.write({"timestamp": float(i), "steps": i * 100})

        # Write one more row (goes to new CSV)
        writer.write({"timestamp": 5.0, "steps": 500})

        csv_path = metrics_dir / "worker_0.csv"
        df = pd.read_csv(csv_path)

        # CSV should only have the new row
        assert len(df) == 1
        assert df["steps"].iloc[0] == 500

    def test_parquet_contains_rotated_data(self, writer: TieredMetricWriter, metrics_dir: Path) -> None:
        """Parquet shard should contain the rotated rows."""
        for i in range(5):
            writer.write({"timestamp": float(i), "steps": i * 100})

        parquet_path = metrics_dir / "worker_0_000000.parquet"
        df = pd.read_parquet(parquet_path)

        assert len(df) == 5
        assert list(df["steps"]) == [0, 100, 200, 300, 400]

    def test_multiple_rotations(self, writer: TieredMetricWriter, metrics_dir: Path) -> None:
        """Multiple rotations should create multiple shards."""
        # Write 12 rows: 5 + 5 + 2 = two rotations, 2 in CSV
        for i in range(12):
            writer.write({"timestamp": float(i), "steps": i * 100})

        # Should have two Parquet shards
        assert (metrics_dir / "worker_0_000000.parquet").exists()
        assert (metrics_dir / "worker_0_000001.parquet").exists()

        # CSV should have remaining 2 rows
        csv_df = pd.read_csv(metrics_dir / "worker_0.csv")
        assert len(csv_df) == 2
        assert list(csv_df["steps"]) == [1000, 1100]

    def test_shard_index_increments(self, writer: TieredMetricWriter, metrics_dir: Path) -> None:
        """Shard index should increment with each rotation."""
        # Three rotations
        for i in range(15):
            writer.write({"timestamp": float(i), "steps": i})

        assert (metrics_dir / "worker_0_000000.parquet").exists()
        assert (metrics_dir / "worker_0_000001.parquet").exists()
        assert (metrics_dir / "worker_0_000002.parquet").exists()


class TestReadAll:
    """Test reading combined data from CSV and Parquet."""

    def test_read_all_combines_parquet_and_csv(self, writer: TieredMetricWriter, metrics_dir: Path) -> None:
        """read_all should combine Parquet shards and CSV."""
        # Write 7 rows: 5 go to Parquet, 2 stay in CSV
        for i in range(7):
            writer.write({"timestamp": float(i), "steps": i * 100})

        df = writer.read_all()

        assert len(df) == 7
        assert list(df["steps"]) == [0, 100, 200, 300, 400, 500, 600]

    def test_read_all_empty(self, writer: TieredMetricWriter, metrics_dir: Path) -> None:
        """read_all on empty writer should return empty DataFrame."""
        df = writer.read_all()
        assert len(df) == 0

    def test_read_all_csv_only(self, writer: TieredMetricWriter, metrics_dir: Path) -> None:
        """read_all with only CSV data should work."""
        writer.write({"timestamp": 1.0, "steps": 100})
        writer.write({"timestamp": 2.0, "steps": 200})

        df = writer.read_all()

        assert len(df) == 2
        assert list(df["steps"]) == [100, 200]

    def test_read_all_parquet_only(self, writer: TieredMetricWriter, metrics_dir: Path) -> None:
        """read_all with only Parquet data (CSV empty) should work."""
        # Exactly 5 rows triggers rotation, CSV is then empty
        for i in range(5):
            writer.write({"timestamp": float(i), "steps": i * 100})

        df = writer.read_all()

        assert len(df) == 5


class TestClose:
    """Test cleanup and close behavior."""

    def test_close_flushes_csv(self, writer: TieredMetricWriter, metrics_dir: Path) -> None:
        """close() should flush any pending CSV data."""
        writer.write({"timestamp": 1.0, "steps": 100})
        writer.close()

        # Should be able to read the data after close
        csv_path = metrics_dir / "worker_0.csv"
        df = pd.read_csv(csv_path)
        assert len(df) == 1

    def test_context_manager(self, metrics_dir: Path) -> None:
        """Should support context manager protocol."""
        with TieredMetricWriter(
            base_path=metrics_dir,
            source_id="worker_0",
            max_csv_rows=5,
        ) as writer:
            writer.write({"timestamp": 1.0, "steps": 100})

        # File should be properly closed
        csv_path = metrics_dir / "worker_0.csv"
        df = pd.read_csv(csv_path)
        assert len(df) == 1


class TestSchemaHandling:
    """Test handling of varying schemas."""

    def test_new_columns_added(self, writer: TieredMetricWriter, metrics_dir: Path) -> None:
        """New columns appearing mid-stream should be handled."""
        writer.write({"timestamp": 1.0, "steps": 100})
        writer.write({"timestamp": 2.0, "steps": 200, "loss": 0.5})  # New column

        csv_path = metrics_dir / "worker_0.csv"
        df = pd.read_csv(csv_path)

        assert "loss" in df.columns
        assert pd.isna(df["loss"].iloc[0])  # First row missing loss
        assert df["loss"].iloc[1] == 0.5


class TestRecovery:
    """Test recovery and state persistence."""

    def test_discovers_existing_shards(self, metrics_dir: Path) -> None:
        """New writer should discover existing Parquet shards."""
        # Create some pre-existing shards
        df1 = pd.DataFrame({"timestamp": [1.0, 2.0], "steps": [100, 200]})
        df1.to_parquet(metrics_dir / "worker_0_000000.parquet", index=False)

        df2 = pd.DataFrame({"timestamp": [3.0, 4.0], "steps": [300, 400]})
        df2.to_parquet(metrics_dir / "worker_0_000001.parquet", index=False)

        # Create writer - should detect 2 existing shards
        writer = TieredMetricWriter(
            base_path=metrics_dir,
            source_id="worker_0",
            max_csv_rows=5,
        )

        # Write new data
        writer.write({"timestamp": 5.0, "steps": 500})

        # read_all should include all data
        df = writer.read_all()
        assert len(df) == 5
        assert list(df["steps"]) == [100, 200, 300, 400, 500]

    def test_new_shard_after_existing(self, metrics_dir: Path) -> None:
        """New rotation should create shard with correct index after existing."""
        # Create pre-existing shard
        df1 = pd.DataFrame({"timestamp": [1.0, 2.0], "steps": [100, 200]})
        shard_0_path = metrics_dir / "worker_0_000000.parquet"
        df1.to_parquet(shard_0_path, index=False)
        original_mtime = shard_0_path.stat().st_mtime

        writer = TieredMetricWriter(
            base_path=metrics_dir,
            source_id="worker_0",
            max_csv_rows=3,  # Low threshold
        )

        # Write enough to trigger rotation
        for i in range(4):
            writer.write({"timestamp": float(10 + i), "steps": 1000 + i * 100})

        # Should create shard 000001 (not 000000 which exists)
        assert (metrics_dir / "worker_0_000001.parquet").exists()
        # Original shard should be unchanged
        assert shard_0_path.stat().st_mtime == original_mtime


class TestStaticRead:
    """Test static read functions for dashboard use."""

    def test_read_source_combines_all(self, metrics_dir: Path) -> None:
        """read_source should read all data for a source."""
        from mario_rl.metrics.tiered_writer import read_source

        # Create data
        writer = TieredMetricWriter(
            base_path=metrics_dir,
            source_id="worker_0",
            max_csv_rows=3,
        )
        for i in range(5):
            writer.write({"timestamp": float(i), "value": i})
        writer.close()

        # Read back with static function
        df = read_source(metrics_dir, "worker_0")
        assert len(df) == 5

    def test_read_all_sources(self, metrics_dir: Path) -> None:
        """read_all_sources should combine data from multiple sources."""
        from mario_rl.metrics.tiered_writer import read_all_sources

        # Create data for two workers
        for worker_id in [0, 1]:
            writer = TieredMetricWriter(
                base_path=metrics_dir,
                source_id=f"worker_{worker_id}",
                max_csv_rows=5,
            )
            for i in range(3):
                writer.write({"timestamp": float(i), "value": worker_id * 100 + i})
            writer.close()

        # Read all sources
        df = read_all_sources(metrics_dir, pattern="worker_*")
        assert len(df) == 6
        assert "source_id" in df.columns
        assert set(df["source_id"]) == {"worker_0", "worker_1"}

    def test_read_all_sources_empty(self, metrics_dir: Path) -> None:
        """read_all_sources on empty dir should return empty DataFrame."""
        from mario_rl.metrics.tiered_writer import read_all_sources

        df = read_all_sources(metrics_dir, pattern="worker_*")
        assert len(df) == 0
