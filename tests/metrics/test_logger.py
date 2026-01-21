"""Tests for MetricLogger."""

import csv
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from mario_rl.metrics.schema import DDQNMetrics
from mario_rl.metrics.logger import MetricLogger


@pytest.fixture
def tmp_csv(tmp_path: Path) -> Path:
    """Create temporary CSV path."""
    return tmp_path / "metrics.csv"


@pytest.fixture
def logger(tmp_csv: Path) -> MetricLogger:
    """Create a basic logger with DDQN schema."""
    return MetricLogger(
        source_id="worker.0",
        schema=DDQNMetrics,
        csv_path=tmp_csv,
    )


# =============================================================================
# Counter Tests
# =============================================================================


def test_counter_starts_at_zero(logger: MetricLogger) -> None:
    """Counters start at zero."""
    snap = logger.snapshot()
    assert snap["episodes"] == 0


def test_counter_increment_by_one(logger: MetricLogger) -> None:
    """count() increments counter by 1 by default."""
    logger.count("episodes")
    snap = logger.snapshot()
    assert snap["episodes"] == 1


def test_counter_increment_by_n(logger: MetricLogger) -> None:
    """count() increments counter by n."""
    logger.count("steps", n=64)
    snap = logger.snapshot()
    assert snap["steps"] == 64


def test_counter_accumulates(logger: MetricLogger) -> None:
    """Multiple count() calls accumulate."""
    logger.count("episodes")
    logger.count("episodes")
    logger.count("episodes")
    snap = logger.snapshot()
    assert snap["episodes"] == 3


# =============================================================================
# Gauge Tests
# =============================================================================


def test_gauge_starts_at_zero(logger: MetricLogger) -> None:
    """Gauges start at zero."""
    snap = logger.snapshot()
    assert snap["epsilon"] == 0.0


def test_gauge_set(logger: MetricLogger) -> None:
    """gauge() sets the current value."""
    logger.gauge("epsilon", 0.5)
    snap = logger.snapshot()
    assert snap["epsilon"] == 0.5


def test_gauge_overwrites(logger: MetricLogger) -> None:
    """gauge() overwrites previous value."""
    logger.gauge("epsilon", 0.5)
    logger.gauge("epsilon", 0.1)
    snap = logger.snapshot()
    assert snap["epsilon"] == 0.1


# =============================================================================
# Rolling Average Tests
# =============================================================================


def test_rolling_starts_at_zero(logger: MetricLogger) -> None:
    """Rolling averages start at zero (empty buffer)."""
    snap = logger.snapshot()
    assert snap["reward"] == 0.0


def test_rolling_single_value(logger: MetricLogger) -> None:
    """observe() with single value returns that value."""
    logger.observe("reward", 100.0)
    snap = logger.snapshot()
    assert snap["reward"] == 100.0


def test_rolling_average_of_multiple(logger: MetricLogger) -> None:
    """observe() computes average of multiple values."""
    logger.observe("reward", 100.0)
    logger.observe("reward", 200.0)
    logger.observe("reward", 300.0)
    snap = logger.snapshot()
    assert snap["reward"] == 200.0  # (100 + 200 + 300) / 3


def test_rolling_respects_window_size(tmp_csv: Path) -> None:
    """Rolling average respects configured window size."""
    logger = MetricLogger(
        source_id="test",
        schema=DDQNMetrics,
        csv_path=tmp_csv,
    )

    # The default window is 100, add more than that
    # First, add 100 values of 100.0
    for _ in range(100):
        logger.observe("reward", 100.0)

    # Then add 100 more values of 200.0 (these should push out the old ones)
    for _ in range(100):
        logger.observe("reward", 200.0)

    snap = logger.snapshot()
    # Window should only keep last 100 values (all 200s)
    assert snap["reward"] == 200.0


# =============================================================================
# Snapshot Tests
# =============================================================================


def test_snapshot_includes_timestamp(logger: MetricLogger) -> None:
    """Snapshot includes timestamp."""
    before = time.time()
    snap = logger.snapshot()
    after = time.time()
    assert before <= snap["timestamp"] <= after


def test_snapshot_includes_all_schema_metrics(logger: MetricLogger) -> None:
    """Snapshot includes all metrics from schema."""
    snap = logger.snapshot()
    for defn in DDQNMetrics.definitions():
        assert defn.name in snap, f"Missing {defn.name}"


# =============================================================================
# CSV Tests
# =============================================================================


def test_flush_creates_csv(logger: MetricLogger, tmp_csv: Path) -> None:
    """flush() creates CSV file."""
    logger.flush()
    assert tmp_csv.exists()


def test_flush_writes_header(logger: MetricLogger, tmp_csv: Path) -> None:
    """flush() writes CSV header on first call."""
    logger.flush()
    with open(tmp_csv) as f:
        reader = csv.reader(f)
        header = next(reader)
        assert "timestamp" in header
        assert "episodes" in header
        assert "reward" in header


def test_flush_writes_data_row(logger: MetricLogger, tmp_csv: Path) -> None:
    """flush() writes data row."""
    logger.count("episodes")
    logger.gauge("epsilon", 0.5)
    logger.observe("reward", 100.0)
    logger.flush()

    with open(tmp_csv) as f:
        reader = csv.DictReader(f)
        row = next(reader)
        assert row["episodes"] == "1"
        assert row["epsilon"] == "0.5"
        assert row["reward"] == "100.0"


def test_multiple_flushes_append_rows(logger: MetricLogger, tmp_csv: Path) -> None:
    """Multiple flush() calls append rows."""
    logger.count("episodes")
    logger.flush()

    logger.count("episodes")
    logger.flush()

    with open(tmp_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["episodes"] == "1"
        assert rows[1]["episodes"] == "2"


# =============================================================================
# ZMQ Publishing Tests
# =============================================================================


def test_flush_publishes_when_publisher_provided(tmp_csv: Path) -> None:
    """flush() publishes snapshot via EventPublisher."""
    mock_pub = Mock()
    logger = MetricLogger(
        source_id="worker.0",
        schema=DDQNMetrics,
        csv_path=tmp_csv,
        publisher=mock_pub,
    )

    logger.count("episodes")
    logger.flush()

    mock_pub.publish.assert_called_once()
    call_args = mock_pub.publish.call_args
    assert call_args[0][0] == "metrics"  # msg_type
    assert "snapshot" in call_args[0][1]
    assert call_args[0][1]["source"] == "worker.0"


def test_flush_without_publisher_succeeds(logger: MetricLogger, tmp_csv: Path) -> None:
    """flush() works without publisher (CSV only)."""
    logger.count("episodes")
    logger.flush()  # Should not raise
    assert tmp_csv.exists()


# =============================================================================
# Close Tests
# =============================================================================


def test_close_flushes_file(logger: MetricLogger, tmp_csv: Path) -> None:
    """close() flushes and closes file."""
    logger.count("episodes")
    logger.flush()
    logger.close()

    # File should be readable
    with open(tmp_csv) as f:
        content = f.read()
        assert "episodes" in content
