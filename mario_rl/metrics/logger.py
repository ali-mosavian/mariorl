"""
MetricLogger - tracks metrics and writes to CSV/Parquet / publishes via ZMQ.

Single responsibility: track metric values and output them.
Does NOT aggregate across sources (that's MetricAggregator's job).

Uses TieredMetricWriter for storage: CSV hot tier with Parquet cold tier.
"""

from __future__ import annotations

import time
from typing import Any
from pathlib import Path
from typing import Protocol
from collections import deque
from dataclasses import field
from dataclasses import dataclass

from mario_rl.metrics.schema import MetricDef
from mario_rl.metrics.schema import MetricType
from mario_rl.metrics.tiered_writer import TieredMetricWriter


class MetricSchema(Protocol):
    """Protocol for metric schema classes."""

    @classmethod
    def definitions(cls) -> list[MetricDef]: ...


class Publisher(Protocol):
    """Protocol for event publishers (e.g., EventPublisher)."""

    def publish(self, msg_type: str, data: dict[str, Any]) -> None: ...


@dataclass
class MetricLogger:
    """Tracks metrics and writes to CSV/Parquet / publishes via ZMQ.

    Uses tiered storage: CSV for hot data (fast appends), rotates to
    Parquet for cold data (efficient storage and reads).

    Usage:
        logger = MetricLogger(
            source_id="worker.0",
            schema=DDQNMetrics,
            csv_path=Path("metrics/worker_0.csv"),
            publisher=event_publisher,  # Optional
        )

        # During training
        logger.count("episodes")
        logger.gauge("epsilon", 0.1)
        logger.observe("reward", 150.0)

        # Periodically
        logger.flush()  # Writes to storage + publishes snapshot
    """

    source_id: str
    schema: type[MetricSchema]
    csv_path: Path
    publisher: Publisher | None = None
    max_csv_rows: int = 5000  # Rotate to Parquet after this many rows

    # Internal state
    _counters: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _gauges: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _rolling: dict[str, deque] = field(default_factory=dict, init=False, repr=False)
    _rolling_windows: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _text: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    # Tiered storage writer
    _writer: TieredMetricWriter | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize metric storage and tiered writer."""
        # Initialize metric storage based on schema
        for defn in self.schema.definitions():
            match defn.metric_type:
                case MetricType.COUNTER:
                    self._counters[defn.name] = 0
                case MetricType.GAUGE:
                    self._gauges[defn.name] = 0.0
                case MetricType.ROLLING:
                    self._rolling[defn.name] = deque(maxlen=defn.window)
                    self._rolling_windows[defn.name] = defn.window
                case MetricType.TEXT:
                    self._text[defn.name] = ""

        # Initialize tiered writer (CSV hot tier + Parquet cold tier)
        # Derive base_path and source_id from csv_path for compatibility
        csv_path = Path(self.csv_path)
        self._writer = TieredMetricWriter(
            base_path=csv_path.parent,
            source_id=csv_path.stem,
            max_csv_rows=self.max_csv_rows,
        )

    def count(self, name: str, n: int = 1) -> None:
        """Increment a counter metric.

        Args:
            name: Metric name (must be COUNTER type in schema)
            n: Amount to increment (default 1)
        """
        if name in self._counters:
            self._counters[name] += n

    def gauge(self, name: str, value: float) -> None:
        """Set a gauge metric.

        Args:
            name: Metric name (must be GAUGE type in schema)
            value: Current value
        """
        if name in self._gauges:
            self._gauges[name] = value

    def observe(self, name: str, value: float) -> None:
        """Add a value to a rolling average metric.

        Args:
            name: Metric name (must be ROLLING type in schema)
            value: Value to add
        """
        if name in self._rolling:
            self._rolling[name].append(value)

    def text(self, name: str, value: str) -> None:
        """Set a text metric (e.g., comma-separated positions).

        Args:
            name: Metric name (must be TEXT type in schema)
            value: Text value to store
        """
        if name in self._text:
            self._text[name] = value

    def snapshot(self) -> dict[str, Any]:
        """Get current snapshot of all metrics.

        Returns:
            Dict with timestamp and all metric values.
            Rolling metrics return their current average.
            Text metrics return their current string value.
        """
        snap: dict[str, Any] = {"timestamp": time.time()}

        for defn in self.schema.definitions():
            match defn.metric_type:
                case MetricType.COUNTER:
                    snap[defn.name] = self._counters.get(defn.name, 0)
                case MetricType.GAUGE:
                    snap[defn.name] = self._gauges.get(defn.name, 0.0)
                case MetricType.ROLLING:
                    buf = self._rolling.get(defn.name, deque())
                    snap[defn.name] = sum(buf) / len(buf) if buf else 0.0
                case MetricType.TEXT:
                    snap[defn.name] = self._text.get(defn.name, "")

        return snap

    def flush(self) -> None:
        """Write current snapshot to storage and publish via ZMQ."""
        snap = self.snapshot()

        # Write to tiered storage (CSV hot tier, rotates to Parquet)
        self._write_row(snap)

        # Publish via ZMQ if publisher provided
        if self.publisher is not None:
            self.publisher.publish(
                "metrics",
                {
                    "source": self.source_id,
                    "snapshot": snap,
                },
            )

    def _write_row(self, snap: dict[str, Any]) -> None:
        """Write a row to tiered storage (CSV hot tier, rotates to Parquet)."""
        if self._writer is not None:
            self._writer.write(snap)

    def close(self) -> None:
        """Close the storage writer."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def save_state(self) -> dict[str, Any]:
        """Serialize state for checkpointing.

        Returns:
            Dict that can be saved with torch.save() or json.dump().
        """
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "rolling": {k: list(v) for k, v in self._rolling.items()},
            "text": dict(self._text),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore state from checkpoint.

        Args:
            state: Dict from save_state() or loaded from checkpoint.
        """
        # Restore counters
        for k, v in state.get("counters", {}).items():
            if k in self._counters:
                self._counters[k] = v

        # Restore gauges
        for k, v in state.get("gauges", {}).items():
            if k in self._gauges:
                self._gauges[k] = v

        # Restore rolling buffers
        for k, values in state.get("rolling", {}).items():
            if k in self._rolling:
                self._rolling[k].clear()
                self._rolling[k].extend(values)

        # Restore text fields
        for k, v in state.get("text", {}).items():
            if k in self._text:
                self._text[k] = v
