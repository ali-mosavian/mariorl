"""
MetricAggregator - aggregates metrics from all sources in main process.

Single responsibility: receive snapshots and provide aggregated views.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Metrics that should be summed across workers (counters)
COUNTER_METRICS = {"episodes", "steps", "deaths", "flags", "attempts", "completions"}


@dataclass
class MetricAggregator:
    """Aggregates metrics from workers and coordinator.
    
    Receives snapshots from all sources via ZMQ events and provides
    aggregated views for the UI and logging.
    
    Usage:
        agg = MetricAggregator(num_workers=4)
        
        # On receiving metric events
        agg.update("worker.0", snapshot)
        agg.update("coordinator", snapshot)
        
        # For UI display
        summary = agg.summary()
    """
    
    num_workers: int
    
    # Latest snapshot per source
    _snapshots: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)
    
    def update(self, source: str, snapshot: dict[str, Any]) -> None:
        """Update snapshot for a source.
        
        Args:
            source: Source identifier ("worker.0", "coordinator", etc.)
            snapshot: Snapshot dict from that source
        """
        self._snapshots[source] = snapshot
    
    def worker_snapshot(self, worker_id: int) -> dict[str, Any] | None:
        """Get latest snapshot for a worker.
        
        Args:
            worker_id: Worker index (0, 1, 2, ...)
            
        Returns:
            Snapshot dict or None if not yet received.
        """
        return self._snapshots.get(f"worker.{worker_id}")
    
    def coordinator_snapshot(self) -> dict[str, Any] | None:
        """Get latest snapshot for coordinator.
        
        Returns:
            Snapshot dict or None if not yet received.
        """
        return self._snapshots.get("coordinator")
    
    def aggregate(self) -> dict[str, Any]:
        """Compute aggregated metrics across all workers.
        
        Counters are summed, other metrics are averaged and max tracked.
        
        Returns:
            Dict with aggregated metrics (total_*, mean_*, max_*).
        """
        worker_snaps = [
            self._snapshots.get(f"worker.{i}")
            for i in range(self.num_workers)
        ]
        worker_snaps = [s for s in worker_snaps if s is not None]
        
        if not worker_snaps:
            return {}
        
        result: dict[str, Any] = {}
        
        # Collect all keys (excluding timestamp and levels)
        all_keys: set[str] = set()
        for snap in worker_snaps:
            all_keys.update(k for k in snap.keys() if k not in ("timestamp", "levels"))
        
        for key in all_keys:
            values = [s[key] for s in worker_snaps if key in s]
            if not values:
                continue
            
            if key in COUNTER_METRICS:
                # Sum counters
                result[f"total_{key}"] = sum(values)
            else:
                # Average and max for others
                result[f"mean_{key}"] = sum(values) / len(values)
                result[f"max_{key}"] = max(values)
        
        return result
    
    def aggregate_levels(self) -> dict[str, dict[str, Any]]:
        """Aggregate per-level stats across all workers.
        
        Combines level statistics from all workers, summing counters
        and tracking maximums.
        
        Returns:
            Dict mapping level_id to aggregated level stats.
        """
        combined: dict[str, dict[str, Any]] = {}
        
        for source, snap in self._snapshots.items():
            if not source.startswith("worker."):
                continue
            
            levels = snap.get("levels", {})
            for level_id, level_data in levels.items():
                if level_id not in combined:
                    combined[level_id] = {
                        "attempts": 0,
                        "completions": 0,
                        "deaths": 0,
                        "best_x": 0,
                    }
                
                # Sum counters
                combined[level_id]["attempts"] += level_data.get("attempts", 0)
                combined[level_id]["completions"] += level_data.get("completions", 0)
                combined[level_id]["deaths"] += level_data.get("deaths", 0)
                
                # Track max
                if level_data.get("best_x", 0) > combined[level_id]["best_x"]:
                    combined[level_id]["best_x"] = level_data["best_x"]
        
        return combined
    
    def summary(self) -> dict[str, Any]:
        """Get full summary of all metrics.
        
        Returns:
            Dict with workers, coordinator, aggregated, and levels sections.
        """
        return {
            "workers": {
                f"worker.{i}": self.worker_snapshot(i)
                for i in range(self.num_workers)
                if self.worker_snapshot(i) is not None
            },
            "coordinator": self.coordinator_snapshot(),
            "aggregated": self.aggregate(),
            "levels": self.aggregate_levels(),
        }
