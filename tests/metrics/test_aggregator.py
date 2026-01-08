"""Tests for MetricAggregator."""

import pytest

from mario_rl.metrics.aggregator import MetricAggregator


# =============================================================================
# Basic Update Tests
# =============================================================================


def test_aggregator_stores_snapshot():
    """update() stores snapshot for source."""
    agg = MetricAggregator(num_workers=2)
    snap = {"timestamp": 1000.0, "episodes": 10, "reward": 100.0}
    agg.update("worker.0", snap)
    
    assert agg.worker_snapshot(0) == snap


def test_aggregator_worker_snapshot_returns_none_if_missing():
    """worker_snapshot() returns None for missing worker."""
    agg = MetricAggregator(num_workers=2)
    assert agg.worker_snapshot(0) is None


def test_aggregator_coordinator_snapshot():
    """coordinator_snapshot() returns coordinator data."""
    agg = MetricAggregator(num_workers=2)
    snap = {"timestamp": 1000.0, "update_count": 50}
    agg.update("coordinator", snap)
    
    assert agg.coordinator_snapshot() == snap


def test_aggregator_updates_overwrite():
    """update() overwrites previous snapshot."""
    agg = MetricAggregator(num_workers=2)
    agg.update("worker.0", {"episodes": 10})
    agg.update("worker.0", {"episodes": 20})
    
    assert agg.worker_snapshot(0)["episodes"] == 20


# =============================================================================
# Aggregation Tests
# =============================================================================


def test_aggregate_empty_returns_empty():
    """aggregate() returns empty dict when no workers."""
    agg = MetricAggregator(num_workers=2)
    assert agg.aggregate() == {}


def test_aggregate_sums_counters():
    """aggregate() sums counter metrics across workers."""
    agg = MetricAggregator(num_workers=2)
    agg.update("worker.0", {"episodes": 10, "steps": 1000})
    agg.update("worker.1", {"episodes": 15, "steps": 1500})
    
    result = agg.aggregate()
    assert result["total_episodes"] == 25
    assert result["total_steps"] == 2500


def test_aggregate_averages_gauges():
    """aggregate() averages gauge metrics across workers."""
    agg = MetricAggregator(num_workers=2)
    agg.update("worker.0", {"epsilon": 0.2, "reward": 100.0})
    agg.update("worker.1", {"epsilon": 0.4, "reward": 200.0})
    
    result = agg.aggregate()
    assert result["mean_epsilon"] == pytest.approx(0.3)
    assert result["mean_reward"] == pytest.approx(150.0)


def test_aggregate_tracks_max():
    """aggregate() tracks max values."""
    agg = MetricAggregator(num_workers=2)
    agg.update("worker.0", {"best_x": 500})
    agg.update("worker.1", {"best_x": 800})
    
    result = agg.aggregate()
    assert result["max_best_x"] == 800


def test_aggregate_partial_workers():
    """aggregate() works with partial worker data."""
    agg = MetricAggregator(num_workers=3)
    agg.update("worker.0", {"episodes": 10})
    # worker.1 and worker.2 have no data yet
    
    result = agg.aggregate()
    assert result["total_episodes"] == 10


# =============================================================================
# Level Aggregation Tests
# =============================================================================


def test_aggregate_levels_empty():
    """aggregate_levels() returns empty when no level data."""
    agg = MetricAggregator(num_workers=2)
    agg.update("worker.0", {"episodes": 10})
    
    assert agg.aggregate_levels() == {}


def test_aggregate_levels_combines_workers():
    """aggregate_levels() combines level stats across workers."""
    agg = MetricAggregator(num_workers=2)
    
    agg.update("worker.0", {
        "episodes": 10,
        "levels": {
            "1-1": {"attempts": 5, "completions": 2, "deaths": 3, "best_x": 500},
        }
    })
    agg.update("worker.1", {
        "episodes": 8,
        "levels": {
            "1-1": {"attempts": 4, "completions": 1, "deaths": 3, "best_x": 700},
        }
    })
    
    levels = agg.aggregate_levels()
    assert levels["1-1"]["attempts"] == 9  # 5 + 4
    assert levels["1-1"]["completions"] == 3  # 2 + 1
    assert levels["1-1"]["deaths"] == 6  # 3 + 3
    assert levels["1-1"]["best_x"] == 700  # max


def test_aggregate_levels_different_levels():
    """aggregate_levels() handles different levels per worker."""
    agg = MetricAggregator(num_workers=2)
    
    agg.update("worker.0", {
        "levels": {
            "1-1": {"attempts": 5, "completions": 2, "deaths": 3, "best_x": 500},
        }
    })
    agg.update("worker.1", {
        "levels": {
            "1-2": {"attempts": 3, "completions": 0, "deaths": 3, "best_x": 300},
        }
    })
    
    levels = agg.aggregate_levels()
    assert "1-1" in levels
    assert "1-2" in levels
    assert levels["1-1"]["attempts"] == 5
    assert levels["1-2"]["attempts"] == 3


# =============================================================================
# Summary Tests
# =============================================================================


def test_summary_includes_workers():
    """summary() includes per-worker data."""
    agg = MetricAggregator(num_workers=2)
    agg.update("worker.0", {"episodes": 10})
    
    result = agg.summary()
    assert "workers" in result
    assert result["workers"]["worker.0"]["episodes"] == 10


def test_summary_includes_coordinator():
    """summary() includes coordinator data."""
    agg = MetricAggregator(num_workers=2)
    agg.update("coordinator", {"update_count": 100})
    
    result = agg.summary()
    assert "coordinator" in result
    assert result["coordinator"]["update_count"] == 100


def test_summary_includes_aggregated():
    """summary() includes aggregated metrics."""
    agg = MetricAggregator(num_workers=2)
    agg.update("worker.0", {"episodes": 10})
    agg.update("worker.1", {"episodes": 15})
    
    result = agg.summary()
    assert "aggregated" in result
    assert result["aggregated"]["total_episodes"] == 25


def test_summary_includes_levels():
    """summary() includes aggregated level data."""
    agg = MetricAggregator(num_workers=2)
    agg.update("worker.0", {
        "levels": {"1-1": {"attempts": 5, "completions": 2, "deaths": 3, "best_x": 500}}
    })
    
    result = agg.summary()
    assert "levels" in result
    assert "1-1" in result["levels"]
