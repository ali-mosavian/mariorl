"""Tests for checkpoint/resume support in metrics."""

import pytest
from pathlib import Path

from mario_rl.metrics.schema import DDQNMetrics
from mario_rl.metrics.logger import MetricLogger
from mario_rl.metrics.levels import LevelStats, LevelTracker


# =============================================================================
# MetricLogger save_state/load_state Tests
# =============================================================================


def test_logger_save_state_returns_dict(tmp_path):
    """save_state() returns serializable dict."""
    logger = MetricLogger(
        source_id="worker.0",
        schema=DDQNMetrics,
        csv_path=tmp_path / "metrics.csv",
    )
    logger.count("episodes", n=10)
    logger.gauge("epsilon", 0.5)
    logger.observe("reward", 100.0)
    
    state = logger.save_state()
    assert isinstance(state, dict)
    assert "counters" in state
    assert "gauges" in state
    assert "rolling" in state


def test_logger_save_state_contains_counters(tmp_path):
    """save_state() includes counter values."""
    logger = MetricLogger(
        source_id="worker.0",
        schema=DDQNMetrics,
        csv_path=tmp_path / "metrics.csv",
    )
    logger.count("episodes", n=10)
    logger.count("steps", n=1000)
    
    state = logger.save_state()
    assert state["counters"]["episodes"] == 10
    assert state["counters"]["steps"] == 1000


def test_logger_save_state_contains_gauges(tmp_path):
    """save_state() includes gauge values."""
    logger = MetricLogger(
        source_id="worker.0",
        schema=DDQNMetrics,
        csv_path=tmp_path / "metrics.csv",
    )
    logger.gauge("epsilon", 0.1)
    logger.gauge("buffer_size", 5000)
    
    state = logger.save_state()
    assert state["gauges"]["epsilon"] == 0.1
    assert state["gauges"]["buffer_size"] == 5000


def test_logger_save_state_contains_rolling(tmp_path):
    """save_state() includes rolling buffer contents."""
    logger = MetricLogger(
        source_id="worker.0",
        schema=DDQNMetrics,
        csv_path=tmp_path / "metrics.csv",
    )
    logger.observe("reward", 100.0)
    logger.observe("reward", 200.0)
    logger.observe("reward", 150.0)
    
    state = logger.save_state()
    assert state["rolling"]["reward"] == [100.0, 200.0, 150.0]


def test_logger_load_state_restores_counters(tmp_path):
    """load_state() restores counter values."""
    logger = MetricLogger(
        source_id="worker.0",
        schema=DDQNMetrics,
        csv_path=tmp_path / "metrics.csv",
    )
    
    state = {
        "counters": {"episodes": 50, "steps": 5000},
        "gauges": {},
        "rolling": {},
    }
    logger.load_state(state)
    
    snap = logger.snapshot()
    assert snap["episodes"] == 50
    assert snap["steps"] == 5000


def test_logger_load_state_restores_gauges(tmp_path):
    """load_state() restores gauge values."""
    logger = MetricLogger(
        source_id="worker.0",
        schema=DDQNMetrics,
        csv_path=tmp_path / "metrics.csv",
    )
    
    state = {
        "counters": {},
        "gauges": {"epsilon": 0.05},
        "rolling": {},
    }
    logger.load_state(state)
    
    snap = logger.snapshot()
    assert snap["epsilon"] == 0.05


def test_logger_load_state_restores_rolling(tmp_path):
    """load_state() restores rolling buffer contents."""
    logger = MetricLogger(
        source_id="worker.0",
        schema=DDQNMetrics,
        csv_path=tmp_path / "metrics.csv",
    )
    
    state = {
        "counters": {},
        "gauges": {},
        "rolling": {"reward": [100.0, 200.0, 150.0]},
    }
    logger.load_state(state)
    
    snap = logger.snapshot()
    assert snap["reward"] == 150.0  # avg of restored values


def test_logger_roundtrip(tmp_path):
    """save_state() followed by load_state() preserves data."""
    # Create and populate logger
    logger1 = MetricLogger(
        source_id="worker.0",
        schema=DDQNMetrics,
        csv_path=tmp_path / "metrics1.csv",
    )
    logger1.count("episodes", n=42)
    logger1.gauge("epsilon", 0.123)
    logger1.observe("reward", 100.0)
    logger1.observe("reward", 200.0)
    
    # Save state
    state = logger1.save_state()
    
    # Load into new logger
    logger2 = MetricLogger(
        source_id="worker.0",
        schema=DDQNMetrics,
        csv_path=tmp_path / "metrics2.csv",
    )
    logger2.load_state(state)
    
    # Verify
    snap = logger2.snapshot()
    assert snap["episodes"] == 42
    assert snap["epsilon"] == 0.123
    assert snap["reward"] == 150.0  # avg of 100, 200


# =============================================================================
# LevelStats save_state/load_state Tests
# =============================================================================


def test_level_stats_save_state():
    """LevelStats.save_state() returns serializable dict."""
    stats = LevelStats(level_id="1-1")
    stats.start_attempt()
    stats.start_attempt()
    stats.record_death(x_pos=500)
    stats.record_completion(game_time=300, reward=100.0)
    
    state = stats.save_state()
    assert state["level_id"] == "1-1"
    assert state["attempts"] == 2
    assert state["deaths"] == 1
    assert state["completions"] == 1
    assert state["best_x"] == 500
    assert state["death_positions"] == [500]
    assert state["completion_times"] == [300]
    assert state["rewards"] == [100.0]


def test_level_stats_load_state():
    """LevelStats.load_state() restores all values."""
    state = {
        "level_id": "1-1",
        "attempts": 10,
        "completions": 3,
        "deaths": 7,
        "best_x": 800,
        "best_completion_time": 280,
        "death_positions": [400, 500, 600],
        "completion_times": [300, 290, 280],
        "rewards": [100.0, 120.0, 150.0],
        "speeds": [1.5, 1.8],
    }
    
    stats = LevelStats(level_id="1-1")
    stats.load_state(state)
    
    assert stats.attempts == 10
    assert stats.completions == 3
    assert stats.deaths == 7
    assert stats.best_x == 800
    assert stats.best_completion_time == 280
    assert list(stats.death_positions) == [400, 500, 600]
    assert list(stats.completion_times) == [300, 290, 280]


def test_level_stats_roundtrip():
    """save_state() followed by load_state() preserves data."""
    stats1 = LevelStats(level_id="4-2")
    stats1.start_attempt()
    stats1.record_death(x_pos=300)
    stats1.start_attempt()
    stats1.record_completion(game_time=400, reward=200.0)
    
    state = stats1.save_state()
    
    stats2 = LevelStats(level_id="4-2")
    stats2.load_state(state)
    
    assert stats2.attempts == 2
    assert stats2.deaths == 1
    assert stats2.completions == 1
    assert stats2.completion_rate == 0.5


# =============================================================================
# LevelTracker save_state/load_state Tests
# =============================================================================


def test_tracker_save_state():
    """LevelTracker.save_state() returns dict of level states."""
    tracker = LevelTracker()
    tracker.get("1-1").start_attempt()
    tracker.get("1-2").start_attempt()
    tracker.get("1-2").record_death(x_pos=200)
    
    state = tracker.save_state()
    assert "1-1" in state
    assert "1-2" in state
    assert state["1-1"]["attempts"] == 1
    assert state["1-2"]["deaths"] == 1


def test_tracker_load_state():
    """LevelTracker.load_state() restores all levels."""
    state = {
        "1-1": {
            "level_id": "1-1",
            "attempts": 10,
            "completions": 5,
            "deaths": 5,
            "best_x": 1200,
            "best_completion_time": 280,
            "death_positions": [],
            "completion_times": [],
            "rewards": [],
            "speeds": [],
        },
        "4-1": {
            "level_id": "4-1",
            "attempts": 3,
            "completions": 0,
            "deaths": 3,
            "best_x": 400,
            "best_completion_time": 999999,
            "death_positions": [100, 200, 400],
            "completion_times": [],
            "rewards": [],
            "speeds": [],
        },
    }
    
    tracker = LevelTracker()
    tracker.load_state(state)
    
    assert tracker.get("1-1").attempts == 10
    assert tracker.get("1-1").completion_rate == 0.5
    assert tracker.get("4-1").deaths == 3
    assert list(tracker.get("4-1").death_positions) == [100, 200, 400]


def test_tracker_roundtrip():
    """save_state() followed by load_state() preserves data."""
    tracker1 = LevelTracker()
    tracker1.get("1-1").start_attempt()
    tracker1.get("1-1").record_completion(game_time=300, reward=100.0)
    tracker1.get("1-2").start_attempt()
    tracker1.get("1-2").record_death(x_pos=500)
    
    state = tracker1.save_state()
    
    tracker2 = LevelTracker()
    tracker2.load_state(state)
    
    assert tracker2.get("1-1").completions == 1
    assert tracker2.get("1-2").deaths == 1
    assert len(tracker2.all_levels) == 2
