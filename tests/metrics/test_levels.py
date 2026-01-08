"""Tests for LevelStats and LevelTracker."""

import pytest

from mario_rl.metrics.levels import LevelStats, LevelTracker


# =============================================================================
# LevelStats Tests
# =============================================================================


def test_level_stats_stores_level_id():
    """LevelStats stores level identifier."""
    stats = LevelStats(level_id="1-1")
    assert stats.level_id == "1-1"


def test_level_stats_starts_with_zero_counters():
    """LevelStats starts with zero counters."""
    stats = LevelStats(level_id="1-1")
    assert stats.attempts == 0
    assert stats.completions == 0
    assert stats.deaths == 0


def test_start_attempt_increments_attempts():
    """start_attempt() increments attempts counter."""
    stats = LevelStats(level_id="1-1")
    stats.start_attempt()
    stats.start_attempt()
    assert stats.attempts == 2


def test_record_death_increments_deaths():
    """record_death() increments deaths counter."""
    stats = LevelStats(level_id="1-1")
    stats.record_death(x_pos=500)
    stats.record_death(x_pos=600)
    assert stats.deaths == 2


def test_record_death_stores_position():
    """record_death() stores death position."""
    stats = LevelStats(level_id="1-1")
    stats.record_death(x_pos=500)
    stats.record_death(x_pos=600)
    assert list(stats.death_positions) == [500, 600]


def test_record_death_updates_best_x():
    """record_death() updates best_x if position is further."""
    stats = LevelStats(level_id="1-1")
    stats.record_death(x_pos=500)
    assert stats.best_x == 500
    stats.record_death(x_pos=300)
    assert stats.best_x == 500  # Should not decrease
    stats.record_death(x_pos=700)
    assert stats.best_x == 700


def test_record_completion_increments_completions():
    """record_completion() increments completions counter."""
    stats = LevelStats(level_id="1-1")
    stats.record_completion(game_time=300, reward=100.0)
    stats.record_completion(game_time=280, reward=120.0)
    assert stats.completions == 2


def test_record_completion_stores_time():
    """record_completion() stores completion time."""
    stats = LevelStats(level_id="1-1")
    stats.record_completion(game_time=300, reward=100.0)
    stats.record_completion(game_time=280, reward=120.0)
    assert list(stats.completion_times) == [300, 280]


def test_record_completion_stores_reward():
    """record_completion() stores reward."""
    stats = LevelStats(level_id="1-1")
    stats.record_completion(game_time=300, reward=100.0)
    stats.record_completion(game_time=280, reward=120.0)
    assert list(stats.rewards) == [100.0, 120.0]


def test_record_completion_updates_best_time():
    """record_completion() tracks best (fastest) completion time."""
    stats = LevelStats(level_id="1-1")
    stats.record_completion(game_time=300, reward=100.0)
    assert stats.best_completion_time == 300
    stats.record_completion(game_time=350, reward=80.0)
    assert stats.best_completion_time == 300  # Should not increase
    stats.record_completion(game_time=250, reward=150.0)
    assert stats.best_completion_time == 250


def test_completion_rate_zero_when_no_attempts():
    """completion_rate is 0 when no attempts."""
    stats = LevelStats(level_id="1-1")
    assert stats.completion_rate == 0.0


def test_completion_rate_correct_ratio():
    """completion_rate is completions / attempts."""
    stats = LevelStats(level_id="1-1")
    stats.start_attempt()
    stats.start_attempt()
    stats.start_attempt()
    stats.start_attempt()
    stats.record_completion(game_time=300, reward=100.0)
    assert stats.completion_rate == 0.25  # 1/4


def test_avg_death_x_zero_when_empty():
    """avg_death_x is 0 when no deaths."""
    stats = LevelStats(level_id="1-1")
    assert stats.avg_death_x == 0.0


def test_avg_death_x_computes_average():
    """avg_death_x computes average of death positions."""
    stats = LevelStats(level_id="1-1")
    stats.record_death(x_pos=400)
    stats.record_death(x_pos=600)
    assert stats.avg_death_x == 500.0


def test_death_histogram_buckets_by_100():
    """death_histogram buckets positions by 100 pixels."""
    stats = LevelStats(level_id="1-1")
    stats.record_death(x_pos=50)    # bucket 0
    stats.record_death(x_pos=120)   # bucket 100
    stats.record_death(x_pos=150)   # bucket 100
    stats.record_death(x_pos=250)   # bucket 200
    
    hist = stats.death_histogram
    assert hist[0] == 1
    assert hist[100] == 2
    assert hist[200] == 1


def test_snapshot_returns_dict():
    """snapshot() returns dict with all stats."""
    stats = LevelStats(level_id="1-1")
    stats.start_attempt()
    stats.record_death(x_pos=500)
    
    snap = stats.snapshot()
    assert snap["level_id"] == "1-1"
    assert snap["attempts"] == 1
    assert snap["deaths"] == 1
    assert snap["completions"] == 0
    assert "completion_rate" in snap
    assert "best_x" in snap


# =============================================================================
# LevelTracker Tests
# =============================================================================


def test_tracker_get_creates_on_demand():
    """LevelTracker.get() creates LevelStats on demand."""
    tracker = LevelTracker()
    stats = tracker.get("1-1")
    assert isinstance(stats, LevelStats)
    assert stats.level_id == "1-1"


def test_tracker_get_returns_same_instance():
    """LevelTracker.get() returns same instance for same level."""
    tracker = LevelTracker()
    stats1 = tracker.get("1-1")
    stats2 = tracker.get("1-1")
    assert stats1 is stats2


def test_tracker_get_different_levels():
    """LevelTracker.get() returns different instances for different levels."""
    tracker = LevelTracker()
    stats1 = tracker.get("1-1")
    stats2 = tracker.get("1-2")
    assert stats1 is not stats2
    assert stats1.level_id == "1-1"
    assert stats2.level_id == "1-2"


def test_tracker_all_levels_empty_initially():
    """LevelTracker.all_levels returns empty dict initially."""
    tracker = LevelTracker()
    assert tracker.all_levels == {}


def test_tracker_all_levels_contains_created():
    """LevelTracker.all_levels contains created levels."""
    tracker = LevelTracker()
    tracker.get("1-1")
    tracker.get("4-2")
    assert "1-1" in tracker.all_levels
    assert "4-2" in tracker.all_levels


def test_tracker_snapshot_all():
    """LevelTracker.snapshot_all() returns snapshots of all levels."""
    tracker = LevelTracker()
    tracker.get("1-1").start_attempt()
    tracker.get("1-2").start_attempt()
    
    snaps = tracker.snapshot_all()
    assert "1-1" in snaps
    assert "1-2" in snaps
    assert snaps["1-1"]["attempts"] == 1


def test_tracker_hardest_levels():
    """LevelTracker.hardest_levels() returns levels with lowest completion rate."""
    tracker = LevelTracker()
    
    # 1-1: 50% completion
    s1 = tracker.get("1-1")
    s1.start_attempt()
    s1.start_attempt()
    s1.record_completion(game_time=300, reward=100.0)
    
    # 1-2: 0% completion
    s2 = tracker.get("1-2")
    s2.start_attempt()
    s2.record_death(x_pos=200)
    
    # 4-1: 100% completion
    s3 = tracker.get("4-1")
    s3.start_attempt()
    s3.record_completion(game_time=400, reward=50.0)
    
    hardest = tracker.hardest_levels(n=2)
    assert len(hardest) == 2
    # Should be sorted by completion rate (lowest first)
    assert hardest[0][0] == "1-2"  # 0%
    assert hardest[1][0] == "1-1"  # 50%


def test_tracker_death_hotspots():
    """LevelTracker.death_hotspots() returns histogram for specific level."""
    tracker = LevelTracker()
    stats = tracker.get("1-1")
    stats.record_death(x_pos=150)
    stats.record_death(x_pos=180)
    
    hotspots = tracker.death_hotspots("1-1")
    assert hotspots[100] == 2
