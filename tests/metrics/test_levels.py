"""Tests for LevelStats, LevelTracker, and DeathHotspotAggregate."""

from pathlib import Path

from mario_rl.metrics.levels import LevelStats
from mario_rl.metrics.levels import LevelTracker
from mario_rl.metrics.levels import DeathHotspotAggregate

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
    stats.record_death(x_pos=50)  # bucket 0
    stats.record_death(x_pos=120)  # bucket 100
    stats.record_death(x_pos=150)  # bucket 100
    stats.record_death(x_pos=250)  # bucket 200

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


# =============================================================================
# DeathHotspotAggregate Tests
# =============================================================================


def test_hotspot_record_death_creates_level():
    """record_death creates level entry if not exists."""
    agg = DeathHotspotAggregate()
    agg.record_death("1-1", 100)
    assert "1-1" in agg._data


def test_hotspot_record_death_buckets_by_25():
    """record_death uses 25-pixel buckets by default."""
    agg = DeathHotspotAggregate(bucket_size=25)
    agg.record_death("1-1", 10)  # bucket 0
    agg.record_death("1-1", 30)  # bucket 25
    agg.record_death("1-1", 40)  # bucket 25
    agg.record_death("1-1", 100)  # bucket 100

    hist = agg.get_histogram("1-1")
    assert hist[0] == 1
    assert hist[25] == 2
    assert hist[100] == 1


def test_hotspot_record_deaths_batch():
    """record_deaths_batch records multiple deaths at once."""
    agg = DeathHotspotAggregate()
    agg.record_deaths_batch("1-1", [10, 30, 40, 100])

    hist = agg.get_histogram("1-1")
    assert hist[0] == 1
    assert hist[25] == 2
    assert hist[100] == 1


def test_hotspot_get_histogram_empty_for_unknown_level():
    """get_histogram returns empty dict for unknown level."""
    agg = DeathHotspotAggregate()
    assert agg.get_histogram("1-1") == {}


def test_hotspot_get_hotspots_filters_by_min_deaths():
    """get_hotspots filters buckets below min_deaths."""
    agg = DeathHotspotAggregate()
    agg.record_deaths_batch("1-1", [10, 30, 30, 30, 30, 100])  # 1 at 0, 4 at 25, 1 at 100

    hotspots = agg.get_hotspots("1-1", min_deaths=3)
    assert len(hotspots) == 1
    assert hotspots[0] == (25, 4)


def test_hotspot_get_hotspots_sorted_by_count():
    """get_hotspots returns hotspots sorted by count descending."""
    agg = DeathHotspotAggregate()
    # 2 at bucket 0, 5 at bucket 50, 3 at bucket 100
    agg.record_deaths_batch("1-1", [10, 10, 60, 60, 60, 60, 60, 110, 110, 110])

    hotspots = agg.get_hotspots("1-1", min_deaths=2)
    assert len(hotspots) == 3
    assert hotspots[0][0] == 50  # 5 deaths
    assert hotspots[1][0] == 100  # 3 deaths
    assert hotspots[2][0] == 0  # 2 deaths


def test_hotspot_get_hotspots_top_n():
    """get_hotspots limits to top_n results."""
    agg = DeathHotspotAggregate()
    agg.record_deaths_batch("1-1", [10, 10, 60, 60, 60, 110, 110, 110, 110])

    hotspots = agg.get_hotspots("1-1", min_deaths=2, top_n=2)
    assert len(hotspots) == 2


def test_hotspot_suggest_snapshot_positions():
    """suggest_snapshot_positions returns positions before death hotspots."""
    agg = DeathHotspotAggregate(bucket_size=25)
    # Deaths at bucket 200 (x=200-224), suggest snapshot at 175 (200-25)
    agg.record_deaths_batch("1-1", [210, 215, 220, 225])

    positions = agg.suggest_snapshot_positions("1-1", count=1, min_deaths=2)
    assert len(positions) == 1
    assert positions[0] == 175  # 200 - 25


def test_hotspot_suggest_snapshot_positions_respects_spacing():
    """suggest_snapshot_positions respects minimum spacing."""
    agg = DeathHotspotAggregate(bucket_size=25)
    # Deaths at buckets 200, 225, 300
    agg.record_deaths_batch("1-1", [210, 210, 210, 230, 230, 230, 310, 310, 310])

    # With min_spacing=100, should not select positions 50 apart
    positions = agg.suggest_snapshot_positions("1-1", count=3, min_spacing=100, min_deaths=2)

    # Should only have 2 positions due to spacing
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            assert abs(positions[i] - positions[j]) >= 100


def test_hotspot_suggest_snapshot_empty_when_no_deaths():
    """suggest_snapshot_positions returns empty list when no deaths."""
    agg = DeathHotspotAggregate()
    positions = agg.suggest_snapshot_positions("1-1")
    assert positions == []


def test_hotspot_suggest_restore_position():
    """suggest_restore_position finds position before nearest hotspot."""
    agg = DeathHotspotAggregate(bucket_size=25)
    # Hotspot at bucket 200
    agg.record_deaths_batch("1-1", [210, 215, 220])

    # Death at 250, should suggest restore before bucket 200
    restore = agg.suggest_restore_position("1-1", death_x=250)
    assert restore is not None
    assert restore == 150  # 200 - 25*2


def test_hotspot_suggest_restore_position_none_when_no_hotspots():
    """suggest_restore_position returns None when no hotspots nearby."""
    agg = DeathHotspotAggregate()
    restore = agg.suggest_restore_position("1-1", death_x=500)
    assert restore is None


def test_hotspot_merge_combines_data():
    """merge combines data from another aggregate."""
    agg1 = DeathHotspotAggregate()
    agg1.record_deaths_batch("1-1", [100, 100, 200])

    agg2 = DeathHotspotAggregate()
    agg2.record_deaths_batch("1-1", [100, 300])
    agg2.record_deaths_batch("1-2", [50])

    agg1.merge(agg2)

    hist = agg1.get_histogram("1-1")
    assert hist[100] == 3  # 2 + 1
    assert hist[200] == 1
    assert hist[300] == 1

    assert "1-2" in agg1._data


def test_hotspot_to_dict_and_from_dict():
    """to_dict and from_dict roundtrip correctly."""
    agg = DeathHotspotAggregate(bucket_size=25)
    agg.record_deaths_batch("1-1", [100, 100, 200])
    agg.record_deaths_batch("1-2", [50])

    data = agg.to_dict()
    restored = DeathHotspotAggregate.from_dict(data)

    assert restored.bucket_size == 25
    assert restored.get_histogram("1-1") == agg.get_histogram("1-1")
    assert restored.get_histogram("1-2") == agg.get_histogram("1-2")


def test_hotspot_save_and_load(tmp_path: Path):
    """save and load roundtrip correctly."""
    save_path = tmp_path / "hotspots.json"

    agg = DeathHotspotAggregate(bucket_size=25, save_path=save_path)
    agg.record_deaths_batch("1-1", [100, 100, 200])
    agg.save()

    assert save_path.exists()

    loaded = DeathHotspotAggregate.load(save_path)
    assert loaded.get_histogram("1-1") == agg.get_histogram("1-1")


def test_hotspot_load_or_create_loads_existing(tmp_path: Path):
    """load_or_create loads existing file."""
    save_path = tmp_path / "hotspots.json"

    # Create and save
    agg = DeathHotspotAggregate(save_path=save_path)
    agg.record_death("1-1", 100)
    agg.save()

    # Load or create
    loaded = DeathHotspotAggregate.load_or_create(save_path)
    assert "1-1" in loaded._data


def test_hotspot_load_or_create_creates_new(tmp_path: Path):
    """load_or_create creates new if file doesn't exist."""
    save_path = tmp_path / "nonexistent.json"

    agg = DeathHotspotAggregate.load_or_create(save_path, bucket_size=50)
    assert agg.bucket_size == 50
    assert agg._data == {}


def test_hotspot_save_if_dirty_only_saves_when_modified(tmp_path: Path):
    """save_if_dirty only saves when data has been modified."""
    save_path = tmp_path / "hotspots.json"

    agg = DeathHotspotAggregate(save_path=save_path)

    # Not dirty initially
    assert not agg.save_if_dirty()
    assert not save_path.exists()

    # Dirty after recording
    agg.record_death("1-1", 100)
    assert agg.save_if_dirty()
    assert save_path.exists()

    # Not dirty after save
    assert not agg.save_if_dirty()


def test_hotspot_summary():
    """summary returns per-level statistics."""
    agg = DeathHotspotAggregate()
    agg.record_deaths_batch("1-1", [100, 100, 100, 200, 200])

    summary = agg.summary()
    assert "1-1" in summary
    assert summary["1-1"]["total_deaths"] == 5
    assert summary["1-1"]["hotspot_count"] == 1  # Only bucket 100 has >= 3
    assert summary["1-1"]["worst_bucket"] == (100, 3)
