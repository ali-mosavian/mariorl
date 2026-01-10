"""Tests for dashboard data aggregators using DuckDB.

These tests verify the aggregation logic works correctly
for various data patterns and edge cases.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mario_rl.dashboard.aggregators import (
    LevelStats,
    RatePoint,
    ActionDistPoint,
    aggregate_level_stats,
    aggregate_action_distribution,
    aggregate_rate_data,
    aggregate_death_hotspots_from_csv,
    level_sort_key,
    sample_data,
)


# =============================================================================
# RatePoint Tests
# =============================================================================


class TestRatePoint:
    """Tests for RatePoint dataclass and computed properties."""

    def test_deaths_per_episode_calculation(self) -> None:
        """deaths_per_episode should divide deaths by episodes."""
        point = RatePoint(steps=1000, deaths=10, flags=2, episodes=5, timeouts=1)
        assert point.deaths_per_episode == 2.0

    def test_deaths_per_episode_zero_episodes(self) -> None:
        """deaths_per_episode should return 0 when no episodes."""
        point = RatePoint(steps=1000, deaths=10, flags=0, episodes=0)
        assert point.deaths_per_episode == 0.0

    def test_timeouts_per_episode_calculation(self) -> None:
        """timeouts_per_episode should divide timeouts by episodes."""
        point = RatePoint(steps=1000, deaths=5, flags=1, episodes=10, timeouts=3)
        assert point.timeouts_per_episode == 0.3

    def test_timeouts_per_episode_zero_episodes(self) -> None:
        """timeouts_per_episode should return 0 when no episodes."""
        point = RatePoint(steps=1000, deaths=0, flags=0, episodes=0, timeouts=5)
        assert point.timeouts_per_episode == 0.0

    def test_completion_rate_calculation(self) -> None:
        """completion_rate should be flags/episodes * 100."""
        point = RatePoint(steps=1000, deaths=5, flags=3, episodes=10)
        assert point.completion_rate == 30.0

    def test_completion_rate_zero_episodes(self) -> None:
        """completion_rate should return 0 when no episodes."""
        point = RatePoint(steps=1000, deaths=0, flags=5, episodes=0)
        assert point.completion_rate == 0.0

    def test_completion_rate_100_percent(self) -> None:
        """completion_rate should reach 100% when all episodes complete."""
        point = RatePoint(steps=1000, deaths=0, flags=10, episodes=10)
        assert point.completion_rate == 100.0


# =============================================================================
# Level Sort Key Tests
# =============================================================================


class TestLevelSortKey:
    """Tests for level sorting."""

    def test_sort_key_basic(self) -> None:
        """level_sort_key should parse level strings correctly."""
        assert level_sort_key("1-1") == (1, 1)
        assert level_sort_key("2-3") == (2, 3)
        assert level_sort_key("8-4") == (8, 4)

    def test_sort_key_invalid(self) -> None:
        """level_sort_key should return (999, 999) for invalid levels."""
        assert level_sort_key("invalid") == (999, 999)
        assert level_sort_key("") == (999, 999)
        assert level_sort_key("abc") == (999, 999)

    def test_sort_order(self) -> None:
        """Levels should sort correctly using level_sort_key."""
        levels = ["2-1", "1-2", "1-1", "3-4", "2-3"]
        sorted_levels = sorted(levels, key=level_sort_key)
        assert sorted_levels == ["1-1", "1-2", "2-1", "2-3", "3-4"]


# =============================================================================
# Sample Data Tests
# =============================================================================


class TestSampleData:
    """Tests for data sampling utility."""

    def test_sample_returns_all_when_under_limit(self) -> None:
        """sample_data should return all data when under max_points."""
        data = [1, 2, 3, 4, 5]
        result = sample_data(data, max_points=10)
        assert result == data

    def test_sample_reduces_when_over_limit(self) -> None:
        """sample_data should reduce data when over max_points."""
        data = list(range(100))
        result = sample_data(data, max_points=10)
        assert len(result) == 10

    def test_sample_empty_list(self) -> None:
        """sample_data should handle empty list."""
        result = sample_data([], max_points=10)
        assert result == []


# =============================================================================
# Aggregate Level Stats Tests
# =============================================================================


class TestAggregateLevelStats:
    """Tests for aggregate_level_stats using DuckDB."""

    def test_empty_workers(self) -> None:
        """Should return empty dict for no workers."""
        result = aggregate_level_stats({})
        assert result == {}

    def test_single_worker_single_level(self) -> None:
        """Should aggregate stats for single worker, single level."""
        workers = {
            0: pd.DataFrame({
                "world": [1, 1, 1],
                "stage": [1, 1, 1],
                "deaths": [1, 2, 3],
                "flags": [0, 1, 0],
                "best_x": [100, 200, 300],
                "reward": [10.0, 20.0, 30.0],
                "speed": [1.0, 1.5, 2.0],
            })
        }
        
        result = aggregate_level_stats(workers)
        
        assert "1-1" in result
        stats = result["1-1"]
        assert stats.episodes == 3
        assert stats.deaths == 6  # 1 + 2 + 3
        assert stats.flags == 1
        assert stats.best_x == 300
        assert stats.avg_reward == 20.0
        assert stats.min_reward == 10.0
        assert stats.max_reward == 30.0

    def test_multiple_workers_same_level(self) -> None:
        """Should aggregate across multiple workers for same level."""
        workers = {
            0: pd.DataFrame({
                "world": [1], "stage": [1],
                "deaths": [5], "flags": [1],
                "best_x": [100], "reward": [10.0], "speed": [1.0],
            }),
            1: pd.DataFrame({
                "world": [1], "stage": [1],
                "deaths": [3], "flags": [2],
                "best_x": [200], "reward": [20.0], "speed": [2.0],
            }),
        }
        
        result = aggregate_level_stats(workers)
        
        assert "1-1" in result
        stats = result["1-1"]
        assert stats.deaths == 8  # 5 + 3
        assert stats.flags == 3  # 1 + 2
        assert stats.best_x == 200

    def test_multiple_levels(self) -> None:
        """Should separate stats by level."""
        workers = {
            0: pd.DataFrame({
                "world": [1, 1, 2],
                "stage": [1, 2, 1],
                "deaths": [1, 2, 3],
                "flags": [1, 0, 1],
                "best_x": [100, 150, 200],
                "reward": [10.0, 15.0, 20.0],
                "speed": [1.0, 1.2, 1.5],
            })
        }
        
        result = aggregate_level_stats(workers)
        
        assert len(result) == 3
        assert "1-1" in result
        assert "1-2" in result
        assert "2-1" in result

    def test_missing_columns_fallback(self) -> None:
        """Should handle CSVs without world/stage columns."""
        workers = {
            0: pd.DataFrame({
                "deaths": [5],
                "flags": [1],
                "reward": [100.0],
            })
        }
        
        result = aggregate_level_stats(workers)
        
        # Should fallback to "1-1" as default level
        assert "1-1" in result


# =============================================================================
# Aggregate Rate Data Tests
# =============================================================================


class TestAggregateRateData:
    """Tests for aggregate_rate_data using DuckDB."""

    def test_empty_workers(self) -> None:
        """Should return empty dict for no workers."""
        result = aggregate_rate_data({})
        assert result == {}

    def test_single_worker_bucketing(self) -> None:
        """Should bucket data by step intervals."""
        workers = {
            0: pd.DataFrame({
                "world": [1, 1, 1],
                "stage": [1, 1, 1],
                "steps": [5000, 15000, 25000],
                "deaths": [1, 2, 3],
                "flags": [0, 1, 1],
                "episodes": [5, 10, 15],
                "timeouts": [0, 1, 0],
            })
        }
        
        result = aggregate_rate_data(workers, step_bucket_size=10000)
        
        assert "1-1" in result
        points = result["1-1"]
        assert len(points) == 3
        
        # Check buckets are correct
        buckets = [p.steps for p in points]
        assert 0 in buckets
        assert 10000 in buckets
        assert 20000 in buckets

    def test_multiple_workers_aggregation(self) -> None:
        """Should sum across workers at each bucket."""
        workers = {
            0: pd.DataFrame({
                "world": [1], "stage": [1],
                "steps": [5000],
                "deaths": [5], "flags": [1], "episodes": [10], "timeouts": [1],
            }),
            1: pd.DataFrame({
                "world": [1], "stage": [1],
                "steps": [7000],
                "deaths": [3], "flags": [2], "episodes": [8], "timeouts": [0],
            }),
        }
        
        result = aggregate_rate_data(workers, step_bucket_size=10000)
        
        assert "1-1" in result
        point = result["1-1"][0]
        
        # Should sum across workers
        assert point.deaths == 8  # 5 + 3
        assert point.flags == 3  # 1 + 2
        assert point.episodes == 18  # 10 + 8
        assert point.timeouts == 1  # 1 + 0

    def test_rate_calculations(self) -> None:
        """Should calculate rates correctly from aggregated data."""
        workers = {
            0: pd.DataFrame({
                "world": [1], "stage": [1],
                "steps": [10000],
                "deaths": [20], "flags": [5], "episodes": [100], "timeouts": [10],
            })
        }
        
        result = aggregate_rate_data(workers)
        point = result["1-1"][0]
        
        assert point.deaths_per_episode == 0.2  # 20/100
        assert point.timeouts_per_episode == 0.1  # 10/100
        assert point.completion_rate == 5.0  # 5/100 * 100


# =============================================================================
# Aggregate Action Distribution Tests
# =============================================================================


class TestAggregateActionDistribution:
    """Tests for aggregate_action_distribution."""

    def test_empty_workers(self) -> None:
        """Should return empty dict for no workers."""
        result = aggregate_action_distribution({})
        assert result == {}

    def test_no_action_dist_column(self) -> None:
        """Should return empty dict if action_dist column missing."""
        workers = {
            0: pd.DataFrame({
                "world": [1], "stage": [1], "steps": [1000],
            })
        }
        
        result = aggregate_action_distribution(workers)
        assert result == {}

    def test_parses_action_distribution(self) -> None:
        """Should parse comma-separated action distribution."""
        # 12 action percentages
        dist_str = ",".join(str(i * 0.08) for i in range(1, 13))  # 0.08,0.16,...,0.96
        
        workers = {
            0: pd.DataFrame({
                "world": [1], "stage": [1],
                "steps": [1000],
                "action_dist": [dist_str],
            })
        }
        
        result = aggregate_action_distribution(workers)
        
        assert "1-1" in result
        assert len(result["1-1"]) == 1
        assert len(result["1-1"][0].percentages) == 12


# =============================================================================
# Aggregate Death Hotspots Tests
# =============================================================================


class TestAggregateDeathHotspots:
    """Tests for aggregate_death_hotspots_from_csv."""

    def test_empty_workers(self) -> None:
        """Should return empty dict for no workers."""
        result = aggregate_death_hotspots_from_csv({})
        assert result == {}

    def test_no_death_positions_column(self) -> None:
        """Should return empty dict if death_positions column missing."""
        workers = {
            0: pd.DataFrame({
                "world": [1], "stage": [1],
            })
        }
        
        result = aggregate_death_hotspots_from_csv(workers)
        assert result == {}

    def test_parses_death_positions(self) -> None:
        """Should parse death positions from 'level:pos1,pos2' format."""
        workers = {
            0: pd.DataFrame({
                "death_positions": ["1-1:100,150,200"],
            })
        }
        
        result = aggregate_death_hotspots_from_csv(workers)
        
        assert "1-1" in result
        # Positions bucketed by 25: 100->100, 150->150, 200->200
        assert 100 in result["1-1"]
        assert 150 in result["1-1"]
        assert 200 in result["1-1"]

    def test_buckets_by_25(self) -> None:
        """Should bucket death positions by 25 pixels."""
        workers = {
            0: pd.DataFrame({
                "death_positions": ["1-1:101,102,103,110,140"],
            })
        }
        
        result = aggregate_death_hotspots_from_csv(workers)
        
        # 101,102,103,110 -> bucket 100, 140 -> bucket 125
        assert result["1-1"][100] == 4
        assert result["1-1"][125] == 1

    def test_aggregates_across_workers(self) -> None:
        """Should sum death counts across workers."""
        workers = {
            0: pd.DataFrame({"death_positions": ["1-1:100,100"]}),
            1: pd.DataFrame({"death_positions": ["1-1:100"]}),
        }
        
        result = aggregate_death_hotspots_from_csv(workers)
        
        assert result["1-1"][100] == 3


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataframe(self) -> None:
        """Should handle empty dataframes gracefully."""
        workers = {0: pd.DataFrame()}
        
        assert aggregate_level_stats(workers) == {}
        assert aggregate_rate_data(workers) == {}
        assert aggregate_action_distribution(workers) == {}

    def test_nan_values_in_data(self) -> None:
        """Should handle NaN values without crashing."""
        workers = {
            0: pd.DataFrame({
                "world": [1], "stage": [1],
                "deaths": [None], "flags": [1],
                "reward": [float("nan")], "speed": [None],
            })
        }
        
        # Should not raise
        result = aggregate_level_stats(workers)
        assert "1-1" in result

    def test_mixed_level_data(self) -> None:
        """Should handle workers with different levels."""
        workers = {
            0: pd.DataFrame({
                "world": [1, 2], "stage": [1, 1],
                "deaths": [5, 3], "flags": [1, 2],
                "reward": [10.0, 20.0], "speed": [1.0, 1.5],
            }),
            1: pd.DataFrame({
                "world": [1, 3], "stage": [2, 1],
                "deaths": [2, 4], "flags": [1, 0],
                "reward": [15.0, 25.0], "speed": [1.2, 1.8],
            }),
        }
        
        result = aggregate_level_stats(workers)
        
        # Should have all 4 unique levels
        assert len(result) == 4
        assert "1-1" in result
        assert "1-2" in result
        assert "2-1" in result
        assert "3-1" in result
