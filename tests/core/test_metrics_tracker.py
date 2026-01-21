"""Tests for MetricsTracker.

These tests verify the metrics tracking functionality,
especially the timeout counter.
"""

from __future__ import annotations

from mario_rl.core.metrics import MetricsTracker


def test_initial_values() -> None:
    """MetricsTracker should initialize all counters to zero."""
    tracker = MetricsTracker()

    assert tracker.episode_count == 0
    assert tracker.total_steps == 0
    assert tracker.deaths == 0
    assert tracker.timeouts == 0
    assert tracker.flags == 0
    assert tracker.best_x_ever == 0
    assert tracker.gradients_sent == 0


def test_initial_histories_empty() -> None:
    """MetricsTracker should initialize all histories as empty lists."""
    tracker = MetricsTracker()

    assert tracker.reward_history == []
    assert tracker.speed_history == []
    assert tracker.entropy_history == []
    assert tracker.x_at_death_history == []
    assert tracker.time_to_flag_history == []


def test_add_timeout_increments_counter() -> None:
    """add_timeout() should increment the timeouts counter."""
    tracker = MetricsTracker()
    assert tracker.timeouts == 0

    tracker.add_timeout()
    assert tracker.timeouts == 1

    tracker.add_timeout()
    assert tracker.timeouts == 2


def test_add_timeout_does_not_affect_deaths() -> None:
    """add_timeout() should NOT increment deaths counter.

    Timeouts are distinct from skill-based deaths.
    """
    tracker = MetricsTracker()

    tracker.add_timeout()
    tracker.add_timeout()

    assert tracker.timeouts == 2
    assert tracker.deaths == 0  # Deaths should remain 0


def test_timeout_and_death_independent() -> None:
    """Timeouts and deaths should track independently."""
    tracker = MetricsTracker()

    tracker.add_death(x_pos=100)
    tracker.add_timeout()
    tracker.add_death(x_pos=200)
    tracker.add_timeout()
    tracker.add_timeout()

    assert tracker.deaths == 2
    assert tracker.timeouts == 3


def test_add_death_increments_counter() -> None:
    """add_death() should increment the deaths counter."""
    tracker = MetricsTracker()

    tracker.add_death(x_pos=100)
    assert tracker.deaths == 1

    tracker.add_death(x_pos=200)
    assert tracker.deaths == 2


def test_add_death_records_position() -> None:
    """add_death() should record the death position in history."""
    tracker = MetricsTracker()

    tracker.add_death(x_pos=100)
    tracker.add_death(x_pos=250)
    tracker.add_death(x_pos=500)

    assert tracker.x_at_death_history == [100, 250, 500]


def test_death_history_limit() -> None:
    """Death history should be limited to 20 entries."""
    tracker = MetricsTracker()

    for i in range(25):
        tracker.add_death(x_pos=i * 10)

    assert len(tracker.x_at_death_history) == 20
    # Should keep the most recent 20
    assert tracker.x_at_death_history[0] == 50  # 5th death position


def test_add_flag_increments_counter() -> None:
    """add_flag() should increment the flags counter."""
    tracker = MetricsTracker()

    tracker.add_flag(game_time=200)
    assert tracker.flags == 1

    tracker.add_flag(game_time=150)
    assert tracker.flags == 2


def test_add_flag_records_time() -> None:
    """add_flag() should record the game time in history."""
    tracker = MetricsTracker()

    tracker.add_flag(game_time=200)
    tracker.add_flag(game_time=150)

    assert tracker.time_to_flag_history == [200, 150]


def test_episode_end_types_sum_correctly() -> None:
    """Deaths + timeouts + flags should equal total episode ends."""
    tracker = MetricsTracker()

    # Simulate various episode endings
    tracker.add_death(x_pos=100)  # Episode 1: death
    tracker.add_timeout()  # Episode 2: timeout
    tracker.add_flag(game_time=200)  # Episode 3: flag
    tracker.add_death(x_pos=150)  # Episode 4: death
    tracker.add_timeout()  # Episode 5: timeout
    tracker.add_timeout()  # Episode 6: timeout

    total_episodes = tracker.deaths + tracker.timeouts + tracker.flags
    assert total_episodes == 6


def test_timeout_not_recorded_as_death_position() -> None:
    """Timeout should not add to x_at_death_history.

    Since timeout is not a skill-based death, we don't want
    to track its position in death hotspots.
    """
    tracker = MetricsTracker()

    tracker.add_death(x_pos=100)
    tracker.add_timeout()  # Should NOT add to x_at_death_history
    tracker.add_death(x_pos=200)

    assert len(tracker.x_at_death_history) == 2
    assert tracker.x_at_death_history == [100, 200]


def test_avg_x_at_death() -> None:
    """avg_x_at_death should calculate correctly."""
    tracker = MetricsTracker()

    tracker.add_death(x_pos=100)
    tracker.add_death(x_pos=200)
    tracker.add_death(x_pos=300)

    assert tracker.avg_x_at_death == 200.0


def test_avg_x_at_death_empty() -> None:
    """avg_x_at_death should return 0 when no deaths."""
    tracker = MetricsTracker()

    assert tracker.avg_x_at_death == 0.0


def test_avg_time_to_flag() -> None:
    """avg_time_to_flag should calculate correctly."""
    tracker = MetricsTracker()

    tracker.add_flag(game_time=200)
    tracker.add_flag(game_time=100)

    assert tracker.avg_time_to_flag == 150.0
