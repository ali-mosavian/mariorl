"""Tests for MarioCollector - Mario-specific metrics."""

from unittest.mock import MagicMock

import pytest

from mario_rl.metrics.collectors.mario import MarioCollector


@pytest.fixture
def mock_logger() -> MagicMock:
    """Create a mock MetricLogger."""
    logger = MagicMock()
    return logger


@pytest.fixture
def mario_collector(mock_logger: MagicMock) -> MarioCollector:
    """Create MarioCollector with mock logger."""
    return MarioCollector(logger=mock_logger)


def test_tracks_x_pos(mario_collector: MarioCollector, mock_logger: MagicMock) -> None:
    """Should track x_pos as gauge."""
    mario_collector.on_step({"x_pos": 150})
    mock_logger.gauge.assert_called_with("x_pos", 150)


def test_tracks_game_time_from_nested_state(mario_collector: MarioCollector, mock_logger: MagicMock) -> None:
    """Should extract game_time from nested state dict."""
    mario_collector.on_step({"x_pos": 100, "state": {"time": 380}})

    # Should have called gauge for both x_pos and game_time
    calls = mock_logger.gauge.call_args_list
    assert any(c[0] == ("game_time", 380) for c in calls)


def test_updates_best_x_tracking(mario_collector: MarioCollector, mock_logger: MagicMock) -> None:
    """Should track best_x seen during episode."""
    mario_collector.on_step({"x_pos": 100})
    mario_collector.on_step({"x_pos": 200})
    mario_collector.on_step({"x_pos": 150})  # Goes back

    # Internal best_x should be 200
    assert mario_collector._episode_x_pos == 200


def test_counts_death_when_is_dead(mario_collector: MarioCollector, mock_logger: MagicMock) -> None:
    """Should increment deaths counter when Mario dies."""
    mario_collector.on_episode_end({"is_dead": True})
    mock_logger.count.assert_called_with("deaths")


def test_counts_death_when_is_dying(mario_collector: MarioCollector, mock_logger: MagicMock) -> None:
    """Should increment deaths counter when is_dying."""
    mario_collector.on_episode_end({"is_dying": True})
    mock_logger.count.assert_called_with("deaths")


def test_counts_flag_when_captured(mario_collector: MarioCollector, mock_logger: MagicMock) -> None:
    """Should increment flags counter when flag captured."""
    mario_collector.on_episode_end({"flag_get": True})
    mock_logger.count.assert_called_with("flags")


def test_flag_takes_precedence_over_death(mario_collector: MarioCollector, mock_logger: MagicMock) -> None:
    """Flag capture should be counted even if is_dead is also True."""
    # This can happen in the game state
    mario_collector.on_episode_end({"flag_get": True, "is_dead": True})

    # Should count flag, not death
    assert mock_logger.count.call_args[0] == ("flags",)


def test_calculates_speed_from_position_and_time(mario_collector: MarioCollector, mock_logger: MagicMock) -> None:
    """Should calculate speed = x_pos / time_spent."""
    # Simulate some steps to build up x_pos
    mario_collector.on_step({"x_pos": 200})

    # End episode with time remaining (timer counts down from 400)
    mario_collector.on_episode_end(
        {
            "state": {"time": 380}  # 20 time units spent
        }
    )

    # speed = 200 / 20 = 10.0
    mock_logger.observe.assert_called_with("speed", 10.0)


def test_resets_episode_tracking_after_end(mario_collector: MarioCollector, mock_logger: MagicMock) -> None:
    """Should reset episode state for next episode."""
    mario_collector.on_step({"x_pos": 300})
    mario_collector.on_episode_end({"state": {"time": 350}})

    # Internal tracking should be reset
    assert mario_collector._episode_x_pos == 0
    assert mario_collector._episode_start_time == 400


def test_no_speed_when_no_time_elapsed(mario_collector: MarioCollector, mock_logger: MagicMock) -> None:
    """Should not calculate speed if time_spent is 0."""
    mario_collector.on_step({"x_pos": 100})
    mario_collector.on_episode_end(
        {
            "state": {"time": 400}  # No time spent
        }
    )

    # observe should not be called for speed
    speed_calls = [c for c in mock_logger.observe.call_args_list if c[0][0] == "speed"]
    assert len(speed_calls) == 0


def test_ignores_train_metrics(mario_collector: MarioCollector, mock_logger: MagicMock) -> None:
    """MarioCollector should not process training metrics."""
    mario_collector.on_train_step({"loss": 0.5, "q_mean": 10.0})

    # No calls related to loss/q_mean
    for call in mock_logger.observe.call_args_list:
        assert call[0][0] not in ("loss", "q_mean", "td_error")


def test_flushes_logger(mario_collector: MarioCollector, mock_logger: MagicMock) -> None:
    """Should call flush on the logger."""
    mario_collector.flush()
    mock_logger.flush.assert_called_once()


def test_satisfies_protocol(mario_collector: MarioCollector) -> None:
    """MarioCollector should satisfy MetricCollector protocol."""
    from mario_rl.metrics.collectors import MetricCollector

    assert isinstance(mario_collector, MetricCollector)
