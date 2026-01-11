"""Tests for DDQNCollector - DDQN-specific training metrics."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from mario_rl.metrics.collectors.ddqn import DDQNCollector


@pytest.fixture
def mock_logger() -> MagicMock:
    """Create a mock MetricLogger."""
    return MagicMock()


@pytest.fixture
def ddqn_collector(mock_logger: MagicMock) -> DDQNCollector:
    """Create DDQNCollector with mock logger."""
    return DDQNCollector(logger=mock_logger)


def test_ignores_step_info(ddqn_collector: DDQNCollector, mock_logger: MagicMock) -> None:
    """DDQNCollector should not process step info."""
    ddqn_collector.on_step({"x_pos": 100, "state": {"time": 380}})
    
    # No gauge calls for x_pos or game_time
    x_pos_calls = [c for c in mock_logger.gauge.call_args_list 
                  if c[0][0] == "x_pos"]
    assert len(x_pos_calls) == 0


def test_ignores_episode_end(ddqn_collector: DDQNCollector, mock_logger: MagicMock) -> None:
    """DDQNCollector should not process episode end."""
    ddqn_collector.on_episode_end({"is_dead": True, "flag_get": False})
    
    # No count calls for deaths/flags
    deaths_calls = [c for c in mock_logger.count.call_args_list 
                   if "deaths" in str(c)]
    assert len(deaths_calls) == 0


def test_tracks_loss(ddqn_collector: DDQNCollector, mock_logger: MagicMock) -> None:
    """Should track loss as rolling metric."""
    ddqn_collector.on_train_step({"loss": 0.5})
    mock_logger.observe.assert_called_with("loss", 0.5)


def test_tracks_q_mean(ddqn_collector: DDQNCollector, mock_logger: MagicMock) -> None:
    """Should track q_mean as rolling metric."""
    ddqn_collector.on_train_step({"q_mean": 15.5})
    
    calls = mock_logger.observe.call_args_list
    assert any(c[0] == ("q_mean", 15.5) for c in calls)


def test_tracks_td_error(ddqn_collector: DDQNCollector, mock_logger: MagicMock) -> None:
    """Should track td_error as rolling metric."""
    ddqn_collector.on_train_step({"td_error": 0.25})
    
    calls = mock_logger.observe.call_args_list
    assert any(c[0] == ("td_error", 0.25) for c in calls)


def test_tracks_q_max_as_gauge(ddqn_collector: DDQNCollector, mock_logger: MagicMock) -> None:
    """Should track q_max as gauge (not rolling)."""
    ddqn_collector.on_train_step({"q_max": 25.0})
    mock_logger.gauge.assert_called_with("q_max", 25.0)


def test_tracks_grad_norm(ddqn_collector: DDQNCollector, mock_logger: MagicMock) -> None:
    """Should track gradient norm as gauge."""
    ddqn_collector.on_train_step({"grad_norm": 1.5})
    
    calls = mock_logger.gauge.call_args_list
    assert any(c[0] == ("grad_norm", 1.5) for c in calls)


def test_handles_missing_metrics(ddqn_collector: DDQNCollector, mock_logger: MagicMock) -> None:
    """Should handle missing metrics gracefully."""
    ddqn_collector.on_train_step({})  # Empty metrics
    # No exceptions, no calls
    assert mock_logger.observe.call_count == 0
    assert mock_logger.gauge.call_count == 0


def test_converts_tensor_to_float(ddqn_collector: DDQNCollector, mock_logger: MagicMock) -> None:
    """Should handle tensor values by converting to float."""
    import torch
    ddqn_collector.on_train_step({"loss": torch.tensor(0.5)})
    
    # Should have called with float, not tensor
    call_args = mock_logger.observe.call_args[0]
    assert call_args[0] == "loss"
    assert isinstance(call_args[1], float)


def test_does_not_flush_logger(ddqn_collector: DDQNCollector, mock_logger: MagicMock) -> None:
    """DDQNCollector should not flush - let composite handle it."""
    ddqn_collector.flush()
    # Flush is a no-op for DDQNCollector
    mock_logger.flush.assert_not_called()


def test_satisfies_protocol(ddqn_collector: DDQNCollector) -> None:
    """DDQNCollector should satisfy MetricCollector protocol."""
    from mario_rl.metrics.collectors import MetricCollector
    assert isinstance(ddqn_collector, MetricCollector)
