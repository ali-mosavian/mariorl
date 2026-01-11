"""Tests for SystemCollector - system/training metrics."""

import time
from unittest.mock import MagicMock

import pytest

from mario_rl.metrics.collectors.system import SystemCollector


@pytest.fixture
def mock_logger() -> MagicMock:
    """Create a mock MetricLogger."""
    return MagicMock()


@pytest.fixture
def system_collector(mock_logger: MagicMock) -> SystemCollector:
    """Create SystemCollector with mock logger."""
    return SystemCollector(logger=mock_logger)


def test_counts_steps(system_collector: SystemCollector, mock_logger: MagicMock) -> None:
    """Should count each step."""
    system_collector.on_step({})
    system_collector.on_step({})
    system_collector.on_step({})
    
    # Should have called count("steps") 3 times
    steps_calls = [c for c in mock_logger.count.call_args_list 
                  if c[0] == ("steps",) or (len(c[0]) >= 1 and c[0][0] == "steps")]
    assert len(steps_calls) == 3


def test_tracks_buffer_size(system_collector: SystemCollector, mock_logger: MagicMock) -> None:
    """Should track buffer_size as gauge."""
    system_collector.on_step({"buffer_size": 500})
    
    calls = mock_logger.gauge.call_args_list
    assert any(c[0] == ("buffer_size", 500) for c in calls)


def test_tracks_epsilon(system_collector: SystemCollector, mock_logger: MagicMock) -> None:
    """Should track epsilon as gauge."""
    system_collector.on_step({"epsilon": 0.15})
    
    calls = mock_logger.gauge.call_args_list
    assert any(c[0] == ("epsilon", 0.15) for c in calls)


def test_counts_episodes(system_collector: SystemCollector, mock_logger: MagicMock) -> None:
    """Should count completed episodes."""
    system_collector.on_episode_end({})
    system_collector.on_episode_end({})
    
    episodes_calls = [c for c in mock_logger.count.call_args_list 
                     if c[0] == ("episodes",) or (len(c[0]) >= 1 and c[0][0] == "episodes")]
    assert len(episodes_calls) == 2


def test_tracks_episode_reward(system_collector: SystemCollector, mock_logger: MagicMock) -> None:
    """Should track episode reward as rolling metric."""
    system_collector.on_episode_end({"episode_reward": 150.0})
    mock_logger.observe.assert_called_with("reward", 150.0)


def test_tracks_episode_length(system_collector: SystemCollector, mock_logger: MagicMock) -> None:
    """Should track episode length as rolling metric."""
    system_collector.on_episode_end({"episode_length": 300})
    
    calls = mock_logger.observe.call_args_list
    assert any(c[0] == ("episode_length", 300) for c in calls)


def test_counts_gradient_sends(system_collector: SystemCollector, mock_logger: MagicMock) -> None:
    """Should count gradient sends."""
    system_collector.on_train_step({"grads_sent": True})
    
    calls = mock_logger.count.call_args_list
    assert any(c[0] == ("grads_sent",) for c in calls)


def test_does_not_count_if_no_grads(system_collector: SystemCollector, mock_logger: MagicMock) -> None:
    """Should not count if grads_sent is False."""
    system_collector.on_train_step({"grads_sent": False})
    
    grads_calls = [c for c in mock_logger.count.call_args_list 
                  if c[0] == ("grads_sent",)]
    assert len(grads_calls) == 0


def test_calculates_steps_per_sec(mock_logger: MagicMock) -> None:
    """Should calculate and track steps_per_sec."""
    collector = SystemCollector(logger=mock_logger)
    
    # Call update_steps_per_sec with timing info
    collector.update_steps_per_sec(num_steps=100, elapsed=0.5)
    
    # Should track 200 steps/sec
    calls = mock_logger.gauge.call_args_list
    sps_calls = [c for c in calls if c[0][0] == "steps_per_sec"]
    assert len(sps_calls) == 1
    assert abs(sps_calls[0][0][1] - 200.0) < 0.01


def test_does_not_flush_logger(system_collector: SystemCollector, mock_logger: MagicMock) -> None:
    """SystemCollector should not flush - let composite handle it."""
    system_collector.flush()
    mock_logger.flush.assert_not_called()


def test_satisfies_protocol(system_collector: SystemCollector) -> None:
    """SystemCollector should satisfy MetricCollector protocol."""
    from mario_rl.metrics.collectors import MetricCollector
    assert isinstance(system_collector, MetricCollector)
