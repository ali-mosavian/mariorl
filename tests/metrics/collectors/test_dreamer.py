"""Tests for DreamerCollector."""

from typing import Any
from unittest.mock import Mock

import pytest
import torch

from mario_rl.metrics.collectors import DreamerCollector


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_logger() -> Mock:
    """Create a mock MetricLogger."""
    return Mock()


@pytest.fixture
def dreamer_collector(mock_logger: Mock) -> DreamerCollector:
    """Create a DreamerCollector with mock logger."""
    return DreamerCollector(logger=mock_logger)


# =============================================================================
# on_step Tests
# =============================================================================


def test_ignores_step_info(dreamer_collector: DreamerCollector, mock_logger: Mock) -> None:
    """DreamerCollector should ignore environment step info."""
    dreamer_collector.on_step({"x_pos": 100, "time": 350})
    
    # Should not call any logger methods
    mock_logger.gauge.assert_not_called()
    mock_logger.count.assert_not_called()
    mock_logger.observe.assert_not_called()


# =============================================================================
# on_episode_end Tests
# =============================================================================


def test_ignores_episode_end(dreamer_collector: DreamerCollector, mock_logger: Mock) -> None:
    """DreamerCollector should ignore episode end."""
    dreamer_collector.on_episode_end({"is_dead": True, "flag_get": False})
    
    # Should not call any logger methods
    mock_logger.gauge.assert_not_called()
    mock_logger.count.assert_not_called()
    mock_logger.observe.assert_not_called()


# =============================================================================
# on_train_step Tests
# =============================================================================


def test_tracks_total_loss(dreamer_collector: DreamerCollector, mock_logger: Mock) -> None:
    """Should track total loss as rolling average."""
    dreamer_collector.on_train_step({"loss": 0.5})
    
    mock_logger.observe.assert_any_call("loss", 0.5)


def test_tracks_world_model_losses(dreamer_collector: DreamerCollector, mock_logger: Mock) -> None:
    """Should track world model component losses."""
    metrics = {
        "dynamics_loss": 0.1,
        "reward_loss": 0.2,
        "wm_loss": 0.3,
    }
    dreamer_collector.on_train_step(metrics)
    
    mock_logger.observe.assert_any_call("dynamics_loss", 0.1)
    mock_logger.observe.assert_any_call("reward_loss", 0.2)
    mock_logger.observe.assert_any_call("wm_loss", 0.3)


def test_tracks_behavior_losses(dreamer_collector: DreamerCollector, mock_logger: Mock) -> None:
    """Should track actor-critic losses."""
    metrics = {
        "actor_loss": 0.4,
        "critic_loss": 0.5,
        "behavior_loss": 0.9,
    }
    dreamer_collector.on_train_step(metrics)
    
    mock_logger.observe.assert_any_call("actor_loss", 0.4)
    mock_logger.observe.assert_any_call("critic_loss", 0.5)
    mock_logger.observe.assert_any_call("behavior_loss", 0.9)


def test_tracks_entropy(dreamer_collector: DreamerCollector, mock_logger: Mock) -> None:
    """Should track entropy bonus."""
    dreamer_collector.on_train_step({"entropy": 0.7})
    
    mock_logger.observe.assert_any_call("entropy", 0.7)


def test_tracks_value_stats(dreamer_collector: DreamerCollector, mock_logger: Mock) -> None:
    """Should track value and return statistics."""
    metrics = {
        "value_mean": 10.5,
        "return_mean": 12.3,
    }
    dreamer_collector.on_train_step(metrics)
    
    mock_logger.gauge.assert_any_call("value_mean", 10.5)
    mock_logger.gauge.assert_any_call("return_mean", 12.3)


def test_handles_tensor_values(dreamer_collector: DreamerCollector, mock_logger: Mock) -> None:
    """Should convert tensor values to floats."""
    metrics = {
        "loss": torch.tensor(0.5),
        "actor_loss": torch.tensor(0.3),
    }
    dreamer_collector.on_train_step(metrics)
    
    # Should convert to float
    mock_logger.observe.assert_any_call("loss", 0.5)
    mock_logger.observe.assert_any_call("actor_loss", 0.30000001192092896)  # Float conversion


def test_ignores_missing_metrics(dreamer_collector: DreamerCollector, mock_logger: Mock) -> None:
    """Should not fail on missing metrics."""
    dreamer_collector.on_train_step({})
    
    # Should not raise


# =============================================================================
# flush Tests
# =============================================================================


def test_does_not_flush_logger(dreamer_collector: DreamerCollector, mock_logger: Mock) -> None:
    """Flush is no-op - CompositeCollector handles it."""
    dreamer_collector.flush()
    
    mock_logger.flush.assert_not_called()


# =============================================================================
# save_state / load_state Tests
# =============================================================================


def test_save_state_delegates_to_logger(dreamer_collector: DreamerCollector, mock_logger: Mock) -> None:
    """Should save logger state."""
    mock_logger.save_state.return_value = {"counters": {}}
    
    state = dreamer_collector.save_state()
    
    assert "logger_state" in state
    mock_logger.save_state.assert_called_once()


def test_load_state_restores_logger(dreamer_collector: DreamerCollector, mock_logger: Mock) -> None:
    """Should restore logger state."""
    state = {"logger_state": {"counters": {"steps": 100}}}
    
    dreamer_collector.load_state(state)
    
    mock_logger.load_state.assert_called_once_with(state["logger_state"])


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


def test_satisfies_protocol(dreamer_collector: DreamerCollector) -> None:
    """Should satisfy MetricCollector protocol."""
    from mario_rl.metrics.collectors import MetricCollector
    
    assert isinstance(dreamer_collector, MetricCollector)
