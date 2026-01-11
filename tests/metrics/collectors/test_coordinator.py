"""Tests for coordinator-side collectors."""

import time
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from mario_rl.metrics.collectors.coordinator import AggregatorCollector
from mario_rl.metrics.collectors.coordinator import CoordinatorComposite
from mario_rl.metrics.collectors.coordinator import CoordinatorCollector
from mario_rl.metrics.collectors.coordinator import GradientCollector


@pytest.fixture
def mock_logger() -> MagicMock:
    """Create a mock MetricLogger."""
    return MagicMock()


# =============================================================================
# GradientCollector Tests
# =============================================================================

@pytest.fixture
def gradient_collector(mock_logger: MagicMock) -> GradientCollector:
    """Create GradientCollector with mock logger."""
    return GradientCollector(logger=mock_logger)


def test_counts_gradients(gradient_collector: GradientCollector, mock_logger: MagicMock) -> None:
    """Should count each gradient packet received."""
    gradient_collector.on_gradients_received(0, {})
    gradient_collector.on_gradients_received(1, {})
    
    count_calls = [c for c in mock_logger.count.call_args_list 
                  if c[0] == ("gradients_received",)]
    assert len(count_calls) == 2


def test_tracks_timesteps_from_packet(gradient_collector: GradientCollector, mock_logger: MagicMock) -> None:
    """Should track timesteps from gradient packet."""
    gradient_collector.on_gradients_received(0, {"timesteps": 100})
    
    timestep_calls = [c for c in mock_logger.count.call_args_list 
                    if c[0][0] == "total_timesteps"]
    assert len(timestep_calls) == 1
    assert timestep_calls[0][1]["n"] == 100


def test_counts_updates(gradient_collector: GradientCollector, mock_logger: MagicMock) -> None:
    """Should count each optimizer update."""
    gradient_collector.on_update_applied({})
    
    update_calls = [c for c in mock_logger.count.call_args_list 
                   if c[0] == ("update_count",)]
    assert len(update_calls) == 1


def test_tracks_weight_version(gradient_collector: GradientCollector, mock_logger: MagicMock) -> None:
    """Should track weight version as gauge."""
    gradient_collector.on_update_applied({"weight_version": 42})
    
    calls = mock_logger.gauge.call_args_list
    assert any(c[0] == ("weight_version", 42) for c in calls)


def test_tracks_learning_rate(gradient_collector: GradientCollector, mock_logger: MagicMock) -> None:
    """Should track learning rate as gauge."""
    gradient_collector.on_update_applied({"lr": 0.001})
    
    calls = mock_logger.gauge.call_args_list
    assert any(c[0] == ("learning_rate", 0.001) for c in calls)


def test_counts_checkpoints(gradient_collector: GradientCollector, mock_logger: MagicMock) -> None:
    """Should count checkpoint saves."""
    gradient_collector.on_checkpoint_saved("/path/to/checkpoint")
    
    count_calls = [c for c in mock_logger.count.call_args_list 
                  if c[0] == ("checkpoints_saved",)]
    assert len(count_calls) == 1


def test_gradient_collector_satisfies_protocol(gradient_collector: GradientCollector) -> None:
    """GradientCollector should satisfy CoordinatorCollector protocol."""
    assert isinstance(gradient_collector, CoordinatorCollector)


# =============================================================================
# AggregatorCollector Tests
# =============================================================================

@pytest.fixture
def aggregator_collector(mock_logger: MagicMock) -> AggregatorCollector:
    """Create AggregatorCollector with mock logger."""
    return AggregatorCollector(logger=mock_logger)


def test_stores_worker_snapshots(aggregator_collector: AggregatorCollector) -> None:
    """Should store each worker's snapshot."""
    aggregator_collector.on_worker_metrics(0, {"reward": 100})
    aggregator_collector.on_worker_metrics(1, {"reward": 200})
    
    assert len(aggregator_collector._worker_snapshots) == 2
    assert aggregator_collector._worker_snapshots[0]["reward"] == 100
    assert aggregator_collector._worker_snapshots[1]["reward"] == 200


def test_updates_snapshot_on_same_worker(aggregator_collector: AggregatorCollector) -> None:
    """Should update existing worker's snapshot."""
    aggregator_collector.on_worker_metrics(0, {"reward": 100})
    aggregator_collector.on_worker_metrics(0, {"reward": 150})
    
    assert aggregator_collector._worker_snapshots[0]["reward"] == 150


def test_computes_average_reward(aggregator_collector: AggregatorCollector, mock_logger: MagicMock) -> None:
    """Should compute average reward across workers."""
    aggregator_collector.on_worker_metrics(0, {"reward": 100})
    aggregator_collector.on_worker_metrics(1, {"reward": 200})
    
    calls = mock_logger.gauge.call_args_list
    avg_reward_calls = [c for c in calls if c[0][0] == "avg_reward"]
    assert len(avg_reward_calls) >= 1
    # Last call should have avg of 100 and 200 = 150
    assert abs(avg_reward_calls[-1][0][1] - 150.0) < 0.01


def test_computes_average_speed(aggregator_collector: AggregatorCollector, mock_logger: MagicMock) -> None:
    """Should compute average speed across workers."""
    aggregator_collector.on_worker_metrics(0, {"speed": 10.0})
    aggregator_collector.on_worker_metrics(1, {"speed": 20.0})
    
    calls = mock_logger.gauge.call_args_list
    avg_speed_calls = [c for c in calls if c[0][0] == "avg_speed"]
    assert len(avg_speed_calls) >= 1
    assert abs(avg_speed_calls[-1][0][1] - 15.0) < 0.01


def test_computes_total_deaths(aggregator_collector: AggregatorCollector, mock_logger: MagicMock) -> None:
    """Should sum deaths across workers."""
    aggregator_collector.on_worker_metrics(0, {"deaths": 5})
    aggregator_collector.on_worker_metrics(1, {"deaths": 3})
    
    calls = mock_logger.gauge.call_args_list
    death_calls = [c for c in calls if c[0][0] == "total_deaths"]
    assert len(death_calls) >= 1
    assert death_calls[-1][0][1] == 8


def test_computes_total_flags(aggregator_collector: AggregatorCollector, mock_logger: MagicMock) -> None:
    """Should sum flags across workers."""
    aggregator_collector.on_worker_metrics(0, {"flags": 2})
    aggregator_collector.on_worker_metrics(1, {"flags": 1})
    
    calls = mock_logger.gauge.call_args_list
    flag_calls = [c for c in calls if c[0][0] == "total_flags"]
    assert len(flag_calls) >= 1
    assert flag_calls[-1][0][1] == 3


def test_aggregator_collector_satisfies_protocol(aggregator_collector: AggregatorCollector) -> None:
    """AggregatorCollector should satisfy CoordinatorCollector protocol."""
    assert isinstance(aggregator_collector, CoordinatorCollector)


# =============================================================================
# CoordinatorComposite Tests
# =============================================================================

@pytest.fixture
def coord_composite(mock_logger: MagicMock) -> CoordinatorComposite:
    """Create CoordinatorComposite with real collectors."""
    return CoordinatorComposite(collectors=[
        GradientCollector(logger=mock_logger),
        AggregatorCollector(logger=mock_logger),
    ])


def test_delegates_gradients_received(coord_composite: CoordinatorComposite, mock_logger: MagicMock) -> None:
    """Should delegate to all collectors."""
    coord_composite.on_gradients_received(0, {"timesteps": 50})
    
    # GradientCollector should have counted
    count_calls = [c for c in mock_logger.count.call_args_list 
                  if c[0] == ("gradients_received",)]
    assert len(count_calls) == 1


def test_delegates_worker_metrics(coord_composite: CoordinatorComposite, mock_logger: MagicMock) -> None:
    """Should delegate to all collectors."""
    coord_composite.on_worker_metrics(0, {"reward": 100, "deaths": 2})
    
    # AggregatorCollector should have tracked
    calls = mock_logger.gauge.call_args_list
    assert any("avg_reward" in str(c) or "total_deaths" in str(c) for c in calls)


def test_coordinator_composite_satisfies_protocol(coord_composite: CoordinatorComposite) -> None:
    """CoordinatorComposite should satisfy protocol."""
    assert isinstance(coord_composite, CoordinatorCollector)
