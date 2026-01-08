"""Tests for coordinator-side collectors."""

import pytest
import time
from unittest.mock import MagicMock
from dataclasses import dataclass


@pytest.fixture
def mock_logger():
    """Create a mock MetricLogger."""
    return MagicMock()


# =============================================================================
# GradientCollector Tests
# =============================================================================

@pytest.fixture
def gradient_collector(mock_logger):
    """Create GradientCollector with mock logger."""
    from mario_rl.metrics.collectors.coordinator import GradientCollector
    return GradientCollector(logger=mock_logger)


class TestGradientCollectorOnGradientsReceived:
    """Tests for gradient reception tracking."""
    
    def test_counts_gradients(self, gradient_collector, mock_logger):
        """Should count each gradient packet received."""
        gradient_collector.on_gradients_received(0, {})
        gradient_collector.on_gradients_received(1, {})
        
        count_calls = [c for c in mock_logger.count.call_args_list 
                      if c[0] == ("gradients_received",)]
        assert len(count_calls) == 2
    
    def test_tracks_timesteps_from_packet(self, gradient_collector, mock_logger):
        """Should track timesteps from gradient packet."""
        gradient_collector.on_gradients_received(0, {"timesteps": 100})
        
        timestep_calls = [c for c in mock_logger.count.call_args_list 
                        if c[0][0] == "total_timesteps"]
        assert len(timestep_calls) == 1
        assert timestep_calls[0][1]["n"] == 100


class TestGradientCollectorOnUpdateApplied:
    """Tests for update tracking."""
    
    def test_counts_updates(self, gradient_collector, mock_logger):
        """Should count each optimizer update."""
        gradient_collector.on_update_applied({})
        
        update_calls = [c for c in mock_logger.count.call_args_list 
                       if c[0] == ("update_count",)]
        assert len(update_calls) == 1
    
    def test_tracks_weight_version(self, gradient_collector, mock_logger):
        """Should track weight version as gauge."""
        gradient_collector.on_update_applied({"weight_version": 42})
        
        calls = mock_logger.gauge.call_args_list
        assert any(c[0] == ("weight_version", 42) for c in calls)
    
    def test_tracks_learning_rate(self, gradient_collector, mock_logger):
        """Should track learning rate as gauge."""
        gradient_collector.on_update_applied({"lr": 0.001})
        
        calls = mock_logger.gauge.call_args_list
        assert any(c[0] == ("learning_rate", 0.001) for c in calls)


class TestGradientCollectorOnCheckpointSaved:
    """Tests for checkpoint tracking."""
    
    def test_counts_checkpoints(self, gradient_collector, mock_logger):
        """Should count checkpoint saves."""
        gradient_collector.on_checkpoint_saved("/path/to/checkpoint")
        
        count_calls = [c for c in mock_logger.count.call_args_list 
                      if c[0] == ("checkpoints_saved",)]
        assert len(count_calls) == 1


class TestGradientCollectorProtocol:
    """Protocol compliance tests."""
    
    def test_satisfies_protocol(self, gradient_collector):
        """GradientCollector should satisfy CoordinatorCollector protocol."""
        from mario_rl.metrics.collectors.coordinator import CoordinatorCollector
        assert isinstance(gradient_collector, CoordinatorCollector)


# =============================================================================
# AggregatorCollector Tests
# =============================================================================

@pytest.fixture
def aggregator_collector(mock_logger):
    """Create AggregatorCollector with mock logger."""
    from mario_rl.metrics.collectors.coordinator import AggregatorCollector
    return AggregatorCollector(logger=mock_logger)


class TestAggregatorCollectorOnWorkerMetrics:
    """Tests for worker metrics aggregation."""
    
    def test_stores_worker_snapshots(self, aggregator_collector):
        """Should store each worker's snapshot."""
        aggregator_collector.on_worker_metrics(0, {"reward": 100})
        aggregator_collector.on_worker_metrics(1, {"reward": 200})
        
        assert len(aggregator_collector._worker_snapshots) == 2
        assert aggregator_collector._worker_snapshots[0]["reward"] == 100
        assert aggregator_collector._worker_snapshots[1]["reward"] == 200
    
    def test_updates_snapshot_on_same_worker(self, aggregator_collector):
        """Should update existing worker's snapshot."""
        aggregator_collector.on_worker_metrics(0, {"reward": 100})
        aggregator_collector.on_worker_metrics(0, {"reward": 150})
        
        assert aggregator_collector._worker_snapshots[0]["reward"] == 150
    
    def test_computes_average_reward(self, aggregator_collector, mock_logger):
        """Should compute average reward across workers."""
        aggregator_collector.on_worker_metrics(0, {"reward": 100})
        aggregator_collector.on_worker_metrics(1, {"reward": 200})
        
        calls = mock_logger.gauge.call_args_list
        avg_reward_calls = [c for c in calls if c[0][0] == "avg_reward"]
        assert len(avg_reward_calls) >= 1
        # Last call should have avg of 100 and 200 = 150
        assert abs(avg_reward_calls[-1][0][1] - 150.0) < 0.01
    
    def test_computes_average_speed(self, aggregator_collector, mock_logger):
        """Should compute average speed across workers."""
        aggregator_collector.on_worker_metrics(0, {"speed": 10.0})
        aggregator_collector.on_worker_metrics(1, {"speed": 20.0})
        
        calls = mock_logger.gauge.call_args_list
        avg_speed_calls = [c for c in calls if c[0][0] == "avg_speed"]
        assert len(avg_speed_calls) >= 1
        assert abs(avg_speed_calls[-1][0][1] - 15.0) < 0.01
    
    def test_computes_total_deaths(self, aggregator_collector, mock_logger):
        """Should sum deaths across workers."""
        aggregator_collector.on_worker_metrics(0, {"deaths": 5})
        aggregator_collector.on_worker_metrics(1, {"deaths": 3})
        
        calls = mock_logger.gauge.call_args_list
        death_calls = [c for c in calls if c[0][0] == "total_deaths"]
        assert len(death_calls) >= 1
        assert death_calls[-1][0][1] == 8
    
    def test_computes_total_flags(self, aggregator_collector, mock_logger):
        """Should sum flags across workers."""
        aggregator_collector.on_worker_metrics(0, {"flags": 2})
        aggregator_collector.on_worker_metrics(1, {"flags": 1})
        
        calls = mock_logger.gauge.call_args_list
        flag_calls = [c for c in calls if c[0][0] == "total_flags"]
        assert len(flag_calls) >= 1
        assert flag_calls[-1][0][1] == 3


class TestAggregatorCollectorProtocol:
    """Protocol compliance tests."""
    
    def test_satisfies_protocol(self, aggregator_collector):
        """AggregatorCollector should satisfy CoordinatorCollector protocol."""
        from mario_rl.metrics.collectors.coordinator import CoordinatorCollector
        assert isinstance(aggregator_collector, CoordinatorCollector)


# =============================================================================
# CoordinatorComposite Tests
# =============================================================================

@pytest.fixture
def coord_composite(mock_logger):
    """Create CoordinatorComposite with real collectors."""
    from mario_rl.metrics.collectors.coordinator import (
        CoordinatorComposite,
        GradientCollector,
        AggregatorCollector,
    )
    return CoordinatorComposite(collectors=[
        GradientCollector(logger=mock_logger),
        AggregatorCollector(logger=mock_logger),
    ])


class TestCoordinatorComposite:
    """Tests for CoordinatorComposite delegation."""
    
    def test_delegates_gradients_received(self, coord_composite, mock_logger):
        """Should delegate to all collectors."""
        coord_composite.on_gradients_received(0, {"timesteps": 50})
        
        # GradientCollector should have counted
        count_calls = [c for c in mock_logger.count.call_args_list 
                      if c[0] == ("gradients_received",)]
        assert len(count_calls) == 1
    
    def test_delegates_worker_metrics(self, coord_composite, mock_logger):
        """Should delegate to all collectors."""
        coord_composite.on_worker_metrics(0, {"reward": 100, "deaths": 2})
        
        # AggregatorCollector should have tracked
        calls = mock_logger.gauge.call_args_list
        assert any("avg_reward" in str(c) or "total_deaths" in str(c) for c in calls)
    
    def test_satisfies_protocol(self, coord_composite):
        """CoordinatorComposite should satisfy protocol."""
        from mario_rl.metrics.collectors.coordinator import CoordinatorCollector
        assert isinstance(coord_composite, CoordinatorCollector)
