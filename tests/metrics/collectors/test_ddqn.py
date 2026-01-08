"""Tests for DDQNCollector - DDQN-specific training metrics."""

import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_logger():
    """Create a mock MetricLogger."""
    return MagicMock()


@pytest.fixture
def ddqn_collector(mock_logger):
    """Create DDQNCollector with mock logger."""
    from mario_rl.metrics.collectors.ddqn import DDQNCollector
    return DDQNCollector(logger=mock_logger)


class TestDDQNCollectorOnStep:
    """Tests for on_step method."""
    
    def test_ignores_step_info(self, ddqn_collector, mock_logger):
        """DDQNCollector should not process step info."""
        ddqn_collector.on_step({"x_pos": 100, "state": {"time": 380}})
        
        # No gauge calls for x_pos or game_time
        x_pos_calls = [c for c in mock_logger.gauge.call_args_list 
                      if c[0][0] == "x_pos"]
        assert len(x_pos_calls) == 0


class TestDDQNCollectorOnEpisodeEnd:
    """Tests for on_episode_end method."""
    
    def test_ignores_episode_end(self, ddqn_collector, mock_logger):
        """DDQNCollector should not process episode end."""
        ddqn_collector.on_episode_end({"is_dead": True, "flag_get": False})
        
        # No count calls for deaths/flags
        deaths_calls = [c for c in mock_logger.count.call_args_list 
                       if "deaths" in str(c)]
        assert len(deaths_calls) == 0


class TestDDQNCollectorOnTrainStep:
    """Tests for on_train_step method."""
    
    def test_tracks_loss(self, ddqn_collector, mock_logger):
        """Should track loss as rolling metric."""
        ddqn_collector.on_train_step({"loss": 0.5})
        mock_logger.observe.assert_called_with("loss", 0.5)
    
    def test_tracks_q_mean(self, ddqn_collector, mock_logger):
        """Should track q_mean as rolling metric."""
        ddqn_collector.on_train_step({"q_mean": 15.5})
        
        calls = mock_logger.observe.call_args_list
        assert any(c[0] == ("q_mean", 15.5) for c in calls)
    
    def test_tracks_td_error(self, ddqn_collector, mock_logger):
        """Should track td_error as rolling metric."""
        ddqn_collector.on_train_step({"td_error": 0.25})
        
        calls = mock_logger.observe.call_args_list
        assert any(c[0] == ("td_error", 0.25) for c in calls)
    
    def test_tracks_q_max_as_gauge(self, ddqn_collector, mock_logger):
        """Should track q_max as gauge (not rolling)."""
        ddqn_collector.on_train_step({"q_max": 25.0})
        mock_logger.gauge.assert_called_with("q_max", 25.0)
    
    def test_tracks_grad_norm(self, ddqn_collector, mock_logger):
        """Should track gradient norm as gauge."""
        ddqn_collector.on_train_step({"grad_norm": 1.5})
        
        calls = mock_logger.gauge.call_args_list
        assert any(c[0] == ("grad_norm", 1.5) for c in calls)
    
    def test_handles_missing_metrics(self, ddqn_collector, mock_logger):
        """Should handle missing metrics gracefully."""
        ddqn_collector.on_train_step({})  # Empty metrics
        # No exceptions, no calls
        assert mock_logger.observe.call_count == 0
        assert mock_logger.gauge.call_count == 0
    
    def test_converts_tensor_to_float(self, ddqn_collector, mock_logger):
        """Should handle tensor values by converting to float."""
        import torch
        ddqn_collector.on_train_step({"loss": torch.tensor(0.5)})
        
        # Should have called with float, not tensor
        call_args = mock_logger.observe.call_args[0]
        assert call_args[0] == "loss"
        assert isinstance(call_args[1], float)


class TestDDQNCollectorFlush:
    """Tests for flush method."""
    
    def test_does_not_flush_logger(self, ddqn_collector, mock_logger):
        """DDQNCollector should not flush - let composite handle it."""
        ddqn_collector.flush()
        # Flush is a no-op for DDQNCollector
        mock_logger.flush.assert_not_called()


class TestDDQNCollectorProtocol:
    """Tests that DDQNCollector satisfies MetricCollector protocol."""
    
    def test_satisfies_protocol(self, ddqn_collector):
        """DDQNCollector should satisfy MetricCollector protocol."""
        from mario_rl.metrics.collectors import MetricCollector
        assert isinstance(ddqn_collector, MetricCollector)
