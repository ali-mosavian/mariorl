"""Tests for CompositeCollector - combines multiple collectors."""

import pytest
from unittest.mock import MagicMock, call


@pytest.fixture
def mock_collectors():
    """Create list of mock collectors."""
    collectors = [MagicMock() for _ in range(3)]
    return collectors


@pytest.fixture
def composite(mock_collectors):
    """Create CompositeCollector with mock collectors."""
    from mario_rl.metrics.collectors import CompositeCollector
    return CompositeCollector(collectors=mock_collectors)


class TestCompositeCollectorOnStep:
    """Tests for on_step delegation."""
    
    def test_delegates_to_all_collectors(self, composite, mock_collectors):
        """Should call on_step on all collectors."""
        info = {"x_pos": 100}
        composite.on_step(info)
        
        for collector in mock_collectors:
            collector.on_step.assert_called_once_with(info)
    
    def test_empty_collectors_no_error(self):
        """Should handle empty collector list."""
        from mario_rl.metrics.collectors import CompositeCollector
        composite = CompositeCollector(collectors=[])
        composite.on_step({})  # Should not raise


class TestCompositeCollectorOnEpisodeEnd:
    """Tests for on_episode_end delegation."""
    
    def test_delegates_to_all_collectors(self, composite, mock_collectors):
        """Should call on_episode_end on all collectors."""
        info = {"is_dead": True}
        composite.on_episode_end(info)
        
        for collector in mock_collectors:
            collector.on_episode_end.assert_called_once_with(info)


class TestCompositeCollectorOnTrainStep:
    """Tests for on_train_step delegation."""
    
    def test_delegates_to_all_collectors(self, composite, mock_collectors):
        """Should call on_train_step on all collectors."""
        metrics = {"loss": 0.5}
        composite.on_train_step(metrics)
        
        for collector in mock_collectors:
            collector.on_train_step.assert_called_once_with(metrics)


class TestCompositeCollectorFlush:
    """Tests for flush delegation."""
    
    def test_delegates_to_all_collectors(self, composite, mock_collectors):
        """Should call flush on all collectors."""
        composite.flush()
        
        for collector in mock_collectors:
            collector.flush.assert_called_once()


class TestCompositeCollectorProtocol:
    """Tests that CompositeCollector satisfies MetricCollector protocol."""
    
    def test_satisfies_protocol(self, composite):
        """CompositeCollector should satisfy MetricCollector protocol."""
        from mario_rl.metrics.collectors import MetricCollector
        assert isinstance(composite, MetricCollector)


class TestCompositeCollectorIntegration:
    """Integration tests with real collectors."""
    
    def test_mario_and_ddqn_collectors_together(self):
        """Test MarioCollector and DDQNCollector working together."""
        from mario_rl.metrics.collectors import CompositeCollector
        from mario_rl.metrics.collectors.mario import MarioCollector
        from mario_rl.metrics.collectors.ddqn import DDQNCollector
        
        mock_logger = MagicMock()
        
        mario = MarioCollector(logger=mock_logger)
        ddqn = DDQNCollector(logger=mock_logger)
        composite = CompositeCollector(collectors=[mario, ddqn])
        
        # Step - only Mario should track
        composite.on_step({"x_pos": 100, "state": {"time": 380}})
        
        # Episode end - only Mario should track
        composite.on_episode_end({"is_dead": True})
        
        # Train step - only DDQN should track
        composite.on_train_step({"loss": 0.5, "q_mean": 10.0})
        
        # Verify x_pos was tracked (Mario)
        x_pos_calls = [c for c in mock_logger.gauge.call_args_list 
                      if c[0][0] == "x_pos"]
        assert len(x_pos_calls) == 1
        
        # Verify death was tracked (Mario)
        death_calls = [c for c in mock_logger.count.call_args_list 
                      if c[0] == ("deaths",)]
        assert len(death_calls) == 1
        
        # Verify loss was tracked (DDQN)
        loss_calls = [c for c in mock_logger.observe.call_args_list 
                     if c[0][0] == "loss"]
        assert len(loss_calls) == 1
    
    def test_all_worker_collectors_together(self):
        """Test all worker-side collectors working together."""
        from mario_rl.metrics.collectors import CompositeCollector
        from mario_rl.metrics.collectors.mario import MarioCollector
        from mario_rl.metrics.collectors.ddqn import DDQNCollector
        from mario_rl.metrics.collectors.system import SystemCollector
        
        mock_logger = MagicMock()
        
        mario = MarioCollector(logger=mock_logger)
        ddqn = DDQNCollector(logger=mock_logger)
        system = SystemCollector(logger=mock_logger)
        composite = CompositeCollector(collectors=[mario, ddqn, system])
        
        # Simulate a full cycle
        composite.on_step({"x_pos": 100, "buffer_size": 500, "epsilon": 0.1})
        composite.on_episode_end({"is_dead": True, "episode_reward": 50.0})
        composite.on_train_step({"loss": 0.5, "grads_sent": True})
        composite.flush()
        
        # Verify metrics from each collector were tracked
        gauge_names = [c[0][0] for c in mock_logger.gauge.call_args_list]
        assert "x_pos" in gauge_names  # Mario
        assert "buffer_size" in gauge_names  # System
        assert "epsilon" in gauge_names  # System
        
        count_names = [c[0][0] for c in mock_logger.count.call_args_list]
        assert "deaths" in count_names  # Mario
        assert "steps" in count_names  # System
        assert "episodes" in count_names  # System
        assert "grads_sent" in count_names  # System
        
        observe_names = [c[0][0] for c in mock_logger.observe.call_args_list]
        assert "loss" in observe_names  # DDQN
        assert "reward" in observe_names  # System
