"""Tests for MetricCollector protocol."""

import pytest
from typing import Any, Protocol, runtime_checkable


def test_protocol_defines_required_methods():
    """Protocol should define all required collector methods."""
    from mario_rl.metrics.collectors import MetricCollector
    
    # Protocol should be runtime checkable
    assert hasattr(MetricCollector, '__protocol_attrs__') or hasattr(MetricCollector, '_is_protocol')
    
    # Check required methods exist in protocol
    required_methods = ['on_step', 'on_episode_end', 'on_train_step', 'flush']
    for method in required_methods:
        assert hasattr(MetricCollector, method)


def test_class_implementing_protocol_is_recognized():
    """A class implementing all methods should satisfy the protocol."""
    from mario_rl.metrics.collectors import MetricCollector
    
    class ValidCollector:
        def on_step(self, info: dict[str, Any]) -> None:
            pass
        
        def on_episode_end(self, info: dict[str, Any]) -> None:
            pass
        
        def on_train_step(self, metrics: dict[str, Any]) -> None:
            pass
        
        def flush(self) -> None:
            pass
    
    collector = ValidCollector()
    assert isinstance(collector, MetricCollector)


def test_class_missing_method_not_recognized():
    """A class missing methods should not satisfy the protocol."""
    from mario_rl.metrics.collectors import MetricCollector
    
    class InvalidCollector:
        def on_step(self, info: dict[str, Any]) -> None:
            pass
        # Missing other methods
    
    collector = InvalidCollector()
    assert not isinstance(collector, MetricCollector)
