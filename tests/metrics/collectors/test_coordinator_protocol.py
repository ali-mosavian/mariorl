"""Tests for CoordinatorCollector protocol."""

import pytest
from typing import Any


def test_protocol_defines_required_methods():
    """Protocol should define all required coordinator methods."""
    from mario_rl.metrics.collectors.coordinator import CoordinatorCollector
    
    required_methods = [
        'on_gradients_received',
        'on_update_applied',
        'on_checkpoint_saved',
        'on_worker_metrics',
        'flush',
    ]
    for method in required_methods:
        assert hasattr(CoordinatorCollector, method)


def test_class_implementing_protocol_is_recognized():
    """A class implementing all methods should satisfy the protocol."""
    from mario_rl.metrics.collectors.coordinator import CoordinatorCollector
    
    class ValidCoordCollector:
        def on_gradients_received(self, worker_id: int, packet: Any) -> None:
            pass
        
        def on_update_applied(self, metrics: dict[str, Any]) -> None:
            pass
        
        def on_checkpoint_saved(self, path: str) -> None:
            pass
        
        def on_worker_metrics(self, worker_id: int, snapshot: dict[str, Any]) -> None:
            pass
        
        def flush(self) -> None:
            pass
    
    collector = ValidCoordCollector()
    assert isinstance(collector, CoordinatorCollector)
