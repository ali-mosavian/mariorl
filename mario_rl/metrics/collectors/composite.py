"""Composite collector that delegates to multiple collectors."""

from dataclasses import dataclass, field
from typing import Any

from mario_rl.metrics.collectors.protocol import MetricCollector


@dataclass
class CompositeCollector:
    """Combines multiple collectors into one.
    
    Delegates each method call to all registered collectors.
    """
    
    collectors: list[MetricCollector] = field(default_factory=list)
    
    def on_step(self, info: dict[str, Any]) -> None:
        """Delegate to all collectors."""
        for collector in self.collectors:
            collector.on_step(info)
    
    def on_episode_end(self, info: dict[str, Any]) -> None:
        """Delegate to all collectors."""
        for collector in self.collectors:
            collector.on_episode_end(info)
    
    def on_train_step(self, metrics: dict[str, Any]) -> None:
        """Delegate to all collectors."""
        for collector in self.collectors:
            collector.on_train_step(metrics)
    
    def flush(self) -> None:
        """Delegate to all collectors."""
        for collector in self.collectors:
            collector.flush()
