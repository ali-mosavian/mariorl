"""DDQNCollector - DDQN-specific training metrics extraction.

Tracks:
- Loss (rolling average)
- Q-values (q_mean, q_max, td_error)
- Gradient norm
"""

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mario_rl.metrics.logger import MetricLogger


def _to_float(value: Any) -> float:
    """Convert value to float, handling tensors."""
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


@dataclass
class DDQNCollector:
    """Collects DDQN-specific training metrics.
    
    Extracts metrics from training step results without polluting
    the core training logic.
    
    Attributes:
        logger: MetricLogger to record metrics
    """
    
    logger: "MetricLogger"
    
    def on_step(self, info: dict[str, Any]) -> None:
        """DDQNCollector ignores environment step info.
        
        Game metrics are handled by MarioCollector.
        """
        pass
    
    def on_episode_end(self, info: dict[str, Any]) -> None:
        """DDQNCollector ignores episode end.
        
        Death/flag tracking is handled by MarioCollector.
        """
        pass
    
    def on_train_step(self, metrics: dict[str, Any]) -> None:
        """Extract training metrics.
        
        Args:
            metrics: Dict from learner.compute_loss() containing
                     loss, q_mean, q_max, td_error, grad_norm, etc.
        """
        # Rolling averages
        if "loss" in metrics:
            self.logger.observe("loss", _to_float(metrics["loss"]))
        
        if "q_mean" in metrics:
            self.logger.observe("q_mean", _to_float(metrics["q_mean"]))
        
        if "td_error" in metrics:
            self.logger.observe("td_error", _to_float(metrics["td_error"]))
        
        # Gauges (current value, not averaged)
        if "q_max" in metrics:
            self.logger.gauge("q_max", _to_float(metrics["q_max"]))
        
        if "grad_norm" in metrics:
            self.logger.gauge("grad_norm", _to_float(metrics["grad_norm"]))
    
    def flush(self) -> None:
        """No-op - let CompositeCollector handle flushing."""
        pass
