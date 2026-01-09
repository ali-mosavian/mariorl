"""DreamerCollector - Dreamer-specific training metrics extraction.

Tracks world model and actor-critic metrics:
- Total loss
- World model: dynamics_loss, reward_loss, wm_loss
- Behavior: actor_loss, critic_loss, behavior_loss
- Entropy bonus
- Value/return statistics
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
class DreamerCollector:
    """Collects Dreamer-specific training metrics.
    
    Extracts metrics from training step results without polluting
    the core training logic.
    
    Attributes:
        logger: MetricLogger to record metrics
    """
    
    logger: "MetricLogger"
    
    def on_step(self, info: dict[str, Any]) -> None:
        """DreamerCollector ignores environment step info.
        
        Game metrics are handled by MarioCollector.
        """
        pass
    
    def on_episode_end(self, info: dict[str, Any]) -> None:
        """DreamerCollector ignores episode end.
        
        Death/flag tracking is handled by MarioCollector.
        """
        pass
    
    def on_train_step(self, metrics: dict[str, Any]) -> None:
        """Extract training metrics.
        
        Args:
            metrics: Dict from learner.compute_loss() containing
                     world model and behavior learning metrics.
        """
        # Rolling averages for losses
        rolling_metrics = [
            "loss",           # Total loss
            "dynamics_loss",  # World model - dynamics
            "reward_loss",    # World model - reward prediction
            "wm_loss",        # World model - combined
            "actor_loss",     # Behavior - actor
            "critic_loss",    # Behavior - critic
            "behavior_loss",  # Behavior - combined
            "entropy",        # Entropy bonus
        ]
        
        for name in rolling_metrics:
            if name in metrics:
                self.logger.observe(name, _to_float(metrics[name]))
        
        # Gauges for current values (not averaged)
        gauge_metrics = [
            "value_mean",   # Average predicted value
            "return_mean",  # Average computed return
        ]
        
        for name in gauge_metrics:
            if name in metrics:
                self.logger.gauge(name, _to_float(metrics[name]))
    
    def flush(self) -> None:
        """No-op - let CompositeCollector handle flushing."""
        pass
    
    def save_state(self) -> dict[str, Any]:
        """DreamerCollector is stateless - delegates to logger."""
        return {"logger_state": self.logger.save_state()}
    
    def load_state(self, state: dict[str, Any]) -> None:
        """Restore logger state."""
        if "logger_state" in state:
            self.logger.load_state(state["logger_state"])
