"""SystemCollector - system/training infrastructure metrics.

Tracks:
- Steps and episodes counts
- Buffer size
- Epsilon (exploration rate)
- Steps per second
- Gradients sent
"""

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from mario_rl.metrics.logger import MetricLogger


@dataclass
class SystemCollector:
    """Collects system/infrastructure metrics.
    
    Tracks training system state independent of game or model.
    
    Attributes:
        logger: MetricLogger to record metrics
    """
    
    logger: "MetricLogger"
    
    def on_step(self, info: dict[str, Any]) -> None:
        """Track per-step system metrics.
        
        Args:
            info: Step info containing buffer_size, epsilon, etc.
        """
        # Count steps
        self.logger.count("steps")
        
        # Track buffer state
        if "buffer_size" in info:
            self.logger.gauge("buffer_size", info["buffer_size"])
        
        # Track exploration
        if "epsilon" in info:
            self.logger.gauge("epsilon", info["epsilon"])
    
    def on_episode_end(self, info: dict[str, Any]) -> None:
        """Track episode completion metrics.
        
        Args:
            info: Episode end info containing reward, length, etc.
        """
        self.logger.count("episodes")
        
        if "episode_reward" in info:
            self.logger.observe("reward", info["episode_reward"])
        
        if "episode_length" in info:
            self.logger.observe("episode_length", info["episode_length"])
    
    def on_train_step(self, metrics: dict[str, Any]) -> None:
        """Track training step metrics.
        
        Args:
            metrics: Training metrics including grads_sent flag
        """
        if metrics.get("grads_sent"):
            self.logger.count("grads_sent")
    
    def update_steps_per_sec(self, num_steps: int, elapsed: float) -> None:
        """Update steps per second metric.
        
        Called externally with timing information.
        
        Args:
            num_steps: Number of steps taken
            elapsed: Time elapsed in seconds
        """
        if elapsed > 0:
            sps = num_steps / elapsed
            self.logger.gauge("steps_per_sec", sps)
    
    def flush(self) -> None:
        """No-op - let CompositeCollector handle flushing."""
        pass
