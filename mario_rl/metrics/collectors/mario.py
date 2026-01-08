"""MarioCollector - Mario-specific metrics extraction.

Tracks:
- Position (x_pos, best_x)
- Game time
- Deaths and flag captures
- Speed (x_pos / time_spent per episode)
"""

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from mario_rl.metrics.logger import MetricLogger


@dataclass
class MarioCollector:
    """Collects Mario-specific game metrics.
    
    Extracts metrics from environment info without polluting
    the core training logic.
    
    Attributes:
        logger: MetricLogger to record metrics
    """
    
    logger: "MetricLogger"
    
    # Episode tracking (reset on episode end)
    _episode_x_pos: int = field(init=False, default=0)
    _episode_start_time: int = field(init=False, default=400)  # Mario timer starts at 400
    
    def on_step(self, info: dict[str, Any]) -> None:
        """Extract metrics from step info.
        
        Args:
            info: Environment step info dict
        """
        # Track position
        x_pos = info.get("x_pos", 0)
        self.logger.gauge("x_pos", x_pos)
        
        # Update best x this episode
        self._episode_x_pos = max(self._episode_x_pos, x_pos)
        
        # Extract game time from nested state
        state = info.get("state", {})
        if isinstance(state, dict) and "time" in state:
            self.logger.gauge("game_time", state["time"])
    
    def on_episode_end(self, info: dict[str, Any]) -> None:
        """Extract end-of-episode metrics.
        
        Args:
            info: Final step info dict
        """
        # Check for flag capture vs death
        got_flag = info.get("flag_get", False)
        is_dead = info.get("is_dead", False) or info.get("is_dying", False)
        
        if got_flag:
            self.logger.count("flags")
        elif is_dead:
            self.logger.count("deaths")
        
        # Calculate speed (x_pos / time_spent)
        state = info.get("state", {})
        game_time = state.get("time", self._episode_start_time) if isinstance(state, dict) else self._episode_start_time
        time_spent = self._episode_start_time - game_time
        
        if time_spent > 0:
            speed = self._episode_x_pos / time_spent
            self.logger.observe("speed", speed)
        
        # Reset for next episode
        self._episode_x_pos = 0
        self._episode_start_time = 400
    
    def on_train_step(self, metrics: dict[str, Any]) -> None:
        """MarioCollector ignores training metrics.
        
        Training metrics are handled by model-specific collectors.
        """
        pass
    
    def flush(self) -> None:
        """Flush metrics to logger."""
        self.logger.flush()
