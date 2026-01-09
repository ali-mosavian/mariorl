"""Metric collectors for clean separation of concerns.

Collectors observe training events and extract metrics without
polluting the core training logic.

Worker-side collectors (MetricCollector protocol):
- MarioCollector: Game metrics (x_pos, deaths, flags, speed)
- DDQNCollector: DDQN training metrics (loss, q_mean, td_error)
- DreamerCollector: Dreamer training metrics (world model, actor-critic)
- SystemCollector: System metrics (steps, episodes, buffer_size)
- CompositeCollector: Combines multiple collectors

Coordinator-side collectors (CoordinatorCollector protocol):
- GradientCollector: Gradient flow (grads_received, grads_per_sec)
- AggregatorCollector: Aggregates worker metrics
- CoordinatorComposite: Combines coordinator collectors
"""

from mario_rl.metrics.collectors.protocol import MetricCollector
from mario_rl.metrics.collectors.composite import CompositeCollector
from mario_rl.metrics.collectors.mario import MarioCollector
from mario_rl.metrics.collectors.ddqn import DDQNCollector
from mario_rl.metrics.collectors.dreamer import DreamerCollector
from mario_rl.metrics.collectors.system import SystemCollector
from mario_rl.metrics.collectors.coordinator import (
    CoordinatorCollector,
    GradientCollector,
    AggregatorCollector,
    CoordinatorComposite,
)

__all__ = [
    # Worker-side
    "MetricCollector",
    "CompositeCollector",
    "MarioCollector",
    "DDQNCollector",
    "DreamerCollector",
    "SystemCollector",
    # Coordinator-side
    "CoordinatorCollector",
    "GradientCollector",
    "AggregatorCollector",
    "CoordinatorComposite",
]
