"""Unified metrics system for distributed training."""

from mario_rl.metrics.schema import (
    MetricType,
    MetricDef,
    CommonMetrics,
    DDQNMetrics,
    DreamerMetrics,
    CoordinatorMetrics,
)
from mario_rl.metrics.logger import MetricLogger
from mario_rl.metrics.levels import LevelStats, LevelTracker
from mario_rl.metrics.aggregator import MetricAggregator

__all__ = [
    # Schema
    "MetricType",
    "MetricDef",
    "CommonMetrics",
    "DDQNMetrics",
    "DreamerMetrics",
    "CoordinatorMetrics",
    # Logger
    "MetricLogger",
    # Levels
    "LevelStats",
    "LevelTracker",
    # Aggregator
    "MetricAggregator",
]
