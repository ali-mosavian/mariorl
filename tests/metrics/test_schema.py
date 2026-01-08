"""Tests for metrics schema definitions."""

import pytest

from mario_rl.metrics.schema import (
    MetricType,
    MetricDef,
    CommonMetrics,
    DDQNMetrics,
    DreamerMetrics,
    CoordinatorMetrics,
)


# =============================================================================
# MetricType Tests
# =============================================================================


def test_metric_type_has_counter():
    """MetricType has COUNTER variant."""
    assert MetricType.COUNTER is not None


def test_metric_type_has_gauge():
    """MetricType has GAUGE variant."""
    assert MetricType.GAUGE is not None


def test_metric_type_has_rolling():
    """MetricType has ROLLING variant."""
    assert MetricType.ROLLING is not None


# =============================================================================
# MetricDef Tests
# =============================================================================


def test_metric_def_stores_name():
    """MetricDef stores the metric name."""
    defn = MetricDef("test_metric", MetricType.COUNTER)
    assert defn.name == "test_metric"


def test_metric_def_stores_type():
    """MetricDef stores the metric type."""
    defn = MetricDef("test_metric", MetricType.GAUGE)
    assert defn.metric_type == MetricType.GAUGE


def test_metric_def_default_window():
    """MetricDef has default window of 100 for rolling metrics."""
    defn = MetricDef("test_metric", MetricType.ROLLING)
    assert defn.window == 100


def test_metric_def_custom_window():
    """MetricDef accepts custom window size."""
    defn = MetricDef("test_metric", MetricType.ROLLING, window=50)
    assert defn.window == 50


def test_metric_def_is_frozen():
    """MetricDef is immutable (frozen dataclass)."""
    defn = MetricDef("test_metric", MetricType.COUNTER)
    with pytest.raises(AttributeError):
        defn.name = "changed"


# =============================================================================
# CommonMetrics Tests
# =============================================================================


def test_common_metrics_has_episodes():
    """CommonMetrics defines episodes counter."""
    assert CommonMetrics.EPISODES.name == "episodes"
    assert CommonMetrics.EPISODES.metric_type == MetricType.COUNTER


def test_common_metrics_has_steps():
    """CommonMetrics defines steps counter."""
    assert CommonMetrics.STEPS.name == "steps"
    assert CommonMetrics.STEPS.metric_type == MetricType.COUNTER


def test_common_metrics_has_reward():
    """CommonMetrics defines reward as rolling average."""
    assert CommonMetrics.REWARD.name == "reward"
    assert CommonMetrics.REWARD.metric_type == MetricType.ROLLING


def test_common_metrics_has_epsilon():
    """CommonMetrics defines epsilon gauge."""
    assert CommonMetrics.EPSILON.name == "epsilon"
    assert CommonMetrics.EPSILON.metric_type == MetricType.GAUGE


def test_common_metrics_has_steps_per_sec():
    """CommonMetrics defines steps_per_sec gauge."""
    assert CommonMetrics.STEPS_PER_SEC.name == "steps_per_sec"
    assert CommonMetrics.STEPS_PER_SEC.metric_type == MetricType.GAUGE


def test_common_metrics_definitions_returns_list():
    """CommonMetrics.definitions() returns list of MetricDef."""
    defs = CommonMetrics.definitions()
    assert isinstance(defs, list)
    assert all(isinstance(d, MetricDef) for d in defs)


def test_common_metrics_definitions_includes_core_metrics():
    """CommonMetrics.definitions() includes all core metrics."""
    defs = CommonMetrics.definitions()
    names = [d.name for d in defs]
    assert "episodes" in names
    assert "steps" in names
    assert "reward" in names
    assert "epsilon" in names
    assert "steps_per_sec" in names


# =============================================================================
# DDQNMetrics Tests
# =============================================================================


def test_ddqn_metrics_has_loss():
    """DDQNMetrics defines loss as rolling average."""
    assert DDQNMetrics.LOSS.name == "loss"
    assert DDQNMetrics.LOSS.metric_type == MetricType.ROLLING


def test_ddqn_metrics_has_q_mean():
    """DDQNMetrics defines q_mean as rolling average."""
    assert DDQNMetrics.Q_MEAN.name == "q_mean"
    assert DDQNMetrics.Q_MEAN.metric_type == MetricType.ROLLING


def test_ddqn_metrics_has_q_max():
    """DDQNMetrics defines q_max as gauge."""
    assert DDQNMetrics.Q_MAX.name == "q_max"
    assert DDQNMetrics.Q_MAX.metric_type == MetricType.GAUGE


def test_ddqn_metrics_has_td_error():
    """DDQNMetrics defines td_error as rolling average."""
    assert DDQNMetrics.TD_ERROR.name == "td_error"
    assert DDQNMetrics.TD_ERROR.metric_type == MetricType.ROLLING


def test_ddqn_metrics_has_grad_norm():
    """DDQNMetrics defines grad_norm as rolling average."""
    assert DDQNMetrics.GRAD_NORM.name == "grad_norm"
    assert DDQNMetrics.GRAD_NORM.metric_type == MetricType.ROLLING


def test_ddqn_metrics_has_buffer_size():
    """DDQNMetrics defines buffer_size as gauge."""
    assert DDQNMetrics.BUFFER_SIZE.name == "buffer_size"
    assert DDQNMetrics.BUFFER_SIZE.metric_type == MetricType.GAUGE


def test_ddqn_metrics_definitions_includes_common():
    """DDQNMetrics.definitions() includes CommonMetrics."""
    defs = DDQNMetrics.definitions()
    names = [d.name for d in defs]
    # Common metrics should be included
    assert "episodes" in names
    assert "steps" in names
    assert "reward" in names


def test_ddqn_metrics_definitions_includes_ddqn_specific():
    """DDQNMetrics.definitions() includes DDQN-specific metrics."""
    defs = DDQNMetrics.definitions()
    names = [d.name for d in defs]
    assert "loss" in names
    assert "q_mean" in names
    assert "td_error" in names
    assert "grad_norm" in names


# =============================================================================
# DreamerMetrics Tests
# =============================================================================


def test_dreamer_metrics_has_world_loss():
    """DreamerMetrics defines world_loss as rolling average."""
    assert DreamerMetrics.WORLD_LOSS.name == "world_loss"
    assert DreamerMetrics.WORLD_LOSS.metric_type == MetricType.ROLLING


def test_dreamer_metrics_has_actor_loss():
    """DreamerMetrics defines actor_loss as rolling average."""
    assert DreamerMetrics.ACTOR_LOSS.name == "actor_loss"
    assert DreamerMetrics.ACTOR_LOSS.metric_type == MetricType.ROLLING


def test_dreamer_metrics_has_critic_loss():
    """DreamerMetrics defines critic_loss as rolling average."""
    assert DreamerMetrics.CRITIC_LOSS.name == "critic_loss"
    assert DreamerMetrics.CRITIC_LOSS.metric_type == MetricType.ROLLING


def test_dreamer_metrics_has_kl_div():
    """DreamerMetrics defines kl_div as rolling average."""
    assert DreamerMetrics.KL_DIV.name == "kl_div"
    assert DreamerMetrics.KL_DIV.metric_type == MetricType.ROLLING


def test_dreamer_metrics_has_entropy():
    """DreamerMetrics defines entropy as rolling average."""
    assert DreamerMetrics.ENTROPY.name == "entropy"
    assert DreamerMetrics.ENTROPY.metric_type == MetricType.ROLLING


def test_dreamer_metrics_definitions_includes_common():
    """DreamerMetrics.definitions() includes CommonMetrics."""
    defs = DreamerMetrics.definitions()
    names = [d.name for d in defs]
    assert "episodes" in names
    assert "reward" in names


def test_dreamer_metrics_definitions_includes_dreamer_specific():
    """DreamerMetrics.definitions() includes Dreamer-specific metrics."""
    defs = DreamerMetrics.definitions()
    names = [d.name for d in defs]
    assert "world_loss" in names
    assert "actor_loss" in names
    assert "kl_div" in names


# =============================================================================
# CoordinatorMetrics Tests
# =============================================================================


def test_coordinator_metrics_has_update_count():
    """CoordinatorMetrics defines update_count as counter."""
    assert CoordinatorMetrics.UPDATE_COUNT.name == "update_count"
    assert CoordinatorMetrics.UPDATE_COUNT.metric_type == MetricType.COUNTER


def test_coordinator_metrics_has_grads_per_sec():
    """CoordinatorMetrics defines grads_per_sec as gauge."""
    assert CoordinatorMetrics.GRADS_PER_SEC.name == "grads_per_sec"
    assert CoordinatorMetrics.GRADS_PER_SEC.metric_type == MetricType.GAUGE


def test_coordinator_metrics_has_learning_rate():
    """CoordinatorMetrics defines learning_rate as gauge."""
    assert CoordinatorMetrics.LEARNING_RATE.name == "learning_rate"
    assert CoordinatorMetrics.LEARNING_RATE.metric_type == MetricType.GAUGE


def test_coordinator_metrics_has_total_steps():
    """CoordinatorMetrics defines total_steps as counter."""
    assert CoordinatorMetrics.TOTAL_STEPS.name == "total_steps"
    assert CoordinatorMetrics.TOTAL_STEPS.metric_type == MetricType.COUNTER


def test_coordinator_metrics_has_weight_version():
    """CoordinatorMetrics defines weight_version as counter."""
    assert CoordinatorMetrics.WEIGHT_VERSION.name == "weight_version"
    assert CoordinatorMetrics.WEIGHT_VERSION.metric_type == MetricType.COUNTER


def test_coordinator_metrics_definitions_returns_list():
    """CoordinatorMetrics.definitions() returns list of MetricDef."""
    defs = CoordinatorMetrics.definitions()
    assert isinstance(defs, list)
    names = [d.name for d in defs]
    assert "update_count" in names
    assert "grads_per_sec" in names
    assert "learning_rate" in names
