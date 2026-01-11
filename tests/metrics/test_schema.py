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


def test_metric_type_has_counter() -> None:
    """MetricType has COUNTER variant."""
    assert MetricType.COUNTER is not None


def test_metric_type_has_gauge() -> None:
    """MetricType has GAUGE variant."""
    assert MetricType.GAUGE is not None


def test_metric_type_has_rolling() -> None:
    """MetricType has ROLLING variant."""
    assert MetricType.ROLLING is not None


# =============================================================================
# MetricDef Tests
# =============================================================================


def test_metric_def_stores_name() -> None:
    """MetricDef stores the metric name."""
    defn = MetricDef("test_metric", MetricType.COUNTER)
    assert defn.name == "test_metric"


def test_metric_def_stores_type() -> None:
    """MetricDef stores the metric type."""
    defn = MetricDef("test_metric", MetricType.GAUGE)
    assert defn.metric_type == MetricType.GAUGE


def test_metric_def_default_window() -> None:
    """MetricDef has default window of 100 for rolling metrics."""
    defn = MetricDef("test_metric", MetricType.ROLLING)
    assert defn.window == 100


def test_metric_def_custom_window() -> None:
    """MetricDef accepts custom window size."""
    defn = MetricDef("test_metric", MetricType.ROLLING, window=50)
    assert defn.window == 50


def test_metric_def_is_frozen() -> None:
    """MetricDef is immutable (frozen dataclass)."""
    defn = MetricDef("test_metric", MetricType.COUNTER)
    with pytest.raises(AttributeError):
        defn.name = "changed"


# =============================================================================
# CommonMetrics Tests
# =============================================================================


def test_common_metrics_has_episodes() -> None:
    """CommonMetrics defines episodes counter."""
    assert CommonMetrics.EPISODES.name == "episodes"
    assert CommonMetrics.EPISODES.metric_type == MetricType.COUNTER


def test_common_metrics_has_steps() -> None:
    """CommonMetrics defines steps counter."""
    assert CommonMetrics.STEPS.name == "steps"
    assert CommonMetrics.STEPS.metric_type == MetricType.COUNTER


def test_common_metrics_has_reward() -> None:
    """CommonMetrics defines reward as rolling average."""
    assert CommonMetrics.REWARD.name == "reward"
    assert CommonMetrics.REWARD.metric_type == MetricType.ROLLING


def test_common_metrics_has_epsilon() -> None:
    """CommonMetrics defines epsilon gauge."""
    assert CommonMetrics.EPSILON.name == "epsilon"
    assert CommonMetrics.EPSILON.metric_type == MetricType.GAUGE


def test_common_metrics_has_steps_per_sec() -> None:
    """CommonMetrics defines steps_per_sec gauge."""
    assert CommonMetrics.STEPS_PER_SEC.name == "steps_per_sec"
    assert CommonMetrics.STEPS_PER_SEC.metric_type == MetricType.GAUGE


def test_common_metrics_definitions_returns_list() -> None:
    """CommonMetrics.definitions() returns list of MetricDef."""
    defs = CommonMetrics.definitions()
    assert isinstance(defs, list)
    assert all(isinstance(d, MetricDef) for d in defs)


def test_common_metrics_definitions_includes_core_metrics() -> None:
    """CommonMetrics.definitions() includes all core metrics."""
    defs = CommonMetrics.definitions()
    names = [d.name for d in defs]
    assert "episodes" in names
    assert "steps" in names
    assert "reward" in names
    assert "epsilon" in names
    assert "steps_per_sec" in names


def test_common_metrics_has_deaths() -> None:
    """CommonMetrics defines deaths counter."""
    assert CommonMetrics.DEATHS.name == "deaths"
    assert CommonMetrics.DEATHS.metric_type == MetricType.COUNTER


def test_common_metrics_has_timeouts() -> None:
    """CommonMetrics defines timeouts counter.
    
    Timeouts are distinct from deaths - they represent
    running out of game time rather than skill-based deaths.
    """
    assert CommonMetrics.TIMEOUTS.name == "timeouts"
    assert CommonMetrics.TIMEOUTS.metric_type == MetricType.COUNTER


def test_common_metrics_has_flags() -> None:
    """CommonMetrics defines flags counter."""
    assert CommonMetrics.FLAGS.name == "flags"
    assert CommonMetrics.FLAGS.metric_type == MetricType.COUNTER


def test_common_metrics_definitions_includes_episode_end_metrics() -> None:
    """CommonMetrics.definitions() includes all episode end metrics.
    
    Episodes can end in three ways: death, timeout, or flag.
    All three should be tracked separately.
    """
    defs = CommonMetrics.definitions()
    names = [d.name for d in defs]
    assert "deaths" in names
    assert "timeouts" in names
    assert "flags" in names


# =============================================================================
# DDQNMetrics Tests
# =============================================================================


def test_ddqn_metrics_has_loss() -> None:
    """DDQNMetrics defines loss as rolling average."""
    assert DDQNMetrics.LOSS.name == "loss"
    assert DDQNMetrics.LOSS.metric_type == MetricType.ROLLING


def test_ddqn_metrics_has_q_mean() -> None:
    """DDQNMetrics defines q_mean as rolling average."""
    assert DDQNMetrics.Q_MEAN.name == "q_mean"
    assert DDQNMetrics.Q_MEAN.metric_type == MetricType.ROLLING


def test_ddqn_metrics_has_q_max() -> None:
    """DDQNMetrics defines q_max as gauge."""
    assert DDQNMetrics.Q_MAX.name == "q_max"
    assert DDQNMetrics.Q_MAX.metric_type == MetricType.GAUGE


def test_ddqn_metrics_has_td_error() -> None:
    """DDQNMetrics defines td_error as rolling average."""
    assert DDQNMetrics.TD_ERROR.name == "td_error"
    assert DDQNMetrics.TD_ERROR.metric_type == MetricType.ROLLING


def test_ddqn_metrics_has_grad_norm() -> None:
    """DDQNMetrics defines grad_norm as rolling average."""
    assert DDQNMetrics.GRAD_NORM.name == "grad_norm"
    assert DDQNMetrics.GRAD_NORM.metric_type == MetricType.ROLLING


def test_ddqn_metrics_has_buffer_size() -> None:
    """DDQNMetrics defines buffer_size as gauge."""
    assert DDQNMetrics.BUFFER_SIZE.name == "buffer_size"
    assert DDQNMetrics.BUFFER_SIZE.metric_type == MetricType.GAUGE


def test_ddqn_metrics_definitions_includes_common() -> None:
    """DDQNMetrics.definitions() includes CommonMetrics."""
    defs = DDQNMetrics.definitions()
    names = [d.name for d in defs]
    # Common metrics should be included
    assert "episodes" in names
    assert "steps" in names
    assert "reward" in names


def test_ddqn_metrics_definitions_includes_ddqn_specific() -> None:
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


def test_dreamer_metrics_has_wm_loss() -> None:
    """DreamerMetrics defines wm_loss as rolling average."""
    assert DreamerMetrics.WM_LOSS.name == "wm_loss"
    assert DreamerMetrics.WM_LOSS.metric_type == MetricType.ROLLING


def test_dreamer_metrics_has_dynamics_loss() -> None:
    """DreamerMetrics defines dynamics_loss as rolling average."""
    assert DreamerMetrics.DYNAMICS_LOSS.name == "dynamics_loss"
    assert DreamerMetrics.DYNAMICS_LOSS.metric_type == MetricType.ROLLING


def test_dreamer_metrics_has_behavior_loss() -> None:
    """DreamerMetrics defines behavior_loss as rolling average."""
    assert DreamerMetrics.BEHAVIOR_LOSS.name == "behavior_loss"
    assert DreamerMetrics.BEHAVIOR_LOSS.metric_type == MetricType.ROLLING


def test_dreamer_metrics_has_actor_loss() -> None:
    """DreamerMetrics defines actor_loss as rolling average."""
    assert DreamerMetrics.ACTOR_LOSS.name == "actor_loss"
    assert DreamerMetrics.ACTOR_LOSS.metric_type == MetricType.ROLLING


def test_dreamer_metrics_has_critic_loss() -> None:
    """DreamerMetrics defines critic_loss as rolling average."""
    assert DreamerMetrics.CRITIC_LOSS.name == "critic_loss"
    assert DreamerMetrics.CRITIC_LOSS.metric_type == MetricType.ROLLING


def test_dreamer_metrics_has_entropy() -> None:
    """DreamerMetrics defines entropy as rolling average."""
    assert DreamerMetrics.ENTROPY.name == "entropy"
    assert DreamerMetrics.ENTROPY.metric_type == MetricType.ROLLING


def test_dreamer_metrics_has_value_mean() -> None:
    """DreamerMetrics defines value_mean as rolling average."""
    assert DreamerMetrics.VALUE_MEAN.name == "value_mean"
    assert DreamerMetrics.VALUE_MEAN.metric_type == MetricType.ROLLING


def test_dreamer_metrics_definitions_includes_common() -> None:
    """DreamerMetrics.definitions() includes CommonMetrics."""
    defs = DreamerMetrics.definitions()
    names = [d.name for d in defs]
    assert "episodes" in names
    assert "reward" in names


def test_dreamer_metrics_definitions_includes_dreamer_specific() -> None:
    """DreamerMetrics.definitions() includes Dreamer-specific metrics."""
    defs = DreamerMetrics.definitions()
    names = [d.name for d in defs]
    assert "wm_loss" in names
    assert "actor_loss" in names
    assert "behavior_loss" in names
    assert "entropy" in names


# =============================================================================
# CoordinatorMetrics Tests
# =============================================================================


def test_coordinator_metrics_has_update_count() -> None:
    """CoordinatorMetrics defines update_count as counter."""
    assert CoordinatorMetrics.UPDATE_COUNT.name == "update_count"
    assert CoordinatorMetrics.UPDATE_COUNT.metric_type == MetricType.COUNTER


def test_coordinator_metrics_has_grads_per_sec() -> None:
    """CoordinatorMetrics defines grads_per_sec as gauge."""
    assert CoordinatorMetrics.GRADS_PER_SEC.name == "grads_per_sec"
    assert CoordinatorMetrics.GRADS_PER_SEC.metric_type == MetricType.GAUGE


def test_coordinator_metrics_has_learning_rate() -> None:
    """CoordinatorMetrics defines learning_rate as gauge."""
    assert CoordinatorMetrics.LEARNING_RATE.name == "learning_rate"
    assert CoordinatorMetrics.LEARNING_RATE.metric_type == MetricType.GAUGE


def test_coordinator_metrics_has_total_steps() -> None:
    """CoordinatorMetrics defines total_steps as counter."""
    assert CoordinatorMetrics.TOTAL_STEPS.name == "total_steps"
    assert CoordinatorMetrics.TOTAL_STEPS.metric_type == MetricType.COUNTER


def test_coordinator_metrics_has_weight_version() -> None:
    """CoordinatorMetrics defines weight_version as counter."""
    assert CoordinatorMetrics.WEIGHT_VERSION.name == "weight_version"
    assert CoordinatorMetrics.WEIGHT_VERSION.metric_type == MetricType.COUNTER


def test_coordinator_metrics_definitions_returns_list() -> None:
    """CoordinatorMetrics.definitions() returns list of MetricDef."""
    defs = CoordinatorMetrics.definitions()
    assert isinstance(defs, list)
    names = [d.name for d in defs]
    assert "update_count" in names
    assert "grads_per_sec" in names
    assert "learning_rate" in names
