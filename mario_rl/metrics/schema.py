"""
Metric schema definitions.

Defines metric types and model-specific metric schemas.
Follows Open/Closed principle: extend by adding new schema classes.
"""

from enum import Enum, auto
from dataclasses import dataclass


class MetricType(Enum):
    """Type of metric tracking."""
    
    COUNTER = auto()   # Monotonically increasing (episodes, deaths)
    GAUGE = auto()     # Current value (epsilon, learning_rate)
    ROLLING = auto()   # Rolling average (reward, loss)


@dataclass(frozen=True)
class MetricDef:
    """Definition of a single metric.
    
    Attributes:
        name: Metric identifier (used as CSV column, dict key)
        metric_type: How the metric is tracked
        window: Size of rolling window (only for ROLLING type)
    """
    
    name: str
    metric_type: MetricType
    window: int = 100


# =============================================================================
# Common Metrics (shared by all models)
# =============================================================================


class CommonMetrics:
    """Metrics shared by all model types."""
    
    EPISODES = MetricDef("episodes", MetricType.COUNTER)
    STEPS = MetricDef("steps", MetricType.COUNTER)
    REWARD = MetricDef("reward", MetricType.ROLLING)  # Rolling average of episode rewards
    EPISODE_REWARD = MetricDef("episode_reward", MetricType.GAUGE)  # Last episode's reward
    EPISODE_LENGTH = MetricDef("episode_length", MetricType.ROLLING)
    EPSILON = MetricDef("epsilon", MetricType.GAUGE)
    STEPS_PER_SEC = MetricDef("steps_per_sec", MetricType.GAUGE)
    
    # Mario-specific metrics (game state)
    X_POS = MetricDef("x_pos", MetricType.GAUGE)
    BEST_X = MetricDef("best_x", MetricType.GAUGE)
    BEST_X_EVER = MetricDef("best_x_ever", MetricType.GAUGE)
    GAME_TIME = MetricDef("game_time", MetricType.GAUGE)
    SPEED = MetricDef("speed", MetricType.ROLLING)  # x_pos / game_time (game advance rate)
    WORLD = MetricDef("world", MetricType.GAUGE)
    STAGE = MetricDef("stage", MetricType.GAUGE)
    DEATHS = MetricDef("deaths", MetricType.COUNTER)
    FLAGS = MetricDef("flags", MetricType.COUNTER)
    GRADS_SENT = MetricDef("grads_sent", MetricType.COUNTER)
    
    # Snapshot metrics
    SNAPSHOT_SAVES = MetricDef("snapshot_saves", MetricType.COUNTER)
    SNAPSHOT_RESTORES = MetricDef("snapshot_restores", MetricType.COUNTER)
    
    @classmethod
    def definitions(cls) -> list[MetricDef]:
        """Return list of all common metric definitions."""
        return [
            cls.EPISODES,
            cls.STEPS,
            cls.REWARD,
            cls.EPISODE_REWARD,
            cls.EPISODE_LENGTH,
            cls.EPSILON,
            cls.STEPS_PER_SEC,
            cls.X_POS,
            cls.BEST_X,
            cls.BEST_X_EVER,
            cls.GAME_TIME,
            cls.SPEED,
            cls.WORLD,
            cls.STAGE,
            cls.DEATHS,
            cls.FLAGS,
            cls.GRADS_SENT,
            cls.SNAPSHOT_SAVES,
            cls.SNAPSHOT_RESTORES,
        ]


# =============================================================================
# DDQN-Specific Metrics
# =============================================================================


class DDQNMetrics(CommonMetrics):
    """DDQN-specific metrics (extends CommonMetrics)."""
    
    LOSS = MetricDef("loss", MetricType.ROLLING)
    Q_MEAN = MetricDef("q_mean", MetricType.ROLLING)
    Q_MAX = MetricDef("q_max", MetricType.GAUGE)
    TD_ERROR = MetricDef("td_error", MetricType.ROLLING)
    GRAD_NORM = MetricDef("grad_norm", MetricType.ROLLING)
    BUFFER_SIZE = MetricDef("buffer_size", MetricType.GAUGE)
    PER_BETA = MetricDef("per_beta", MetricType.GAUGE)
    
    @classmethod
    def definitions(cls) -> list[MetricDef]:
        """Return list of all DDQN metric definitions."""
        return super().definitions() + [
            cls.LOSS,
            cls.Q_MEAN,
            cls.Q_MAX,
            cls.TD_ERROR,
            cls.GRAD_NORM,
            cls.BUFFER_SIZE,
            cls.PER_BETA,
        ]


# =============================================================================
# Dreamer-Specific Metrics
# =============================================================================


class DreamerMetrics(CommonMetrics):
    """Dreamer-specific metrics (extends CommonMetrics)."""
    
    WORLD_LOSS = MetricDef("world_loss", MetricType.ROLLING)
    ACTOR_LOSS = MetricDef("actor_loss", MetricType.ROLLING)
    CRITIC_LOSS = MetricDef("critic_loss", MetricType.ROLLING)
    KL_DIV = MetricDef("kl_div", MetricType.ROLLING)
    IMAGINATION_REWARD = MetricDef("imagination_reward", MetricType.ROLLING)
    ENTROPY = MetricDef("entropy", MetricType.ROLLING)
    
    @classmethod
    def definitions(cls) -> list[MetricDef]:
        """Return list of all Dreamer metric definitions."""
        return super().definitions() + [
            cls.WORLD_LOSS,
            cls.ACTOR_LOSS,
            cls.CRITIC_LOSS,
            cls.KL_DIV,
            cls.IMAGINATION_REWARD,
            cls.ENTROPY,
        ]


# =============================================================================
# Coordinator Metrics
# =============================================================================


class CoordinatorMetrics:
    """Metrics specific to the training coordinator."""
    
    UPDATE_COUNT = MetricDef("update_count", MetricType.COUNTER)
    GRADS_PER_SEC = MetricDef("grads_per_sec", MetricType.GAUGE)
    LEARNING_RATE = MetricDef("learning_rate", MetricType.GAUGE)
    TOTAL_STEPS = MetricDef("total_steps", MetricType.COUNTER)
    WEIGHT_VERSION = MetricDef("weight_version", MetricType.COUNTER)
    LOSS = MetricDef("loss", MetricType.ROLLING)
    GRAD_NORM = MetricDef("grad_norm", MetricType.ROLLING)
    
    # Aggregated metrics from workers (for graphs/UI)
    AVG_REWARD = MetricDef("avg_reward", MetricType.GAUGE)
    AVG_SPEED = MetricDef("avg_speed", MetricType.GAUGE)
    AVG_LOSS = MetricDef("avg_loss", MetricType.GAUGE)
    Q_MEAN = MetricDef("q_mean", MetricType.GAUGE)
    TD_ERROR = MetricDef("td_error", MetricType.GAUGE)
    TOTAL_EPISODES = MetricDef("total_episodes", MetricType.COUNTER)
    
    @classmethod
    def definitions(cls) -> list[MetricDef]:
        """Return list of all coordinator metric definitions."""
        return [
            cls.UPDATE_COUNT,
            cls.GRADS_PER_SEC,
            cls.LEARNING_RATE,
            cls.TOTAL_STEPS,
            cls.WEIGHT_VERSION,
            cls.LOSS,
            cls.GRAD_NORM,
            cls.AVG_REWARD,
            cls.AVG_SPEED,
            cls.AVG_LOSS,
            cls.Q_MEAN,
            cls.TD_ERROR,
            cls.TOTAL_EPISODES,
        ]
