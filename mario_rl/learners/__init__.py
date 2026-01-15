"""Learner implementations for distributed RL training."""

from mario_rl.learners.base import Learner
from mario_rl.learners.ddqn import DDQNLearner
from mario_rl.learners.dreamer import DreamerLearner
from mario_rl.learners.muzero import MuZeroLearner
from mario_rl.learners.muzero import MuZeroTrajectory
from mario_rl.learners.muzero import MuZeroTrajectoryCollector
from mario_rl.learners.muzero import compute_value_target
from mario_rl.learners.muzero import run_mcts
from mario_rl.learners.muzero import scale_gradient

__all__ = [
    "DDQNLearner",
    "DreamerLearner",
    "Learner",
    "MuZeroLearner",
    "MuZeroTrajectory",
    "MuZeroTrajectoryCollector",
    "compute_value_target",
    "run_mcts",
    "scale_gradient",
]
