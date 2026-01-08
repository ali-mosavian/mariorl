"""Learner implementations for distributed RL training."""

from mario_rl.learners.base import Learner
from mario_rl.learners.ddqn import DDQNLearner
from mario_rl.learners.dreamer import DreamerLearner

__all__ = [
    "DDQNLearner",
    "DreamerLearner",
    "Learner",
]
