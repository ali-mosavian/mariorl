"""
Mario RL - Reinforcement Learning for Super Mario Bros with World Models

A modular RL framework featuring:
- World model with latent representations
- Distributed training with multiple workers
- Dueling Double DQN architecture
- Prioritized experience replay
"""

__version__ = "0.1.0"

from mario_rl.agent.world_model import MarioWorldModel
from mario_rl.agent.world_model import LatentDDQN
from mario_rl.agent.neural import DuelingDDQNNet

__all__ = [
    "MarioWorldModel",
    "LatentDDQN",
    "DuelingDDQNNet",
]

