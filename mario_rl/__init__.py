"""
Mario RL - Reinforcement Learning for Super Mario Bros with World Models

A modular RL framework featuring:
- World model with latent representations
- Distributed training with multiple workers
- Dreamer and DDQN architectures
"""

__version__ = "0.1.0"

from mario_rl.agent.world_model import LatentDDQN
from mario_rl.agent.world_model import MarioWorldModel

__all__ = [
    "MarioWorldModel",
    "LatentDDQN",
]
