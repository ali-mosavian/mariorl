"""
MCTS (Monte Carlo Tree Search) module for enhanced exploration.

This module provides MCTS-based exploration that works with any RL algorithm
(DDQN, Dreamer) through protocol-based adapters.

Two modes of operation:
1. Real MCTS (training): Uses emulator save/restore for ground-truth exploration
2. Imagined MCTS (inference): Uses learned world model for planning without emulator

Key components:
- MCTSExplorer: Main explorer that collects transitions from all branches
- MCTSConfig: Configuration for MCTS parameters
- Adapters: DDQNAdapter, DreamerAdapter for network integration
- Protocols: PolicyAdapter, ValueAdapter, WorldModelAdapter interfaces
"""

from mario_rl.mcts.node import MCTSNode
from mario_rl.mcts.node import LatentNode
from mario_rl.mcts.config import MCTSConfig
from mario_rl.mcts.node import EmulatorNode
from mario_rl.mcts.adapters import DDQNAdapter
from mario_rl.mcts.explorer import MCTSExplorer
from mario_rl.mcts.adapters import MuZeroAdapter
from mario_rl.mcts.protocols import ValueAdapter
from mario_rl.mcts.adapters import DreamerAdapter
from mario_rl.mcts.protocols import PolicyAdapter
from mario_rl.mcts.protocols import WorldModelAdapter

__all__ = [
    "MCTSExplorer",
    "MCTSConfig",
    "MCTSNode",
    "EmulatorNode",
    "LatentNode",
    "DDQNAdapter",
    "DreamerAdapter",
    "MuZeroAdapter",
    "PolicyAdapter",
    "ValueAdapter",
    "WorldModelAdapter",
]
