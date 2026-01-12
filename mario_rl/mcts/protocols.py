"""
Protocol definitions for MCTS integration with any RL algorithm.

These protocols define the interfaces that adapters must implement
to work with the MCTSExplorer. Using protocols enables duck typing
and allows any compatible network to be used without inheritance.
"""

from typing import Protocol

import torch
import numpy as np


class PolicyAdapter(Protocol):
    """
    Protocol for policy networks usable with MCTS.

    Any class implementing these methods can be used as a policy
    for guiding MCTS action selection during rollouts.
    """

    def get_action(self, state: np.ndarray) -> int:
        """
        Get best action for state (greedy).

        Args:
            state: Observation array (C, H, W) or (N, C, H, W)

        Returns:
            Best action index
        """
        ...

    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        """
        Get action probability distribution for guided exploration.

        Used for biasing MCTS expansion toward promising actions.

        Args:
            state: Observation array

        Returns:
            Array of action probabilities (sums to 1)
        """
        ...


class ValueAdapter(Protocol):
    """
    Protocol for value estimation usable with MCTS.

    Used for leaf node evaluation instead of full rollouts,
    making MCTS much more efficient.
    """

    def get_value(self, state: np.ndarray) -> float:
        """
        Get value estimate for state (for leaf evaluation).

        Args:
            state: Observation array

        Returns:
            Estimated state value
        """
        ...


class WorldModelAdapter(Protocol):
    """
    Protocol for world models usable with imagined MCTS.

    Only Dreamer-style networks implement this - allows MCTS
    to plan in latent space without needing the real emulator.
    """

    def encode(self, obs: np.ndarray) -> torch.Tensor:
        """
        Encode observation to latent space.

        Args:
            obs: Observation array (C, H, W)

        Returns:
            Latent tensor
        """
        ...

    def imagine_step(
        self,
        latent: torch.Tensor,
        action: int,
    ) -> tuple[torch.Tensor, float, bool]:
        """
        Predict next state in latent space without environment.

        Args:
            latent: Current latent state
            action: Action to take

        Returns:
            Tuple of (next_latent, predicted_reward, predicted_done)
        """
        ...

    def decode(self, latent: torch.Tensor) -> np.ndarray:
        """
        Decode latent back to observation space.

        Optional - only needed if we want to visualize imagined states.

        Args:
            latent: Latent tensor

        Returns:
            Reconstructed observation
        """
        ...
