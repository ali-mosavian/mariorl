"""
Concrete adapters for DDQN and Dreamer networks.

These adapters implement the MCTS protocols to allow any supported
network architecture to be used with the MCTSExplorer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

import torch
import numpy as np
import torch.nn.functional as F

from mario_rl.agent.ddqn_net import DoubleDQN

if TYPE_CHECKING:
    from mario_rl.agent.world_model import DreamerDDQN


@dataclass
class DDQNAdapter:
    """
    Adapter for using DDQN with MCTS.

    Implements PolicyAdapter and ValueAdapter protocols.
    DDQN doesn't have a world model, so WorldModelAdapter is not implemented.

    Attributes:
        net: DoubleDQN network (uses online network for inference)
        device: Torch device for inference
        temperature: Softmax temperature for action probabilities
            Higher = more uniform, lower = more greedy
    """

    net: DoubleDQN
    device: torch.device
    temperature: float = 1.0

    def get_action(self, state: np.ndarray) -> int:
        """
        Get greedy action from Q-values.

        Args:
            state: Observation (C, H, W) in [0, 1] range

        Returns:
            Action with highest Q-value
        """
        with torch.no_grad():
            state_t = self._prepare_state(state)
            q_values = self.net(state_t)
            return int(q_values.argmax(dim=1).item())

    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        """
        Get softmax probabilities over Q-values.

        Used for policy-guided MCTS expansion.

        Args:
            state: Observation

        Returns:
            Action probability distribution
        """
        with torch.no_grad():
            state_t = self._prepare_state(state)
            q_values = self.net(state_t)
            probs = F.softmax(q_values / self.temperature, dim=1)
            return probs.cpu().numpy().flatten()

    def get_value(self, state: np.ndarray) -> float:
        """
        Get max Q-value as state value estimate.

        Args:
            state: Observation

        Returns:
            Maximum Q-value (estimated state value)
        """
        with torch.no_grad():
            state_t = self._prepare_state(state)
            q_values = self.net(state_t)
            return float(q_values.max(dim=1).values.item())

    def _prepare_state(self, state: np.ndarray) -> torch.Tensor:
        """Convert numpy state to batched tensor on device."""
        if state.ndim == 3:
            state = state[np.newaxis, ...]  # Add batch dim
        return torch.from_numpy(state).float().to(self.device)


@dataclass
class DreamerAdapter:
    """
    Adapter for using Dreamer/DreamerDDQN with MCTS.

    Implements PolicyAdapter, ValueAdapter, and WorldModelAdapter protocols.
    The world model allows for imagined MCTS without the real environment.

    Attributes:
        net: DreamerDDQN network
        device: Torch device for inference
        temperature: Softmax temperature for action probabilities
    """

    net: "DreamerDDQN"  # Forward reference to avoid circular import
    device: torch.device
    temperature: float = 1.0

    def get_action(self, state: np.ndarray) -> int:
        """
        Get greedy action from Q-network.

        Args:
            state: Observation (C, H, W)

        Returns:
            Best action
        """
        with torch.no_grad():
            state_t = self._prepare_state(state)
            q_values = self.net(state_t)
            return int(q_values.argmax(dim=1).item())

    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        """
        Get action probability distribution.

        Args:
            state: Observation

        Returns:
            Action probabilities
        """
        with torch.no_grad():
            state_t = self._prepare_state(state)
            q_values = self.net(state_t)
            probs = F.softmax(q_values / self.temperature, dim=1)
            return probs.cpu().numpy().flatten()

    def get_value(self, state: np.ndarray) -> float:
        """
        Get value estimate (max Q-value).

        Args:
            state: Observation

        Returns:
            Estimated state value
        """
        with torch.no_grad():
            state_t = self._prepare_state(state)
            q_values = self.net(state_t)
            return float(q_values.max(dim=1).values.item())

    def encode(self, obs: np.ndarray) -> torch.Tensor:
        """
        Encode observation to latent space.

        Args:
            obs: Observation (C, H, W)

        Returns:
            Latent tensor
        """
        with torch.no_grad():
            obs_t = self._prepare_state(obs)
            # Access encoder through the network
            encoder = getattr(self.net, "encoder", None)
            if encoder is not None and callable(encoder):
                result: torch.Tensor = encoder(obs_t)
                return result
            # Fallback: use features from Q-network backbone
            online = getattr(self.net, "online", None)
            if online is not None:
                backbone = getattr(online, "backbone", None)
                if backbone is not None and callable(backbone):
                    result = backbone(obs_t)
                    return result
            raise NotImplementedError("Dreamer network needs encoder or backbone")

    def imagine_step(
        self,
        latent: torch.Tensor,
        action: int,
    ) -> tuple[torch.Tensor, float, bool]:
        """
        Predict next latent state using dynamics model.

        Args:
            latent: Current latent state
            action: Action to take

        Returns:
            Tuple of (next_latent, predicted_reward, predicted_done)
        """
        with torch.no_grad():
            # One-hot encode action
            num_actions = getattr(self.net, "num_actions", 12)
            action_t = torch.zeros(1, num_actions, device=self.device)
            action_t[0, action] = 1.0

            # Use dynamics model if available
            dynamics = getattr(self.net, "dynamics", None)
            if dynamics is not None and callable(dynamics):
                next_latent: torch.Tensor = dynamics(latent, action_t)

                # Predict reward
                reward = 0.0
                reward_head = getattr(self.net, "reward_head", None)
                if reward_head is not None and callable(reward_head):
                    reward = float(reward_head(next_latent).item())

                # Predict done
                done = False
                done_head = getattr(self.net, "done_head", None)
                if done_head is not None and callable(done_head):
                    done = bool(done_head(next_latent).sigmoid().item() > 0.5)

                return next_latent, reward, done

            raise NotImplementedError("Dreamer network needs dynamics model for imagination")

    def decode(self, latent: torch.Tensor) -> np.ndarray:
        """
        Decode latent back to observation space.

        Args:
            latent: Latent tensor

        Returns:
            Reconstructed observation
        """
        with torch.no_grad():
            decoder = getattr(self.net, "decoder", None)
            if decoder is not None and callable(decoder):
                recon: torch.Tensor = decoder(latent)
                return np.asarray(recon.cpu().numpy().squeeze())
            raise NotImplementedError("Dreamer network needs decoder")

    def _prepare_state(self, state: np.ndarray) -> torch.Tensor:
        """Convert numpy state to batched tensor on device."""
        if state.ndim == 3:
            state = state[np.newaxis, ...]
        return torch.from_numpy(state).float().to(self.device)


def create_adapter(
    net: torch.nn.Module,
    device: torch.device,
    temperature: float = 1.0,
) -> DDQNAdapter | DreamerAdapter:
    """
    Factory function to create appropriate adapter for network type.

    Args:
        net: Neural network (DoubleDQN or DreamerDDQN)
        device: Torch device
        temperature: Softmax temperature

    Returns:
        Appropriate adapter instance
    """
    # Import here to avoid circular imports
    from mario_rl.agent.world_model import DreamerDDQN

    if isinstance(net, DreamerDDQN):
        return DreamerAdapter(net=net, device=device, temperature=temperature)
    if isinstance(net, DoubleDQN):
        return DDQNAdapter(net=net, device=device, temperature=temperature)

    raise TypeError(f"Unsupported network type: {type(net)}")


@dataclass
class RandomAdapter:
    """
    Random adapter for pure MCTS without policy network.

    Used when MCTS should do pure tree search without any
    learned policy guidance. Actions are selected uniformly
    at random.

    Attributes:
        num_actions: Number of available actions
    """

    num_actions: int

    def get_action(self, state: np.ndarray) -> int:
        """Get random action."""
        return np.random.randint(self.num_actions)

    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        """Get uniform action probabilities."""
        return np.ones(self.num_actions) / self.num_actions

    def get_value(self, state: np.ndarray) -> float:
        """Return zero value (use rollout returns instead)."""
        return 0.0
