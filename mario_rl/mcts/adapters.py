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
from mario_rl.models.muzero import symexp

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
            num_actions = getattr(self.net, "num_actions", 7)  # SIMPLE_MOVEMENT default
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


@dataclass
class MuZeroAdapter:
    """
    Adapter for using MuZero with latent-space MCTS.

    Unlike DDQN/Dreamer adapters that work with environment-based MCTS,
    MuZero performs MCTS in the learned latent space using its dynamics model.

    This adapter provides policy/value from the learned prediction network,
    and action selection uses the full MuZero MCTS in latent space.

    Attributes:
        model: MuZeroModel (has online, target networks)
        device: Torch device for inference
        num_simulations: MCTS simulations per action selection
        temperature: Temperature for final policy selection
        add_noise: Whether to add Dirichlet noise at root
    """

    model: "MuZeroModel"  # Forward reference
    device: torch.device
    num_simulations: int = 50
    temperature: float = 1.0
    add_noise: bool = True

    def get_action(self, state: np.ndarray) -> int:
        """
        Get action using MuZero's latent-space MCTS.

        Args:
            state: Observation (C, H, W) in [0, 255] range

        Returns:
            Best action from MCTS
        """
        action, _, _ = self.get_action_with_targets(state)
        return action

    def get_action_with_targets(self, state: np.ndarray) -> tuple[int, np.ndarray, float]:
        """
        Get action AND MCTS targets for MuZero training.

        This is the primary method for data collection. Returns:
        - Selected action
        - MCTS policy target (visit count distribution)
        - MCTS value target (root value estimate)

        Args:
            state: Observation (C, H, W) in [0, 255] range

        Returns:
            action: Selected action
            policy: MCTS visit count distribution (num_actions,)
            value: Root value estimate from MCTS
        """
        # Import here to avoid circular imports
        from mario_rl.learners.muzero import run_mcts

        state_t = self._prepare_state(state)

        policy, value, root = run_mcts(
            model=self.model,
            s=state_t,
            num_simulations=self.num_simulations,
            temperature=self.temperature,
            add_noise=self.add_noise,
        )

        # Sample from policy (or take argmax if temperature is very low)
        if self.temperature < 0.1:
            action = int(np.argmax(policy))
        else:
            action = int(np.random.choice(len(policy), p=policy))

        return action, policy, value

    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        """
        Get MCTS policy from latent-space planning.

        Args:
            state: Observation

        Returns:
            Action probability distribution from MCTS visit counts
        """
        from mario_rl.learners.muzero import run_mcts

        state_t = self._prepare_state(state)

        policy, value, root = run_mcts(
            model=self.model,
            s=state_t,
            num_simulations=self.num_simulations,
            temperature=self.temperature,
            add_noise=self.add_noise,
        )

        return policy

    def get_value(self, state: np.ndarray) -> float:
        """
        Get value estimate from prediction network.

        Note: Network outputs symlog-scaled values, we convert to real space.

        Args:
            state: Observation

        Returns:
            Value estimate (in real space)
        """
        with torch.no_grad():
            state_t = self._prepare_state(state)
            _, _, value_symlog = self.model.initial_inference(state_t)
            # Convert from symlog space to real space
            return float(symexp(value_symlog).item())

    def get_action_fast(self, state: np.ndarray) -> int:
        """
        Get action without full MCTS (fast, for evaluation).

        Uses only the policy network prediction without MCTS search.

        Args:
            state: Observation

        Returns:
            Action from policy network
        """
        with torch.no_grad():
            state_t = self._prepare_state(state)
            _, policy_logits, _ = self.model.initial_inference(state_t)
            policy = F.softmax(policy_logits / self.temperature, dim=-1)
            return int(torch.multinomial(policy, num_samples=1).item())

    def _prepare_state(self, state: np.ndarray) -> torch.Tensor:
        """Convert numpy state to batched tensor on device."""
        if state.ndim == 3:
            state = state[np.newaxis, ...]
        return torch.from_numpy(state).float().to(self.device)


@dataclass 
class MarioRolloutAdapter:
    """
    Mario-specific adapter for MCTS rollouts.

    Uses domain knowledge that Mario should mostly go RIGHT.
    This makes rollouts meaningful - RIGHT actions will accumulate
    more reward than LEFT/NOOP, allowing MCTS to distinguish them.

    Action distribution during rollouts:
    - 70% RIGHT variants (RIGHT, R+A, R+B, R+A+B)
    - 20% JUMP (A) for obstacles
    - 10% other actions

    Attributes:
        num_actions: Number of available actions (7 for SIMPLE_MOVEMENT)
    """

    num_actions: int = 7  # SIMPLE_MOVEMENT

    # Mario action indices
    RIGHT_ACTIONS: tuple = (1, 2, 3, 4)  # RIGHT, R+A, R+B, R+A+B
    JUMP_ACTION: int = 5  # A (jump)

    def get_action(self, state: np.ndarray) -> int:
        """Get RIGHT-biased action for Mario rollouts."""
        r = np.random.random()
        if r < 0.70:
            # 70% RIGHT variants
            return np.random.choice(self.RIGHT_ACTIONS)
        elif r < 0.90:
            # 20% jump
            return self.JUMP_ACTION
        else:
            # 10% random (exploration)
            return np.random.randint(self.num_actions)

    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        """Get RIGHT-biased action probabilities."""
        probs = np.ones(self.num_actions) * 0.01  # Small base probability
        
        # 70% for RIGHT variants
        for action in self.RIGHT_ACTIONS:
            probs[action] = 0.70 / len(self.RIGHT_ACTIONS)
        
        # 20% for jump
        probs[self.JUMP_ACTION] = 0.20
        
        # Normalize
        return probs / probs.sum()

    def get_value(self, state: np.ndarray) -> float:
        """Return zero value (use rollout returns)."""
        return 0.0
