"""MuZero Learner for MuZero-style training with MCTS.

Training in MuZero works by:
1. Collecting trajectories using MCTS for action selection
2. Storing trajectories with MCTS targets in replay buffer
3. Training by unrolling dynamics K steps and matching predictions

Naming convention:
- s: observation/state (from environment)
- z: latent state (internal representation)
- a: action
- r: reward
- π: policy
- v: value

The learner handles:
- Unrolling the dynamics model
- Computing policy, value, and reward losses
- Target network updates
"""

from dataclasses import dataclass
from dataclasses import field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from mario_rl.mcts import LatentNode
from mario_rl.mcts import MCTSNode
from mario_rl.models.muzero import MuZeroConfig
from mario_rl.models.muzero import MuZeroModel
from mario_rl.models.muzero import info_nce_loss


@dataclass(frozen=True)
class MuZeroTrajectory:
    """A trajectory segment for MuZero training.

    Stores K+1 steps of data starting from an observation.

    Attributes:
        obs: Initial observation (C, H, W)
        actions: Sequence of K actions taken
        rewards: Sequence of K rewards received
        policies: MCTS policy targets at each step (K+1, num_actions)
        values: Value targets at each step (K+1,)
        dones: Whether episode ended at each step (K+1,)
    """

    obs: np.ndarray  # (C, H, W)
    actions: np.ndarray  # (K,)
    rewards: np.ndarray  # (K,)
    policies: np.ndarray  # (K+1, num_actions)
    values: np.ndarray  # (K+1,)
    dones: np.ndarray  # (K+1,)


@dataclass
class MuZeroLearner:
    """Learner for MuZero training with latent grounding.

    Computes MuZero loss by unrolling dynamics model K steps:
    - Policy loss: Cross-entropy between predicted and MCTS policies
    - Value loss: MSE between predicted and target values
    - Reward loss: MSE between predicted and actual rewards
    - Consistency loss: SimSiam-style alignment (predicted z ≈ encoded z)
    - Contrastive loss: InfoNCE to prevent representation collapse

    The loss is computed for the initial state plus K unrolled steps.
    """

    model: MuZeroModel
    unroll_steps: int = 5
    value_loss_weight: float = 1.0
    reward_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    consistency_loss_weight: float | None = None  # Uses config if None
    contrastive_loss_weight: float | None = None  # Uses config if None

    # Internal state
    _device: torch.device = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize device from model."""
        self._device = next(self.model.parameters()).device
        # Use config weights if not overridden
        if self.consistency_loss_weight is None:
            self.consistency_loss_weight = self.model.config.consistency_weight
        if self.contrastive_loss_weight is None:
            self.contrastive_loss_weight = self.model.config.contrastive_weight

    def compute_trajectory_loss(
        self,
        s: Tensor,
        actions: Tensor,
        rewards: Tensor,
        target_policies: Tensor,
        target_values: Tensor,
        next_states: Tensor | None = None,
        dones: Tensor | None = None,
        weights: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Compute MuZero loss for a batch of trajectory segments.

        This is for full MuZero training with MCTS policy targets.
        For single-step training (Learner protocol), use compute_loss().

        Args:
            s: Initial observations (N, C, H, W)
            actions: Action sequences (N, K)
            rewards: Reward sequences (N, K)
            target_policies: MCTS policy targets (N, K+1, num_actions)
            target_values: Value targets (N, K+1)
            next_states: Sequence of next observations (N, K, C, H, W) for grounding
            dones: Episode done flags (N, K+1) - optional, for masking
            weights: Importance sampling weights (N,) - optional for PER

        Returns:
            loss: Scalar loss tensor
            metrics: Dict with training metrics
        """
        batch_size = s.shape[0]
        K = actions.shape[1]

        if weights is None:
            weights = torch.ones(batch_size, device=s.device)

        # Track losses per step for proper averaging
        policy_losses = []
        value_losses = []
        reward_losses = []
        consistency_losses = []
        contrastive_losses = []

        # Initial inference: s → z
        z, policy_logits, v = self.model.initial_inference(s)

        # Loss at root (step 0)
        policy_loss_0 = F.cross_entropy(
            policy_logits,
            target_policies[:, 0],
            reduction="none",
        )
        value_loss_0 = F.mse_loss(v, target_values[:, 0], reduction="none")

        policy_losses.append(policy_loss_0)
        value_losses.append(value_loss_0)

        # Unroll K steps in latent space
        for k in range(K):
            z_pred, r, policy_logits, v = self.model.recurrent_inference(
                z, actions[:, k]
            )

            # Policy loss
            policy_loss_k = F.cross_entropy(
                policy_logits,
                target_policies[:, k + 1],
                reduction="none",
            )
            policy_losses.append(policy_loss_k)

            # Value loss
            value_loss_k = F.mse_loss(v, target_values[:, k + 1], reduction="none")
            value_losses.append(value_loss_k)

            # Reward loss
            reward_loss_k = F.mse_loss(r, rewards[:, k], reduction="none")
            reward_losses.append(reward_loss_k)

            # Latent grounding losses (if next_states provided)
            if next_states is not None:
                # Encode actual next observation: s_{k+1} → z_target
                with torch.no_grad():
                    z_target = self.model.encode(next_states[:, k])
                    z_target = z_target.detach()  # Stop gradient on target

                # Consistency loss: pred(z_pred) should match z_target
                z_pred_transformed = self.model.predict(z_pred)
                consistency_k = F.mse_loss(
                    z_pred_transformed,
                    z_target,
                    reduction="none",
                ).mean(dim=-1)
                consistency_losses.append(consistency_k)

                # Contrastive loss: InfoNCE with in-batch negatives
                e_pred = self.model.project(z_pred)
                e_target = self.model.project(z_target)
                contrastive_k = info_nce_loss(
                    e_pred,
                    e_target,
                    temperature=self.model.config.contrastive_temp,
                )
                contrastive_losses.append(contrastive_k)

            # Update z for next iteration
            z = z_pred

        # Stack and average losses
        policy_loss = torch.stack(policy_losses, dim=1).mean(dim=1)  # (N,)
        value_loss = torch.stack(value_losses, dim=1).mean(dim=1)  # (N,)
        reward_loss = torch.stack(reward_losses, dim=1).mean(dim=1)  # (N,)

        # Weighted combination of task losses
        total_loss = (
            self.policy_loss_weight * policy_loss
            + self.value_loss_weight * value_loss
            + self.reward_loss_weight * reward_loss
        )

        # Add grounding losses if computed
        consistency_loss_mean = 0.0
        contrastive_loss_mean = 0.0

        if consistency_losses:
            consistency_loss = torch.stack(consistency_losses, dim=1).mean(dim=1)
            total_loss = total_loss + self.consistency_loss_weight * consistency_loss
            consistency_loss_mean = (weights * consistency_loss).mean().item()

        if contrastive_losses:
            contrastive_loss = torch.stack(contrastive_losses, dim=1).mean(dim=1)
            total_loss = total_loss + self.contrastive_loss_weight * contrastive_loss
            contrastive_loss_mean = (weights * contrastive_loss).mean().item()

        # Apply importance weights and average over batch
        loss = (weights * total_loss).mean()

        # Metrics
        metrics = {
            "loss": loss.item(),
            "policy_loss": (weights * policy_loss).mean().item(),
            "value_loss": (weights * value_loss).mean().item(),
            "reward_loss": (weights * reward_loss).mean().item(),
            "consistency_loss": consistency_loss_mean,
            "contrastive_loss": contrastive_loss_mean,
            "value_pred_mean": v.mean().item(),
            "value_target_mean": target_values[:, -1].mean().item(),
        }

        return loss, metrics

    def compute_loss(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        dones: Tensor,
        weights: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Compute loss from single-step transitions (Learner protocol).

        This is the standard interface for distributed training.
        Uses 1-step dynamics unroll with latent grounding losses.

        For full MuZero training with MCTS targets, use compute_trajectory_loss().

        Args:
            states: Current observations (N, C, H, W)
            actions: Actions taken (N,)
            rewards: Rewards received (N,)
            next_states: Next observations (N, C, H, W)
            dones: Episode done flags (N,)
            weights: Importance weights (N,)

        Returns:
            loss: Scalar loss
            metrics: Training metrics
        """
        # Get initial predictions: s → z
        z, policy_logits, v = self.model.initial_inference(states)

        # Get next latent predictions (1-step unroll): z, a → z', r
        z_pred, r_pred, _, v_next = self.model.recurrent_inference(z, actions)

        # Get target value from target network
        with torch.no_grad():
            _, _, v_target = self.model.target.initial_inference(next_states)
            # Bootstrap target: r + γ * V(s') * (1 - done)
            value_target = rewards + self.model.config.discount * v_target * (
                1.0 - dones.float()
            )

        # Task losses
        value_loss = F.mse_loss(v, value_target, reduction="none")
        reward_loss = F.mse_loss(r_pred, rewards, reduction="none")

        # No policy target in single-step mode, use entropy regularization
        policy = F.softmax(policy_logits, dim=-1)
        log_policy = F.log_softmax(policy_logits, dim=-1)
        entropy = -(policy * log_policy).sum(dim=-1)
        policy_loss = -entropy  # Maximize entropy when no MCTS target

        # Latent grounding losses
        with torch.no_grad():
            z_target = self.model.encode(next_states).detach()

        # Consistency loss: pred(z_pred) should match z_target
        z_pred_transformed = self.model.predict(z_pred)
        consistency_loss = F.mse_loss(
            z_pred_transformed,
            z_target,
            reduction="none",
        ).mean(dim=-1)

        # Contrastive loss: InfoNCE with in-batch negatives
        e_pred = self.model.project(z_pred)
        e_target = self.model.project(z_target)
        contrastive_loss = info_nce_loss(
            e_pred,
            e_target,
            temperature=self.model.config.contrastive_temp,
        )

        # Combine losses
        total_loss = (
            value_loss
            + reward_loss
            + 0.01 * policy_loss
            + self.consistency_loss_weight * consistency_loss
            + self.contrastive_loss_weight * contrastive_loss
        )

        if weights is not None:
            loss = (weights * total_loss).mean()
        else:
            loss = total_loss.mean()

        metrics = {
            "loss": loss.item(),
            "value_loss": value_loss.mean().item(),
            "reward_loss": reward_loss.mean().item(),
            "consistency_loss": consistency_loss.mean().item(),
            "contrastive_loss": contrastive_loss.mean().item(),
            "entropy": entropy.mean().item(),
            "value_pred_mean": v.mean().item(),
            "value_target_mean": value_target.mean().item(),
        }

        return loss, metrics

    def update_targets(self, tau: float = 0.005) -> None:
        """Update target network weights.

        Args:
            tau: Interpolation coefficient (1.0 = hard copy, <1.0 = soft update)
        """
        if tau >= 1.0:
            self.model.sync_target()
        else:
            self.model.soft_update(tau)


def run_mcts(
    model: MuZeroModel,
    s: Tensor,
    num_simulations: int = 50,
    exploration: float = 1.25,
    temperature: float = 1.0,
    add_noise: bool = True,
    noise_alpha: float = 0.3,
    noise_frac: float = 0.25,
) -> tuple[np.ndarray, float, MCTSNode]:
    """Run MCTS from an observation to get action policy and value.

    Args:
        model: MuZero model for inference
        s: Single observation (1, C, H, W) or (C, H, W)
        num_simulations: Number of MCTS simulations
        exploration: PUCT exploration constant
        temperature: Temperature for final action selection
        add_noise: Whether to add Dirichlet noise at root
        noise_alpha: Dirichlet alpha parameter
        noise_frac: Fraction of noise to mix with prior

    Returns:
        policy: Action probabilities from visit counts (num_actions,)
        value: Root value estimate
        root: Root node of MCTS tree
    """
    device = next(model.parameters()).device

    # Ensure batch dimension
    if s.dim() == 3:
        s = s.unsqueeze(0)
    s = s.to(device)

    # Initial inference for root: s → z, π, v
    with torch.no_grad():
        z, policy_logits, v = model.initial_inference(s)
        root_policy = F.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()
        root_value = v.item()
        z_root = z.squeeze(0)  # (latent_dim,)

    num_actions = model.num_actions

    # Add Dirichlet noise at root for exploration
    if add_noise:
        noise = np.random.dirichlet([noise_alpha] * num_actions)
        root_policy = (1 - noise_frac) * root_policy + noise_frac * noise

    # Create root node (stores latent z, not observation s)
    root: MCTSNode[Tensor] = MCTSNode(
        state=z_root,
        obs=s.squeeze(0).cpu().numpy(),
        visits=1,
        total_value=root_value,
    )

    # Expand root with all actions
    for a in range(num_actions):
        with torch.no_grad():
            z_next, r, policy_logits, v = model.recurrent_inference(
                z_root.unsqueeze(0), torch.tensor([a], device=device)
            )

        child: MCTSNode[Tensor] = MCTSNode(
            state=z_next.squeeze(0),
            obs=root.obs,  # Obs not needed for latent nodes
            parent=root,
            action=a,
            prior=root_policy[a],
            reward=r.item(),
        )
        root.children.append(child)

    # Run simulations
    for _ in range(num_simulations):
        node = root

        # Selection: traverse tree using PUCT until we reach a leaf
        while node.children:
            node = node.best_child(exploration=exploration, use_puct=True)

        # Expansion: expand all actions from leaf
        if not node.terminal:
            with torch.no_grad():
                # Get policy from current node's latent state z
                policy_logits, v_leaf = model.online.prediction(node.state.unsqueeze(0))
                leaf_policy = F.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()
                v_leaf = v_leaf.item()

            for a in range(num_actions):
                with torch.no_grad():
                    z_next, r, _, _ = model.recurrent_inference(
                        node.state.unsqueeze(0),
                        torch.tensor([a], device=device),
                    )

                child = MCTSNode(
                    state=z_next.squeeze(0),
                    obs=node.obs,
                    parent=node,
                    action=a,
                    prior=leaf_policy[a],
                    reward=r.item(),
                )
                node.children.append(child)

            # Use leaf value for backprop
            value_to_backprop = v_leaf
        else:
            value_to_backprop = 0.0

        # Backpropagation
        discount = model.config.discount
        current = node
        cumulative_reward = 0.0

        while current is not None:
            current.visits += 1
            # Add discounted value
            current.total_value += value_to_backprop + cumulative_reward
            # Accumulate discounted reward going up
            cumulative_reward = current.reward + discount * cumulative_reward
            current = current.parent

    # Get policy from visit counts
    policy = root.get_policy_target(num_actions, temperature)

    return policy, root.value, root


def compute_value_target(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    discount: float,
    td_steps: int,
) -> list[float]:
    """Compute n-step value targets for MuZero training.

    Uses n-step bootstrapping:
        V_target(t) = r_t + γ*r_{t+1} + ... + γ^{n-1}*r_{t+n-1} + γ^n * V(s_{t+n})

    Args:
        rewards: List of rewards [r_0, r_1, ..., r_T]
        values: List of MCTS values [v_0, v_1, ..., v_T]
        dones: List of done flags
        discount: Discount factor γ
        td_steps: Number of steps for bootstrapping

    Returns:
        targets: Value targets for each timestep
    """
    T = len(rewards)
    targets = []

    for t in range(T):
        # Compute n-step return
        value_target = 0.0
        discount_power = 1.0

        for n in range(td_steps):
            if t + n >= T:
                break
            if dones[t + n]:
                # Episode ended, no more future rewards
                value_target += discount_power * rewards[t + n]
                break
            value_target += discount_power * rewards[t + n]
            discount_power *= discount

        # Bootstrap with value estimate if we didn't hit done
        if t + td_steps < T and not any(dones[t : t + td_steps]):
            value_target += discount_power * values[t + td_steps]

        targets.append(value_target)

    return targets
