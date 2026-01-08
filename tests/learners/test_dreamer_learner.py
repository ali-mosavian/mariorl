"""Tests for DreamerLearner behavior.

These tests verify the DreamerLearner correctly:
- Implements the Learner protocol
- Computes world model loss (reconstruction, dynamics, reward)
- Computes actor-critic loss on imagined trajectories
- Returns proper training metrics
"""

from dataclasses import dataclass

import pytest
import torch
from torch import Tensor

from mario_rl.learners import Learner
from mario_rl.models import DreamerModel


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class LearnerTestConfig:
    """Test configuration for DreamerLearner."""

    input_shape: tuple[int, int, int] = (4, 64, 64)
    num_actions: int = 12
    latent_dim: int = 128
    gamma: float = 0.99
    lambda_gae: float = 0.95
    imagination_horizon: int = 15
    batch_size: int = 8


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config() -> LearnerTestConfig:
    """Default test configuration."""
    return LearnerTestConfig()


@pytest.fixture
def dreamer_model(config: LearnerTestConfig) -> DreamerModel:
    """Create a DreamerModel for testing."""
    return DreamerModel(
        input_shape=config.input_shape,
        num_actions=config.num_actions,
        latent_dim=config.latent_dim,
    )


@pytest.fixture
def dreamer_learner(dreamer_model: DreamerModel, config: LearnerTestConfig):
    """Create a DreamerLearner for testing."""
    from mario_rl.learners.dreamer import DreamerLearner

    return DreamerLearner(
        model=dreamer_model,
        gamma=config.gamma,
        lambda_gae=config.lambda_gae,
        imagination_horizon=config.imagination_horizon,
    )


@pytest.fixture
def sample_batch(config: LearnerTestConfig) -> dict[str, Tensor]:
    """Create a sample transition batch."""
    return {
        "states": torch.randn(config.batch_size, *config.input_shape),
        "actions": torch.randint(0, config.num_actions, (config.batch_size,)),
        "rewards": torch.randn(config.batch_size),
        "next_states": torch.randn(config.batch_size, *config.input_shape),
        "dones": torch.zeros(config.batch_size),
    }


# =============================================================================
# Protocol Conformance Tests
# =============================================================================


def test_dreamer_learner_implements_learner_protocol(dreamer_learner) -> None:
    """DreamerLearner should satisfy the Learner protocol."""
    assert isinstance(dreamer_learner, Learner)


def test_dreamer_learner_has_model_attribute(
    dreamer_learner, dreamer_model: DreamerModel
) -> None:
    """DreamerLearner must expose its model as an attribute."""
    assert hasattr(dreamer_learner, "model")
    assert dreamer_learner.model is dreamer_model


# =============================================================================
# Loss Computation Tests
# =============================================================================


def test_compute_loss_returns_scalar_tensor(
    dreamer_learner, sample_batch: dict[str, Tensor]
) -> None:
    """compute_loss should return a scalar loss tensor."""
    loss, _ = dreamer_learner.compute_loss(**sample_batch)

    assert isinstance(loss, Tensor)
    assert loss.dim() == 0  # Scalar
    assert loss.dtype == torch.float32


def test_compute_loss_returns_metrics_dict(
    dreamer_learner, sample_batch: dict[str, Tensor]
) -> None:
    """compute_loss should return a metrics dictionary."""
    _, metrics = dreamer_learner.compute_loss(**sample_batch)

    assert isinstance(metrics, dict)
    assert "loss" in metrics


def test_compute_loss_loss_is_finite(
    dreamer_learner, sample_batch: dict[str, Tensor]
) -> None:
    """Loss should be finite (not NaN or Inf)."""
    loss, _ = dreamer_learner.compute_loss(**sample_batch)

    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_compute_loss_requires_grad(
    dreamer_learner, sample_batch: dict[str, Tensor]
) -> None:
    """Loss tensor should require gradients for backprop."""
    loss, _ = dreamer_learner.compute_loss(**sample_batch)

    assert loss.requires_grad


def test_compute_loss_allows_backprop(
    dreamer_learner, sample_batch: dict[str, Tensor]
) -> None:
    """Backpropagation should compute gradients for model parameters."""
    dreamer_learner.model.zero_grad()

    loss, _ = dreamer_learner.compute_loss(**sample_batch)
    loss.backward()

    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in dreamer_learner.model.parameters()
    )
    assert has_grad


# =============================================================================
# World Model Loss Tests
# =============================================================================


def test_compute_world_model_loss(
    dreamer_learner, sample_batch: dict[str, Tensor]
) -> None:
    """Should be able to compute world model loss separately."""
    wm_loss, wm_metrics = dreamer_learner.compute_world_model_loss(
        states=sample_batch["states"],
        actions=sample_batch["actions"],
        rewards=sample_batch["rewards"],
        next_states=sample_batch["next_states"],
    )

    assert isinstance(wm_loss, Tensor)
    assert wm_loss.dim() == 0
    assert not torch.isnan(wm_loss)


def test_world_model_loss_includes_dynamics(
    dreamer_learner, sample_batch: dict[str, Tensor]
) -> None:
    """World model loss metrics should include dynamics loss."""
    _, metrics = dreamer_learner.compute_world_model_loss(
        states=sample_batch["states"],
        actions=sample_batch["actions"],
        rewards=sample_batch["rewards"],
        next_states=sample_batch["next_states"],
    )

    assert "dynamics_loss" in metrics


def test_world_model_loss_includes_reward(
    dreamer_learner, sample_batch: dict[str, Tensor]
) -> None:
    """World model loss metrics should include reward prediction loss."""
    _, metrics = dreamer_learner.compute_world_model_loss(
        states=sample_batch["states"],
        actions=sample_batch["actions"],
        rewards=sample_batch["rewards"],
        next_states=sample_batch["next_states"],
    )

    assert "reward_loss" in metrics


# =============================================================================
# Behavior Loss Tests
# =============================================================================


def test_compute_behavior_loss(dreamer_learner, config: LearnerTestConfig) -> None:
    """Should be able to compute behavior loss on imagined trajectories."""
    z_start = torch.randn(config.batch_size, config.latent_dim)

    behavior_loss, metrics = dreamer_learner.compute_behavior_loss(z_start)

    assert isinstance(behavior_loss, Tensor)
    assert behavior_loss.dim() == 0
    assert not torch.isnan(behavior_loss)


def test_behavior_loss_includes_actor_loss(
    dreamer_learner, config: LearnerTestConfig
) -> None:
    """Behavior loss metrics should include actor loss."""
    z_start = torch.randn(config.batch_size, config.latent_dim)

    _, metrics = dreamer_learner.compute_behavior_loss(z_start)

    assert "actor_loss" in metrics


def test_behavior_loss_includes_critic_loss(
    dreamer_learner, config: LearnerTestConfig
) -> None:
    """Behavior loss metrics should include critic loss."""
    z_start = torch.randn(config.batch_size, config.latent_dim)

    _, metrics = dreamer_learner.compute_behavior_loss(z_start)

    assert "critic_loss" in metrics


def test_behavior_loss_trains_on_imagination(
    dreamer_learner, config: LearnerTestConfig
) -> None:
    """Behavior loss should be computed on imagined trajectories."""
    z_start = torch.randn(config.batch_size, config.latent_dim)

    # Compute loss - internally this should use imagine_trajectory
    loss, metrics = dreamer_learner.compute_behavior_loss(z_start)

    # Should have computed returns over imagination horizon
    assert loss.requires_grad


# =============================================================================
# Target Update Tests
# =============================================================================


def test_update_targets_is_callable(dreamer_learner) -> None:
    """update_targets should be callable (may be no-op for Dreamer)."""
    # Dreamer may not have separate target networks
    dreamer_learner.update_targets(tau=0.005)
    dreamer_learner.update_targets(tau=1.0)
    # Should not raise


# =============================================================================
# Metrics Tests
# =============================================================================


def test_metrics_include_world_model_metrics(
    dreamer_learner, sample_batch: dict[str, Tensor]
) -> None:
    """Total metrics should include world model components."""
    _, metrics = dreamer_learner.compute_loss(**sample_batch)

    # Should have dynamics and reward losses
    assert any("dynamics" in k or "reward" in k for k in metrics.keys())


def test_metrics_include_behavior_metrics(
    dreamer_learner, sample_batch: dict[str, Tensor]
) -> None:
    """Total metrics should include actor-critic components."""
    _, metrics = dreamer_learner.compute_loss(**sample_batch)

    # Should have actor and critic losses
    assert any("actor" in k or "critic" in k for k in metrics.keys())


# =============================================================================
# Edge Cases
# =============================================================================


def test_single_sample_batch(dreamer_learner, config: LearnerTestConfig) -> None:
    """Learner should handle batch size of 1."""
    states = torch.randn(1, *config.input_shape)
    actions = torch.randint(0, config.num_actions, (1,))
    rewards = torch.randn(1)
    next_states = torch.randn(1, *config.input_shape)
    dones = torch.zeros(1)

    loss, metrics = dreamer_learner.compute_loss(
        states, actions, rewards, next_states, dones
    )

    assert loss.dim() == 0
    assert isinstance(metrics, dict)


def test_all_done_batch(dreamer_learner, config: LearnerTestConfig) -> None:
    """Learner should handle batch where all episodes are done."""
    states = torch.randn(4, *config.input_shape)
    actions = torch.randint(0, config.num_actions, (4,))
    rewards = torch.randn(4)
    next_states = torch.randn(4, *config.input_shape)
    dones = torch.ones(4)

    loss, _ = dreamer_learner.compute_loss(states, actions, rewards, next_states, dones)

    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


@pytest.mark.parametrize("horizon", [1, 5, 15, 30])
def test_different_imagination_horizons(
    dreamer_model: DreamerModel, horizon: int, config: LearnerTestConfig
) -> None:
    """Learner should work with different imagination horizons."""
    from mario_rl.learners.dreamer import DreamerLearner

    learner = DreamerLearner(
        model=dreamer_model,
        imagination_horizon=horizon,
    )

    states = torch.randn(4, *config.input_shape)
    actions = torch.randint(0, config.num_actions, (4,))
    rewards = torch.randn(4)
    next_states = torch.randn(4, *config.input_shape)
    dones = torch.zeros(4)

    loss, _ = learner.compute_loss(states, actions, rewards, next_states, dones)

    assert not torch.isnan(loss)
