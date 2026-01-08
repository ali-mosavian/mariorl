"""Tests for DDQNLearner behavior.

These tests verify the DDQNLearner correctly:
- Implements the Learner protocol
- Computes Double DQN TD loss
- Supports soft and hard target updates
- Returns proper training metrics
- Allows gradient backpropagation
"""

from dataclasses import dataclass

import pytest
import torch
from torch import Tensor

from mario_rl.learners import Learner
from mario_rl.models import DoubleDQN


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class LearnerTestConfig:
    """Test configuration for DDQNLearner."""

    input_shape: tuple[int, int, int] = (4, 64, 64)
    num_actions: int = 12
    gamma: float = 0.99
    batch_size: int = 8


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config() -> LearnerTestConfig:
    """Default test configuration."""
    return LearnerTestConfig()


@pytest.fixture
def ddqn_model(config: LearnerTestConfig) -> DoubleDQN:
    """Create a DoubleDQN model for testing."""
    return DoubleDQN(
        input_shape=config.input_shape,
        num_actions=config.num_actions,
        dropout=0.0,  # Disable dropout for deterministic tests
    )


@pytest.fixture
def ddqn_learner(ddqn_model: DoubleDQN, config: LearnerTestConfig):
    """Create a DDQNLearner for testing."""
    from mario_rl.learners.ddqn import DDQNLearner

    return DDQNLearner(model=ddqn_model, gamma=config.gamma)


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


def test_ddqn_learner_implements_learner_protocol(ddqn_learner) -> None:
    """DDQNLearner should satisfy the Learner protocol."""
    assert isinstance(ddqn_learner, Learner)


def test_ddqn_learner_has_model_attribute(ddqn_learner, ddqn_model: DoubleDQN) -> None:
    """DDQNLearner must expose its model as an attribute."""
    assert hasattr(ddqn_learner, "model")
    assert ddqn_learner.model is ddqn_model


# =============================================================================
# Loss Computation Tests
# =============================================================================


def test_compute_loss_returns_scalar_tensor(ddqn_learner, sample_batch: dict[str, Tensor]) -> None:
    """compute_loss should return a scalar loss tensor."""
    loss, _ = ddqn_learner.compute_loss(**sample_batch)

    assert isinstance(loss, Tensor)
    assert loss.dim() == 0  # Scalar
    assert loss.dtype == torch.float32


def test_compute_loss_returns_metrics_dict(ddqn_learner, sample_batch: dict[str, Tensor]) -> None:
    """compute_loss should return a metrics dictionary."""
    _, metrics = ddqn_learner.compute_loss(**sample_batch)

    assert isinstance(metrics, dict)
    assert "loss" in metrics
    assert "q_mean" in metrics


def test_compute_loss_loss_is_positive(ddqn_learner, sample_batch: dict[str, Tensor]) -> None:
    """Loss should be non-negative (it's a squared/huber loss)."""
    loss, _ = ddqn_learner.compute_loss(**sample_batch)

    assert loss.item() >= 0


def test_compute_loss_requires_grad(ddqn_learner, sample_batch: dict[str, Tensor]) -> None:
    """Loss tensor should require gradients for backprop."""
    loss, _ = ddqn_learner.compute_loss(**sample_batch)

    assert loss.requires_grad


def test_compute_loss_allows_backprop(ddqn_learner, sample_batch: dict[str, Tensor]) -> None:
    """Backpropagation through loss should compute gradients for online network."""
    # Zero any existing gradients
    ddqn_learner.model.online.zero_grad()

    loss, _ = ddqn_learner.compute_loss(**sample_batch)
    loss.backward()

    # Check that online network has gradients
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in ddqn_learner.model.online.parameters()
    )
    assert has_grad


def test_compute_loss_target_has_no_grad(ddqn_learner, sample_batch: dict[str, Tensor]) -> None:
    """Target network should not have gradients after loss computation."""
    loss, _ = ddqn_learner.compute_loss(**sample_batch)
    loss.backward()

    # Target network should have no gradients (frozen)
    target_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in ddqn_learner.model.target.parameters()
    )
    assert not target_has_grad


def test_compute_loss_uses_double_dqn_targets(ddqn_learner, sample_batch: dict[str, Tensor]) -> None:
    """Loss should use Double DQN target: Q_target(s', argmax Q_online(s')).

    Verify by checking that the loss changes when we modify target network
    but keep online network the same.
    """
    # Compute initial loss
    loss1, _ = ddqn_learner.compute_loss(**sample_batch)

    # Modify target network weights
    with torch.no_grad():
        for p in ddqn_learner.model.target.parameters():
            p.add_(10.0)

    # Recompute loss - should be different because target changed
    loss2, _ = ddqn_learner.compute_loss(**sample_batch)

    assert not torch.isclose(loss1, loss2)


def test_compute_loss_respects_done_flag(ddqn_learner, config: LearnerTestConfig) -> None:
    """When done=1, next state Q-value should be zeroed (no bootstrap)."""
    states = torch.randn(2, *config.input_shape)
    actions = torch.tensor([0, 0])
    rewards = torch.tensor([1.0, 1.0])
    next_states = torch.randn(2, *config.input_shape)

    # First sample: not done (bootstrap)
    # Second sample: done (no bootstrap)
    dones_different = torch.tensor([0.0, 1.0])

    loss_diff, _ = ddqn_learner.compute_loss(states, actions, rewards, next_states, dones_different)

    # Both done - should give different loss if bootstrap matters
    dones_same = torch.tensor([0.0, 0.0])
    loss_same, _ = ddqn_learner.compute_loss(states, actions, rewards, next_states, dones_same)

    # Losses should differ because done flag affects target
    assert not torch.isclose(loss_diff, loss_same)


# =============================================================================
# Metrics Tests
# =============================================================================


def test_metrics_loss_matches_tensor(ddqn_learner, sample_batch: dict[str, Tensor]) -> None:
    """Metrics 'loss' should match the loss tensor value."""
    loss, metrics = ddqn_learner.compute_loss(**sample_batch)

    assert abs(metrics["loss"] - loss.item()) < 1e-6


def test_metrics_q_mean_is_reasonable(ddqn_learner, sample_batch: dict[str, Tensor]) -> None:
    """Q-mean should be within bounded range due to softsign scaling."""
    _, metrics = ddqn_learner.compute_loss(**sample_batch)

    # Q-values are bounded by q_scale (default 100)
    assert -100 <= metrics["q_mean"] <= 100


def test_metrics_include_td_error(ddqn_learner, sample_batch: dict[str, Tensor]) -> None:
    """Metrics should include TD error for prioritized replay."""
    _, metrics = ddqn_learner.compute_loss(**sample_batch)

    assert "td_error_mean" in metrics
    assert metrics["td_error_mean"] >= 0  # Absolute error


# =============================================================================
# Target Update Tests
# =============================================================================


def test_update_targets_hard_sync(ddqn_learner) -> None:
    """update_targets with tau=1.0 should hard sync target to online."""
    # Modify online weights
    with torch.no_grad():
        for p in ddqn_learner.model.online.parameters():
            p.add_(5.0)

    # Hard sync
    ddqn_learner.update_targets(tau=1.0)

    # Weights should match
    for online_p, target_p in zip(
        ddqn_learner.model.online.parameters(),
        ddqn_learner.model.target.parameters(),
    ):
        assert torch.allclose(online_p.data, target_p.data)


def test_update_targets_soft_update(ddqn_learner) -> None:
    """update_targets with tau<1.0 should interpolate weights."""
    # Save original target weights
    original_target = {k: v.clone() for k, v in ddqn_learner.model.target.state_dict().items()}

    # Modify online weights
    with torch.no_grad():
        for p in ddqn_learner.model.online.parameters():
            p.add_(10.0)

    # Soft update with tau=0.5
    ddqn_learner.update_targets(tau=0.5)

    # Verify interpolation
    for name, target_p in ddqn_learner.model.target.named_parameters():
        online_p = dict(ddqn_learner.model.online.named_parameters())[name]
        expected = 0.5 * online_p.data + 0.5 * original_target[name]
        assert torch.allclose(target_p.data, expected, atol=1e-6)


def test_update_targets_default_tau(ddqn_learner) -> None:
    """update_targets with default tau should perform soft update."""
    original_target = {k: v.clone() for k, v in ddqn_learner.model.target.state_dict().items()}

    # Modify online weights
    with torch.no_grad():
        for p in ddqn_learner.model.online.parameters():
            p.add_(10.0)

    # Default soft update
    ddqn_learner.update_targets()

    # Weights should have changed but not be equal to online
    weights_changed = False
    for name, target_p in ddqn_learner.model.target.named_parameters():
        if not torch.equal(target_p.data, original_target[name]):
            weights_changed = True
            break

    assert weights_changed


# =============================================================================
# Edge Cases
# =============================================================================


def test_single_sample_batch(ddqn_learner, config: LearnerTestConfig) -> None:
    """Learner should handle batch size of 1."""
    states = torch.randn(1, *config.input_shape)
    actions = torch.randint(0, config.num_actions, (1,))
    rewards = torch.randn(1)
    next_states = torch.randn(1, *config.input_shape)
    dones = torch.zeros(1)

    loss, metrics = ddqn_learner.compute_loss(states, actions, rewards, next_states, dones)

    assert loss.dim() == 0
    assert isinstance(metrics, dict)


def test_all_done_batch(ddqn_learner, config: LearnerTestConfig) -> None:
    """Learner should handle batch where all episodes are done."""
    states = torch.randn(4, *config.input_shape)
    actions = torch.randint(0, config.num_actions, (4,))
    rewards = torch.randn(4)
    next_states = torch.randn(4, *config.input_shape)
    dones = torch.ones(4)  # All done

    loss, _ = ddqn_learner.compute_loss(states, actions, rewards, next_states, dones)

    # Should still compute valid loss (targets = rewards only)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_extreme_rewards(ddqn_learner, config: LearnerTestConfig) -> None:
    """Learner should handle extreme reward values."""
    states = torch.randn(4, *config.input_shape)
    actions = torch.randint(0, config.num_actions, (4,))
    rewards = torch.tensor([1000.0, -1000.0, 0.0, 0.001])
    next_states = torch.randn(4, *config.input_shape)
    dones = torch.zeros(4)

    loss, _ = ddqn_learner.compute_loss(states, actions, rewards, next_states, dones)

    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


@pytest.mark.parametrize("gamma", [0.0, 0.5, 0.99, 1.0])
def test_different_gamma_values(gamma: float, config: LearnerTestConfig) -> None:
    """Learner should work with different discount factors."""
    from mario_rl.learners.ddqn import DDQNLearner

    model = DoubleDQN(
        input_shape=config.input_shape,
        num_actions=config.num_actions,
        dropout=0.0,
    )
    learner = DDQNLearner(model=model, gamma=gamma)

    states = torch.randn(4, *config.input_shape)
    actions = torch.randint(0, config.num_actions, (4,))
    rewards = torch.randn(4)
    next_states = torch.randn(4, *config.input_shape)
    dones = torch.zeros(4)

    loss, _ = learner.compute_loss(states, actions, rewards, next_states, dones)

    assert not torch.isnan(loss)
