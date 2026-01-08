"""Tests for DoubleDQN model behavior.

These tests verify the DoubleDQN model correctly:
- Produces Q-values with correct shape and bounds
- Selects actions via epsilon-greedy policy
- Syncs target network (hard and soft)
- Implements the Model protocol
"""

from dataclasses import dataclass

import pytest
import torch
from torch import Tensor

from mario_rl.models import Model


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class DDQNTestConfig:
    """Test configuration for DoubleDQN."""

    input_shape: tuple[int, int, int] = (4, 64, 64)
    num_actions: int = 12
    feature_dim: int = 512
    hidden_dim: int = 256
    dropout: float = 0.0  # Disable dropout for deterministic tests
    q_scale: float = 100.0


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config() -> DDQNTestConfig:
    """Default test configuration."""
    return DDQNTestConfig()


@pytest.fixture
def ddqn_model(config: DDQNTestConfig):
    """Create a DoubleDQN model for testing."""
    # Import here to test the actual implementation
    from mario_rl.models.ddqn import DoubleDQN

    return DoubleDQN(
        input_shape=config.input_shape,
        num_actions=config.num_actions,
        feature_dim=config.feature_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        q_scale=config.q_scale,
    )


@pytest.fixture
def sample_batch(config: DDQNTestConfig) -> Tensor:
    """Create a sample observation batch."""
    return torch.randn(8, *config.input_shape)


# =============================================================================
# Protocol Conformance Tests
# =============================================================================


def test_ddqn_implements_model_protocol(ddqn_model) -> None:
    """DoubleDQN should satisfy the Model protocol."""
    assert isinstance(ddqn_model, Model)


def test_ddqn_has_num_actions_attribute(ddqn_model, config: DDQNTestConfig) -> None:
    """DoubleDQN must expose num_actions as an attribute."""
    assert hasattr(ddqn_model, "num_actions")
    assert ddqn_model.num_actions == config.num_actions


# =============================================================================
# Forward Pass Tests
# =============================================================================


def test_forward_returns_correct_shape(ddqn_model, sample_batch: Tensor, config: DDQNTestConfig) -> None:
    """Forward pass should return Q-values with shape (batch, num_actions)."""
    q_values = ddqn_model(sample_batch)

    assert q_values.shape == (8, config.num_actions)
    assert q_values.dtype == torch.float32


def test_forward_online_network(ddqn_model, sample_batch: Tensor) -> None:
    """Forward with network='online' should use online network."""
    q_online = ddqn_model(sample_batch, network="online")
    q_default = ddqn_model(sample_batch)

    assert torch.equal(q_online, q_default)


def test_forward_target_network(ddqn_model, sample_batch: Tensor) -> None:
    """Forward with network='target' should use target network."""
    # Initially target and online are synced
    q_online = ddqn_model(sample_batch, network="online")
    q_target = ddqn_model(sample_batch, network="target")

    assert torch.allclose(q_online, q_target)


def test_forward_invalid_network_raises(ddqn_model, sample_batch: Tensor) -> None:
    """Forward with invalid network name should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown network"):
        ddqn_model(sample_batch, network="invalid")


def test_q_values_are_bounded(ddqn_model, sample_batch: Tensor, config: DDQNTestConfig) -> None:
    """Q-values should be bounded by [-q_scale, q_scale] due to softsign activation."""
    q_values = ddqn_model(sample_batch)

    assert q_values.min() >= -config.q_scale
    assert q_values.max() <= config.q_scale


def test_q_values_gradient_flows(ddqn_model, sample_batch: Tensor) -> None:
    """Gradients should flow through Q-value computation."""
    q_values = ddqn_model(sample_batch)
    loss = q_values.mean()
    loss.backward()

    # Check that online network has gradients
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in ddqn_model.online.parameters())
    assert has_grad


# =============================================================================
# Action Selection Tests
# =============================================================================


def test_select_action_returns_correct_shape(ddqn_model, sample_batch: Tensor) -> None:
    """get_action should return tensor with shape (batch,)."""
    actions = ddqn_model.select_action(sample_batch, epsilon=0.0)

    assert actions.shape == (8,)
    assert actions.dtype == torch.int64


def test_select_action_greedy_selects_max_q(ddqn_model, sample_batch: Tensor) -> None:
    """With epsilon=0, get_action should select action with highest Q-value."""
    actions = ddqn_model.select_action(sample_batch, epsilon=0.0)
    q_values = ddqn_model(sample_batch)
    expected = q_values.argmax(dim=1)

    assert torch.equal(actions, expected)


def test_select_action_with_full_exploration(ddqn_model) -> None:
    """With epsilon=1.0, get_action should return random actions."""
    batch = torch.randn(1000, 4, 64, 64)
    actions = ddqn_model.select_action(batch, epsilon=1.0)

    # With 1000 samples and 12 actions, we should see most actions
    unique_actions = actions.unique()
    assert len(unique_actions) > 5  # At least half the actions should appear


def test_select_action_values_in_valid_range(ddqn_model, sample_batch: Tensor) -> None:
    """All returned actions should be valid action indices."""
    for epsilon in [0.0, 0.5, 1.0]:
        actions = ddqn_model.select_action(sample_batch, epsilon=epsilon)
        assert (actions >= 0).all()
        assert (actions < ddqn_model.num_actions).all()


# =============================================================================
# Target Network Tests
# =============================================================================


def test_sync_target_copies_weights(ddqn_model) -> None:
    """sync_target should copy online weights to target exactly."""
    # Modify online weights
    with torch.no_grad():
        for p in ddqn_model.online.parameters():
            p.add_(1.0)

    # Sync
    ddqn_model.sync_target()

    # Verify weights match
    for online_p, target_p in zip(
        ddqn_model.online.parameters(),
        ddqn_model.target.parameters(),
    ):
        assert torch.equal(online_p.data, target_p.data)


def test_target_parameters_are_frozen(ddqn_model) -> None:
    """Target network parameters should have requires_grad=False."""
    ddqn_model.sync_target()

    for p in ddqn_model.target.parameters():
        assert not p.requires_grad


def test_soft_update_interpolates_weights(ddqn_model) -> None:
    """soft_update should interpolate between online and target weights."""
    # Save original target weights
    original_target = {k: v.clone() for k, v in ddqn_model.target.state_dict().items()}

    # Modify online weights significantly
    with torch.no_grad():
        for p in ddqn_model.online.parameters():
            p.add_(10.0)

    # Soft update with tau=0.5
    ddqn_model.soft_update(tau=0.5)

    # Verify interpolation: new_target = 0.5 * online + 0.5 * old_target
    for name, target_p in ddqn_model.target.named_parameters():
        online_p = dict(ddqn_model.online.named_parameters())[name]
        expected = 0.5 * online_p.data + 0.5 * original_target[name]
        assert torch.allclose(target_p.data, expected, atol=1e-6)


def test_soft_update_with_tau_one_equals_hard_sync(ddqn_model) -> None:
    """soft_update with tau=1.0 should be equivalent to hard sync."""
    # Modify online weights
    with torch.no_grad():
        for p in ddqn_model.online.parameters():
            p.add_(5.0)

    # Soft update with tau=1.0
    ddqn_model.soft_update(tau=1.0)

    # Should match online exactly
    for online_p, target_p in zip(
        ddqn_model.online.parameters(),
        ddqn_model.target.parameters(),
    ):
        assert torch.allclose(online_p.data, target_p.data)


def test_soft_update_with_tau_zero_preserves_target(ddqn_model) -> None:
    """soft_update with tau=0.0 should not change target weights."""
    original_target = {k: v.clone() for k, v in ddqn_model.target.state_dict().items()}

    # Modify online weights
    with torch.no_grad():
        for p in ddqn_model.online.parameters():
            p.add_(10.0)

    # Soft update with tau=0.0
    ddqn_model.soft_update(tau=0.0)

    # Target should be unchanged
    for name, target_p in ddqn_model.target.named_parameters():
        assert torch.equal(target_p.data, original_target[name])


# =============================================================================
# State Dict Tests
# =============================================================================


def test_state_dict_is_serializable(ddqn_model) -> None:
    """state_dict should return a dict that can be saved and loaded."""
    state = ddqn_model.state_dict()

    assert isinstance(state, dict)
    assert len(state) > 0


def test_load_state_dict_restores_weights(ddqn_model, sample_batch: Tensor) -> None:
    """load_state_dict should restore model to saved state."""
    # Get initial output
    initial_output = ddqn_model(sample_batch).clone()
    initial_state = {k: v.clone() for k, v in ddqn_model.state_dict().items()}

    # Modify weights
    with torch.no_grad():
        for p in ddqn_model.parameters():
            p.add_(1.0)

    # Verify output changed
    modified_output = ddqn_model(sample_batch)
    assert not torch.equal(modified_output, initial_output)

    # Restore and verify
    ddqn_model.load_state_dict(initial_state)
    restored_output = ddqn_model(sample_batch)
    assert torch.equal(restored_output, initial_output)


# =============================================================================
# Edge Cases
# =============================================================================


def test_single_sample_batch(ddqn_model, config: DDQNTestConfig) -> None:
    """Model should handle batch size of 1."""
    single = torch.randn(1, *config.input_shape)

    q_values = ddqn_model(single)
    actions = ddqn_model.select_action(single, epsilon=0.0)

    assert q_values.shape == (1, config.num_actions)
    assert actions.shape == (1,)


def test_large_batch(ddqn_model, config: DDQNTestConfig) -> None:
    """Model should handle large batch sizes."""
    large_batch = torch.randn(256, *config.input_shape)

    q_values = ddqn_model(large_batch)

    assert q_values.shape == (256, config.num_actions)


@pytest.mark.parametrize("q_scale", [1.0, 10.0, 100.0, 1000.0])
def test_different_q_scales(q_scale: float) -> None:
    """Model should respect different q_scale values."""
    from mario_rl.models.ddqn import DoubleDQN

    model = DoubleDQN(
        input_shape=(4, 64, 64),
        num_actions=12,
        q_scale=q_scale,
    )
    batch = torch.randn(8, 4, 64, 64)
    q_values = model(batch)

    assert q_values.min() >= -q_scale
    assert q_values.max() <= q_scale
