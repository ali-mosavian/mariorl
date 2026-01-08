"""Tests for DreamerModel behavior.

These tests verify the DreamerModel correctly:
- Encodes observations to latent space
- Imagines future trajectories using learned dynamics
- Produces actor (policy) and critic (value) outputs
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
class DreamerTestConfig:
    """Test configuration for DreamerModel."""

    input_shape: tuple[int, int, int] = (4, 64, 64)
    num_actions: int = 12
    latent_dim: int = 128
    hidden_dim: int = 256
    imagination_horizon: int = 15


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config() -> DreamerTestConfig:
    """Default test configuration."""
    return DreamerTestConfig()


@pytest.fixture
def dreamer_model(config: DreamerTestConfig):
    """Create a DreamerModel for testing."""
    from mario_rl.models.dreamer import DreamerModel

    return DreamerModel(
        input_shape=config.input_shape,
        num_actions=config.num_actions,
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
    )


@pytest.fixture
def sample_batch(config: DreamerTestConfig) -> Tensor:
    """Create a sample observation batch."""
    return torch.randn(8, *config.input_shape)


# =============================================================================
# Protocol Conformance Tests
# =============================================================================


def test_dreamer_implements_model_protocol(dreamer_model) -> None:
    """DreamerModel should satisfy the Model protocol."""
    assert isinstance(dreamer_model, Model)


def test_dreamer_has_num_actions_attribute(dreamer_model, config: DreamerTestConfig) -> None:
    """DreamerModel must expose num_actions as an attribute."""
    assert hasattr(dreamer_model, "num_actions")
    assert dreamer_model.num_actions == config.num_actions


# =============================================================================
# Encoder Tests
# =============================================================================


def test_encode_returns_correct_shape(dreamer_model, sample_batch: Tensor, config: DreamerTestConfig) -> None:
    """Encoder should produce latent vectors with shape (batch, latent_dim)."""
    z = dreamer_model.encode(sample_batch)

    assert z.shape == (8, config.latent_dim)
    assert z.dtype == torch.float32


def test_encode_deterministic_mode(dreamer_model, sample_batch: Tensor) -> None:
    """Deterministic encoding should give same result each time."""
    z1 = dreamer_model.encode(sample_batch, deterministic=True)
    z2 = dreamer_model.encode(sample_batch, deterministic=True)

    assert torch.equal(z1, z2)


def test_encode_stochastic_mode(dreamer_model, sample_batch: Tensor) -> None:
    """Stochastic encoding should give different results (due to sampling)."""
    z1 = dreamer_model.encode(sample_batch, deterministic=False)
    z2 = dreamer_model.encode(sample_batch, deterministic=False)

    # With probability ~1, samples should differ
    assert not torch.equal(z1, z2)


# =============================================================================
# Imagination Tests
# =============================================================================


def test_imagine_step_returns_correct_shapes(
    dreamer_model, config: DreamerTestConfig
) -> None:
    """imagine_step should return next latent, reward, and done predictions."""
    batch_size = 4
    z = torch.randn(batch_size, config.latent_dim)
    actions = torch.randint(0, config.num_actions, (batch_size,))

    z_next, reward_pred, done_pred = dreamer_model.imagine_step(z, actions)

    assert z_next.shape == (batch_size, config.latent_dim)
    assert reward_pred.shape == (batch_size,)
    assert done_pred.shape == (batch_size,)


def test_imagine_trajectory_returns_correct_shapes(
    dreamer_model, config: DreamerTestConfig
) -> None:
    """imagine_trajectory should return sequence of latents, rewards, and dones."""
    batch_size = 4
    horizon = config.imagination_horizon
    z_start = torch.randn(batch_size, config.latent_dim)

    z_traj, rewards, dones = dreamer_model.imagine_trajectory(z_start, horizon=horizon)

    # Trajectory has horizon+1 states (including start)
    assert z_traj.shape == (batch_size, horizon + 1, config.latent_dim)
    # Rewards and dones for each step
    assert rewards.shape == (batch_size, horizon)
    assert dones.shape == (batch_size, horizon)


def test_imagine_trajectory_first_state_matches_start(
    dreamer_model, config: DreamerTestConfig
) -> None:
    """First state in imagined trajectory should match starting latent."""
    z_start = torch.randn(4, config.latent_dim)

    z_traj, _, _ = dreamer_model.imagine_trajectory(z_start, horizon=5)

    assert torch.equal(z_traj[:, 0], z_start)


def test_imagine_trajectory_is_differentiable(
    dreamer_model, config: DreamerTestConfig
) -> None:
    """Imagined trajectory should support gradient computation."""
    z_start = torch.randn(4, config.latent_dim, requires_grad=True)

    z_traj, rewards, _ = dreamer_model.imagine_trajectory(z_start, horizon=5)

    # Should be able to backprop through trajectory
    loss = z_traj.sum() + rewards.sum()
    loss.backward()

    assert z_start.grad is not None


# =============================================================================
# Actor Tests
# =============================================================================


def test_actor_returns_action_logits(dreamer_model, config: DreamerTestConfig) -> None:
    """Actor should return logits for each action."""
    z = torch.randn(8, config.latent_dim)

    logits = dreamer_model.actor(z)

    assert logits.shape == (8, config.num_actions)


def test_actor_logits_are_unbounded(dreamer_model, config: DreamerTestConfig) -> None:
    """Actor logits should not be bounded (softmax applied later)."""
    z = torch.randn(8, config.latent_dim) * 10  # Large input

    logits = dreamer_model.actor(z)

    # Logits can be any real number
    assert logits.abs().max() > 0


def test_actor_supports_gradient_flow(dreamer_model, config: DreamerTestConfig) -> None:
    """Gradients should flow through actor."""
    z = torch.randn(8, config.latent_dim, requires_grad=True)

    logits = dreamer_model.actor(z)
    loss = logits.sum()
    loss.backward()

    assert z.grad is not None


# =============================================================================
# Critic Tests
# =============================================================================


def test_critic_returns_value_estimate(dreamer_model, config: DreamerTestConfig) -> None:
    """Critic should return scalar value estimate for each state."""
    z = torch.randn(8, config.latent_dim)

    values = dreamer_model.critic(z)

    assert values.shape == (8,)


def test_critic_supports_gradient_flow(dreamer_model, config: DreamerTestConfig) -> None:
    """Gradients should flow through critic."""
    z = torch.randn(8, config.latent_dim, requires_grad=True)

    values = dreamer_model.critic(z)
    loss = values.sum()
    loss.backward()

    assert z.grad is not None


# =============================================================================
# Forward Pass / Action Selection Tests
# =============================================================================


def test_forward_returns_action_logits(
    dreamer_model, sample_batch: Tensor, config: DreamerTestConfig
) -> None:
    """Forward pass should return action logits with shape (batch, num_actions)."""
    logits = dreamer_model(sample_batch)

    assert logits.shape == (8, config.num_actions)


def test_get_action_returns_correct_shape(dreamer_model, sample_batch: Tensor) -> None:
    """get_action should return actions with shape (batch,)."""
    actions = dreamer_model.select_action(sample_batch, epsilon=0.0)

    assert actions.shape == (8,)
    assert actions.dtype == torch.int64


def test_get_action_greedy_samples_from_policy(dreamer_model, sample_batch: Tensor) -> None:
    """With epsilon=0, get_action should sample from actor's distribution."""
    # Run multiple times - greedy should still sample from softmax
    actions = dreamer_model.select_action(sample_batch, epsilon=0.0)

    assert (actions >= 0).all()
    assert (actions < dreamer_model.num_actions).all()


def test_get_action_with_exploration(dreamer_model) -> None:
    """With epsilon=1.0, get_action should return random actions."""
    batch = torch.randn(100, 4, 64, 64)
    actions = dreamer_model.select_action(batch, epsilon=1.0)

    unique_actions = actions.unique()
    assert len(unique_actions) > 1


# =============================================================================
# World Model Components Tests
# =============================================================================


def test_has_dynamics_model(dreamer_model) -> None:
    """DreamerModel should have a dynamics model for state prediction."""
    assert hasattr(dreamer_model, "dynamics")


def test_has_reward_predictor(dreamer_model) -> None:
    """DreamerModel should have a reward predictor."""
    assert hasattr(dreamer_model, "reward_pred")


def test_has_encoder(dreamer_model) -> None:
    """DreamerModel should have an encoder."""
    assert hasattr(dreamer_model, "encoder")


# =============================================================================
# State Dict Tests
# =============================================================================


def test_state_dict_is_serializable(dreamer_model) -> None:
    """state_dict should be serializable."""
    state = dreamer_model.state_dict()

    assert isinstance(state, dict)
    assert len(state) > 0


def test_load_state_dict_restores_weights(dreamer_model, sample_batch: Tensor) -> None:
    """load_state_dict should restore model."""
    initial_output = dreamer_model(sample_batch).clone()
    initial_state = {k: v.clone() for k, v in dreamer_model.state_dict().items()}

    # Modify weights
    with torch.no_grad():
        for p in dreamer_model.parameters():
            p.add_(1.0)

    # Restore
    dreamer_model.load_state_dict(initial_state)
    restored_output = dreamer_model(sample_batch)

    assert torch.allclose(restored_output, initial_output)


# =============================================================================
# Edge Cases
# =============================================================================


def test_single_sample(dreamer_model, config: DreamerTestConfig) -> None:
    """Model should handle batch size of 1."""
    single = torch.randn(1, *config.input_shape)

    z = dreamer_model.encode(single)
    logits = dreamer_model(single)
    actions = dreamer_model.select_action(single)

    assert z.shape == (1, config.latent_dim)
    assert logits.shape == (1, config.num_actions)
    assert actions.shape == (1,)


def test_large_batch(dreamer_model, config: DreamerTestConfig) -> None:
    """Model should handle large batches."""
    large = torch.randn(256, *config.input_shape)

    z = dreamer_model.encode(large)
    logits = dreamer_model(large)

    assert z.shape == (256, config.latent_dim)
    assert logits.shape == (256, config.num_actions)
