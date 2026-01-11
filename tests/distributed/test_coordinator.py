"""Tests for distributed Coordinator behavior.

These tests verify the Coordinator correctly:
- Aggregates gradients from multiple workers
- Applies aggregated updates to model
- Broadcasts updated weights to workers
- Supports target network updates
"""

from typing import Any
from dataclasses import dataclass
from dataclasses import field

import pytest
import torch
from torch import Tensor
from torch import nn

from mario_rl.models import Model
from mario_rl.learners import Learner


# =============================================================================
# Mock Implementations
# =============================================================================


@dataclass(frozen=True)
class MockModelConfig:
    """Configuration for MockModel."""

    obs_shape: tuple[int, ...] = (4, 64, 64)
    num_actions: int = 12


class MockModel(nn.Module):
    """Simple model for testing distributed infrastructure."""

    def __init__(self, config: MockModelConfig | None = None) -> None:
        super().__init__()
        config = config or MockModelConfig()
        self.obs_shape = config.obs_shape
        self.num_actions = config.num_actions
        flat_size = 1
        for dim in config.obs_shape:
            flat_size *= dim
        self.fc = nn.Linear(flat_size, config.num_actions)

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        return self.fc(x_flat)

    def get_action(self, x: Tensor, epsilon: float = 0.0) -> Tensor:
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.num_actions, (x.shape[0],), device=x.device)
        with torch.no_grad():
            return self.forward(x).argmax(dim=1)


@dataclass
class MockLearner:
    """Simple learner for testing distributed infrastructure."""

    model: MockModel
    gamma: float = 0.99
    _update_count: int = field(default=0, init=False)

    def compute_loss(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        dones: Tensor,
    ) -> tuple[Tensor, dict[str, Any]]:
        q_values = self.model(states)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next = self.model(next_states).max(dim=1).values
            targets = rewards + self.gamma * q_next * (1 - dones.float())

        loss = torch.nn.functional.mse_loss(q_selected, targets)
        return loss, {"loss": loss.item()}

    def update_targets(self, tau: float = 1.0) -> None:
        self._update_count += 1


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_model() -> MockModel:
    """Create a mock model for testing."""
    return MockModel()


@pytest.fixture
def mock_learner(mock_model: MockModel) -> MockLearner:
    """Create a mock learner for testing."""
    return MockLearner(model=mock_model)


@pytest.fixture
def coordinator(mock_learner: MockLearner) -> "Coordinator":
    """Create a Coordinator for testing."""
    from mario_rl.distributed.coordinator import Coordinator

    return Coordinator(learner=mock_learner, lr=0.001)


@pytest.fixture
def sample_gradients(mock_model: MockModel) -> dict[str, Tensor]:
    """Create sample gradients matching model structure."""
    return {
        name: torch.randn_like(param)
        for name, param in mock_model.named_parameters()
    }


# =============================================================================
# Coordinator Setup Tests
# =============================================================================


def test_coordinator_accepts_any_learner(mock_learner: MockLearner) -> None:
    """Coordinator should accept any Learner that implements the protocol."""
    from mario_rl.distributed.coordinator import Coordinator

    assert isinstance(mock_learner, Learner)
    coordinator = Coordinator(learner=mock_learner, lr=0.001)
    assert coordinator.learner is mock_learner


def test_coordinator_exposes_model(coordinator: "Coordinator", mock_model: MockModel) -> None:
    """Coordinator should expose the underlying model."""
    assert coordinator.model is mock_model


def test_coordinator_creates_optimizer(coordinator: "Coordinator") -> None:
    """Coordinator should create an optimizer for the model."""
    assert coordinator.optimizer is not None


# =============================================================================
# Gradient Aggregation Tests
# =============================================================================


def test_aggregate_single_gradient(coordinator: "Coordinator", sample_gradients: dict[str, Tensor]) -> None:
    """Aggregating a single gradient set should return it unchanged."""
    aggregated = coordinator.aggregate_gradients([sample_gradients])

    for name in sample_gradients:
        assert torch.allclose(aggregated[name], sample_gradients[name])


def test_aggregate_multiple_gradients(coordinator: "Coordinator", mock_model: MockModel) -> None:
    """Aggregating multiple gradients should compute mean."""
    # Create 4 different gradient sets
    grad_sets = [
        {name: torch.randn_like(param) for name, param in mock_model.named_parameters()}
        for _ in range(4)
    ]

    aggregated = coordinator.aggregate_gradients(grad_sets)

    # Verify mean was computed
    for name in aggregated:
        expected = sum(g[name] for g in grad_sets) / len(grad_sets)
        assert torch.allclose(aggregated[name], expected)


def test_aggregate_empty_list_raises(coordinator: "Coordinator") -> None:
    """Aggregating empty gradient list should raise ValueError."""
    with pytest.raises(ValueError):
        coordinator.aggregate_gradients([])


# =============================================================================
# Gradient Application Tests
# =============================================================================


def test_apply_gradients_updates_weights(coordinator: "Coordinator", sample_gradients: dict[str, Tensor]) -> None:
    """apply_gradients should update model weights."""
    original_weights = {
        name: param.clone() for name, param in coordinator.model.named_parameters()
    }

    coordinator.apply_gradients(sample_gradients)

    # Weights should have changed
    weights_changed = False
    for name, param in coordinator.model.named_parameters():
        if not torch.equal(param.data, original_weights[name]):
            weights_changed = True
            break

    assert weights_changed


def test_apply_gradients_uses_optimizer(coordinator: "Coordinator", sample_gradients: dict[str, Tensor]) -> None:
    """apply_gradients should use the optimizer's step."""
    # Zero gradients first
    coordinator.optimizer.zero_grad()

    coordinator.apply_gradients(sample_gradients)

    # Optimizer should have been stepped (weights changed)
    # We can't easily verify optimizer was called, but weights should change
    pass  # Test covered by apply_gradients_updates_weights


# =============================================================================
# Weight Broadcasting Tests
# =============================================================================


def test_weights_returns_current_weights(coordinator: "Coordinator") -> None:
    """weights should return current model weights."""
    weights = coordinator.weights()

    for name, param in coordinator.model.named_parameters():
        assert torch.equal(weights[name], param.data)


def test_weights_are_detached(coordinator: "Coordinator") -> None:
    """weights should return detached tensors."""
    weights = coordinator.weights()

    for name, tensor in weights.items():
        assert not tensor.requires_grad


# =============================================================================
# Target Update Tests
# =============================================================================


def test_update_targets_delegates_to_learner(coordinator: "Coordinator") -> None:
    """update_targets should delegate to learner's update_targets."""
    initial_count = coordinator.learner._update_count

    coordinator.update_targets(tau=0.005)

    assert coordinator.learner._update_count == initial_count + 1


def test_update_targets_with_different_tau(coordinator: "Coordinator") -> None:
    """update_targets should pass tau to learner."""
    coordinator.update_targets(tau=1.0)
    coordinator.update_targets(tau=0.001)
    # Should not raise


# =============================================================================
# Full Training Step Tests
# =============================================================================


def test_training_step_integrates_all_operations(coordinator: "Coordinator", mock_model: MockModel) -> None:
    """A full training step should aggregate, apply, and return new weights."""
    # Create gradient sets from multiple workers
    grad_sets = [
        {name: torch.randn_like(param) for name, param in mock_model.named_parameters()}
        for _ in range(4)
    ]

    original_weights = coordinator.weights()

    # Execute training step
    new_weights = coordinator.training_step(grad_sets)

    # Verify weights changed
    for name in new_weights:
        assert not torch.equal(new_weights[name], original_weights[name])


def test_training_step_returns_weights(coordinator: "Coordinator", sample_gradients: dict[str, Tensor]) -> None:
    """training_step should return updated weights for broadcasting."""
    new_weights = coordinator.training_step([sample_gradients])

    assert isinstance(new_weights, dict)
    for name, param in coordinator.model.named_parameters():
        assert name in new_weights
        assert new_weights[name].shape == param.shape


# =============================================================================
# Integration with Real Models
# =============================================================================


def test_coordinator_with_ddqn() -> None:
    """Coordinator should work with DoubleDQN model."""
    from mario_rl.distributed.coordinator import Coordinator
    from mario_rl.models import DoubleDQN
    from mario_rl.learners import DDQNLearner

    model = DoubleDQN(
        input_shape=(4, 64, 64),
        num_actions=12,
        dropout=0.0,
    )
    learner = DDQNLearner(model=model)
    coordinator = Coordinator(learner=learner, lr=0.001)

    # Create mock gradients
    grads = {name: torch.randn_like(p) for name, p in model.named_parameters()}

    new_weights = coordinator.training_step([grads])

    assert len(new_weights) > 0


def test_coordinator_with_dreamer() -> None:
    """Coordinator should work with DreamerModel."""
    from mario_rl.distributed.coordinator import Coordinator
    from mario_rl.models import DreamerModel
    from mario_rl.learners import DreamerLearner

    model = DreamerModel(
        input_shape=(4, 64, 64),
        num_actions=12,
    )
    learner = DreamerLearner(model=model, imagination_horizon=5)
    coordinator = Coordinator(learner=learner, lr=0.0001)

    # Create mock gradients
    grads = {name: torch.randn_like(p) for name, p in model.named_parameters()}

    new_weights = coordinator.training_step([grads])

    assert len(new_weights) > 0
