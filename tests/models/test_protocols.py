"""Tests for Model and Learner protocol contracts.

These tests verify that the Model and Learner protocols are properly defined
and that implementations correctly satisfy their contracts.
"""

from typing import Any
from dataclasses import field
from dataclasses import dataclass

import torch
import pytest
from torch import nn
from torch import Tensor

from mario_rl.models import Model
from mario_rl.learners import Learner

# =============================================================================
# Mock Implementations for Testing
# =============================================================================


@dataclass
class MockModelConfig:
    """Configuration for MockModel."""

    obs_shape: tuple[int, ...]
    num_actions: int


class MockModel(nn.Module):
    """Minimal model implementation for protocol testing.

    Note: nn.Module subclasses cannot be dataclasses due to metaclass conflicts.
    We use a separate config dataclass for initialization parameters.
    """

    def __init__(self, config: MockModelConfig) -> None:
        super().__init__()
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

    def select_action(self, x: Tensor, epsilon: float = 0.0) -> Tensor:
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.num_actions, (x.shape[0],), device=x.device)
        with torch.no_grad():
            q_values = self.forward(x)
            return q_values.argmax(dim=1)


@dataclass
class MockLearner:
    """Minimal learner implementation for protocol testing."""

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
        return loss, {"loss": loss.item(), "q_mean": q_values.mean().item()}

    def update_targets(self, tau: float = 1.0) -> None:
        self._update_count += 1


# =============================================================================
# Protocol Contract Tests
# =============================================================================


@pytest.fixture
def mock_model() -> MockModel:
    """Create a mock model for testing."""
    config = MockModelConfig(obs_shape=(4, 64, 64), num_actions=12)
    return MockModel(config)


@pytest.fixture
def mock_learner(mock_model: MockModel) -> MockLearner:
    """Create a mock learner for testing."""
    return MockLearner(model=mock_model)


def test_model_protocol_is_runtime_checkable() -> None:
    """Model protocol should be checkable with isinstance at runtime."""
    assert hasattr(Model, "__protocol_attrs__") or hasattr(Model, "__subclasshook__")


def test_learner_protocol_is_runtime_checkable() -> None:
    """Learner protocol should be checkable with isinstance at runtime."""
    assert hasattr(Learner, "__protocol_attrs__") or hasattr(Learner, "__subclasshook__")


def test_mock_model_implements_model_protocol(mock_model: MockModel) -> None:
    """MockModel should satisfy the Model protocol."""
    assert isinstance(mock_model, Model)


def test_mock_learner_implements_learner_protocol(mock_learner: MockLearner) -> None:
    """MockLearner should satisfy the Learner protocol."""
    assert isinstance(mock_learner, Learner)


def test_model_has_num_actions_attribute(mock_model: MockModel) -> None:
    """Models must expose num_actions as an attribute."""
    assert hasattr(mock_model, "num_actions")
    assert mock_model.num_actions == 12


def test_model_forward_returns_correct_shape(mock_model: MockModel) -> None:
    """Forward pass should return tensor with shape (batch, num_actions)."""
    batch = torch.randn(8, 4, 64, 64)

    output = mock_model(batch)

    assert output.shape == (8, 12)
    assert output.dtype == torch.float32


def test_model_select_action_returns_correct_shape(mock_model: MockModel) -> None:
    """get_action should return tensor with shape (batch,)."""
    batch = torch.randn(4, 4, 64, 64)

    actions = mock_model.select_action(batch, epsilon=0.0)

    assert actions.shape == (4,)
    assert actions.dtype == torch.int64


def test_model_select_action_greedy_selects_max(mock_model: MockModel) -> None:
    """With epsilon=0, get_action should select argmax of Q-values."""
    batch = torch.randn(4, 4, 64, 64)

    actions = mock_model.select_action(batch, epsilon=0.0)
    q_values = mock_model(batch)
    expected = q_values.argmax(dim=1)

    assert torch.equal(actions, expected)


def test_model_select_action_with_epsilon_explores(mock_model: MockModel) -> None:
    """With epsilon=1.0, get_action should return random actions."""
    batch = torch.randn(100, 4, 64, 64)

    # With full exploration, we should see variety in actions
    actions = mock_model.select_action(batch, epsilon=1.0)
    unique_actions = actions.unique()

    # With 100 samples and 12 actions, we expect most actions to appear
    assert len(unique_actions) > 1


def test_model_state_dict_is_serializable(mock_model: MockModel) -> None:
    """state_dict should return a picklable dict."""
    state = mock_model.state_dict()

    assert isinstance(state, dict)
    assert len(state) > 0


def test_model_load_state_dict_restores_weights(mock_model: MockModel) -> None:
    """load_state_dict should restore model to saved state."""
    # Save initial state
    initial_state = {k: v.clone() for k, v in mock_model.state_dict().items()}

    # Modify weights
    with torch.no_grad():
        for param in mock_model.parameters():
            param.add_(1.0)

    # Restore
    mock_model.load_state_dict(initial_state)

    # Verify restoration
    for key, value in mock_model.state_dict().items():
        assert torch.equal(value, initial_state[key])


def test_model_parameters_returns_generator(mock_model: MockModel) -> None:
    """parameters() should return an iterable of Parameter tensors."""
    params = list(mock_model.parameters())

    assert len(params) > 0
    assert all(isinstance(p, torch.nn.Parameter) for p in params)


def test_learner_has_model_attribute(mock_learner: MockLearner) -> None:
    """Learners must expose their model as an attribute."""
    assert hasattr(mock_learner, "model")
    assert isinstance(mock_learner.model, Model)


def test_learner_compute_loss_returns_tensor_and_metrics(
    mock_learner: MockLearner,
) -> None:
    """compute_loss should return (loss_tensor, metrics_dict)."""
    states = torch.randn(8, 4, 64, 64)
    actions = torch.randint(0, 12, (8,))
    rewards = torch.randn(8)
    next_states = torch.randn(8, 4, 64, 64)
    dones = torch.zeros(8)

    loss, metrics = mock_learner.compute_loss(states, actions, rewards, next_states, dones)

    assert isinstance(loss, Tensor)
    assert loss.dim() == 0  # Scalar
    assert loss.requires_grad
    assert isinstance(metrics, dict)
    assert "loss" in metrics


def test_learner_compute_loss_allows_backprop(mock_learner: MockLearner) -> None:
    """Loss from compute_loss should support backpropagation."""
    states = torch.randn(4, 4, 64, 64)
    actions = torch.randint(0, 12, (4,))
    rewards = torch.randn(4)
    next_states = torch.randn(4, 4, 64, 64)
    dones = torch.zeros(4)

    loss, _ = mock_learner.compute_loss(states, actions, rewards, next_states, dones)
    loss.backward()

    # Check gradients were computed
    has_grad = any(p.grad is not None for p in mock_learner.model.parameters())
    assert has_grad


def test_learner_update_targets_is_callable(mock_learner: MockLearner) -> None:
    """update_targets should be callable without error."""
    mock_learner.update_targets(tau=0.005)
    mock_learner.update_targets(tau=1.0)

    # Just verify it doesn't raise
    assert mock_learner._update_count == 2
