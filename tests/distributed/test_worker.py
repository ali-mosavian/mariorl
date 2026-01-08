"""Tests for distributed Worker behavior.

These tests verify the Worker correctly:
- Uses any Model/Learner that implements the protocols
- Computes gradients from batches
- Shares gradients with coordinator
- Receives and applies updated weights
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
# Mock Implementations for Protocol Testing
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
            q_values = self.forward(x)
            return q_values.argmax(dim=1)


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
        return loss, {"loss": loss.item(), "q_mean": q_values.mean().item()}

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
def worker(mock_learner: MockLearner):
    """Create a Worker for testing."""
    from mario_rl.distributed.worker import Worker

    return Worker(learner=mock_learner)


@pytest.fixture
def sample_batch() -> dict[str, Tensor]:
    """Create a sample transition batch."""
    return {
        "states": torch.randn(8, 4, 64, 64),
        "actions": torch.randint(0, 12, (8,)),
        "rewards": torch.randn(8),
        "next_states": torch.randn(8, 4, 64, 64),
        "dones": torch.zeros(8),
    }


# =============================================================================
# Worker Protocol Tests
# =============================================================================


def test_worker_accepts_any_learner(mock_learner: MockLearner) -> None:
    """Worker should accept any Learner that implements the protocol."""
    from mario_rl.distributed.worker import Worker

    assert isinstance(mock_learner, Learner)
    worker = Worker(learner=mock_learner)
    assert worker.learner is mock_learner


def test_worker_exposes_model(worker, mock_model: MockModel) -> None:
    """Worker should expose the underlying model."""
    assert worker.model is mock_model


# =============================================================================
# Gradient Computation Tests
# =============================================================================


def test_compute_gradients_returns_dict(worker, sample_batch: dict[str, Tensor]) -> None:
    """compute_gradients should return a dict of gradients."""
    grads, metrics = worker.compute_gradients(**sample_batch)

    assert isinstance(grads, dict)
    assert len(grads) > 0


def test_compute_gradients_returns_metrics(worker, sample_batch: dict[str, Tensor]) -> None:
    """compute_gradients should return training metrics."""
    _, metrics = worker.compute_gradients(**sample_batch)

    assert isinstance(metrics, dict)
    assert "loss" in metrics


def test_gradients_match_parameter_names(worker, sample_batch: dict[str, Tensor]) -> None:
    """Gradient dict keys should match model parameter names."""
    grads, _ = worker.compute_gradients(**sample_batch)

    param_names = set(name for name, _ in worker.model.named_parameters())
    grad_names = set(grads.keys())

    assert grad_names == param_names


def test_gradients_are_tensors(worker, sample_batch: dict[str, Tensor]) -> None:
    """All gradients should be tensors."""
    grads, _ = worker.compute_gradients(**sample_batch)

    for name, grad in grads.items():
        assert isinstance(grad, Tensor), f"Gradient {name} is not a tensor"


def test_gradients_match_parameter_shapes(worker, sample_batch: dict[str, Tensor]) -> None:
    """Gradient shapes should match parameter shapes."""
    grads, _ = worker.compute_gradients(**sample_batch)

    for name, param in worker.model.named_parameters():
        assert grads[name].shape == param.shape


def test_gradients_are_detached(worker, sample_batch: dict[str, Tensor]) -> None:
    """Gradients should be detached from computation graph."""
    grads, _ = worker.compute_gradients(**sample_batch)

    for name, grad in grads.items():
        assert not grad.requires_grad, f"Gradient {name} should not require grad"


# =============================================================================
# Weight Synchronization Tests
# =============================================================================


def test_apply_weights_updates_model(worker) -> None:
    """apply_weights should update model parameters."""
    # Save original weights
    original_weights = {
        name: param.clone() for name, param in worker.model.named_parameters()
    }

    # Create new weights
    new_weights = {name: param + 1.0 for name, param in original_weights.items()}

    # Apply new weights
    worker.apply_weights(new_weights)

    # Verify update
    for name, param in worker.model.named_parameters():
        expected = original_weights[name] + 1.0
        assert torch.allclose(param.data, expected)


def test_apply_weights_with_state_dict_format(worker) -> None:
    """apply_weights should work with state_dict format."""
    # Get current state dict (clone to avoid shared memory issues)
    original_state = {k: v.clone() for k, v in worker.model.state_dict().items()}

    # Modify and apply
    new_state = {k: v + 1.0 for k, v in original_state.items()}
    worker.apply_weights(new_state)

    # Verify
    for name, param in worker.model.named_parameters():
        expected = original_state[name] + 1.0
        assert torch.allclose(param.data, expected)


def test_weights_returns_current_weights(worker) -> None:
    """weights should return current model weights."""
    weights = worker.weights()

    for name, param in worker.model.named_parameters():
        assert torch.equal(weights[name], param.data)


def test_weights_roundtrip(worker) -> None:
    """weights -> modify -> apply_weights should work."""
    weights = worker.weights()
    modified = {k: v + 5.0 for k, v in weights.items()}
    worker.apply_weights(modified)
    new_weights = worker.weights()

    for name in weights:
        expected = weights[name] + 5.0
        assert torch.allclose(new_weights[name], expected)


# =============================================================================
# Integration with Real Models
# =============================================================================


def test_worker_with_ddqn() -> None:
    """Worker should work with DoubleDQN model."""
    from mario_rl.distributed.worker import Worker
    from mario_rl.models import DoubleDQN
    from mario_rl.learners import DDQNLearner

    model = DoubleDQN(
        input_shape=(4, 64, 64),
        num_actions=12,
        dropout=0.0,
    )
    learner = DDQNLearner(model=model)
    worker = Worker(learner=learner)

    batch = {
        "states": torch.randn(4, 4, 64, 64),
        "actions": torch.randint(0, 12, (4,)),
        "rewards": torch.randn(4),
        "next_states": torch.randn(4, 4, 64, 64),
        "dones": torch.zeros(4),
    }

    grads, metrics = worker.compute_gradients(**batch)

    assert len(grads) > 0
    assert "loss" in metrics


def test_worker_with_dreamer() -> None:
    """Worker should work with DreamerModel."""
    from mario_rl.distributed.worker import Worker
    from mario_rl.models import DreamerModel
    from mario_rl.learners import DreamerLearner

    model = DreamerModel(
        input_shape=(4, 64, 64),
        num_actions=12,
    )
    learner = DreamerLearner(model=model, imagination_horizon=5)
    worker = Worker(learner=learner)

    batch = {
        "states": torch.randn(4, 4, 64, 64),
        "actions": torch.randint(0, 12, (4,)),
        "rewards": torch.randn(4),
        "next_states": torch.randn(4, 4, 64, 64),
        "dones": torch.zeros(4),
    }

    grads, metrics = worker.compute_gradients(**batch)

    assert len(grads) > 0
    assert "loss" in metrics
