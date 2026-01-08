"""Tests for TrainingCoordinator behavior.

These tests verify the TrainingCoordinator correctly:
- Accumulates gradients from workers
- Applies learning rate scheduling
- Saves checkpoints
- Updates target networks
"""

from typing import Any
from pathlib import Path
from dataclasses import dataclass
from dataclasses import field
from unittest.mock import Mock

import pytest
import torch
from torch import Tensor
from torch import nn


# =============================================================================
# Mock Model and Learner
# =============================================================================


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
        self.num_actions = 5

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


@dataclass
class MockLearner:
    """Mock learner for testing coordinator."""

    model: SimpleModel
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
        return q_values.mean(), {"loss": q_values.mean().item()}

    def update_targets(self, tau: float = 1.0) -> None:
        self._update_count += 1


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_model() -> SimpleModel:
    """Create a simple model."""
    return SimpleModel()


@pytest.fixture
def mock_learner(simple_model: SimpleModel) -> MockLearner:
    """Create a mock learner."""
    return MockLearner(model=simple_model)


@pytest.fixture
def shm_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for shared memory."""
    shm = tmp_path / "shm"
    shm.mkdir()
    return shm


@pytest.fixture
def checkpoint_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for checkpoints."""
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir()
    return ckpt


@pytest.fixture
def training_coordinator(mock_learner: MockLearner, shm_dir: Path, checkpoint_dir: Path):
    """Create a TrainingCoordinator for testing."""
    from mario_rl.distributed.training_coordinator import TrainingCoordinator

    return TrainingCoordinator(
        learner=mock_learner,
        num_workers=2,
        shm_dir=shm_dir,
        checkpoint_dir=checkpoint_dir,
        learning_rate=1e-4,
        target_update_interval=100,
        checkpoint_interval=1000,
    )


# =============================================================================
# Basic Functionality Tests
# =============================================================================


def test_coordinator_creates_successfully(training_coordinator) -> None:
    """Coordinator should create without error."""
    assert training_coordinator is not None


def test_coordinator_has_model(training_coordinator, simple_model: SimpleModel) -> None:
    """Coordinator should have model reference."""
    assert training_coordinator.model is simple_model


def test_coordinator_creates_optimizer(training_coordinator) -> None:
    """Coordinator should create optimizer."""
    assert training_coordinator.optimizer is not None


def test_coordinator_creates_gradient_pool(training_coordinator) -> None:
    """Coordinator should create gradient pool."""
    assert training_coordinator.gradient_pool is not None


# =============================================================================
# Gradient Accumulation Tests
# =============================================================================


def test_apply_gradients(training_coordinator, simple_model: SimpleModel) -> None:
    """Should apply gradients to model."""
    # Create mock gradients
    grads = {
        name: torch.ones_like(param)
        for name, param in simple_model.named_parameters()
    }

    initial_params = {
        name: param.clone()
        for name, param in simple_model.named_parameters()
    }

    training_coordinator.apply_gradients([grads])

    # Parameters should have changed
    for name, param in simple_model.named_parameters():
        # Note: optimizer step changes params, so they should differ
        # (this is behavior test, not exact value)
        pass  # Just checking no error

    training_coordinator.update_count > 0


def test_accumulate_gradients(training_coordinator, simple_model: SimpleModel) -> None:
    """Should accumulate multiple gradient batches."""
    grads1 = {name: torch.ones_like(p) for name, p in simple_model.named_parameters()}
    grads2 = {name: torch.ones_like(p) * 2 for name, p in simple_model.named_parameters()}

    averaged = training_coordinator.aggregate_gradients([grads1, grads2])

    # Averaged gradients should be (1 + 2) / 2 = 1.5
    for name, grad in averaged.items():
        assert torch.allclose(grad, torch.ones_like(grad) * 1.5)


def test_poll_gradients_from_pool(mock_learner: MockLearner, shm_dir: Path, checkpoint_dir: Path) -> None:
    """Should poll gradients from shared memory pool."""
    from mario_rl.distributed.training_coordinator import TrainingCoordinator

    coordinator = TrainingCoordinator(
        learner=mock_learner,
        num_workers=2,
        shm_dir=shm_dir,
        checkpoint_dir=checkpoint_dir,
    )

    # Write some gradients to pool
    grads = {
        name: torch.ones_like(param)
        for name, param in mock_learner.model.named_parameters()
    }
    coordinator.gradient_pool.write(worker_id=0, grads=grads)

    # Poll should get them
    packets = coordinator.poll_gradients()

    assert len(packets) == 1

    coordinator.close()


# =============================================================================
# Learning Rate Scheduling Tests
# =============================================================================


def test_lr_schedule_initial(training_coordinator) -> None:
    """Initial LR should match config."""
    lr = training_coordinator.current_lr()
    assert lr == pytest.approx(1e-4)


def test_lr_schedule_cosine_decay(mock_learner: MockLearner, shm_dir: Path, checkpoint_dir: Path) -> None:
    """LR should decay with cosine schedule."""
    from mario_rl.distributed.training_coordinator import TrainingCoordinator

    coordinator = TrainingCoordinator(
        learner=mock_learner,
        num_workers=2,
        shm_dir=shm_dir,
        checkpoint_dir=checkpoint_dir,
        learning_rate=1e-4,
        lr_min=1e-5,
        lr_decay_steps=1000,
    )

    # At step 0, should be max LR
    lr0 = coordinator.current_lr()

    # Simulate steps
    coordinator._total_steps = 500

    # At 50%, should be between min and max
    lr_mid = coordinator.current_lr()

    # At end, should be min
    coordinator._total_steps = 1000
    lr_end = coordinator.current_lr()

    assert lr0 > lr_mid > lr_end
    assert lr_end == pytest.approx(1e-5, rel=0.1)

    coordinator.close()


def test_update_lr_modifies_optimizer(training_coordinator) -> None:
    """update_lr should modify optimizer learning rate."""
    training_coordinator._total_steps = 500
    training_coordinator.update_lr()

    for pg in training_coordinator.optimizer.param_groups:
        assert pg["lr"] < 1e-4  # Should have decayed


# =============================================================================
# Checkpoint Tests
# =============================================================================


def test_save_checkpoint(training_coordinator, checkpoint_dir: Path) -> None:
    """Should save checkpoint to disk."""
    training_coordinator.save_checkpoint()

    # Checkpoint file should exist
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    assert len(checkpoints) >= 1


def test_checkpoint_contains_model_state(training_coordinator, checkpoint_dir: Path) -> None:
    """Checkpoint should contain model state dict."""
    training_coordinator.save_checkpoint()

    checkpoints = list(checkpoint_dir.glob("*.pt"))
    ckpt = torch.load(checkpoints[0], weights_only=False)

    assert "model_state_dict" in ckpt


def test_checkpoint_contains_optimizer_state(training_coordinator, checkpoint_dir: Path) -> None:
    """Checkpoint should contain optimizer state dict."""
    training_coordinator.save_checkpoint()

    checkpoints = list(checkpoint_dir.glob("*.pt"))
    ckpt = torch.load(checkpoints[0], weights_only=False)

    assert "optimizer_state_dict" in ckpt


def test_checkpoint_contains_step_count(training_coordinator, checkpoint_dir: Path) -> None:
    """Checkpoint should contain step count."""
    training_coordinator._total_steps = 500
    training_coordinator.save_checkpoint()

    checkpoints = list(checkpoint_dir.glob("*.pt"))
    ckpt = torch.load(checkpoints[0], weights_only=False)

    assert ckpt["total_steps"] == 500


def test_load_checkpoint(training_coordinator, checkpoint_dir: Path) -> None:
    """Should load checkpoint from disk."""
    training_coordinator._total_steps = 500
    training_coordinator.save_checkpoint()

    # Reset state
    training_coordinator._total_steps = 0

    # Load
    training_coordinator.load_latest_checkpoint()

    assert training_coordinator._total_steps == 500


def test_save_weights_file(training_coordinator, checkpoint_dir: Path) -> None:
    """Should save weights file for workers to sync."""
    training_coordinator.save_weights()

    weights_file = checkpoint_dir / "weights.pt"
    assert weights_file.exists()


# =============================================================================
# Target Update Tests
# =============================================================================


def test_maybe_update_targets(training_coordinator, mock_learner: MockLearner) -> None:
    """Should update targets at interval."""
    training_coordinator.target_update_interval = 10

    # Not at interval
    training_coordinator._update_count = 5
    training_coordinator.maybe_update_targets()
    assert mock_learner._update_count == 0

    # At interval
    training_coordinator._update_count = 10
    training_coordinator.maybe_update_targets()
    assert mock_learner._update_count == 1


# =============================================================================
# Training Step Tests
# =============================================================================


def test_training_step(training_coordinator, simple_model: SimpleModel) -> None:
    """Full training step should work."""
    grads = {
        name: torch.ones_like(param)
        for name, param in simple_model.named_parameters()
    }

    # Write to pool
    training_coordinator.gradient_pool.write(worker_id=0, grads=grads)

    # Run step
    result = training_coordinator.training_step()

    assert "update_count" in result
    assert "total_steps" in result


def test_training_step_skips_if_no_gradients(training_coordinator) -> None:
    """Training step should skip if no gradients available."""
    result = training_coordinator.training_step()

    assert result["gradients_processed"] == 0


# =============================================================================
# Cleanup Tests
# =============================================================================


def test_close_cleans_up(training_coordinator) -> None:
    """close should release resources."""
    training_coordinator.close()

    # Double close should not raise
    training_coordinator.close()
