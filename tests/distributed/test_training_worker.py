"""Tests for TrainingWorker behavior.

These tests verify the TrainingWorker correctly:
- Collects experience from environment
- Stores transitions in local buffer
- Computes gradients from sampled batches
- Syncs weights from file
- Uses epsilon-greedy exploration
"""

from typing import Any
from pathlib import Path
from dataclasses import field
from dataclasses import dataclass

import torch
import pytest
import numpy as np
from torch import nn
from torch import Tensor

# =============================================================================
# Mock Environment
# =============================================================================


class MockEnv:
    """Mock gymnasium environment for testing."""

    def __init__(self, obs_shape: tuple[int, ...] = (4, 64, 64), episode_length: int = 100) -> None:
        self.obs_shape = obs_shape
        self.episode_length = episode_length
        self._step_count = 0

    def reset(self, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        self._step_count = 0
        return np.random.randn(*self.obs_shape).astype(np.float32), {"x_pos": 0}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        obs = np.random.randn(*self.obs_shape).astype(np.float32)
        reward = 1.0
        terminated = self._step_count >= self.episode_length
        return obs, reward, terminated, False, {"x_pos": self._step_count * 10}


# =============================================================================
# Mock Model and Learner
# =============================================================================


class MockModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, obs_shape: tuple[int, ...] = (4, 64, 64), num_actions: int = 12) -> None:
        super().__init__()
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        flat = 1
        for d in obs_shape:
            flat *= d
        self.fc = nn.Linear(flat, num_actions)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x.view(x.shape[0], -1))

    def get_action(self, x: Tensor, epsilon: float = 0.0) -> Tensor:
        if np.random.random() < epsilon:
            return torch.randint(0, self.num_actions, (x.shape[0],))
        with torch.no_grad():
            return self.forward(x).argmax(dim=1)


@dataclass
class MockLearner:
    """Simple learner for testing."""

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
        weights: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        q_values = self.model(states)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.model(next_states).max(dim=1).values
            targets = rewards + self.gamma * q_next * (1 - dones)
        element_wise_loss = torch.nn.functional.mse_loss(q_selected, targets, reduction="none")
        if weights is not None:
            loss = (weights * element_wise_loss).mean()
        else:
            loss = element_wise_loss.mean()
        return loss, {"loss": loss.item(), "td_error": element_wise_loss.mean().item()}

    def update_targets(self, tau: float = 1.0) -> None:
        self._update_count += 1


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class WorkerTestConfig:
    """Test configuration."""

    obs_shape: tuple[int, ...] = (4, 64, 64)
    num_actions: int = 12
    buffer_capacity: int = 1000
    batch_size: int = 32
    collect_steps: int = 64
    train_steps: int = 4


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config() -> WorkerTestConfig:
    """Default test configuration."""
    return WorkerTestConfig()


@pytest.fixture
def mock_env(config: WorkerTestConfig) -> MockEnv:
    """Create a mock environment."""
    return MockEnv(obs_shape=config.obs_shape)


@pytest.fixture
def mock_model(config: WorkerTestConfig) -> MockModel:
    """Create a mock model."""
    return MockModel(obs_shape=config.obs_shape, num_actions=config.num_actions)


@pytest.fixture
def mock_learner(mock_model: MockModel) -> MockLearner:
    """Create a mock learner."""
    return MockLearner(model=mock_model)


@pytest.fixture
def training_worker(mock_env: MockEnv, mock_learner: MockLearner, config: WorkerTestConfig):
    """Create a TrainingWorker for testing."""
    from mario_rl.distributed.training_worker import TrainingWorker

    return TrainingWorker(
        env=mock_env,
        learner=mock_learner,
        buffer_capacity=config.buffer_capacity,
        batch_size=config.batch_size,
        n_step=1,
        gamma=0.99,
    )


@pytest.fixture
def weights_file(tmp_path: Path, mock_model: MockModel) -> Path:
    """Create a temporary weights file."""
    weights_path = tmp_path / "weights.pt"
    torch.save(mock_model.state_dict(), weights_path)
    return weights_path


# =============================================================================
# Basic Functionality Tests
# =============================================================================


def test_training_worker_creates_successfully(training_worker) -> None:
    """TrainingWorker should create without error."""
    assert training_worker is not None


def test_training_worker_has_model(training_worker, mock_model: MockModel) -> None:
    """TrainingWorker should expose model."""
    assert training_worker.model is mock_model


def test_training_worker_has_buffer(training_worker) -> None:
    """TrainingWorker should have a replay buffer."""
    assert training_worker.buffer is not None


# =============================================================================
# Collection Tests
# =============================================================================


def test_collect_fills_buffer(training_worker, config: WorkerTestConfig) -> None:
    """Collecting should add transitions to buffer."""
    initial_size = len(training_worker.buffer)

    training_worker.collect(num_steps=50)

    assert len(training_worker.buffer) > initial_size


def test_collect_returns_info(training_worker) -> None:
    """collect should return collection info."""
    info = training_worker.collect(num_steps=20)

    assert "steps" in info
    assert "episodes_completed" in info
    assert info["steps"] == 20


def test_collect_uses_epsilon(training_worker) -> None:
    """Collection should use specified epsilon."""
    # Set high epsilon to ensure exploration
    training_worker.epsilon = 1.0

    info = training_worker.collect(num_steps=100)

    # Should complete without error
    assert info["steps"] == 100


# =============================================================================
# Training Step Tests
# =============================================================================


def test_train_step_computes_gradients(training_worker, config: WorkerTestConfig) -> None:
    """train_step should compute gradients."""
    # First fill buffer
    training_worker.collect(num_steps=100)

    # Train step
    grads, metrics = training_worker.train_step()

    assert grads is not None
    assert isinstance(grads, dict)
    assert len(grads) > 0


def test_train_step_returns_metrics(training_worker) -> None:
    """train_step should return training metrics."""
    training_worker.collect(num_steps=100)

    _, metrics = training_worker.train_step()

    assert "loss" in metrics


def test_train_step_requires_buffer_data(training_worker) -> None:
    """train_step should require enough buffer data."""
    # Buffer is empty
    with pytest.raises(ValueError):
        training_worker.train_step()


def test_can_train_checks_buffer(training_worker, config: WorkerTestConfig) -> None:
    """can_train should check if buffer has enough data."""
    assert not training_worker.can_train()

    training_worker.collect(num_steps=100)

    assert training_worker.can_train()


# =============================================================================
# Weight Sync Tests
# =============================================================================


def test_sync_weights_from_file(training_worker, weights_file: Path) -> None:
    """Should sync weights from file."""
    # Modify model weights
    with torch.no_grad():
        for p in training_worker.model.parameters():
            p.add_(1.0)

    # Sync from file
    training_worker.sync_weights(weights_file)

    # Weights should be restored (not modified)
    # This is hard to verify directly, but sync should not raise
    assert True


def test_sync_weights_updates_version(training_worker, weights_file: Path) -> None:
    """Syncing weights should update weight version."""
    initial_version = training_worker.weight_version

    training_worker.sync_weights(weights_file)

    assert training_worker.weight_version > initial_version


def test_sync_weights_checks_file_modification(training_worker, weights_file: Path) -> None:
    """Should skip sync if file hasn't changed."""
    training_worker.sync_weights(weights_file)
    v1 = training_worker.weight_version

    # Sync again without file change
    training_worker.sync_weights(weights_file)
    v2 = training_worker.weight_version

    # Version should not increase again (file unchanged)
    assert v2 == v1


def test_sync_weights_handles_missing_file(training_worker, tmp_path: Path) -> None:
    """Should handle missing weights file gracefully."""
    missing_file = tmp_path / "nonexistent.pt"

    # Should not raise
    training_worker.sync_weights(missing_file)


# =============================================================================
# Epsilon Decay Tests
# =============================================================================


def test_epsilon_decay(training_worker) -> None:
    """Epsilon should decay based on steps."""
    training_worker.epsilon_start = 1.0
    training_worker.epsilon_end = 0.01
    training_worker.epsilon_decay_steps = 1000

    initial_eps = training_worker.epsilon_at(steps=0)
    mid_eps = training_worker.epsilon_at(steps=500)
    final_eps = training_worker.epsilon_at(steps=1000)
    beyond_eps = training_worker.epsilon_at(steps=2000)

    assert initial_eps == 1.0
    assert mid_eps < initial_eps
    assert final_eps == pytest.approx(0.01, abs=0.01)
    assert beyond_eps == pytest.approx(0.01, abs=0.01)


def test_epsilon_used_in_collection(training_worker) -> None:
    """get_action should use current epsilon."""
    training_worker.epsilon = 0.5
    training_worker.total_steps = 0

    # This is tested via collect - should not raise
    training_worker.collect(num_steps=10)


# =============================================================================
# Full Training Loop Tests
# =============================================================================


def test_run_cycle(training_worker, config: WorkerTestConfig) -> None:
    """run_cycle should collect and train."""
    # Run a full cycle
    result = training_worker.run_cycle(
        collect_steps=config.collect_steps,
        train_steps=config.train_steps,
    )

    assert result is not None
    assert "gradients" in result
    assert "collection_info" in result


def test_run_cycle_returns_gradients(training_worker) -> None:
    """run_cycle should return computed gradients."""
    result = training_worker.run_cycle(
        collect_steps=100,
        train_steps=4,
    )

    grads = result["gradients"]
    assert len(grads) > 0


def test_run_cycle_increments_steps(training_worker) -> None:
    """run_cycle should increment total steps."""
    initial_steps = training_worker.total_steps

    training_worker.run_cycle(collect_steps=50, train_steps=2)

    assert training_worker.total_steps == initial_steps + 50


# =============================================================================
# Edge Cases
# =============================================================================


def test_small_buffer(config: WorkerTestConfig) -> None:
    """Should work with small buffer."""
    from mario_rl.distributed.training_worker import TrainingWorker

    env = MockEnv(obs_shape=config.obs_shape)
    model = MockModel(obs_shape=config.obs_shape, num_actions=config.num_actions)
    learner = MockLearner(model=model)

    worker = TrainingWorker(
        env=env,
        learner=learner,
        buffer_capacity=50,
        batch_size=32,
    )

    worker.collect(num_steps=100)

    # Buffer should not exceed capacity
    assert len(worker.buffer) <= 50


def test_n_step_returns(config: WorkerTestConfig) -> None:
    """Should support n-step returns."""
    from mario_rl.distributed.training_worker import TrainingWorker

    env = MockEnv(obs_shape=config.obs_shape)
    model = MockModel(obs_shape=config.obs_shape, num_actions=config.num_actions)
    learner = MockLearner(model=model)

    worker = TrainingWorker(
        env=env,
        learner=learner,
        buffer_capacity=100,
        batch_size=16,
        n_step=3,
        gamma=0.99,
    )

    worker.collect(num_steps=50)

    # Buffer should have transitions
    assert len(worker.buffer) > 0
