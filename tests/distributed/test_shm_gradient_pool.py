"""Tests for SharedGradientPool behavior.

These tests verify the SharedGradientPool interface:
- Creates per-worker gradient buffers
- Reads/writes gradients correctly
- Handles multiple workers
"""

from pathlib import Path
from dataclasses import dataclass

import pytest
import torch
from torch import nn


# =============================================================================
# Mock Model
# =============================================================================


class SimpleModel(nn.Module):
    """Simple model for testing gradient transfer."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_model() -> SimpleModel:
    """Create a simple model."""
    return SimpleModel()


@pytest.fixture
def model_with_gradients(simple_model: SimpleModel) -> SimpleModel:
    """Create a model with computed gradients."""
    x = torch.randn(4, 10)
    loss = simple_model(x).sum()
    loss.backward()
    return simple_model


@pytest.fixture
def shm_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for shared memory files."""
    shm = tmp_path / "shm"
    shm.mkdir()
    return shm


@pytest.fixture
def gradient_pool(simple_model: SimpleModel, shm_dir: Path):
    """Create a SharedGradientPool for testing."""
    from mario_rl.distributed.shm_gradient_pool import SharedGradientPool

    pool = SharedGradientPool(
        num_workers=2,
        model=simple_model,
        shm_dir=shm_dir,
    )
    yield pool
    pool.close()


# =============================================================================
# Creation Tests
# =============================================================================


def test_pool_creates_successfully(gradient_pool) -> None:
    """Pool should create without error."""
    assert gradient_pool is not None


def test_pool_creates_per_worker_buffers(gradient_pool) -> None:
    """Pool should have one buffer per worker."""
    assert len(gradient_pool.buffers) == 2


def test_pool_shm_files_exist(gradient_pool, shm_dir: Path) -> None:
    """Shared memory files should be created."""
    files = list(shm_dir.glob("*.shm"))
    assert len(files) == 2


# =============================================================================
# Read/Write Tests
# =============================================================================


def test_write_and_read_worker(model_with_gradients: SimpleModel, shm_dir: Path) -> None:
    """Should write and read from specific worker."""
    from mario_rl.distributed.shm_gradient_pool import SharedGradientPool

    pool = SharedGradientPool(
        num_workers=2,
        model=model_with_gradients,
        shm_dir=shm_dir,
    )

    grads = {
        name: param.grad.clone()
        for name, param in model_with_gradients.named_parameters()
        if param.grad is not None
    }

    # Write to worker 0
    pool.write(worker_id=0, grads=grads)

    # Read from worker 0
    result = pool.read(worker_id=0)

    assert result is not None
    for name, expected in grads.items():
        assert torch.allclose(result.grads[name], expected)

    pool.close()


def test_read_all_workers(model_with_gradients: SimpleModel, shm_dir: Path) -> None:
    """read_all should return gradients from all workers."""
    from mario_rl.distributed.shm_gradient_pool import SharedGradientPool

    pool = SharedGradientPool(
        num_workers=2,
        model=model_with_gradients,
        shm_dir=shm_dir,
    )

    grads = {
        name: param.grad.clone()
        for name, param in model_with_gradients.named_parameters()
        if param.grad is not None
    }

    # Write to both workers
    pool.write(worker_id=0, grads=grads, worker_meta={"id": 0})
    pool.write(worker_id=1, grads=grads, worker_meta={"id": 1})

    # Read all
    results = pool.read_all()

    assert len(results) == 2

    pool.close()


def test_read_empty_returns_none(gradient_pool) -> None:
    """Reading from empty buffer should return None."""
    result = gradient_pool.read(worker_id=0)
    assert result is None


# =============================================================================
# Worker Isolation Tests
# =============================================================================


def test_workers_have_separate_buffers(model_with_gradients: SimpleModel, shm_dir: Path) -> None:
    """Each worker should have its own buffer."""
    from mario_rl.distributed.shm_gradient_pool import SharedGradientPool

    pool = SharedGradientPool(
        num_workers=2,
        model=model_with_gradients,
        shm_dir=shm_dir,
    )

    grads = {
        name: param.grad.clone()
        for name, param in model_with_gradients.named_parameters()
        if param.grad is not None
    }

    # Write to worker 0 only
    pool.write(worker_id=0, grads=grads)

    # Worker 0 should have data
    assert pool.read(worker_id=0) is not None

    # Worker 1 should be empty
    assert pool.read(worker_id=1) is None

    pool.close()


# =============================================================================
# Attach Tests
# =============================================================================


def test_attach_to_existing(simple_model: SimpleModel, shm_dir: Path) -> None:
    """Should attach to existing pool."""
    from mario_rl.distributed.shm_gradient_pool import SharedGradientPool

    # Create pool
    creator = SharedGradientPool(
        num_workers=2,
        model=simple_model,
        shm_dir=shm_dir,
        create=True,
    )

    # Attach to existing
    attacher = SharedGradientPool(
        num_workers=2,
        model=simple_model,
        shm_dir=shm_dir,
        create=False,
    )

    assert len(attacher.buffers) == 2

    creator.close()
    attacher.close()


# =============================================================================
# Cleanup Tests
# =============================================================================


def test_close_cleans_up(simple_model: SimpleModel, shm_dir: Path) -> None:
    """close should release resources."""
    from mario_rl.distributed.shm_gradient_pool import SharedGradientPool

    pool = SharedGradientPool(
        num_workers=2,
        model=simple_model,
        shm_dir=shm_dir,
    )
    pool.close()

    # Should not raise on double close
    pool.close()


def test_unlink_removes_files(simple_model: SimpleModel, shm_dir: Path) -> None:
    """unlink should remove shared memory files."""
    from mario_rl.distributed.shm_gradient_pool import SharedGradientPool

    pool = SharedGradientPool(
        num_workers=2,
        model=simple_model,
        shm_dir=shm_dir,
    )
    pool.unlink()

    files = list(shm_dir.glob("*.shm"))
    assert len(files) == 0
