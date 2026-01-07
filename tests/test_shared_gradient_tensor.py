"""
TDD tests for SharedGradientTensor - zero-copy gradient transfer via shared memory.

Test plan:
1. Basic creation and cleanup
2. Attach to existing buffer
3. Write gradients from model
4. Read gradients back
5. Seqlock: detect concurrent writes
6. Seqlock: reject reads during write-in-progress
7. Multiple write/read cycles
8. Works across separate "processes" (simulated via separate instances)
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from mario_rl.training.shared_gradient_tensor import SharedGradientTensor
from mario_rl.training.shared_gradient_tensor import GradientPacket


class SimpleModel(nn.Module):
    """Simple model for testing gradient transfer."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


@pytest.fixture
def temp_shm_path(tmp_path: Path) -> Path:
    """Create a temporary path for shared memory file."""
    return tmp_path / "test_grads.shm"


@pytest.fixture
def simple_model() -> SimpleModel:
    """Create a simple model for testing."""
    return SimpleModel()


@pytest.fixture
def model_with_gradients(simple_model: SimpleModel) -> SimpleModel:
    """Create a model with computed gradients."""
    x = torch.randn(4, 10)
    loss = simple_model(x).sum()
    loss.backward()
    return simple_model


# === Creation Tests ===


def test_create_buffer_creates_file(
    simple_model: SimpleModel, temp_shm_path: Path
) -> None:
    """Creating a buffer should create the shared memory file."""
    buffer = SharedGradientTensor(
        model=simple_model, shm_path=temp_shm_path, create=True
    )

    assert temp_shm_path.exists()
    assert temp_shm_path.stat().st_size > 0

    buffer.close()


def test_create_buffer_calculates_correct_size(
    simple_model: SimpleModel, temp_shm_path: Path
) -> None:
    """Buffer size should match ring buffer structure."""
    num_slots = 4  # default
    buffer = SharedGradientTensor(
        model=simple_model, shm_path=temp_shm_path, create=True, num_slots=num_slots
    )

    # Calculate expected size: global header + num_slots * (slot header + data)
    expected_numel = sum(p.numel() for p in simple_model.parameters())
    slot_header = 64  # per-slot header
    global_header = 64  # global header
    slot_data = expected_numel * 4  # float32
    expected_bytes = global_header + num_slots * (slot_header + slot_data)

    assert buffer.total_bytes == expected_bytes
    assert buffer.total_numel == expected_numel
    assert buffer.num_slots == num_slots

    buffer.close()


def test_create_buffer_stores_param_layout(
    simple_model: SimpleModel, temp_shm_path: Path
) -> None:
    """Buffer should store parameter layout for reconstruction."""
    buffer = SharedGradientTensor(
        model=simple_model, shm_path=temp_shm_path, create=True
    )

    # Should have layout for each parameter
    assert len(buffer.param_layout) == 4  # fc1.weight, fc1.bias, fc2.weight, fc2.bias

    # Check layout structure
    for name, numel, shape in buffer.param_layout:
        assert isinstance(name, str)
        assert isinstance(numel, int)
        assert isinstance(shape, torch.Size)

    buffer.close()


def test_unlink_removes_file(
    simple_model: SimpleModel, temp_shm_path: Path
) -> None:
    """Unlink should remove the shared memory file."""
    buffer = SharedGradientTensor(
        model=simple_model, shm_path=temp_shm_path, create=True
    )
    buffer.unlink()

    assert not temp_shm_path.exists()


# === Attach Tests ===


def test_attach_to_existing_buffer(
    simple_model: SimpleModel, temp_shm_path: Path
) -> None:
    """Should be able to attach to a buffer created by another instance."""
    creator = SharedGradientTensor(
        model=simple_model, shm_path=temp_shm_path, create=True
    )

    attacher = SharedGradientTensor(
        model=simple_model, shm_path=temp_shm_path, create=False
    )

    assert attacher.total_numel == creator.total_numel
    assert attacher.total_bytes == creator.total_bytes

    attacher.close()
    creator.unlink()


def test_attach_fails_if_file_missing(
    simple_model: SimpleModel, temp_shm_path: Path
) -> None:
    """Attaching to non-existent file should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        SharedGradientTensor(
            model=simple_model, shm_path=temp_shm_path, create=False
        )


# === Write Tests ===


def test_write_gradients_from_model(
    model_with_gradients: SimpleModel, temp_shm_path: Path
) -> None:
    """Writing gradients should succeed and return True."""
    buffer = SharedGradientTensor(
        model=model_with_gradients, shm_path=temp_shm_path, create=True
    )

    grads = {
        name: param.grad.clone()
        for name, param in model_with_gradients.named_parameters()
        if param.grad is not None
    }

    result = buffer.write(grads)

    assert result is True
    buffer.unlink()


def test_write_sets_ready_flag(
    model_with_gradients: SimpleModel, temp_shm_path: Path
) -> None:
    """After write, ready flag should be set."""
    buffer = SharedGradientTensor(
        model=model_with_gradients, shm_path=temp_shm_path, create=True
    )

    grads = {
        name: param.grad.clone()
        for name, param in model_with_gradients.named_parameters()
        if param.grad is not None
    }

    buffer.write(grads)

    assert buffer.is_ready() is True
    buffer.unlink()


def test_write_updates_seq_to_even(
    model_with_gradients: SimpleModel, temp_shm_path: Path
) -> None:
    """After write, sequence number should be even (write complete)."""
    buffer = SharedGradientTensor(
        model=model_with_gradients, shm_path=temp_shm_path, create=True
    )

    grads = {
        name: param.grad.clone()
        for name, param in model_with_gradients.named_parameters()
        if param.grad is not None
    }

    buffer.write(grads)

    seq = buffer._read_seq()
    assert seq % 2 == 0  # Even = write complete

    buffer.unlink()


# === Read Tests ===


def test_read_returns_gradients(
    model_with_gradients: SimpleModel, temp_shm_path: Path
) -> None:
    """Reading after write should return gradient dict."""
    buffer = SharedGradientTensor(
        model=model_with_gradients, shm_path=temp_shm_path, create=True
    )

    grads = {
        name: param.grad.clone()
        for name, param in model_with_gradients.named_parameters()
        if param.grad is not None
    }

    buffer.write(grads)
    result = buffer.read()

    assert result is not None
    assert isinstance(result, GradientPacket)
    assert len(result.grads) == len(grads)

    buffer.unlink()


def test_read_returns_correct_values(
    model_with_gradients: SimpleModel, temp_shm_path: Path
) -> None:
    """Read gradients should match written gradients."""
    buffer = SharedGradientTensor(
        model=model_with_gradients, shm_path=temp_shm_path, create=True
    )

    grads = {
        name: param.grad.clone()
        for name, param in model_with_gradients.named_parameters()
        if param.grad is not None
    }

    buffer.write(grads)
    result = buffer.read()

    for name, expected_grad in grads.items():
        assert name in result.grads
        assert torch.allclose(result.grads[name], expected_grad)

    buffer.unlink()


def test_read_returns_correct_shapes(
    model_with_gradients: SimpleModel, temp_shm_path: Path
) -> None:
    """Read gradients should have correct shapes."""
    buffer = SharedGradientTensor(
        model=model_with_gradients, shm_path=temp_shm_path, create=True
    )

    grads = {
        name: param.grad.clone()
        for name, param in model_with_gradients.named_parameters()
        if param.grad is not None
    }

    buffer.write(grads)
    result = buffer.read()

    for name, expected_grad in grads.items():
        assert result.grads[name].shape == expected_grad.shape

    buffer.unlink()


def test_read_clears_ready_flag(
    model_with_gradients: SimpleModel, temp_shm_path: Path
) -> None:
    """After read, ready flag should be cleared."""
    buffer = SharedGradientTensor(
        model=model_with_gradients, shm_path=temp_shm_path, create=True
    )

    grads = {
        name: param.grad.clone()
        for name, param in model_with_gradients.named_parameters()
        if param.grad is not None
    }

    buffer.write(grads)
    buffer.read()

    assert buffer.is_ready() is False

    buffer.unlink()


def test_read_returns_none_when_not_ready(
    simple_model: SimpleModel, temp_shm_path: Path
) -> None:
    """Reading when no data written should return None."""
    buffer = SharedGradientTensor(
        model=simple_model, shm_path=temp_shm_path, create=True
    )

    result = buffer.read()

    assert result is None

    buffer.unlink()


# === Seqlock Tests ===


def test_read_returns_none_during_write(
    model_with_gradients: SimpleModel, temp_shm_path: Path
) -> None:
    """Reading while seq is odd (write in progress) should return None."""
    buffer = SharedGradientTensor(
        model=model_with_gradients, shm_path=temp_shm_path, create=True
    )

    # Manually set seq to odd (simulating write in progress)
    buffer._write_seq(1)
    buffer._write_ready(1)

    result = buffer.read()

    assert result is None

    buffer.unlink()


def test_read_detects_concurrent_write(
    model_with_gradients: SimpleModel, temp_shm_path: Path
) -> None:
    """Seqlock should detect if data changes during read."""
    buffer = SharedGradientTensor(
        model=model_with_gradients, shm_path=temp_shm_path, create=True
    )

    grads = {
        name: param.grad.clone()
        for name, param in model_with_gradients.named_parameters()
        if param.grad is not None
    }
    buffer.write(grads)

    # Get the slot that was written to
    write_idx, _ = buffer._read_global_header()
    slot_idx = (write_idx - 1) % buffer.num_slots

    # Simulate concurrent write by changing seq after _read_slot_header
    # but before the final seq check
    def mock_read() -> GradientPacket | None:
        # First seq check passes
        seq1, ready, version, worker_id, timesteps, episodes, loss, q_mean, td_error, avg_reward, avg_speed, entropy, deaths, flags, best_x = buffer._read_slot_header(slot_idx)
        if ready != 1 or (seq1 % 2) == 1:
            return None

        # Read the data
        slot_array = buffer._slot_arrays[slot_idx]
        grads_result: dict[str, torch.Tensor] = {}
        offset = 0
        for name, numel, shape in buffer.param_layout:
            grad_flat = torch.from_numpy(slot_array[offset : offset + numel].copy())
            grads_result[name] = grad_flat.view(shape)
            offset += numel

        # Simulate concurrent write by changing seq
        buffer._write_seq(seq1 + 2, slot_idx)

        # Now the second seq check should fail
        seq2 = buffer._read_seq(slot_idx)
        if seq1 != seq2:
            return None

        return GradientPacket(
            grads=grads_result,
            version=version,
            worker_id=worker_id,
            timesteps=timesteps,
            episodes=episodes,
            loss=loss,
            q_mean=q_mean,
            td_error=td_error,
            avg_reward=avg_reward,
            avg_speed=avg_speed,
            entropy=entropy,
            deaths=deaths,
            flags=flags,
            best_x=best_x,
        )

    result = mock_read()
    assert result is None  # Should detect the concurrent write

    buffer.unlink()


# === Multiple Roundtrip Tests ===


def test_multiple_write_read_cycles(
    simple_model: SimpleModel, temp_shm_path: Path
) -> None:
    """Should handle multiple write/read cycles correctly."""
    buffer = SharedGradientTensor(
        model=simple_model, shm_path=temp_shm_path, create=True
    )

    for i in range(5):
        # Generate different gradients each time
        x = torch.randn(4, 10)
        loss = simple_model(x).sum()
        simple_model.zero_grad()
        loss.backward()

        grads = {
            name: param.grad.clone()
            for name, param in simple_model.named_parameters()
            if param.grad is not None
        }

        buffer.write(grads)
        result = buffer.read()

        assert result is not None
        for name, expected in grads.items():
            assert torch.allclose(result.grads[name], expected), f"Mismatch on cycle {i}"

    buffer.unlink()


# === Cross-Instance Tests ===


def test_writer_reader_separate_instances(
    model_with_gradients: SimpleModel, temp_shm_path: Path
) -> None:
    """Writer and reader as separate instances should work."""
    # Writer creates buffer
    writer = SharedGradientTensor(
        model=model_with_gradients, shm_path=temp_shm_path, create=True
    )

    grads = {
        name: param.grad.clone()
        for name, param in model_with_gradients.named_parameters()
        if param.grad is not None
    }
    writer.write(grads)
    writer.close()  # Close but don't unlink

    # Reader attaches to existing buffer
    reader = SharedGradientTensor(
        model=model_with_gradients, shm_path=temp_shm_path, create=False
    )

    result = reader.read()

    assert result is not None
    for name, expected in grads.items():
        assert torch.allclose(result.grads[name], expected)

    reader.unlink()


# === GPU Tests ===


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_write_gpu_gradients(temp_shm_path: Path) -> None:
    """Should handle GPU tensors by copying to CPU."""
    model = SimpleModel().cuda()

    x = torch.randn(4, 10).cuda()
    loss = model(x).sum()
    loss.backward()

    buffer = SharedGradientTensor(
        model=model, shm_path=temp_shm_path, create=True
    )

    grads = {
        name: param.grad.clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }

    result = buffer.write(grads)
    assert result is True

    result = buffer.read()
    assert result is not None

    # Read gradients should be on CPU
    for name, grad in result.grads.items():
        assert grad.device.type == "cpu"
        # Values should match (compare on same device)
        assert torch.allclose(grad, grads[name].cpu())

    buffer.unlink()


# === Ring Buffer Tests ===


def test_ring_buffer_multiple_writes_before_read(
    simple_model: SimpleModel, temp_shm_path: Path
) -> None:
    """Should be able to write multiple times before reading."""
    num_slots = 4
    buffer = SharedGradientTensor(
        model=simple_model, shm_path=temp_shm_path, create=True, num_slots=num_slots
    )

    all_grads = []
    for i in range(3):  # Write 3 times
        x = torch.randn(4, 10)
        loss = simple_model(x).sum()
        simple_model.zero_grad()
        loss.backward()

        grads = {
            name: param.grad.clone()
            for name, param in simple_model.named_parameters()
            if param.grad is not None
        }
        all_grads.append(grads)
        buffer.write(grads)

    # Should be able to read all 3 (in order)
    for i in range(3):
        result = buffer.read()
        assert result is not None
        for name, expected in all_grads[i].items():
            assert torch.allclose(result.grads[name], expected), f"Mismatch on read {i}"

    buffer.unlink()


def test_ring_buffer_wraps_around(
    simple_model: SimpleModel, temp_shm_path: Path
) -> None:
    """Ring buffer should wrap around when full."""
    num_slots = 2
    buffer = SharedGradientTensor(
        model=simple_model, shm_path=temp_shm_path, create=True, num_slots=num_slots
    )

    # Write 4 times (wraps around with 2 slots)
    for i in range(4):
        x = torch.randn(4, 10)
        loss = simple_model(x).sum()
        simple_model.zero_grad()
        loss.backward()

        grads = {
            name: param.grad.clone()
            for name, param in simple_model.named_parameters()
            if param.grad is not None
        }
        buffer.write(grads)

        # Read immediately to free slot
        result = buffer.read()
        assert result is not None

    buffer.unlink()


def test_ring_buffer_read_latest(
    simple_model: SimpleModel, temp_shm_path: Path
) -> None:
    """read_latest should skip old data and return newest."""
    num_slots = 4
    buffer = SharedGradientTensor(
        model=simple_model, shm_path=temp_shm_path, create=True, num_slots=num_slots
    )

    # Write 3 times without reading
    last_grads = None
    for i in range(3):
        x = torch.randn(4, 10)
        loss = simple_model(x).sum()
        simple_model.zero_grad()
        loss.backward()

        grads = {
            name: param.grad.clone()
            for name, param in simple_model.named_parameters()
            if param.grad is not None
        }
        last_grads = grads
        buffer.write(grads)

    # read_latest should return the newest (3rd write)
    result = buffer.read_latest()
    assert result is not None

    for name, expected in last_grads.items():
        assert torch.allclose(result.grads[name], expected)

    # After read_latest, read() should return None (all caught up)
    assert buffer.read() is None

    buffer.unlink()


def test_ring_buffer_count_ready(
    simple_model: SimpleModel, temp_shm_path: Path
) -> None:
    """count_ready should return number of slots with data."""
    num_slots = 4
    buffer = SharedGradientTensor(
        model=simple_model, shm_path=temp_shm_path, create=True, num_slots=num_slots
    )

    assert buffer.count_ready() == 0

    # Write 2 times
    for i in range(2):
        x = torch.randn(4, 10)
        loss = simple_model(x).sum()
        simple_model.zero_grad()
        loss.backward()

        grads = {
            name: param.grad.clone()
            for name, param in simple_model.named_parameters()
            if param.grad is not None
        }
        buffer.write(grads)

    assert buffer.count_ready() == 2

    # Read one
    buffer.read()
    assert buffer.count_ready() == 1

    buffer.unlink()
