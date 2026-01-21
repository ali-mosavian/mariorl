"""Tests for EliteBuffer - protected buffer for high-quality experiences."""

from __future__ import annotations

import torch
import pytest
import numpy as np

from mario_rl.core.types import Transition
from mario_rl.core.elite_buffer import EliteBuffer
from mario_rl.core.elite_buffer import EliteTransition


@pytest.fixture
def sample_transition() -> Transition:
    """Create a sample transition for testing."""
    return Transition(
        state=np.random.rand(4, 64, 64).astype(np.float32),
        action=1,
        reward=1.0,
        next_state=np.random.rand(4, 64, 64).astype(np.float32),
        done=False,
        flag_get=False,
        max_x=100,
    )


@pytest.fixture
def flag_transition() -> Transition:
    """Create a transition with flag capture."""
    return Transition(
        state=np.random.rand(4, 64, 64).astype(np.float32),
        action=2,
        reward=10.0,
        next_state=np.random.rand(4, 64, 64).astype(np.float32),
        done=True,
        flag_get=True,
        max_x=3000,
    )


def test_create_empty_buffer() -> None:
    """Test creating an empty elite buffer."""
    buffer = EliteBuffer(capacity=100)
    assert len(buffer) == 0
    assert buffer.capacity == 100


def test_add_single_transition(sample_transition: Transition) -> None:
    """Test adding a single transition."""
    buffer = EliteBuffer(capacity=100)
    added = buffer.add(sample_transition, quality=500.0)
    assert added is True
    assert len(buffer) == 1


def test_add_episode(sample_transition: Transition) -> None:
    """Test adding an entire episode."""
    buffer = EliteBuffer(capacity=100)
    transitions = [sample_transition for _ in range(10)]

    added = buffer.add_episode(
        transitions=transitions,
        max_x=1500,
        flag_captured=False,
        episode_reward=50.0,
    )

    assert added == 10
    assert len(buffer) == 10


def test_add_episode_with_flag(sample_transition: Transition) -> None:
    """Test adding a flag capture episode gets higher quality."""
    _buffer = EliteBuffer(capacity=100)
    _transitions = [sample_transition for _ in range(5)]

    # Without flag
    quality_no_flag = EliteBuffer.compute_quality(
        max_x=1000,
        flag_captured=False,
        episode_reward=50.0,
    )

    # With flag
    quality_with_flag = EliteBuffer.compute_quality(
        max_x=1000,
        flag_captured=True,
        episode_reward=50.0,
    )

    # Flag should add 1000 points
    assert quality_with_flag == quality_no_flag + 1000.0


def test_fills_to_capacity(sample_transition: Transition) -> None:
    """Test buffer fills up to capacity."""
    buffer = EliteBuffer(capacity=10)

    for i in range(15):
        buffer.add(sample_transition, quality=float(i))

    assert len(buffer) == 10


def test_replaces_lowest_quality(sample_transition: Transition) -> None:
    """Test that lowest quality gets replaced by higher quality."""
    buffer = EliteBuffer(capacity=5)

    # Fill with quality 0-4
    for i in range(5):
        buffer.add(sample_transition, quality=float(i))

    assert len(buffer) == 5
    assert buffer._min_quality == 0.0

    # Add higher quality - should replace lowest (0)
    buffer.add(sample_transition, quality=10.0)

    assert len(buffer) == 5
    assert buffer._min_quality == 1.0  # 0 was replaced


def test_rejects_lower_quality_when_full(sample_transition: Transition) -> None:
    """Test that lower quality transitions are rejected when full."""
    buffer = EliteBuffer(capacity=5)

    # Fill with quality 10-14
    for i in range(5):
        buffer.add(sample_transition, quality=10.0 + i)

    assert len(buffer) == 5
    min_before = buffer._min_quality

    # Try to add lower quality - should be rejected
    added = buffer.add(sample_transition, quality=5.0)

    assert added is False
    assert len(buffer) == 5
    assert buffer._min_quality == min_before


def test_cannot_sample_empty() -> None:
    """Test sampling from empty buffer returns None."""
    buffer = EliteBuffer(capacity=100)
    result = buffer.sample(10)
    assert result is None


def test_sample_returns_tensors(sample_transition: Transition) -> None:
    """Test sampling returns proper tensor batch."""
    buffer = EliteBuffer(capacity=100)

    for _ in range(20):
        buffer.add(sample_transition, quality=100.0)

    batch = buffer.sample(10)

    assert batch is not None
    assert "states" in batch
    assert "actions" in batch
    assert "rewards" in batch
    assert "next_states" in batch
    assert "dones" in batch
    assert "weights" in batch

    assert batch["states"].shape == (10, 4, 64, 64)
    assert batch["actions"].shape == (10,)
    assert batch["rewards"].shape == (10,)
    assert batch["weights"].shape == (10,)


def test_sample_respects_batch_size(sample_transition: Transition) -> None:
    """Test sampling respects batch size limit."""
    buffer = EliteBuffer(capacity=100)

    for _ in range(5):
        buffer.add(sample_transition, quality=100.0)

    # Request more than available
    batch = buffer.sample(10)

    assert batch is not None
    # Should return only what's available
    assert batch["states"].shape[0] == 5


def test_sample_device_placement(sample_transition: Transition) -> None:
    """Test samples are placed on correct device."""
    buffer = EliteBuffer(capacity=100)

    for _ in range(10):
        buffer.add(sample_transition, quality=100.0)

    batch = buffer.sample(5, device="cpu")

    assert batch["states"].device == torch.device("cpu")


def test_can_sample(sample_transition: Transition) -> None:
    """Test can_sample check."""
    buffer = EliteBuffer(capacity=100)

    assert buffer.can_sample(5) is False

    for _ in range(5):
        buffer.add(sample_transition, quality=100.0)

    assert buffer.can_sample(5) is True
    assert buffer.can_sample(10) is False


def test_quality_includes_max_x() -> None:
    """Test quality includes max X position."""
    quality = EliteBuffer.compute_quality(max_x=1500, flag_captured=False, episode_reward=0.0)
    assert quality == 1500.0


def test_quality_includes_flag_bonus() -> None:
    """Test quality includes flag bonus."""
    no_flag = EliteBuffer.compute_quality(max_x=1000, flag_captured=False, episode_reward=0.0)
    with_flag = EliteBuffer.compute_quality(max_x=1000, flag_captured=True, episode_reward=0.0)

    assert with_flag - no_flag == 1000.0


def test_quality_includes_reward_bonus() -> None:
    """Test quality includes reward bonus (scaled by 0.1)."""
    no_reward = EliteBuffer.compute_quality(max_x=1000, flag_captured=False, episode_reward=0.0)
    with_reward = EliteBuffer.compute_quality(max_x=1000, flag_captured=False, episode_reward=100.0)

    assert with_reward - no_reward == 10.0  # 100 * 0.1


def test_quality_composite() -> None:
    """Test full quality computation."""
    quality = EliteBuffer.compute_quality(max_x=2000, flag_captured=True, episode_reward=500.0)

    expected = 2000.0 + 1000.0 + (500.0 * 0.1)  # max_x + flag + reward*0.1
    assert quality == expected


def test_stats_empty() -> None:
    """Test stats on empty buffer."""
    buffer = EliteBuffer(capacity=100)
    stats = buffer.get_stats()

    assert stats["size"] == 0
    assert stats["fill_pct"] == 0.0


def test_stats_with_data(sample_transition: Transition) -> None:
    """Test stats with data in buffer."""
    buffer = EliteBuffer(capacity=100)

    buffer.add(sample_transition, quality=100.0)
    buffer.add(sample_transition, quality=200.0)
    buffer.add(sample_transition, quality=300.0)

    stats = buffer.get_stats()

    assert stats["size"] == 3
    assert stats["capacity"] == 100
    assert stats["fill_pct"] == 3.0
    assert stats["min_quality"] == 100.0
    assert stats["max_quality"] == 300.0
    assert stats["mean_quality"] == 200.0


def test_total_added_count(sample_transition: Transition) -> None:
    """Test total_added tracks all additions."""
    buffer = EliteBuffer(capacity=5)

    for i in range(10):
        buffer.add(sample_transition, quality=float(i))

    stats = buffer.get_stats()
    assert stats["total_added"] == 10  # All 10 were added, some replaced


def test_unique_episodes(sample_transition: Transition) -> None:
    """Test unique episode tracking."""
    buffer = EliteBuffer(capacity=100)

    # Add two episodes
    buffer.add_episode([sample_transition] * 5, max_x=100, flag_captured=False, episode_reward=10.0)
    buffer.add_episode([sample_transition] * 3, max_x=200, flag_captured=False, episode_reward=20.0)

    stats = buffer.get_stats()
    assert stats["unique_episodes"] == 2


def test_rejects_below_threshold(sample_transition: Transition) -> None:
    """Test transitions below min_quality_to_add are rejected."""
    buffer = EliteBuffer(capacity=100, min_quality_to_add=500.0)

    added = buffer.add(sample_transition, quality=400.0)
    assert added is False
    assert len(buffer) == 0


def test_accepts_above_threshold(sample_transition: Transition) -> None:
    """Test transitions above min_quality_to_add are accepted."""
    buffer = EliteBuffer(capacity=100, min_quality_to_add=500.0)

    added = buffer.add(sample_transition, quality=600.0)
    assert added is True
    assert len(buffer) == 1


def test_episode_rejected_below_threshold(sample_transition: Transition) -> None:
    """Test episodes below min_quality_to_add are rejected."""
    buffer = EliteBuffer(capacity=100, min_quality_to_add=5000.0)

    # This episode has quality ~1500 (max_x + reward*0.1)
    added = buffer.add_episode(
        transitions=[sample_transition] * 5,
        max_x=1500,
        flag_captured=False,
        episode_reward=0.0,
    )

    assert added == 0
    assert len(buffer) == 0


def test_create_elite_transition(sample_transition: Transition) -> None:
    """Test creating an EliteTransition."""
    elite = EliteTransition(
        transition=sample_transition,
        quality=500.0,
        episode_id=1,
    )

    assert elite.transition == sample_transition
    assert elite.quality == 500.0
    assert elite.episode_id == 1


def test_elite_transition_frozen(sample_transition: Transition) -> None:
    """Test EliteTransition is frozen."""
    elite = EliteTransition(
        transition=sample_transition,
        quality=500.0,
        episode_id=1,
    )

    with pytest.raises((AttributeError, TypeError)):  # FrozenInstanceError or AttributeError
        elite.quality = 600.0
