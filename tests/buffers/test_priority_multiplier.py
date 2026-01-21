"""Tests for asymmetric priority multiplier in replay buffers."""

from __future__ import annotations

import pytest
import numpy as np

from mario_rl.core.types import Transition
from mario_rl.buffers.prioritized import PrioritizedReplayBuffer


@pytest.fixture
def obs_shape() -> tuple[int, ...]:
    """Standard observation shape."""
    return (4, 64, 64)


@pytest.fixture
def sample_transition(obs_shape: tuple[int, ...]) -> Transition:
    """Create a sample transition for testing."""
    return Transition(
        state=np.random.rand(*obs_shape).astype(np.float32),
        action=1,
        reward=1.0,
        next_state=np.random.rand(*obs_shape).astype(np.float32),
        done=False,
        flag_get=False,
        max_x=100,
    )


@pytest.fixture
def flag_transition(obs_shape: tuple[int, ...]) -> Transition:
    """Create a transition with flag capture."""
    return Transition(
        state=np.random.rand(*obs_shape).astype(np.float32),
        action=2,
        reward=10.0,
        next_state=np.random.rand(*obs_shape).astype(np.float32),
        done=True,
        flag_get=True,
        max_x=3000,
    )


def test_flag_priority_multiplier_default(obs_shape: tuple[int, ...]) -> None:
    """Test default flag priority multiplier is 50."""
    buffer = PrioritizedReplayBuffer(capacity=100, obs_shape=obs_shape)
    assert buffer.flag_priority_multiplier == 50.0


def test_flag_priority_multiplier_custom(obs_shape: tuple[int, ...]) -> None:
    """Test custom flag priority multiplier."""
    buffer = PrioritizedReplayBuffer(
        capacity=100,
        obs_shape=obs_shape,
        flag_priority_multiplier=100.0,
    )
    assert buffer.flag_priority_multiplier == 100.0


def test_stores_flag_get(
    obs_shape: tuple[int, ...],
    sample_transition: Transition,
    flag_transition: Transition,
) -> None:
    """Test that flag_get is stored in the buffer."""
    buffer = PrioritizedReplayBuffer(capacity=100, obs_shape=obs_shape)

    buffer.add(sample_transition)
    buffer.add(flag_transition)

    assert buffer.flag_gets[0] is False or buffer.flag_gets[0] == np.False_
    assert buffer.flag_gets[1] is True or buffer.flag_gets[1] == np.True_


def test_flag_gets_higher_initial_priority(
    obs_shape: tuple[int, ...],
    sample_transition: Transition,
    flag_transition: Transition,
) -> None:
    """Test flag transitions get higher initial priority."""
    buffer = PrioritizedReplayBuffer(
        capacity=100,
        obs_shape=obs_shape,
        flag_priority_multiplier=50.0,
    )

    # Add normal transition first
    buffer.add(sample_transition)
    normal_priority = buffer.tree.tree[buffer.capacity - 1]

    # Add flag transition
    buffer.add(flag_transition)
    flag_priority = buffer.tree.tree[buffer.capacity]

    # Flag should have 50x higher priority
    assert flag_priority == pytest.approx(normal_priority * 50.0)


def test_flag_priority_preserved_on_update(
    obs_shape: tuple[int, ...],
    flag_transition: Transition,
) -> None:
    """Test flag priority boost is preserved when priorities are updated."""
    buffer = PrioritizedReplayBuffer(
        capacity=100,
        obs_shape=obs_shape,
        flag_priority_multiplier=50.0,
    )

    # Add flag transition
    buffer.add(flag_transition)

    # Sample to get indices
    batch = buffer.sample(1)
    _initial_priority = buffer.tree.tree[batch.indices[0]]

    # Update priority with TD error
    td_errors = np.array([1.0])
    buffer.update_priorities(batch.indices, td_errors)

    updated_priority = buffer.tree.tree[batch.indices[0]]

    # Updated priority should still include multiplier
    # Base priority = (1.0 + epsilon) ^ alpha ≈ 1.0 ^ 0.6 ≈ 1.0
    # With multiplier: ~50
    assert updated_priority > 10.0  # Should be significantly boosted


def test_normal_transition_no_boost(
    obs_shape: tuple[int, ...],
    sample_transition: Transition,
) -> None:
    """Test normal transitions don't get flag boost."""
    buffer = PrioritizedReplayBuffer(
        capacity=100,
        obs_shape=obs_shape,
        flag_priority_multiplier=50.0,
    )

    buffer.add(sample_transition)

    # Priority should be base max_priority^alpha, not boosted
    priority = buffer.tree.tree[buffer.capacity - 1]
    expected = buffer.max_priority**buffer.alpha

    assert priority == pytest.approx(expected)


def test_multiplier_disabled_when_one(
    obs_shape: tuple[int, ...],
    sample_transition: Transition,
    flag_transition: Transition,
) -> None:
    """Test multiplier has no effect when set to 1.0."""
    buffer = PrioritizedReplayBuffer(
        capacity=100,
        obs_shape=obs_shape,
        flag_priority_multiplier=1.0,
    )

    buffer.add(sample_transition)
    normal_priority = buffer.tree.tree[buffer.capacity - 1]

    buffer.add(flag_transition)
    flag_priority = buffer.tree.tree[buffer.capacity]

    # Same priority (within floating point tolerance)
    assert flag_priority == pytest.approx(normal_priority)


def test_transition_default_flag_get() -> None:
    """Test Transition defaults flag_get to False."""
    transition = Transition(
        state=np.zeros((4, 64, 64), dtype=np.float32),
        action=1,
        reward=1.0,
        next_state=np.zeros((4, 64, 64), dtype=np.float32),
        done=False,
    )
    assert transition.flag_get is False


def test_transition_default_max_x() -> None:
    """Test Transition defaults max_x to 0."""
    transition = Transition(
        state=np.zeros((4, 64, 64), dtype=np.float32),
        action=1,
        reward=1.0,
        next_state=np.zeros((4, 64, 64), dtype=np.float32),
        done=False,
    )
    assert transition.max_x == 0


def test_transition_with_flag_get() -> None:
    """Test creating Transition with flag_get=True."""
    transition = Transition(
        state=np.zeros((4, 64, 64), dtype=np.float32),
        action=1,
        reward=1.0,
        next_state=np.zeros((4, 64, 64), dtype=np.float32),
        done=True,
        flag_get=True,
        max_x=3000,
    )
    assert transition.flag_get is True
    assert transition.max_x == 3000


def test_transition_frozen() -> None:
    """Test Transition is frozen (immutable)."""
    transition = Transition(
        state=np.zeros((4, 64, 64), dtype=np.float32),
        action=1,
        reward=1.0,
        next_state=np.zeros((4, 64, 64), dtype=np.float32),
        done=False,
    )

    with pytest.raises((AttributeError, TypeError)):  # FrozenInstanceError
        transition.flag_get = True  # type: ignore[misc]


def test_buffer_config_defaults() -> None:
    """Test BufferConfig has flag priority defaults."""
    from mario_rl.core.config import BufferConfig

    config = BufferConfig()
    assert config.flag_priority_multiplier == 50.0
    assert config.elite_capacity == 1000
    assert config.elite_sample_ratio == 0.15


def test_buffer_config_custom() -> None:
    """Test BufferConfig with custom values."""
    from mario_rl.core.config import BufferConfig

    config = BufferConfig(
        flag_priority_multiplier=100.0,
        elite_capacity=500,
        elite_sample_ratio=0.2,
    )
    assert config.flag_priority_multiplier == 100.0
    assert config.elite_capacity == 500
    assert config.elite_sample_ratio == 0.2
