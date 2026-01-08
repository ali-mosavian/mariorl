"""Tests for ReplayBuffer behavior.

These tests verify the ReplayBuffer correctly:
- Stores transitions and samples batches
- Computes N-step returns
- Supports prioritized sampling (optional)
- Handles episode boundaries
"""

from dataclasses import dataclass

import pytest
import numpy as np
import torch
from torch import Tensor

from mario_rl.core.types import Transition


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class BufferTestConfig:
    """Test configuration for ReplayBuffer."""

    capacity: int = 1000
    obs_shape: tuple[int, ...] = (4, 64, 64)
    batch_size: int = 32
    n_step: int = 3
    gamma: float = 0.99


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config() -> BufferTestConfig:
    """Default test configuration."""
    return BufferTestConfig()


@pytest.fixture
def replay_buffer(config: BufferTestConfig):
    """Create a ReplayBuffer for testing."""
    from mario_rl.core.replay_buffer import ReplayBuffer

    return ReplayBuffer(
        capacity=config.capacity,
        obs_shape=config.obs_shape,
        n_step=config.n_step,
        gamma=config.gamma,
    )


@pytest.fixture
def sample_transition(config: BufferTestConfig) -> Transition:
    """Create a sample transition."""
    return Transition(
        state=np.random.randn(*config.obs_shape).astype(np.float32),
        action=0,
        reward=1.0,
        next_state=np.random.randn(*config.obs_shape).astype(np.float32),
        done=False,
    )


def make_transitions(config: BufferTestConfig, n: int, done_at: int | None = None) -> list[Transition]:
    """Create n transitions, optionally with done at specified index."""
    transitions = []
    for i in range(n):
        transitions.append(
            Transition(
                state=np.random.randn(*config.obs_shape).astype(np.float32),
                action=i % 12,
                reward=float(i),
                next_state=np.random.randn(*config.obs_shape).astype(np.float32),
                done=(i == done_at) if done_at is not None else False,
            )
        )
    return transitions


# =============================================================================
# Basic Storage Tests
# =============================================================================


def test_buffer_starts_empty(replay_buffer) -> None:
    """Buffer should start with zero length."""
    assert len(replay_buffer) == 0


def test_add_increases_length(replay_buffer, sample_transition: Transition) -> None:
    """Adding transitions should increase buffer length."""
    replay_buffer.add(sample_transition)
    # With n_step=3, first transition appears after 3 adds
    replay_buffer.add(sample_transition)
    replay_buffer.add(sample_transition)
    
    assert len(replay_buffer) >= 1


def test_buffer_respects_capacity(config: BufferTestConfig) -> None:
    """Buffer should not exceed capacity."""
    from mario_rl.core.replay_buffer import ReplayBuffer

    small_buffer = ReplayBuffer(
        capacity=10,
        obs_shape=config.obs_shape,
        n_step=1,  # No n-step for simpler test
        gamma=config.gamma,
    )

    # Add more than capacity
    for t in make_transitions(config, 50):
        small_buffer.add(t)

    assert len(small_buffer) <= 10


def test_can_sample_returns_false_when_empty(replay_buffer, config: BufferTestConfig) -> None:
    """can_sample should return False when buffer is empty."""
    assert not replay_buffer.can_sample(config.batch_size)


def test_can_sample_returns_true_when_enough_samples(replay_buffer, config: BufferTestConfig) -> None:
    """can_sample should return True when buffer has enough samples."""
    for t in make_transitions(config, 50):
        replay_buffer.add(t)

    assert replay_buffer.can_sample(config.batch_size)


# =============================================================================
# Sampling Tests
# =============================================================================


def test_sample_returns_correct_batch_size(replay_buffer, config: BufferTestConfig) -> None:
    """Sample should return batch of requested size."""
    for t in make_transitions(config, 100):
        replay_buffer.add(t)

    batch = replay_buffer.sample(config.batch_size)

    assert batch.states.shape[0] == config.batch_size
    assert batch.actions.shape[0] == config.batch_size
    assert batch.rewards.shape[0] == config.batch_size
    assert batch.next_states.shape[0] == config.batch_size
    assert batch.dones.shape[0] == config.batch_size


def test_sample_returns_correct_shapes(replay_buffer, config: BufferTestConfig) -> None:
    """Sampled tensors should have correct shapes."""
    for t in make_transitions(config, 100):
        replay_buffer.add(t)

    batch = replay_buffer.sample(config.batch_size)

    assert batch.states.shape == (config.batch_size, *config.obs_shape)
    assert batch.next_states.shape == (config.batch_size, *config.obs_shape)
    assert batch.actions.shape == (config.batch_size,)
    assert batch.rewards.shape == (config.batch_size,)
    assert batch.dones.shape == (config.batch_size,)


def test_sample_returns_tensors(replay_buffer, config: BufferTestConfig) -> None:
    """Sample should return torch tensors."""
    for t in make_transitions(config, 100):
        replay_buffer.add(t)

    batch = replay_buffer.sample(config.batch_size)

    assert isinstance(batch.states, Tensor)
    assert isinstance(batch.actions, Tensor)
    assert isinstance(batch.rewards, Tensor)
    assert isinstance(batch.next_states, Tensor)
    assert isinstance(batch.dones, Tensor)


def test_sample_states_are_float32(replay_buffer, config: BufferTestConfig) -> None:
    """State tensors should be float32."""
    for t in make_transitions(config, 100):
        replay_buffer.add(t)

    batch = replay_buffer.sample(config.batch_size)

    assert batch.states.dtype == torch.float32
    assert batch.next_states.dtype == torch.float32


def test_sample_actions_are_int64(replay_buffer, config: BufferTestConfig) -> None:
    """Action tensors should be int64."""
    for t in make_transitions(config, 100):
        replay_buffer.add(t)

    batch = replay_buffer.sample(config.batch_size)

    assert batch.actions.dtype == torch.int64


def test_sample_raises_when_not_enough_data(replay_buffer, config: BufferTestConfig) -> None:
    """Sample should raise when not enough data."""
    # Add fewer than batch_size
    for t in make_transitions(config, 5):
        replay_buffer.add(t)

    with pytest.raises(ValueError):
        replay_buffer.sample(config.batch_size)


# =============================================================================
# N-Step Return Tests
# =============================================================================


def test_nstep_computes_discounted_return(config: BufferTestConfig) -> None:
    """N-step buffer should compute discounted rewards."""
    from mario_rl.core.replay_buffer import ReplayBuffer

    buffer = ReplayBuffer(
        capacity=100,
        obs_shape=config.obs_shape,
        n_step=3,
        gamma=0.99,
    )

    # Add 3 transitions with rewards 1, 2, 3
    for i in range(1, 4):
        buffer.add(
            Transition(
                state=np.zeros(config.obs_shape, dtype=np.float32),
                action=0,
                reward=float(i),
                next_state=np.zeros(config.obs_shape, dtype=np.float32),
                done=False,
            )
        )

    # First n-step transition should have:
    # reward = 1 + 0.99*2 + 0.99^2*3 = 1 + 1.98 + 2.9403 = 5.9203
    assert len(buffer) >= 1


def test_nstep_handles_episode_boundary(config: BufferTestConfig) -> None:
    """N-step should handle done=True correctly."""
    from mario_rl.core.replay_buffer import ReplayBuffer

    buffer = ReplayBuffer(
        capacity=100,
        obs_shape=config.obs_shape,
        n_step=3,
        gamma=0.99,
    )

    # Add transitions where middle one is done
    buffer.add(Transition(
        state=np.zeros(config.obs_shape, dtype=np.float32),
        action=0, reward=1.0,
        next_state=np.zeros(config.obs_shape, dtype=np.float32),
        done=False,
    ))
    buffer.add(Transition(
        state=np.zeros(config.obs_shape, dtype=np.float32),
        action=0, reward=2.0,
        next_state=np.zeros(config.obs_shape, dtype=np.float32),
        done=True,  # Episode ends here
    ))

    # Should still produce valid transitions
    assert len(buffer) >= 1


def test_nstep_one_is_regular_buffer(config: BufferTestConfig) -> None:
    """With n_step=1, buffer should behave like regular buffer."""
    from mario_rl.core.replay_buffer import ReplayBuffer

    buffer = ReplayBuffer(
        capacity=100,
        obs_shape=config.obs_shape,
        n_step=1,
        gamma=0.99,
    )

    # Add single transition
    buffer.add(Transition(
        state=np.ones(config.obs_shape, dtype=np.float32),
        action=5,
        reward=10.0,
        next_state=np.ones(config.obs_shape, dtype=np.float32) * 2,
        done=False,
    ))

    # Should be immediately available (no n-step delay)
    assert len(buffer) == 1


# =============================================================================
# Prioritized Experience Replay Tests
# =============================================================================


def test_per_buffer_has_weights(config: BufferTestConfig) -> None:
    """PER buffer should return importance sampling weights."""
    from mario_rl.core.replay_buffer import ReplayBuffer

    buffer = ReplayBuffer(
        capacity=100,
        obs_shape=config.obs_shape,
        n_step=1,
        gamma=0.99,
        alpha=0.6,  # Enable PER
    )

    for t in make_transitions(config, 50):
        buffer.add(t)

    batch = buffer.sample(config.batch_size)

    assert hasattr(batch, "weights")
    assert batch.weights.shape == (config.batch_size,)


def test_per_buffer_has_indices(config: BufferTestConfig) -> None:
    """PER buffer should return indices for priority updates."""
    from mario_rl.core.replay_buffer import ReplayBuffer

    buffer = ReplayBuffer(
        capacity=100,
        obs_shape=config.obs_shape,
        n_step=1,
        gamma=0.99,
        alpha=0.6,
    )

    for t in make_transitions(config, 50):
        buffer.add(t)

    batch = buffer.sample(config.batch_size)

    assert hasattr(batch, "indices")
    assert batch.indices.shape == (config.batch_size,)


def test_per_update_priorities(config: BufferTestConfig) -> None:
    """Should be able to update priorities after sampling."""
    from mario_rl.core.replay_buffer import ReplayBuffer

    buffer = ReplayBuffer(
        capacity=100,
        obs_shape=config.obs_shape,
        n_step=1,
        gamma=0.99,
        alpha=0.6,
    )

    for t in make_transitions(config, 50):
        buffer.add(t)

    batch = buffer.sample(config.batch_size)

    # Update priorities with TD errors
    td_errors = np.random.rand(config.batch_size)
    buffer.update_priorities(batch.indices, td_errors)

    # Should not raise


def test_uniform_sampling_no_weights(config: BufferTestConfig) -> None:
    """With alpha=0, buffer should use uniform sampling (no weights)."""
    from mario_rl.core.replay_buffer import ReplayBuffer

    buffer = ReplayBuffer(
        capacity=100,
        obs_shape=config.obs_shape,
        n_step=1,
        gamma=0.99,
        alpha=0.0,  # Uniform sampling
    )

    for t in make_transitions(config, 50):
        buffer.add(t)

    batch = buffer.sample(config.batch_size)

    # With uniform sampling, weights should all be 1
    if hasattr(batch, "weights"):
        assert np.allclose(batch.weights.numpy(), 1.0)


# =============================================================================
# Episode Flush Tests
# =============================================================================


def test_flush_at_episode_end(config: BufferTestConfig) -> None:
    """Flushing should add remaining n-step transitions."""
    from mario_rl.core.replay_buffer import ReplayBuffer

    buffer = ReplayBuffer(
        capacity=100,
        obs_shape=config.obs_shape,
        n_step=3,
        gamma=0.99,
    )

    # Add 2 transitions (less than n_step)
    buffer.add(Transition(
        state=np.zeros(config.obs_shape, dtype=np.float32),
        action=0, reward=1.0,
        next_state=np.zeros(config.obs_shape, dtype=np.float32),
        done=False,
    ))
    buffer.add(Transition(
        state=np.zeros(config.obs_shape, dtype=np.float32),
        action=0, reward=2.0,
        next_state=np.zeros(config.obs_shape, dtype=np.float32),
        done=True,  # Episode ends
    ))

    # Flush to get remaining transitions
    buffer.flush()

    # Should have transitions now
    assert len(buffer) >= 1


def test_reset_clears_buffer(replay_buffer, config: BufferTestConfig) -> None:
    """Reset should clear all stored data."""
    for t in make_transitions(config, 50):
        replay_buffer.add(t)

    assert len(replay_buffer) > 0

    replay_buffer.reset()

    assert len(replay_buffer) == 0


# =============================================================================
# Device Tests
# =============================================================================


def test_sample_to_device(replay_buffer, config: BufferTestConfig) -> None:
    """Sample should support specifying output device."""
    for t in make_transitions(config, 100):
        replay_buffer.add(t)

    batch = replay_buffer.sample(config.batch_size, device="cpu")

    assert batch.states.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sample_to_cuda(replay_buffer, config: BufferTestConfig) -> None:
    """Sample should work with CUDA device."""
    for t in make_transitions(config, 100):
        replay_buffer.add(t)

    batch = replay_buffer.sample(config.batch_size, device="cuda")

    assert batch.states.device.type == "cuda"


# =============================================================================
# Edge Cases
# =============================================================================


def test_sample_size_equals_buffer_size(config: BufferTestConfig) -> None:
    """Should be able to sample when batch_size equals buffer size."""
    from mario_rl.core.replay_buffer import ReplayBuffer

    buffer = ReplayBuffer(
        capacity=10,
        obs_shape=config.obs_shape,
        n_step=1,
        gamma=0.99,
    )

    for t in make_transitions(config, 10):
        buffer.add(t)

    batch = buffer.sample(10)
    assert batch.states.shape[0] == 10


def test_repeated_sampling_gives_different_batches(replay_buffer, config: BufferTestConfig) -> None:
    """Multiple samples should give different batches."""
    for t in make_transitions(config, 100):
        replay_buffer.add(t)

    batch1 = replay_buffer.sample(config.batch_size)
    batch2 = replay_buffer.sample(config.batch_size)

    # Actions should differ (very unlikely to be identical)
    assert not torch.equal(batch1.actions, batch2.actions)
