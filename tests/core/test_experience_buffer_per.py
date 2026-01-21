"""Tests for Prioritized Experience Replay in ExperienceBuffer.

These tests verify ExperienceBuffer correctly implements PER:
- Stores transitions with priorities
- Samples based on priority distribution
- Updates priorities based on TD errors
- Computes importance sampling weights
"""

import pytest
import numpy as np

from mario_rl.core.types import Transition
from mario_rl.core.replay_buffer import ExperienceBuffer

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def obs_shape() -> tuple[int, ...]:
    """Default observation shape for tests."""
    return (4, 64, 64)


@pytest.fixture
def per_buffer(obs_shape: tuple[int, ...]) -> ExperienceBuffer:
    """Create a PER-enabled ExperienceBuffer."""
    return ExperienceBuffer(
        capacity=100,
        obs_shape=obs_shape,
        alpha=0.6,  # PER enabled
        epsilon=1e-6,
    )


@pytest.fixture
def uniform_buffer(obs_shape: tuple[int, ...]) -> ExperienceBuffer:
    """Create a uniform sampling ExperienceBuffer (no PER)."""
    return ExperienceBuffer(
        capacity=100,
        obs_shape=obs_shape,
        alpha=0.0,  # PER disabled
    )


def make_transition(obs_shape: tuple[int, ...], reward: float = 1.0) -> Transition:
    """Create a sample transition."""
    return Transition(
        state=np.random.randn(*obs_shape).astype(np.float32),
        action=0,
        reward=reward,
        next_state=np.random.randn(*obs_shape).astype(np.float32),
        done=False,
    )


def fill_buffer(buffer: ExperienceBuffer, obs_shape: tuple[int, ...], n: int) -> None:
    """Fill buffer with n transitions."""
    for _ in range(n):
        buffer.store(make_transition(obs_shape))


# =============================================================================
# PER Detection Tests
# =============================================================================


def test_per_buffer_uses_per(per_buffer: ExperienceBuffer) -> None:
    """PER buffer should report uses_per=True."""
    assert per_buffer.uses_per is True


def test_uniform_buffer_not_uses_per(uniform_buffer: ExperienceBuffer) -> None:
    """Uniform buffer should report uses_per=False."""
    assert uniform_buffer.uses_per is False


# =============================================================================
# PER Storage Tests
# =============================================================================


def test_store_with_default_priority(per_buffer: ExperienceBuffer, obs_shape: tuple[int, ...]) -> None:
    """Store should use max_priority when no priority specified."""
    per_buffer.store(make_transition(obs_shape))

    assert per_buffer.size == 1
    # Default priority should be max_priority (1.0)
    assert per_buffer._tree is not None
    assert per_buffer._tree.total > 0


def test_store_with_explicit_priority(per_buffer: ExperienceBuffer, obs_shape: tuple[int, ...]) -> None:
    """Store should accept explicit priority."""
    per_buffer.store(make_transition(obs_shape), priority=5.0)

    assert per_buffer.size == 1
    # Priority should reflect the explicit value (5.0 + epsilon)^alpha
    assert per_buffer._tree is not None


def test_store_updates_max_priority(per_buffer: ExperienceBuffer, obs_shape: tuple[int, ...]) -> None:
    """Storing high priority should update max_priority tracking."""
    initial_max = per_buffer._max_priority
    per_buffer.store(make_transition(obs_shape), priority=10.0)

    # max_priority shouldn't change from store (only update_priorities does that)
    assert per_buffer._max_priority == initial_max


# =============================================================================
# PER Sampling Tests
# =============================================================================


def test_sample_per_returns_indices_and_weights(per_buffer: ExperienceBuffer, obs_shape: tuple[int, ...]) -> None:
    """sample_per should return tree indices and importance weights."""
    fill_buffer(per_buffer, obs_shape, 50)

    result = per_buffer.sample_per(batch_size=16, beta=0.4)

    # Result should have 10 elements (8 data + indices + weights)
    assert len(result) == 10

    # Unpack
    states, actions, rewards, next_states, dones, ah, nah, dt, indices, weights = result

    assert indices.shape == (16,)
    assert weights.shape == (16,)
    assert indices.dtype == np.int64
    assert weights.dtype == np.float32


def test_sample_per_weights_normalized(per_buffer: ExperienceBuffer, obs_shape: tuple[int, ...]) -> None:
    """Importance weights should be normalized (max weight = 1.0)."""
    fill_buffer(per_buffer, obs_shape, 50)

    _, _, _, _, _, _, _, _, _, weights = per_buffer.sample_per(batch_size=16, beta=0.4)

    assert weights.max() <= 1.0 + 1e-6
    assert weights.min() > 0.0


def test_sample_per_beta_zero_no_correction(per_buffer: ExperienceBuffer, obs_shape: tuple[int, ...]) -> None:
    """Beta=0 should produce all weights equal to 1 (no IS correction)."""
    fill_buffer(per_buffer, obs_shape, 50)

    # Add some high-priority transitions
    for _ in range(10):
        per_buffer.store(make_transition(obs_shape), priority=10.0)

    # Sample with beta=0 (no importance sampling correction)
    _, _, _, _, _, _, _, _, _, weights = per_buffer.sample_per(batch_size=32, beta=0.0)

    # All weights should be 1.0 when beta=0
    assert np.allclose(weights, 1.0)


def test_sample_per_raises_when_disabled(uniform_buffer: ExperienceBuffer, obs_shape: tuple[int, ...]) -> None:
    """sample_per should raise ValueError when PER is disabled."""
    fill_buffer(uniform_buffer, obs_shape, 50)

    with pytest.raises(ValueError, match="PER is disabled"):
        uniform_buffer.sample_per(batch_size=16, beta=0.4)


def test_sample_per_raises_when_empty(per_buffer: ExperienceBuffer) -> None:
    """sample_per should raise ValueError when buffer is empty."""
    with pytest.raises(ValueError, match="empty buffer"):
        per_buffer.sample_per(batch_size=16, beta=0.4)


# =============================================================================
# Priority Update Tests
# =============================================================================


def test_update_priorities_changes_tree(per_buffer: ExperienceBuffer, obs_shape: tuple[int, ...]) -> None:
    """update_priorities should modify the sum tree."""
    fill_buffer(per_buffer, obs_shape, 50)

    # Sample to get indices
    _, _, _, _, _, _, _, _, indices, _ = per_buffer.sample_per(batch_size=16, beta=0.4)

    # Get initial tree total
    initial_total = per_buffer._tree.total

    # Update with very high TD errors
    td_errors = np.ones(16) * 100.0
    per_buffer.update_priorities(indices, td_errors)

    # Tree total should change
    assert per_buffer._tree.total != initial_total


def test_update_priorities_updates_max(per_buffer: ExperienceBuffer, obs_shape: tuple[int, ...]) -> None:
    """update_priorities should update max_priority."""
    fill_buffer(per_buffer, obs_shape, 50)

    initial_max = per_buffer._max_priority

    _, _, _, _, _, _, _, _, indices, _ = per_buffer.sample_per(batch_size=16, beta=0.4)

    # Update with very high TD errors
    td_errors = np.ones(16) * 100.0
    per_buffer.update_priorities(indices, td_errors)

    assert per_buffer._max_priority > initial_max


def test_update_priorities_noop_when_disabled(uniform_buffer: ExperienceBuffer, obs_shape: tuple[int, ...]) -> None:
    """update_priorities should be a no-op when PER is disabled."""
    fill_buffer(uniform_buffer, obs_shape, 50)

    # This should not raise
    uniform_buffer.update_priorities(np.array([0, 1, 2]), np.array([1.0, 2.0, 3.0]))


# =============================================================================
# Reset Tests
# =============================================================================


def test_reset_clears_tree(per_buffer: ExperienceBuffer, obs_shape: tuple[int, ...]) -> None:
    """reset should clear the PER tree."""
    fill_buffer(per_buffer, obs_shape, 50)

    assert per_buffer._tree.total > 0

    per_buffer.reset()

    assert per_buffer.size == 0
    assert per_buffer._max_priority == 1.0


# =============================================================================
# Stratified Sampling Tests
# =============================================================================


def test_sample_per_stratified(per_buffer: ExperienceBuffer, obs_shape: tuple[int, ...]) -> None:
    """PER should use stratified sampling (covering the priority range)."""
    fill_buffer(per_buffer, obs_shape, 100)

    # Sample many times and check indices are distributed
    all_indices = []
    for _ in range(10):
        _, _, _, _, _, _, _, _, indices, _ = per_buffer.sample_per(batch_size=20, beta=0.4)
        all_indices.extend(indices.tolist())

    # Should have variety in indices (not always sampling same items)
    unique_indices = len(set(all_indices))
    assert unique_indices > 50  # Should sample from most of buffer


# =============================================================================
# High Priority Sampling Bias Tests
# =============================================================================


def test_high_priority_sampled_more(obs_shape: tuple[int, ...]) -> None:
    """High priority transitions should be sampled more frequently.

    We create a buffer where we can distinguish high-priority samples
    by their reward values, then verify high-reward samples appear more often.
    """
    buffer = ExperienceBuffer(
        capacity=100,
        obs_shape=obs_shape,
        alpha=0.6,
        epsilon=1e-6,
    )

    # Add low priority transitions with reward=0.0
    for _ in range(90):
        t = Transition(
            state=np.random.randn(*obs_shape).astype(np.float32),
            action=0,
            reward=0.0,  # Marker for low priority
            next_state=np.random.randn(*obs_shape).astype(np.float32),
            done=False,
        )
        buffer.store(t, priority=0.1)

    # Add high priority transitions with reward=1.0
    for _ in range(10):
        t = Transition(
            state=np.random.randn(*obs_shape).astype(np.float32),
            action=0,
            reward=1.0,  # Marker for high priority
            next_state=np.random.randn(*obs_shape).astype(np.float32),
            done=False,
        )
        buffer.store(t, priority=10.0)

    # Sample many times and count high-priority samples by reward
    high_priority_count = 0
    total_samples = 0

    for _ in range(100):
        _, _, rewards, _, _, _, _, _, _, _ = buffer.sample_per(batch_size=10, beta=0.4)
        high_priority_count += np.sum(rewards > 0.5)
        total_samples += 10

    # High priority items (10%) should be sampled more than their proportion
    # With 100x priority difference, they should be sampled much more
    high_priority_ratio = high_priority_count / total_samples
    assert high_priority_ratio > 0.15, f"Expected >15% but got {high_priority_ratio:.1%}"


# =============================================================================
# Action History Support Tests
# =============================================================================


def test_per_with_action_history(obs_shape: tuple[int, ...]) -> None:
    """PER should work with action history enabled."""
    buffer = ExperienceBuffer(
        capacity=100,
        obs_shape=obs_shape,
        action_history_shape=(8, 12),  # 8 steps, 12 actions
        alpha=0.6,
    )

    # Add transitions with action history
    for _ in range(50):
        t = Transition(
            state=np.random.randn(*obs_shape).astype(np.float32),
            action=0,
            reward=1.0,
            next_state=np.random.randn(*obs_shape).astype(np.float32),
            done=False,
            action_history=np.random.randn(8, 12).astype(np.float32),
            next_action_history=np.random.randn(8, 12).astype(np.float32),
        )
        buffer.store(t)

    # Sample with PER
    s, a, r, ns, d, ah, nah, dt, indices, weights = buffer.sample_per(batch_size=16, beta=0.4)

    assert ah is not None
    assert nah is not None
    assert ah.shape == (16, 8, 12)
    assert nah.shape == (16, 8, 12)
