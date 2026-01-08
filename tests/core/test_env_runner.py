"""Tests for EnvRunner behavior.

These tests verify the EnvRunner correctly:
- Collects steps from environment
- Processes rewards (normalization, clipping)
- Handles episode boundaries
- Returns transitions
"""

from typing import Any
from dataclasses import dataclass
from dataclasses import field
from unittest.mock import Mock
from unittest.mock import MagicMock

import pytest
import numpy as np

from mario_rl.core.types import Transition


# =============================================================================
# Mock Environment
# =============================================================================


@dataclass
class MockEnvConfig:
    """Configuration for mock environment."""

    obs_shape: tuple[int, ...] = (4, 64, 64)
    num_actions: int = 12
    episode_length: int = 100


class MockEnv:
    """Mock gymnasium environment for testing."""

    def __init__(self, config: MockEnvConfig | None = None) -> None:
        self.config = config or MockEnvConfig()
        self._step_count = 0
        self._current_obs = self._make_obs()

    def _make_obs(self) -> np.ndarray:
        return np.random.randn(*self.config.obs_shape).astype(np.float32)

    def reset(self, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        self._step_count = 0
        self._current_obs = self._make_obs()
        return self._current_obs.copy(), {"x_pos": 0}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        next_obs = self._make_obs()
        reward = 1.0  # Simple reward
        terminated = self._step_count >= self.config.episode_length
        truncated = False
        info = {
            "x_pos": self._step_count * 10,
            "flag_get": terminated,
        }
        self._current_obs = next_obs
        return next_obs, reward, terminated, truncated, info


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class RunnerTestConfig:
    """Test configuration for EnvRunner."""

    obs_shape: tuple[int, ...] = (4, 64, 64)
    num_actions: int = 12
    reward_clip: float = 5.0


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config() -> RunnerTestConfig:
    """Default test configuration."""
    return RunnerTestConfig()


@pytest.fixture
def mock_env() -> MockEnv:
    """Create a mock environment."""
    return MockEnv()


@pytest.fixture
def action_fn() -> Mock:
    """Create a mock action selection function."""
    fn = Mock(return_value=0)
    return fn


@pytest.fixture
def env_runner(mock_env: MockEnv, action_fn: Mock):
    """Create an EnvRunner for testing."""
    from mario_rl.core.env_runner import EnvRunner

    return EnvRunner(env=mock_env, action_fn=action_fn)


# =============================================================================
# Basic Collection Tests
# =============================================================================


def test_collect_returns_transitions(env_runner) -> None:
    """collect should return a list of transitions."""
    transitions = env_runner.collect(num_steps=10)

    assert isinstance(transitions, list)
    assert len(transitions) == 10


def test_transitions_have_correct_types(env_runner) -> None:
    """Each transition should have correct field types."""
    transitions = env_runner.collect(num_steps=5)

    for t in transitions:
        assert isinstance(t, Transition)
        assert isinstance(t.state, np.ndarray)
        assert isinstance(t.action, int)
        assert isinstance(t.reward, float)
        assert isinstance(t.next_state, np.ndarray)
        assert isinstance(t.done, bool)


def test_transitions_have_correct_shapes(env_runner, config: RunnerTestConfig) -> None:
    """Transition states should have correct shapes."""
    transitions = env_runner.collect(num_steps=5)

    for t in transitions:
        assert t.state.shape == config.obs_shape
        assert t.next_state.shape == config.obs_shape


def test_collect_calls_action_fn(env_runner, action_fn: Mock) -> None:
    """collect should call action_fn for each step."""
    env_runner.collect(num_steps=10)

    assert action_fn.call_count == 10


def test_collect_passes_state_to_action_fn(env_runner, action_fn: Mock, config: RunnerTestConfig) -> None:
    """action_fn should receive current state."""
    env_runner.collect(num_steps=1)

    call_args = action_fn.call_args
    state = call_args[0][0]  # First positional arg
    assert state.shape == config.obs_shape


# =============================================================================
# Episode Handling Tests
# =============================================================================


def test_handles_episode_end(mock_env: MockEnv, action_fn: Mock) -> None:
    """Should handle episode termination and reset."""
    from mario_rl.core.env_runner import EnvRunner

    # Short episode
    mock_env.config = MockEnvConfig(episode_length=5)
    runner = EnvRunner(env=mock_env, action_fn=action_fn)

    # Collect more steps than episode length
    transitions = runner.collect(num_steps=10)

    assert len(transitions) == 10
    # At least one transition should have done=True
    assert any(t.done for t in transitions)


def test_done_transition_has_correct_next_state(mock_env: MockEnv, action_fn: Mock) -> None:
    """When done, next_state should be the terminal state."""
    from mario_rl.core.env_runner import EnvRunner

    mock_env.config = MockEnvConfig(episode_length=3)
    runner = EnvRunner(env=mock_env, action_fn=action_fn)

    transitions = runner.collect(num_steps=3)

    # Last transition should be done
    assert transitions[-1].done


def test_resets_on_episode_end(mock_env: MockEnv, action_fn: Mock) -> None:
    """Should reset environment when episode ends."""
    from mario_rl.core.env_runner import EnvRunner

    mock_env.config = MockEnvConfig(episode_length=5)
    runner = EnvRunner(env=mock_env, action_fn=action_fn)

    # Collect across episode boundary
    runner.collect(num_steps=7)

    # Environment should have been reset
    # (step count should be less than total steps collected)
    assert mock_env._step_count < 7


def test_episode_callback_called(mock_env: MockEnv, action_fn: Mock) -> None:
    """Episode end callback should be called when episode finishes."""
    from mario_rl.core.env_runner import EnvRunner

    mock_env.config = MockEnvConfig(episode_length=3)
    callback = Mock()
    runner = EnvRunner(env=mock_env, action_fn=action_fn, on_episode_end=callback)

    runner.collect(num_steps=5)

    assert callback.call_count >= 1


# =============================================================================
# Reward Processing Tests
# =============================================================================


def test_reward_clipping(mock_env: MockEnv, action_fn: Mock) -> None:
    """Rewards should be clipped when clip value is set."""
    from mario_rl.core.env_runner import EnvRunner

    # Mock environment to return extreme rewards
    original_step = mock_env.step
    def extreme_reward_step(action):
        obs, _, term, trunc, info = original_step(action)
        return obs, 100.0, term, trunc, info  # Extreme reward

    mock_env.step = extreme_reward_step

    runner = EnvRunner(
        env=mock_env,
        action_fn=action_fn,
        reward_clip=5.0,
    )

    transitions = runner.collect(num_steps=3)

    for t in transitions:
        assert -5.0 <= t.reward <= 5.0


def test_reward_scaling(mock_env: MockEnv, action_fn: Mock) -> None:
    """Rewards should be scaled when scale value is set."""
    from mario_rl.core.env_runner import EnvRunner

    runner = EnvRunner(
        env=mock_env,
        action_fn=action_fn,
        reward_scale=0.1,
    )

    transitions = runner.collect(num_steps=3)

    # Original reward was 1.0, scaled should be 0.1
    for t in transitions:
        assert abs(t.reward) < 1.0  # Scaled down


def test_no_reward_processing_by_default(mock_env: MockEnv, action_fn: Mock) -> None:
    """Without reward processing, rewards should pass through unchanged."""
    from mario_rl.core.env_runner import EnvRunner

    runner = EnvRunner(env=mock_env, action_fn=action_fn)

    transitions = runner.collect(num_steps=3)

    for t in transitions:
        assert t.reward == 1.0  # Original reward from mock


# =============================================================================
# Info/Metrics Tests
# =============================================================================


def test_returns_collected_info(mock_env: MockEnv, action_fn: Mock) -> None:
    """Should return info about collection."""
    from mario_rl.core.env_runner import EnvRunner

    runner = EnvRunner(env=mock_env, action_fn=action_fn)

    transitions, info = runner.collect_with_info(num_steps=10)

    assert "total_reward" in info
    assert "episodes_completed" in info
    assert "steps" in info


def test_info_counts_episodes(mock_env: MockEnv, action_fn: Mock) -> None:
    """Info should count completed episodes."""
    from mario_rl.core.env_runner import EnvRunner

    mock_env.config = MockEnvConfig(episode_length=5)
    runner = EnvRunner(env=mock_env, action_fn=action_fn)

    _, info = runner.collect_with_info(num_steps=12)

    assert info["episodes_completed"] >= 2


def test_info_sums_rewards(mock_env: MockEnv, action_fn: Mock) -> None:
    """Info should sum total reward."""
    from mario_rl.core.env_runner import EnvRunner

    runner = EnvRunner(env=mock_env, action_fn=action_fn)

    _, info = runner.collect_with_info(num_steps=10)

    assert info["total_reward"] == 10.0  # 1.0 per step


# =============================================================================
# State Continuity Tests
# =============================================================================


def test_state_continuity(mock_env: MockEnv, action_fn: Mock) -> None:
    """next_state of one transition should match state of next."""
    from mario_rl.core.env_runner import EnvRunner

    runner = EnvRunner(env=mock_env, action_fn=action_fn)

    transitions = runner.collect(num_steps=5)

    for i in range(len(transitions) - 1):
        if not transitions[i].done:
            # Next state should match next transition's state
            assert np.array_equal(
                transitions[i].next_state,
                transitions[i + 1].state,
            )


def test_preserves_state_between_calls(mock_env: MockEnv, action_fn: Mock) -> None:
    """Runner should preserve state between collect calls."""
    from mario_rl.core.env_runner import EnvRunner

    runner = EnvRunner(env=mock_env, action_fn=action_fn)

    # First collection
    t1 = runner.collect(num_steps=3)

    # Second collection should continue from last state
    t2 = runner.collect(num_steps=3)

    # Last next_state of t1 should match first state of t2
    assert np.array_equal(t1[-1].next_state, t2[0].state)


# =============================================================================
# Edge Cases
# =============================================================================


def test_single_step(env_runner) -> None:
    """Should handle single step collection."""
    transitions = env_runner.collect(num_steps=1)

    assert len(transitions) == 1


def test_zero_steps(env_runner) -> None:
    """Should handle zero step collection."""
    transitions = env_runner.collect(num_steps=0)

    assert len(transitions) == 0


def test_multiple_episodes_in_one_collect(mock_env: MockEnv, action_fn: Mock) -> None:
    """Should handle multiple episodes in single collect call."""
    from mario_rl.core.env_runner import EnvRunner

    mock_env.config = MockEnvConfig(episode_length=3)
    runner = EnvRunner(env=mock_env, action_fn=action_fn)

    transitions = runner.collect(num_steps=20)

    assert len(transitions) == 20
    # Multiple done flags
    done_count = sum(1 for t in transitions if t.done)
    assert done_count >= 6  # At least 6 episodes in 20 steps
