"""Integration tests for RAM-based training.

Tests the complete pipeline of using NES RAM (2048 bytes) as observations
instead of visual frames (4x64x64).
"""

import torch
import pytest
import numpy as np

from mario_rl.models import DDQNModel
from mario_rl.models import RAMDoubleDQN
from mario_rl.learners.ddqn import DDQNLearner
from mario_rl.environment.factory import create_ram_env


class TestRAMObservationWrapper:
    """Tests for RAMObservationWrapper."""

    def test_observation_shape(self) -> None:
        """RAM observations should be shape (2048,) uint8."""
        env = create_ram_env(level=(1, 1))
        obs, info = env.reset()

        assert obs.shape == (2048,), f"Expected (2048,), got {obs.shape}"
        assert obs.dtype == np.uint8, f"Expected uint8, got {obs.dtype}"
        env.close()

    def test_step_returns_ram(self) -> None:
        """Step should also return RAM observations."""
        env = create_ram_env(level=(1, 1))
        obs, info = env.reset()

        # Take a few steps
        for _ in range(10):
            obs, reward, done, truncated, info = env.step(1)  # Right
            assert obs.shape == (2048,)
            assert obs.dtype == np.uint8
            if done:
                break

        env.close()

    def test_ram_changes_with_actions(self) -> None:
        """RAM should change when we take actions."""
        env = create_ram_env(level=(1, 1))
        obs1, _ = env.reset()

        # Take some steps
        for _ in range(10):
            obs2, _, done, _, _ = env.step(1)  # Right
            if done:
                break

        # RAM should be different after moving
        assert not np.array_equal(obs1, obs2), "RAM should change after actions"
        env.close()


class TestRAMDoubleDQNProtocol:
    """Test that RAMDoubleDQN satisfies DDQNModel protocol."""

    def test_implements_protocol(self) -> None:
        """RAMDoubleDQN should implement DDQNModel protocol."""
        model = RAMDoubleDQN(ram_size=2048, num_actions=7)
        assert isinstance(model, DDQNModel)

    def test_has_required_attributes(self) -> None:
        """Model should have num_actions attribute."""
        model = RAMDoubleDQN(ram_size=2048, num_actions=7)
        assert model.num_actions == 7

    def test_has_action_history_len(self) -> None:
        """Model should have action_history_len attribute."""
        model = RAMDoubleDQN(ram_size=2048, num_actions=7, action_history_len=4)
        assert model.action_history_len == 4


class TestRAMDDQNLearner:
    """Test DDQNLearner with RAMDoubleDQN model."""

    @pytest.fixture
    def ram_learner(self) -> DDQNLearner:
        """Create a DDQNLearner with RAMDoubleDQN."""
        model = RAMDoubleDQN(ram_size=2048, num_actions=7, action_history_len=4)
        return DDQNLearner(model=model, gamma=0.99, n_step=1)

    def test_compute_loss_with_ram_input(self, ram_learner: DDQNLearner) -> None:
        """Learner should compute loss with RAM observations."""
        batch_size = 4
        states = torch.randint(0, 255, (batch_size, 2048), dtype=torch.uint8)
        actions = torch.randint(0, 7, (batch_size,))
        rewards = torch.randn(batch_size)
        next_states = torch.randint(0, 255, (batch_size, 2048), dtype=torch.uint8)
        dones = torch.zeros(batch_size)

        loss, metrics = ram_learner.compute_loss(states, actions, rewards, next_states, dones)

        assert loss.shape == ()  # Scalar
        assert loss.requires_grad
        assert "loss" in metrics
        assert "q_mean" in metrics

    def test_compute_loss_with_action_history(self, ram_learner: DDQNLearner) -> None:
        """Learner should accept action_history for RAM model."""
        batch_size = 4
        states = torch.randint(0, 255, (batch_size, 2048), dtype=torch.uint8)
        actions = torch.randint(0, 7, (batch_size,))
        rewards = torch.randn(batch_size)
        next_states = torch.randint(0, 255, (batch_size, 2048), dtype=torch.uint8)
        dones = torch.zeros(batch_size)
        action_history = torch.randn(batch_size, 4, 7)  # (batch, history_len, num_actions)
        next_action_history = torch.randn(batch_size, 4, 7)

        loss, metrics = ram_learner.compute_loss(
            states,
            actions,
            rewards,
            next_states,
            dones,
            action_histories=action_history,
            next_action_histories=next_action_history,
        )

        assert loss.shape == ()
        assert loss.requires_grad

    def test_gradient_flow(self, ram_learner: DDQNLearner) -> None:
        """Gradients should flow through RAM model."""
        batch_size = 4
        states = torch.randint(0, 255, (batch_size, 2048), dtype=torch.uint8)
        actions = torch.randint(0, 7, (batch_size,))
        rewards = torch.randn(batch_size)
        next_states = torch.randint(0, 255, (batch_size, 2048), dtype=torch.uint8)
        dones = torch.zeros(batch_size)

        loss, _ = ram_learner.compute_loss(states, actions, rewards, next_states, dones)
        loss.backward()

        # Check gradients exist
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in ram_learner.model.parameters() if p.requires_grad
        )
        assert has_grad, "No gradients found in model parameters"

    def test_update_targets(self, ram_learner: DDQNLearner) -> None:
        """update_targets should work for RAM model."""
        # Hard sync
        ram_learner.update_targets(tau=1.0)

        # Soft update
        ram_learner.update_targets(tau=0.005)

        # Should not raise


class TestRAMEnvIntegration:
    """Test RAM environment with model integration."""

    def test_env_to_model_forward(self) -> None:
        """Observations from env should work directly with model."""
        env = create_ram_env(level=(1, 1))
        model = RAMDoubleDQN(ram_size=2048, num_actions=7)

        obs, _ = env.reset()

        # Convert to tensor and add batch dimension
        obs_t = torch.from_numpy(obs).unsqueeze(0)
        q_values = model(obs_t, network="online")

        assert q_values.shape == (1, 7)
        env.close()

    def test_action_selection(self) -> None:
        """Model should select actions from RAM observations."""
        env = create_ram_env(level=(1, 1))
        model = RAMDoubleDQN(ram_size=2048, num_actions=7)

        obs, _ = env.reset()

        obs_t = torch.from_numpy(obs).unsqueeze(0)
        action = model.select_action(obs_t, epsilon=0.0)

        assert action.shape == (1,)
        assert 0 <= action.item() < 7
        env.close()

    def test_full_episode_loop(self) -> None:
        """Run a short episode with RAM observations and model actions."""
        env = create_ram_env(level=(1, 1))
        model = RAMDoubleDQN(ram_size=2048, num_actions=7)

        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0

        for _ in range(100):  # Max 100 steps
            obs_t = torch.from_numpy(obs).unsqueeze(0)
            action = model.select_action(obs_t, epsilon=0.1)
            obs, reward, done, truncated, info = env.step(action.item())
            total_reward += reward
            steps += 1

            if done or truncated:
                break

        env.close()
        assert steps > 0, "Should take at least one step"
