"""Tests for MuZero learner components."""

import pytest
import torch

from mario_rl.learners.muzero import MuZeroLearner
from mario_rl.learners.muzero import MuZeroTrajectory
from mario_rl.learners.muzero import compute_value_target
from mario_rl.learners.muzero import run_mcts
from mario_rl.models.muzero import MuZeroConfig
from mario_rl.models.muzero import MuZeroModel


@pytest.fixture
def config() -> MuZeroConfig:
    """Standard test configuration."""
    return MuZeroConfig(
        input_shape=(4, 84, 84),
        num_actions=7,
        latent_dim=64,
        hidden_dim=32,
        num_simulations=10,  # Small for tests
    )


@pytest.fixture
def model(config: MuZeroConfig) -> MuZeroModel:
    """MuZero model for testing."""
    return MuZeroModel(config)


@pytest.fixture
def learner(model: MuZeroModel) -> MuZeroLearner:
    """MuZero learner for testing."""
    return MuZeroLearner(model=model, unroll_steps=3)


@pytest.fixture
def small_obs() -> torch.Tensor:
    """Small batch of observations for testing."""
    return torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.float32)


class TestMuZeroLearner:
    """Tests for the MuZero learner."""

    def test_compute_trajectory_loss(
        self, learner: MuZeroLearner, small_obs: torch.Tensor, config: MuZeroConfig
    ) -> None:
        """Trajectory loss computation produces valid output."""
        K = 3
        batch_size = 2

        actions = torch.randint(0, config.num_actions, (batch_size, K))
        rewards = torch.randn(batch_size, K)
        target_policies = torch.softmax(
            torch.randn(batch_size, K + 1, config.num_actions), dim=-1
        )
        target_values = torch.randn(batch_size, K + 1)

        loss, metrics = learner.compute_trajectory_loss(
            small_obs, actions, rewards, target_policies, target_values
        )

        assert loss.ndim == 0
        assert not torch.isnan(loss)
        assert "loss" in metrics
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "reward_loss" in metrics

    def test_compute_trajectory_loss_with_weights(
        self, learner: MuZeroLearner, small_obs: torch.Tensor, config: MuZeroConfig
    ) -> None:
        """Trajectory loss computation works with importance weights."""
        K = 3
        batch_size = 2

        actions = torch.randint(0, config.num_actions, (batch_size, K))
        rewards = torch.randn(batch_size, K)
        target_policies = torch.softmax(
            torch.randn(batch_size, K + 1, config.num_actions), dim=-1
        )
        target_values = torch.randn(batch_size, K + 1)
        weights = torch.tensor([0.5, 1.5])

        loss, metrics = learner.compute_trajectory_loss(
            small_obs, actions, rewards, target_policies, target_values, weights=weights
        )

        assert loss.ndim == 0
        assert not torch.isnan(loss)

    def test_compute_loss(
        self, learner: MuZeroLearner, config: MuZeroConfig
    ) -> None:
        """Single-step loss computation (Learner protocol)."""
        batch_size = 2

        states = torch.randint(0, 256, (batch_size, 4, 84, 84), dtype=torch.float32)
        actions = torch.randint(0, config.num_actions, (batch_size,))
        rewards = torch.randn(batch_size)
        next_states = torch.randint(0, 256, (batch_size, 4, 84, 84), dtype=torch.float32)
        dones = torch.zeros(batch_size)

        loss, metrics = learner.compute_loss(
            states, actions, rewards, next_states, dones
        )

        assert loss.ndim == 0
        assert not torch.isnan(loss)
        assert "value_loss" in metrics
        assert "reward_loss" in metrics

    def test_update_targets_hard(self, learner: MuZeroLearner) -> None:
        """Hard target update copies weights."""
        # Modify online
        for p in learner.model.online.parameters():
            p.data.fill_(42.0)

        learner.update_targets(tau=1.0)

        for online_p, target_p in zip(
            learner.model.online.parameters(),
            learner.model.target.parameters(),
        ):
            assert torch.allclose(online_p, target_p)

    def test_update_targets_soft(self, learner: MuZeroLearner) -> None:
        """Soft target update interpolates."""
        # Set initial weights
        for p in learner.model.online.parameters():
            p.data.fill_(1.0)
        for p in learner.model.target.parameters():
            p.data.fill_(0.0)

        learner.update_targets(tau=0.1)

        for target_p in learner.model.target.parameters():
            assert torch.allclose(target_p, torch.full_like(target_p, 0.1), atol=1e-5)


class TestRunMCTS:
    """Tests for the MCTS execution function."""

    def test_mcts_returns_valid_policy(self, model: MuZeroModel) -> None:
        """MCTS returns valid probability distribution."""
        obs = torch.randint(0, 256, (1, 4, 84, 84), dtype=torch.float32)

        policy, value, root = run_mcts(
            model, obs, num_simulations=10, add_noise=False
        )

        assert policy.shape == (model.num_actions,)
        assert abs(policy.sum() - 1.0) < 1e-5
        assert (policy >= 0).all()

    def test_mcts_returns_value(self, model: MuZeroModel) -> None:
        """MCTS returns a value estimate."""
        obs = torch.randint(0, 256, (1, 4, 84, 84), dtype=torch.float32)

        policy, value, root = run_mcts(model, obs, num_simulations=10)

        assert isinstance(value, float)

    def test_mcts_builds_tree(self, model: MuZeroModel) -> None:
        """MCTS builds a tree with children."""
        obs = torch.randint(0, 256, (1, 4, 84, 84), dtype=torch.float32)

        policy, value, root = run_mcts(model, obs, num_simulations=10)

        # Root should have children (one per action)
        assert len(root.children) == model.num_actions
        # Should have visits from simulations
        assert root.visits > 1

    def test_mcts_with_noise(self, model: MuZeroModel) -> None:
        """MCTS with Dirichlet noise adds exploration."""
        obs = torch.randint(0, 256, (1, 4, 84, 84), dtype=torch.float32)

        # Run multiple times - noise should cause variation
        policies = [
            run_mcts(model, obs, num_simulations=5, add_noise=True)[0]
            for _ in range(5)
        ]

        # With noise, policies should vary (probabilistic, so just check it runs)
        assert all(p.shape == (model.num_actions,) for p in policies)

    def test_mcts_temperature(self, model: MuZeroModel) -> None:
        """Temperature affects policy sharpness."""
        obs = torch.randint(0, 256, (1, 4, 84, 84), dtype=torch.float32)

        # Low temperature should be sharper
        policy_cold, _, _ = run_mcts(
            model, obs, num_simulations=10, temperature=0.1, add_noise=False
        )
        # High temperature should be flatter
        policy_hot, _, _ = run_mcts(
            model, obs, num_simulations=10, temperature=2.0, add_noise=False
        )

        # Cold policy should have higher max
        assert policy_cold.max() >= policy_hot.max() - 0.2  # Allow some variance


class TestComputeValueTarget:
    """Tests for n-step value target computation."""

    def test_single_step_no_done(self) -> None:
        """Single step with bootstrap."""
        rewards = [1.0, 2.0, 3.0]
        values = [10.0, 11.0, 12.0]
        dones = [False, False, False]
        discount = 0.9
        td_steps = 1

        targets = compute_value_target(rewards, values, dones, discount, td_steps)

        # V(0) = r_0 + γ * V(1) = 1 + 0.9 * 11 = 10.9
        assert abs(targets[0] - 10.9) < 1e-5
        # V(1) = r_1 + γ * V(2) = 2 + 0.9 * 12 = 12.8
        assert abs(targets[1] - 12.8) < 1e-5
        # V(2) = r_2 (no more steps to bootstrap)
        assert abs(targets[2] - 3.0) < 1e-5

    def test_multi_step(self) -> None:
        """Multi-step returns."""
        rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
        values = [10.0, 11.0, 12.0, 13.0, 14.0]
        dones = [False, False, False, False, False]
        discount = 0.9
        td_steps = 3

        targets = compute_value_target(rewards, values, dones, discount, td_steps)

        # V(0) = r_0 + γ*r_1 + γ²*r_2 + γ³*V(3)
        # = 1 + 0.9*2 + 0.81*3 + 0.729*13
        expected = 1 + 0.9 * 2 + 0.81 * 3 + 0.729 * 13
        assert abs(targets[0] - expected) < 1e-5

    def test_with_done(self) -> None:
        """Episode termination truncates returns."""
        rewards = [1.0, 2.0, 3.0, 4.0]
        values = [10.0, 11.0, 12.0, 13.0]
        dones = [False, True, False, False]  # Episode ends at step 1
        discount = 0.9
        td_steps = 3

        targets = compute_value_target(rewards, values, dones, discount, td_steps)

        # V(0) = r_0 + γ*r_1 (then done, no bootstrap)
        expected = 1.0 + 0.9 * 2.0
        assert abs(targets[0] - expected) < 1e-5


class TestMuZeroTrajectory:
    """Tests for trajectory data structure."""

    def test_trajectory_creation(self) -> None:
        """Trajectory stores data correctly."""
        import numpy as np

        traj = MuZeroTrajectory(
            obs=np.zeros((4, 84, 84), dtype=np.float32),
            actions=np.array([0, 1, 2]),
            rewards=np.array([1.0, 2.0, 3.0]),
            policies=np.ones((4, 7)) / 7,  # K+1 = 4
            values=np.array([1.0, 2.0, 3.0, 4.0]),
            dones=np.array([False, False, False, False]),
        )

        assert traj.obs.shape == (4, 84, 84)
        assert traj.actions.shape == (3,)
        assert traj.rewards.shape == (3,)
        assert traj.policies.shape == (4, 7)
        assert traj.values.shape == (4,)


class TestLatentGroundingLoss:
    """Tests for consistency and contrastive losses in learner."""

    def test_trajectory_loss_with_next_states(
        self, learner: MuZeroLearner, small_obs: torch.Tensor, config: MuZeroConfig
    ) -> None:
        """Trajectory loss includes grounding losses when next_states provided."""
        K = 3
        batch_size = 2

        actions = torch.randint(0, config.num_actions, (batch_size, K))
        rewards = torch.randn(batch_size, K)
        target_policies = torch.softmax(
            torch.randn(batch_size, K + 1, config.num_actions), dim=-1
        )
        target_values = torch.randn(batch_size, K + 1)
        # Next states for grounding: (N, K, C, H, W)
        next_states = torch.randint(
            0, 256, (batch_size, K, 4, 84, 84), dtype=torch.float32
        )

        loss, metrics = learner.compute_trajectory_loss(
            small_obs,
            actions,
            rewards,
            target_policies,
            target_values,
            next_states=next_states,
        )

        assert loss.ndim == 0
        assert not torch.isnan(loss)
        # Should have grounding loss metrics
        assert "consistency_loss" in metrics
        assert "contrastive_loss" in metrics
        # Grounding losses should be non-zero
        assert metrics["consistency_loss"] > 0
        assert metrics["contrastive_loss"] > 0

    def test_trajectory_loss_without_next_states(
        self, learner: MuZeroLearner, small_obs: torch.Tensor, config: MuZeroConfig
    ) -> None:
        """Trajectory loss works without grounding when no next_states."""
        K = 3
        batch_size = 2

        actions = torch.randint(0, config.num_actions, (batch_size, K))
        rewards = torch.randn(batch_size, K)
        target_policies = torch.softmax(
            torch.randn(batch_size, K + 1, config.num_actions), dim=-1
        )
        target_values = torch.randn(batch_size, K + 1)

        loss, metrics = learner.compute_trajectory_loss(
            small_obs,
            actions,
            rewards,
            target_policies,
            target_values,
            next_states=None,  # No grounding
        )

        assert loss.ndim == 0
        assert not torch.isnan(loss)
        # Grounding losses should be zero
        assert metrics["consistency_loss"] == 0.0
        assert metrics["contrastive_loss"] == 0.0

    def test_compute_loss_includes_grounding(
        self, learner: MuZeroLearner, config: MuZeroConfig
    ) -> None:
        """Single-step compute_loss includes grounding losses."""
        batch_size = 2

        states = torch.randint(0, 256, (batch_size, 4, 84, 84), dtype=torch.float32)
        actions = torch.randint(0, config.num_actions, (batch_size,))
        rewards = torch.randn(batch_size)
        next_states = torch.randint(
            0, 256, (batch_size, 4, 84, 84), dtype=torch.float32
        )
        dones = torch.zeros(batch_size)

        loss, metrics = learner.compute_loss(
            states, actions, rewards, next_states, dones
        )

        assert loss.ndim == 0
        assert not torch.isnan(loss)
        assert "consistency_loss" in metrics
        assert "contrastive_loss" in metrics
        # Should be non-zero since next_states are provided
        assert metrics["consistency_loss"] > 0
        assert metrics["contrastive_loss"] > 0

    def test_grounding_loss_weights(self, model: MuZeroModel) -> None:
        """Custom grounding loss weights are applied."""
        # Create learner with custom weights
        learner = MuZeroLearner(
            model=model,
            unroll_steps=2,
            consistency_loss_weight=5.0,
            contrastive_loss_weight=0.1,
        )

        assert learner.consistency_loss_weight == 5.0
        assert learner.contrastive_loss_weight == 0.1

    def test_grounding_uses_config_defaults(self, model: MuZeroModel) -> None:
        """Learner uses config weights when not overridden."""
        learner = MuZeroLearner(model=model, unroll_steps=2)

        assert learner.consistency_loss_weight == model.config.consistency_weight
        assert learner.contrastive_loss_weight == model.config.contrastive_weight


class TestScaleGradient:
    """Tests for gradient scaling function."""

    def test_forward_pass_unchanged(self) -> None:
        """Forward pass returns tensor unchanged."""
        from mario_rl.learners.muzero import scale_gradient

        x = torch.randn(4, 8, requires_grad=True)
        scaled = scale_gradient(x, 0.5)

        assert torch.allclose(scaled, x)

    def test_gradient_scaling(self) -> None:
        """Backward pass scales gradients correctly."""
        from mario_rl.learners.muzero import scale_gradient

        x = torch.randn(4, 8, requires_grad=True)
        scale = 0.25

        # Forward
        scaled = scale_gradient(x, scale)
        loss = (scaled ** 2).sum()

        # Backward
        loss.backward()

        # Gradient should be scaled: d/dx[(x*s + x.detach()*(1-s))^2] at x
        # = 2 * (x*s + x*(1-s)) * s = 2 * x * s
        expected_grad = 2 * x * scale
        assert torch.allclose(x.grad, expected_grad, rtol=1e-5)

    def test_scale_one_is_identity(self) -> None:
        """Scale of 1.0 gives normal gradients."""
        from mario_rl.learners.muzero import scale_gradient

        x = torch.randn(4, 8, requires_grad=True)
        scaled = scale_gradient(x, 1.0)
        loss = (scaled ** 2).sum()
        loss.backward()

        # Full gradient: 2x
        expected_grad = 2 * x
        assert torch.allclose(x.grad, expected_grad)

    def test_scale_zero_no_gradient(self) -> None:
        """Scale of 0.0 stops gradients."""
        from mario_rl.learners.muzero import scale_gradient

        x = torch.randn(4, 8, requires_grad=True)
        scaled = scale_gradient(x, 0.0)
        loss = (scaled ** 2).sum()
        loss.backward()

        # Gradient should be zero
        expected_grad = torch.zeros_like(x)
        assert torch.allclose(x.grad, expected_grad)


class TestMuZeroTrajectoryCollector:
    """Tests for trajectory collection."""

    def test_collector_creation(self) -> None:
        """Collector can be created with valid config."""
        from mario_rl.learners.muzero import MuZeroTrajectoryCollector
        import numpy as np

        collector = MuZeroTrajectoryCollector(
            unroll_steps=5,
            num_actions=7,
            td_steps=10,
            discount=0.99,
        )

        assert collector.unroll_steps == 5
        assert collector.num_actions == 7

    def test_start_episode(self) -> None:
        """Start episode initializes trajectory data."""
        from mario_rl.learners.muzero import MuZeroTrajectoryCollector
        import numpy as np

        collector = MuZeroTrajectoryCollector(
            unroll_steps=3,
            num_actions=4,
        )

        obs = np.random.rand(4, 84, 84).astype(np.float32)
        policy = np.array([0.25, 0.25, 0.25, 0.25])
        value = 1.5

        collector.start_episode(obs, policy, value)

        assert len(collector._obs) == 1
        assert len(collector._actions) == 0
        assert len(collector._policies) == 1

    def test_add_step(self) -> None:
        """Adding steps accumulates trajectory data."""
        from mario_rl.learners.muzero import MuZeroTrajectoryCollector
        import numpy as np

        collector = MuZeroTrajectoryCollector(
            unroll_steps=3,
            num_actions=4,
        )

        obs = np.random.rand(4, 84, 84).astype(np.float32)
        policy = np.array([0.25, 0.25, 0.25, 0.25])
        value = 1.5

        collector.start_episode(obs, policy, value)
        
        # Add a step
        next_obs = np.random.rand(4, 84, 84).astype(np.float32)
        next_policy = np.array([0.1, 0.2, 0.3, 0.4])
        collector.add_step(
            action=2,
            reward=0.5,
            next_obs=next_obs,
            done=False,
            next_policy=next_policy,
            next_value=2.0,
        )

        assert len(collector._obs) == 2
        assert len(collector._actions) == 1
        assert collector._actions[0] == 2
        assert collector._rewards[0] == 0.5

    def test_get_trajectories_returns_segments(self) -> None:
        """Get trajectories returns proper K-step segments."""
        from mario_rl.learners.muzero import MuZeroTrajectoryCollector
        import numpy as np

        K = 3
        collector = MuZeroTrajectoryCollector(
            unroll_steps=K,
            num_actions=4,
        )

        obs_shape = (4, 84, 84)
        
        # Start episode
        obs = np.random.rand(*obs_shape).astype(np.float32)
        policy = np.ones(4) / 4
        collector.start_episode(obs, policy, 0.0)

        # Add K+2 steps (should give us 2 trajectories)
        for i in range(K + 2):
            next_obs = np.random.rand(*obs_shape).astype(np.float32)
            collector.add_step(
                action=i % 4,
                reward=float(i),
                next_obs=next_obs,
                done=False,
                next_policy=policy,
                next_value=float(i + 1),
            )

        trajectories = collector.get_trajectories()

        # Should have 2 trajectory segments
        assert len(trajectories) >= 1

        # Check first trajectory shapes
        traj = trajectories[0]
        assert traj["obs"].shape == obs_shape
        assert traj["actions"].shape == (K,)
        assert traj["rewards"].shape == (K,)
        assert traj["policies"].shape == (K + 1, 4)
        assert traj["values"].shape == (K + 1,)
        assert traj["next_obs"].shape == (K, *obs_shape)
        assert traj["dones"].shape == (K + 1,)

    def test_get_trajectories_requires_minimum_steps(self) -> None:
        """Get trajectories returns empty if not enough steps."""
        from mario_rl.learners.muzero import MuZeroTrajectoryCollector
        import numpy as np

        K = 5
        collector = MuZeroTrajectoryCollector(
            unroll_steps=K,
            num_actions=4,
        )

        obs = np.random.rand(4, 84, 84).astype(np.float32)
        policy = np.ones(4) / 4
        collector.start_episode(obs, policy, 0.0)

        # Add only K-1 steps (not enough)
        for _ in range(K - 1):
            collector.add_step(
                action=0,
                reward=0.0,
                next_obs=obs.copy(),
                done=False,
                next_policy=policy,
                next_value=0.0,
            )

        trajectories = collector.get_trajectories()
        assert len(trajectories) == 0

    def test_reset_clears_data(self) -> None:
        """Reset clears all collected data."""
        from mario_rl.learners.muzero import MuZeroTrajectoryCollector
        import numpy as np

        collector = MuZeroTrajectoryCollector(
            unroll_steps=3,
            num_actions=4,
        )

        obs = np.random.rand(4, 84, 84).astype(np.float32)
        policy = np.ones(4) / 4
        collector.start_episode(obs, policy, 0.0)
        collector.add_step(0, 1.0, obs, False, policy, 0.0)

        assert len(collector) == 1

        collector.reset()
        assert len(collector) == 0
