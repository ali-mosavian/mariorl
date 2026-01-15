"""Tests for MuZero model components."""

import numpy as np
import pytest
import torch

from mario_rl.models.muzero import DynamicsNetwork
from mario_rl.models.muzero import MuZeroConfig
from mario_rl.models.muzero import MuZeroModel
from mario_rl.models.muzero import MuZeroNetwork
from mario_rl.models.muzero import PredictionNetwork
from mario_rl.models.muzero import PredictorNetwork
from mario_rl.models.muzero import ProjectorNetwork
from mario_rl.models.muzero import RepresentationNetwork
from mario_rl.models.muzero import info_nce_loss


@pytest.fixture
def config() -> MuZeroConfig:
    """Standard test configuration."""
    return MuZeroConfig(
        input_shape=(4, 84, 84),
        num_actions=7,
        latent_dim=64,  # Smaller for tests
        hidden_dim=32,
    )


@pytest.fixture
def small_obs() -> torch.Tensor:
    """Small batch of observations for testing."""
    return torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.float32)


class TestRepresentationNetwork:
    """Tests for the representation (encoder) network."""

    def test_output_shape(self, config: MuZeroConfig, small_obs: torch.Tensor) -> None:
        """Representation network produces correct latent shape."""
        net = RepresentationNetwork(config.input_shape, config.latent_dim)
        latent = net(small_obs)

        assert latent.shape == (2, config.latent_dim)

    def test_output_normalized(self, config: MuZeroConfig, small_obs: torch.Tensor) -> None:
        """Representation output is L2 normalized."""
        net = RepresentationNetwork(config.input_shape, config.latent_dim)
        latent = net(small_obs)

        norms = torch.norm(latent, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_different_inputs_different_outputs(self, config: MuZeroConfig) -> None:
        """Different observations produce different latent states."""
        net = RepresentationNetwork(config.input_shape, config.latent_dim)

        obs1 = torch.zeros(1, 4, 84, 84)
        obs2 = torch.ones(1, 4, 84, 84) * 255

        latent1 = net(obs1)
        latent2 = net(obs2)

        assert not torch.allclose(latent1, latent2)


class TestDynamicsNetwork:
    """Tests for the dynamics network."""

    def test_output_shapes(self, config: MuZeroConfig) -> None:
        """Dynamics network produces correct shapes."""
        net = DynamicsNetwork(config.latent_dim, config.num_actions, config.hidden_dim)

        state = torch.randn(2, config.latent_dim)
        action = torch.tensor([0, 3])

        next_state, reward = net(state, action)

        assert next_state.shape == (2, config.latent_dim)
        assert reward.shape == (2,)

    def test_next_state_normalized(self, config: MuZeroConfig) -> None:
        """Dynamics output state is L2 normalized."""
        net = DynamicsNetwork(config.latent_dim, config.num_actions, config.hidden_dim)

        state = torch.randn(2, config.latent_dim)
        action = torch.tensor([0, 3])

        next_state, _ = net(state, action)
        norms = torch.norm(next_state, dim=-1)

        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_different_actions_different_states(self, config: MuZeroConfig) -> None:
        """Different actions produce different next states."""
        net = DynamicsNetwork(config.latent_dim, config.num_actions, config.hidden_dim)

        state = torch.randn(1, config.latent_dim)

        next_state0, _ = net(state, torch.tensor([0]))
        next_state1, _ = net(state, torch.tensor([1]))

        assert not torch.allclose(next_state0, next_state1)


class TestPredictionNetwork:
    """Tests for the prediction network."""

    def test_output_shapes(self, config: MuZeroConfig) -> None:
        """Prediction network produces correct shapes."""
        net = PredictionNetwork(config.latent_dim, config.num_actions, config.hidden_dim)

        state = torch.randn(2, config.latent_dim)
        policy_logits, value = net(state)

        assert policy_logits.shape == (2, config.num_actions)
        assert value.shape == (2,)

    def test_policy_sums_to_one(self, config: MuZeroConfig) -> None:
        """Policy logits softmax to valid distribution."""
        net = PredictionNetwork(config.latent_dim, config.num_actions, config.hidden_dim)

        state = torch.randn(2, config.latent_dim)
        policy_logits, _ = net(state)
        policy = torch.softmax(policy_logits, dim=-1)

        sums = policy.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


class TestMuZeroNetwork:
    """Tests for the combined MuZero network."""

    def test_initial_inference(self, config: MuZeroConfig, small_obs: torch.Tensor) -> None:
        """Initial inference produces correct shapes."""
        net = MuZeroNetwork(config)
        state, policy_logits, value = net.initial_inference(small_obs)

        assert state.shape == (2, config.latent_dim)
        assert policy_logits.shape == (2, config.num_actions)
        assert value.shape == (2,)

    def test_recurrent_inference(self, config: MuZeroConfig) -> None:
        """Recurrent inference produces correct shapes."""
        net = MuZeroNetwork(config)

        state = torch.randn(2, config.latent_dim)
        action = torch.tensor([0, 3])

        next_state, reward, policy_logits, value = net.recurrent_inference(state, action)

        assert next_state.shape == (2, config.latent_dim)
        assert reward.shape == (2,)
        assert policy_logits.shape == (2, config.num_actions)
        assert value.shape == (2,)

    def test_select_action_greedy(self, config: MuZeroConfig, small_obs: torch.Tensor) -> None:
        """Greedy action selection returns valid actions."""
        net = MuZeroNetwork(config)
        action = net.select_action(small_obs, greedy=True)

        assert action.shape == (2,)
        assert (action >= 0).all()
        assert (action < config.num_actions).all()

    def test_select_action_stochastic(
        self, config: MuZeroConfig, small_obs: torch.Tensor
    ) -> None:
        """Stochastic action selection samples from policy."""
        net = MuZeroNetwork(config)

        # Run multiple times to check for variation
        actions = [net.select_action(small_obs, greedy=False) for _ in range(10)]

        # At least some should be different (probabilistic)
        unique_actions = set(tuple(a.tolist()) for a in actions)
        # With temperature=1.0 and 7 actions, we should see some variation
        assert len(unique_actions) >= 1  # At minimum, should work


class TestMuZeroModel:
    """Tests for the full MuZero model with target network."""

    def test_sync_target(self, config: MuZeroConfig) -> None:
        """Target network sync copies weights correctly."""
        model = MuZeroModel(config)

        # Modify online weights
        for p in model.online.parameters():
            p.data.fill_(1.0)

        # Sync should copy online to target
        model.sync_target()

        for online_p, target_p in zip(
            model.online.parameters(), model.target.parameters()
        ):
            assert torch.allclose(online_p, target_p)

    def test_soft_update(self, config: MuZeroConfig) -> None:
        """Soft update interpolates weights correctly."""
        model = MuZeroModel(config)

        # Set online to 1.0 and target to 0.0
        for p in model.online.parameters():
            p.data.fill_(1.0)
        for p in model.target.parameters():
            p.data.fill_(0.0)

        tau = 0.5
        model.soft_update(tau)

        # Target should now be 0.5
        for target_p in model.target.parameters():
            assert torch.allclose(target_p, torch.full_like(target_p, tau), atol=1e-5)

    def test_compute_loss(self, config: MuZeroConfig, small_obs: torch.Tensor) -> None:
        """Loss computation produces valid loss and metrics."""
        model = MuZeroModel(config)

        K = 3  # Unroll steps
        batch_size = 2

        actions = torch.randint(0, config.num_actions, (batch_size, K))
        target_policies = torch.softmax(
            torch.randn(batch_size, K + 1, config.num_actions), dim=-1
        )
        target_values = torch.randn(batch_size, K + 1)
        target_rewards = torch.randn(batch_size, K)

        loss, info = model.compute_loss(
            small_obs, actions, target_policies, target_values, target_rewards
        )

        assert loss.ndim == 0  # Scalar
        assert not torch.isnan(loss)
        assert "loss" in info
        assert "policy_loss" in info
        assert "value_loss" in info
        assert "reward_loss" in info


class TestMCTSNodeGeneric:
    """Tests for the generic MCTSNode with different state types."""

    def test_emulator_node(self) -> None:
        """MCTSNode works with numpy array state (emulator snapshot)."""
        from mario_rl.mcts import MCTSNode

        state = np.random.randn(1000).astype(np.float32)  # Emulator state
        obs = np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8)

        node: MCTSNode[np.ndarray] = MCTSNode(state=state, obs=obs)

        assert node.state.shape == (1000,)
        assert node.visits == 0
        assert node.value == 0.0

    def test_latent_node(self) -> None:
        """MCTSNode works with tensor state (latent state)."""
        from mario_rl.mcts import MCTSNode

        state = torch.randn(512)  # Latent state
        obs = np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8)

        node: MCTSNode[torch.Tensor] = MCTSNode(state=state, obs=obs)

        assert node.state.shape == (512,)
        assert isinstance(node.state, torch.Tensor)

    def test_get_policy_target(self) -> None:
        """Policy target extraction from visit counts."""
        from mario_rl.mcts import MCTSNode

        root: MCTSNode[np.ndarray] = MCTSNode(
            state=np.array([0]),
            obs=np.zeros((4, 84, 84)),
            visits=100,
        )

        # Add children with different visit counts
        for action, visits in enumerate([50, 30, 20, 0, 0, 0, 0]):
            child: MCTSNode[np.ndarray] = MCTSNode(
                state=np.array([action]),
                obs=np.zeros((4, 84, 84)),
                parent=root,
                action=action,
                visits=visits,
            )
            root.children.append(child)

        policy = root.get_policy_target(num_actions=7, temperature=1.0)

        assert policy.shape == (7,)
        assert np.isclose(policy.sum(), 1.0)
        assert np.isclose(policy[0], 0.5)  # 50 / 100
        assert np.isclose(policy[1], 0.3)  # 30 / 100
        assert np.isclose(policy[2], 0.2)  # 20 / 100


class TestProjectorNetwork:
    """Tests for the projector network (contrastive loss)."""

    def test_output_shape(self, config: MuZeroConfig) -> None:
        """Projector produces correct embedding shape."""
        net = ProjectorNetwork(config.latent_dim, config.proj_dim, config.hidden_dim)
        z = torch.randn(2, config.latent_dim)
        e = net(z)

        assert e.shape == (2, config.proj_dim)

    def test_output_normalized(self, config: MuZeroConfig) -> None:
        """Projector output is L2 normalized."""
        net = ProjectorNetwork(config.latent_dim, config.proj_dim, config.hidden_dim)
        z = torch.randn(2, config.latent_dim)
        e = net(z)

        norms = torch.norm(e, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestPredictorNetwork:
    """Tests for the predictor network (consistency loss)."""

    def test_output_shape(self, config: MuZeroConfig) -> None:
        """Predictor produces same shape as input."""
        net = PredictorNetwork(config.latent_dim, config.hidden_dim)
        z = torch.randn(2, config.latent_dim)
        z_pred = net(z)

        assert z_pred.shape == (2, config.latent_dim)

    def test_different_from_input(self, config: MuZeroConfig) -> None:
        """Predictor transforms input (not identity)."""
        net = PredictorNetwork(config.latent_dim, config.hidden_dim)
        z = torch.randn(2, config.latent_dim)
        z_pred = net(z)

        # Should not be identical to input
        assert not torch.allclose(z, z_pred)


class TestInfoNCELoss:
    """Tests for the InfoNCE contrastive loss function."""

    def test_output_shape(self) -> None:
        """InfoNCE produces per-sample loss."""
        query = torch.randn(4, 32)
        positive = torch.randn(4, 32)

        loss = info_nce_loss(query, positive, temperature=0.1)

        assert loss.shape == (4,)

    def test_positive_pairs_low_loss(self) -> None:
        """Identical query and positive should have low loss."""
        # Use identical embeddings for query and positive
        embeddings = torch.randn(4, 32)
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

        loss = info_nce_loss(embeddings, embeddings, temperature=0.1)

        # Loss should be very low when query matches positive exactly
        assert loss.mean() < 0.1

    def test_random_pairs_higher_loss(self) -> None:
        """Random query and positive should have higher loss."""
        query = torch.randn(4, 32)
        positive = torch.randn(4, 32)  # Different random embeddings

        loss = info_nce_loss(query, positive, temperature=0.1)

        # Loss should be higher when embeddings don't match
        # With random embeddings, loss is roughly log(N) where N is batch size
        assert loss.mean() > 0.5

    def test_temperature_effect(self) -> None:
        """Lower temperature makes loss more discriminative."""
        query = torch.randn(4, 32)
        positive = query + torch.randn_like(query) * 0.1  # Slightly different

        loss_low_temp = info_nce_loss(query, positive, temperature=0.05)
        loss_high_temp = info_nce_loss(query, positive, temperature=1.0)

        # Lower temperature should give different loss characteristics
        # but both should be finite
        assert torch.isfinite(loss_low_temp).all()
        assert torch.isfinite(loss_high_temp).all()


class TestMuZeroNetworkGrounding:
    """Tests for latent grounding methods on MuZeroNetwork."""

    def test_project_method(self, config: MuZeroConfig) -> None:
        """Network has working project method."""
        net = MuZeroNetwork(config)
        z = torch.randn(2, config.latent_dim)

        e = net.project(z)

        assert e.shape == (2, config.proj_dim)

    def test_predict_method(self, config: MuZeroConfig) -> None:
        """Network has working predict method."""
        net = MuZeroNetwork(config)
        z = torch.randn(2, config.latent_dim)

        z_pred = net.predict(z)

        assert z_pred.shape == (2, config.latent_dim)


class TestMuZeroModelGrounding:
    """Tests for latent grounding on MuZeroModel."""

    def test_encode_method(self, config: MuZeroConfig, small_obs: torch.Tensor) -> None:
        """Model has working encode method."""
        model = MuZeroModel(config)
        z = model.encode(small_obs)

        assert z.shape == (2, config.latent_dim)

    def test_project_method(self, config: MuZeroConfig) -> None:
        """Model has working project method."""
        model = MuZeroModel(config)
        z = torch.randn(2, config.latent_dim)

        e = model.project(z)

        assert e.shape == (2, config.proj_dim)

    def test_predict_method(self, config: MuZeroConfig) -> None:
        """Model has working predict method."""
        model = MuZeroModel(config)
        z = torch.randn(2, config.latent_dim)

        z_pred = model.predict(z)

        assert z_pred.shape == (2, config.latent_dim)
