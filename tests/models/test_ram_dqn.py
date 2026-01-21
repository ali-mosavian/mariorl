"""Tests for RAM-based DQN model."""

import torch

from mario_rl.models.ram_dqn import RAMDQNNet
from mario_rl.models.ram_dqn import RAMBackbone
from mario_rl.models.ram_dqn import RAMDoubleDQN


class TestRAMBackbone:
    """Tests for RAMBackbone."""

    def test_forward_uint8_input(self) -> None:
        """Test forward pass with uint8 input."""
        backbone = RAMBackbone(ram_size=2048, feature_dim=512)
        x = torch.randint(0, 256, (4, 2048), dtype=torch.uint8)
        out = backbone(x)
        assert out.shape == (4, 512)
        assert out.dtype == torch.float32

    def test_forward_float_input(self) -> None:
        """Test forward pass with float input in [0, 255]."""
        backbone = RAMBackbone(ram_size=2048, feature_dim=512)
        x = torch.rand(4, 2048) * 255.0
        out = backbone(x)
        assert out.shape == (4, 512)

    def test_forward_normalized_input(self) -> None:
        """Test forward pass with already normalized [0, 1] input."""
        backbone = RAMBackbone(ram_size=2048, feature_dim=512)
        x = torch.rand(4, 2048)
        out = backbone(x)
        assert out.shape == (4, 512)


class TestRAMDQNNet:
    """Tests for RAMDQNNet."""

    def test_forward(self) -> None:
        """Test forward pass returns Q-values for all actions."""
        net = RAMDQNNet(ram_size=2048, num_actions=7)
        x = torch.randint(0, 256, (4, 2048), dtype=torch.uint8)
        q_values = net(x)
        assert q_values.shape == (4, 7)

    def test_select_action_greedy(self) -> None:
        """Test greedy action selection."""
        net = RAMDQNNet(ram_size=2048, num_actions=7)
        x = torch.randint(0, 256, (4, 2048), dtype=torch.uint8)
        actions = net.select_action(x, epsilon=0.0)
        assert actions.shape == (4,)
        assert actions.dtype == torch.int64
        assert (actions >= 0).all() and (actions < 7).all()

    def test_select_action_random(self) -> None:
        """Test random action selection with epsilon=1."""
        net = RAMDQNNet(ram_size=2048, num_actions=7)
        x = torch.randint(0, 256, (4, 2048), dtype=torch.uint8)
        actions = net.select_action(x, epsilon=1.0)
        assert actions.shape == (4,)
        assert (actions >= 0).all() and (actions < 7).all()


class TestRAMDoubleDQN:
    """Tests for RAMDoubleDQN."""

    def test_forward_online(self) -> None:
        """Test forward pass through online network."""
        model = RAMDoubleDQN(ram_size=2048, num_actions=7)
        x = torch.randint(0, 256, (4, 2048), dtype=torch.uint8)
        q_values = model(x, network="online")
        assert q_values.shape == (4, 7)

    def test_forward_target(self) -> None:
        """Test forward pass through target network."""
        model = RAMDoubleDQN(ram_size=2048, num_actions=7)
        x = torch.randint(0, 256, (4, 2048), dtype=torch.uint8)
        q_values = model(x, network="target")
        assert q_values.shape == (4, 7)

    def test_sync_target(self) -> None:
        """Test target network sync."""
        model = RAMDoubleDQN(ram_size=2048, num_actions=7)
        model.eval()  # Disable dropout for deterministic comparison
        x = torch.randint(0, 256, (4, 2048), dtype=torch.uint8)

        # Initial sync means they should be equal
        online_q = model(x, network="online")
        target_q = model(x, network="target")
        assert torch.allclose(online_q, target_q)

        # Modify online weights
        with torch.no_grad():
            for p in model.online.parameters():
                p.add_(1.0)

        # Now they should differ
        online_q = model(x, network="online")
        target_q = model(x, network="target")
        assert not torch.allclose(online_q, target_q)

        # Sync and they should be equal again
        model.sync_target()
        online_q = model(x, network="online")
        target_q = model(x, network="target")
        assert torch.allclose(online_q, target_q)

    def test_soft_update(self) -> None:
        """Test soft target update."""
        model = RAMDoubleDQN(ram_size=2048, num_actions=7)
        model.eval()  # Disable dropout for deterministic comparison

        # Get initial target weights
        target_weights_before = {k: v.clone() for k, v in model.target.state_dict().items()}

        # Modify online weights
        with torch.no_grad():
            for p in model.online.parameters():
                p.add_(10.0)

        # Soft update with tau=0.5
        model.soft_update(tau=0.5)

        # Check weights moved halfway
        for k, v in model.target.state_dict().items():
            online_v = model.online.state_dict()[k]
            expected = 0.5 * online_v + 0.5 * target_weights_before[k]
            assert torch.allclose(v, expected)

    def test_compute_loss(self) -> None:
        """Test loss computation."""
        model = RAMDoubleDQN(ram_size=2048, num_actions=7)

        batch_size = 8
        states = torch.randint(0, 256, (batch_size, 2048), dtype=torch.uint8)
        actions = torch.randint(0, 7, (batch_size,))
        rewards = torch.randn(batch_size)
        next_states = torch.randint(0, 256, (batch_size, 2048), dtype=torch.uint8)
        dones = torch.zeros(batch_size)

        loss, info = model.compute_loss(states, actions, rewards, next_states, dones)

        assert loss.shape == ()
        assert loss.item() >= 0
        assert "loss" in info
        assert "q_mean" in info
        assert "q_max" in info
        assert "td_error_mean" in info
        assert "target_q_mean" in info

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through the network."""
        model = RAMDoubleDQN(ram_size=2048, num_actions=7)

        batch_size = 4
        states = torch.randint(0, 256, (batch_size, 2048), dtype=torch.uint8)
        actions = torch.randint(0, 7, (batch_size,))
        rewards = torch.randn(batch_size)
        next_states = torch.randint(0, 256, (batch_size, 2048), dtype=torch.uint8)
        dones = torch.zeros(batch_size)

        loss, _ = model.compute_loss(states, actions, rewards, next_states, dones)
        loss.backward()

        # Check online network has gradients
        for p in model.online.parameters():
            assert p.grad is not None
            assert not torch.isnan(p.grad).any()

        # Target network should NOT have gradients (frozen)
        for p in model.target.parameters():
            assert p.grad is None
