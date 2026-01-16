"""
Train a simple encoder to predict x_pos and reward from frames.

Tests the hypothesis that auxiliary supervision (game state prediction)
can learn better representations than pixel reconstruction.

Usage:
    uv run python scripts/train_auxiliary_encoder.py --steps 50000
"""

import click
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from collections import deque
from tqdm import tqdm

from mario_rl.environment.factory import create_mario_env
from mario_rl.models.ddqn import DoubleDQN


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric log: sign(x) * log(|x| + 1)"""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog"""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


class AuxiliaryEncoder(nn.Module):
    """
    Simple CNN encoder that predicts x_pos and reward from frames.

    Architecture:
        Frame (4, 64, 64) → CNN → z (latent_dim) → x_pos, reward
    """

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        # CNN backbone (same as DDQN)
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 4, 64, 64)
            flat_size = self.encoder(dummy).shape[1]

        # Latent projection
        self.fc_latent = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

        # Prediction heads
        self.x_pos_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode frames to latent z."""
        x = x / 255.0  # Normalize
        features = self.encoder(x)
        z = self.fc_latent(features)
        return z

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            z: latent representation
            x_pos_pred: predicted x_pos (symlog scale)
            reward_pred: predicted reward (symlog scale)
        """
        z = self.encode(x)
        x_pos_pred = self.x_pos_head(z)
        reward_pred = self.reward_head(z)
        return z, x_pos_pred, reward_pred


@dataclass
class Transition:
    state: np.ndarray
    x_pos: float
    reward: float


def collect_data(
    env,
    policy_model: DoubleDQN | None,
    device: str,
    num_steps: int,
    epsilon: float = 0.1,
) -> list[Transition]:
    """Collect transitions using policy model or random actions."""
    transitions = []
    state, info = env.reset()

    for _ in range(num_steps):
        if policy_model is not None:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = policy_model(state_tensor).argmax(dim=-1).item()
        else:
            action = env.action_space.sample()

        next_state, reward, done, truncated, info = env.step(action)

        transitions.append(Transition(
            state=state.copy(),
            x_pos=info.get("x_pos", 0),
            reward=reward,
        ))

        state = next_state
        if done or truncated:
            state, info = env.reset()

    return transitions


def train_step(
    model: AuxiliaryEncoder,
    optimizer: torch.optim.Optimizer,
    batch: list[Transition],
    device: str,
) -> dict[str, float]:
    """Single training step."""
    # Prepare batch
    states = torch.tensor(np.stack([t.state for t in batch]), dtype=torch.float32, device=device)
    x_pos = torch.tensor([t.x_pos for t in batch], dtype=torch.float32, device=device)
    rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device)

    # Forward pass
    z, x_pos_pred, reward_pred = model(states)

    # Targets in symlog scale (symlog is the normalization)
    x_pos_target = symlog(x_pos)
    reward_target = symlog(rewards)

    # Losses
    x_pos_loss = F.mse_loss(x_pos_pred.squeeze(), x_pos_target)
    reward_loss = F.mse_loss(reward_pred.squeeze(), reward_target)

    # Total loss
    loss = x_pos_loss + reward_loss

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Compute metrics
    with torch.no_grad():
        # Convert predictions back to original scale for error calculation
        x_pos_pred_actual = symexp(x_pos_pred.squeeze())
        x_pos_error = (x_pos_pred_actual - x_pos).abs().mean()

        reward_pred_actual = symexp(reward_pred.squeeze())
        reward_error = (reward_pred_actual - rewards).abs().mean()

    return {
        "loss": loss.item(),
        "x_pos_loss": x_pos_loss.item(),
        "reward_loss": reward_loss.item(),
        "x_pos_mae": x_pos_error.item(),
        "reward_mae": reward_error.item(),
        "z_mean": z.mean().item(),
        "z_std": z.std().item(),
    }


def evaluate(
    model: AuxiliaryEncoder,
    env,
    policy_model: DoubleDQN | None,
    device: str,
    num_episodes: int = 5,
) -> dict[str, float]:
    """Evaluate encoder predictions."""
    model.eval()

    all_x_pos_errors = []
    all_reward_errors = []

    for _ in range(num_episodes):
        state, info = env.reset()
        done = False

        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

                # Get predictions
                z, x_pos_pred, reward_pred = model(state_tensor)
                x_pos_pred_actual = symexp(x_pos_pred.squeeze())

                # Get action
                if policy_model is not None:
                    action = policy_model(state_tensor).argmax(dim=-1).item()
                else:
                    action = env.action_space.sample()

            next_state, reward, done, truncated, info = env.step(action)

            # Record errors
            actual_x_pos = info.get("x_pos", 0)
            all_x_pos_errors.append(abs(x_pos_pred_actual.item() - actual_x_pos))

            reward_pred_actual = symexp(reward_pred.squeeze()).item()
            all_reward_errors.append(abs(reward_pred_actual - reward))

            state = next_state
            if truncated:
                break

    model.train()

    return {
        "eval_x_pos_mae": np.mean(all_x_pos_errors),
        "eval_reward_mae": np.mean(all_reward_errors),
    }


@click.command()
@click.option("--steps", default=50000, help="Training steps")
@click.option("--batch-size", default=64, help="Batch size")
@click.option("--lr", default=1e-4, help="Learning rate")
@click.option("--latent-dim", default=128, help="Latent dimension")
@click.option("--buffer-size", default=10000, help="Replay buffer size")
@click.option("--policy-checkpoint", default=None, help="DDQN checkpoint for data collection")
@click.option("--device", default="cuda", help="Device")
@click.option("--eval-interval", default=5000, help="Evaluation interval")
@click.option("--save-path", default="/tmp/auxiliary_encoder.pt", help="Save path")
def main(
    steps: int,
    batch_size: int,
    lr: float,
    latent_dim: int,
    buffer_size: int,
    policy_checkpoint: str | None,
    device: str,
    eval_interval: int,
    save_path: str,
):
    """Train auxiliary encoder to predict x_pos and reward."""
    print("=" * 60)
    print("AUXILIARY ENCODER TRAINING")
    print("=" * 60)
    print(f"Goal: Learn latent z that predicts x_pos and reward")
    print(f"Steps: {steps}, Batch: {batch_size}, LR: {lr}")
    print(f"Latent dim: {latent_dim}")
    print()

    # Load policy model if provided
    policy_model = None
    if policy_checkpoint:
        print(f"Loading policy from {policy_checkpoint}...")
        checkpoint = torch.load(policy_checkpoint, map_location=device, weights_only=False)
        policy_model = DoubleDQN(input_shape=(4, 64, 64), num_actions=7).to(device)
        if "model_state_dict" in checkpoint:
            policy_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            policy_model.load_state_dict(checkpoint)
        policy_model.eval()
        print("Policy loaded!")

    # Create environment and model
    env = create_mario_env(level=(1, 1))
    model = AuxiliaryEncoder(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Replay buffer
    buffer: deque[Transition] = deque(maxlen=buffer_size)

    # Initial data collection
    print("Collecting initial data...")
    initial_data = collect_data(env, policy_model, device, num_steps=buffer_size // 2)
    buffer.extend(initial_data)
    print(f"Buffer size: {len(buffer)}")

    # Training loop
    print("\nTraining...")
    metrics_window: deque[dict[str, float]] = deque(maxlen=100)

    pbar = tqdm(range(steps), desc="Training", unit="step")
    for step in pbar:
        # Collect more data periodically
        if step % 500 == 0 and step > 0:
            new_data = collect_data(env, policy_model, device, num_steps=200)
            buffer.extend(new_data)

        # Sample batch
        batch_indices = np.random.choice(len(buffer), size=batch_size, replace=False)
        batch = [buffer[i] for i in batch_indices]

        # Train step
        metrics = train_step(model, optimizer, batch, device)
        metrics_window.append(metrics)

        # Update progress bar
        if len(metrics_window) > 0:
            avg_metrics = {k: np.mean([m[k] for m in metrics_window]) for k in metrics.keys()}
            pbar.set_postfix({
                "loss": f"{avg_metrics['loss']:.4f}",
                "x_mae": f"{avg_metrics['x_pos_mae']:.1f}",
                "r_mae": f"{avg_metrics['reward_mae']:.3f}",
                "z_std": f"{avg_metrics['z_std']:.3f}",
            })

        # Evaluate
        if step % eval_interval == 0 and step > 0:
            eval_metrics = evaluate(model, env, policy_model, device)
            tqdm.write(
                f"  EVAL @ {step} | x_pos MAE: {eval_metrics['eval_x_pos_mae']:.1f} | "
                f"reward MAE: {eval_metrics['eval_reward_mae']:.3f}"
            )

    # Save model
    torch.save({
        "model_state_dict": model.state_dict(),
        "latent_dim": latent_dim,
        "step": steps,
    }, save_path)
    print(f"\nSaved to {save_path}")

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    eval_metrics = evaluate(model, env, policy_model, device, num_episodes=10)
    print(f"X-pos MAE: {eval_metrics['eval_x_pos_mae']:.1f} pixels")
    print(f"Reward MAE: {eval_metrics['eval_reward_mae']:.3f}")

    # Test latent diversity
    print("\nLatent diversity test...")
    test_data = collect_data(env, policy_model, device, num_steps=500)
    with torch.no_grad():
        states = torch.tensor(
            np.stack([t.state for t in test_data]), dtype=torch.float32, device=device
        )
        z = model.encode(states)

        # Compute pairwise distances
        z_np = z.cpu().numpy()
        from scipy.spatial.distance import pdist

        distances = pdist(z_np, metric="euclidean")
        cosine_sims = 1 - pdist(z_np, metric="cosine")

        print(f"Latent L2 distance: mean={distances.mean():.2f}, std={distances.std():.2f}")
        print(f"Latent cosine sim: mean={cosine_sims.mean():.3f}")

        # Check correlation between z and x_pos
        x_positions = np.array([t.x_pos for t in test_data])
        z_pca = z_np[:, 0]  # First latent dimension
        corr = np.corrcoef(z_pca, x_positions)[0, 1]
        print(f"Correlation (z[0] vs x_pos): {corr:.3f}")

    env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
