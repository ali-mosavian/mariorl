"""
Visualize world model dynamics by comparing imagined rollouts with real
environment trajectories.

This tests both the encoder/decoder (reconstruction) and the dynamics model
(prediction of next states given actions).

Usage:
    uv run python scripts/visualize_world_model_dynamics.py --checkpoint path/to/checkpoint.pt --horizon 8
"""

from pathlib import Path

import click
import torch
import numpy as np
import matplotlib.pyplot as plt

from mario_rl.models.dreamer import DreamerModel
from mario_rl.agent.world_model import MarioWorldModel
from mario_rl.environment.factory import create_mario_env


def load_world_model(checkpoint_path: Path, device: str):
    """Load trained world model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "world_model" in checkpoint:
        # MarioWorldModel format
        state_dim = (4, 64, 64, 1)
        action_dim = 12
        latent_dim = 128

        world_model = MarioWorldModel(
            frame_shape=state_dim,
            num_actions=action_dim,
            latent_dim=latent_dim,
        ).to(device)
        world_model.load_state_dict(checkpoint["world_model"])

    elif "model_state_dict" in checkpoint:
        # DreamerModel format
        input_shape = (4, 64, 64)
        action_dim = 12
        latent_dim = 128

        world_model = DreamerModel(
            input_shape=input_shape,
            num_actions=action_dim,
            latent_dim=latent_dim,
        ).to(device)
        world_model.load_state_dict(checkpoint["model_state_dict"])

    else:
        # Direct state dict - try DreamerModel
        input_shape = (4, 64, 64)
        action_dim = 12
        latent_dim = 128

        world_model = DreamerModel(
            input_shape=input_shape,
            num_actions=action_dim,
            latent_dim=latent_dim,
        ).to(device)
        world_model.load_state_dict(checkpoint)

    world_model.eval()
    return world_model


def collect_comparison_rollout(
    env,
    world_model,
    horizon: int,
    device: str,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """
    Collect a rollout comparing real environment dynamics with world model
    imagination.

    Starting from the same initial state, we:
    1. Take the same actions in both real env and imagined trajectory
    2. Record frames from both sources
    3. Compare how predictions diverge from reality

    Returns:
        real_frames: List of real environment frames
        imagined_frames: List of frames predicted by world model
        actions: List of actions taken
    """
    real_frames = []
    imagined_frames = []
    actions_taken = []
    h_state = None  # For GRU hidden state

    # Reset environment
    state, _ = env.reset()

    # Detect model type
    is_dreamer = hasattr(world_model, "actor")

    with torch.no_grad():
        # Get initial latent state from real observation
        state_tensor = torch.from_numpy(state).float().to(device)
        state_tensor = state_tensor.unsqueeze(0)  # (1, 4, 64, 64)

        if is_dreamer:
            # DreamerModel: input (N, C, H, W) in [0, 255]
            z_current = world_model.encode(state_tensor, deterministic=True)
        else:
            # MarioWorldModel: input (N, F, H, W, C) in [0, 1]
            state_tensor = state_tensor.unsqueeze(-1) / 255.0
            z_current = world_model.encode(state_tensor, deterministic=True)

        for _ in range(horizon):
            # Store current real frame
            real_frame = state[-1].copy()
            real_frames.append(real_frame)

            # Get imagined frame from current latent
            if is_dreamer:
                imagined = world_model.decoder(z_current)  # (1, C, H, W)
                imagined_frame = imagined[0, -1, :, :].cpu().numpy()
            else:
                imagined = world_model.decode(z_current)  # (1, F, H, W, C)
                imagined_frame = imagined[0, -1, :, :, 0].cpu().numpy()

            imagined_frames.append(imagined_frame)

            # Select action (random for visualization)
            action = np.random.randint(0, 12)
            actions_taken.append(action)

            # Step real environment
            state, reward, terminated, truncated, info = env.step(action)

            # Imagine next state with dynamics model
            action_tensor = torch.tensor([action], dtype=torch.long, device=device)

            if is_dreamer:
                # DreamerModel dynamics returns (z_next, h_next, z_mu)
                z_next, h_state, _ = world_model.dynamics(z_current, action_tensor, h_state)
            else:
                # MarioWorldModel dynamics
                z_next, _, _ = world_model.dynamics(z_current, action_tensor)

            z_current = z_next

            if terminated or truncated:
                # If episode ends, we can't continue comparison
                break

    return real_frames, imagined_frames, actions_taken


def visualize_dynamics_comparison(
    real_frames: list[np.ndarray],
    imagined_frames: list[np.ndarray],
    actions: list[int],
    save_path: Path | None = None,
) -> None:
    """
    Visualize real vs imagined trajectory side by side.

    Shows:
    - Row 1: Real environment frames
    - Row 2: Imagined frames (from dynamics model)
    - Row 3: Absolute difference
    - Row 4: Cumulative divergence over time
    """
    num_frames = len(real_frames)

    # Display all frames (or max 10)
    max_cols = min(num_frames, 10)
    if num_frames > max_cols:
        indices = np.linspace(0, num_frames - 1, max_cols, dtype=int)
    else:
        indices = range(num_frames)

    num_display = len(indices)

    # Create figure
    fig, axes = plt.subplots(4, num_display, figsize=(2 * num_display, 8))

    if num_display == 1:
        axes = axes.reshape(4, 1)

    action_names = ["NOOP", "RIGHT", "R+A", "R+B", "R+A+B", "A", "LEFT", "L+A", "L+B", "L+A+B", "DOWN", "UP"]

    # Track cumulative error
    cumulative_mse = []
    mse_so_far = 0.0

    for real, imagined in zip(real_frames, imagined_frames, strict=False):
        real_norm = real / 255.0
        mse = np.mean((real_norm - imagined) ** 2)
        mse_so_far += mse
        cumulative_mse.append(mse_so_far)

    for col, idx in enumerate(indices):
        real = real_frames[idx]
        imagined = imagined_frames[idx]
        action = actions[idx]

        real_norm = real / 255.0
        diff = np.abs(real_norm - imagined)
        mse = np.mean(diff**2)

        # Row 1: Real
        axes[0, col].imshow(real_norm, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"Real t={idx}\n{action_names[action]}", fontsize=8)
        axes[0, col].axis("off")

        # Row 2: Imagined
        axes[1, col].imshow(imagined, cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title("Imagined", fontsize=8)
        axes[1, col].axis("off")

        # Row 3: Difference
        im_diff = axes[2, col].imshow(diff, cmap="hot", vmin=0, vmax=0.5)
        axes[2, col].set_title(f"MSE: {mse:.4f}", fontsize=8)
        axes[2, col].axis("off")

        # Row 4: Cumulative error over time (line plot)
        axes[3, col].clear()
        axes[3, col].plot(range(idx + 1), cumulative_mse[: idx + 1], "r-", linewidth=2)
        axes[3, col].set_xlim(0, num_frames - 1)
        axes[3, col].set_ylim(0, max(cumulative_mse) * 1.1)
        axes[3, col].axvline(idx, color="blue", linestyle="--", alpha=0.5)
        axes[3, col].set_xlabel("Step", fontsize=7)
        axes[3, col].set_ylabel("Cumul. MSE", fontsize=7)
        axes[3, col].tick_params(labelsize=6)
        axes[3, col].grid(True, alpha=0.3)

    # Add row labels
    fig.text(0.02, 0.88, "Real", rotation=90, va="center", fontsize=12, weight="bold")
    fig.text(0.02, 0.67, "Imagined\n(Dynamics)", rotation=90, va="center", fontsize=12, weight="bold")
    fig.text(0.02, 0.45, "Difference", rotation=90, va="center", fontsize=12, weight="bold")
    fig.text(0.02, 0.20, "Error\nAccumulation", rotation=90, va="center", fontsize=12, weight="bold")

    # Add colorbar
    cbar = fig.colorbar(im_diff, ax=axes[2, :], orientation="horizontal", pad=0.05, fraction=0.05)
    cbar.set_label("Absolute Difference", fontsize=10)

    plt.tight_layout(rect=[0.03, 0.03, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")

    plt.show()


def compute_dynamics_metrics(
    real_frames: list[np.ndarray],
    imagined_frames: list[np.ndarray],
) -> dict[str, float]:
    """Compute metrics for dynamics model quality."""
    mse_per_step = []

    for _, (real, imagined) in enumerate(zip(real_frames, imagined_frames, strict=False)):
        real_norm = real / 255.0
        mse = np.mean((real_norm - imagined) ** 2)
        mse_per_step.append(mse)

    # Compute divergence rate (how fast error accumulates)
    if len(mse_per_step) > 1:
        divergence_rate = (mse_per_step[-1] - mse_per_step[0]) / len(mse_per_step)
    else:
        divergence_rate = 0.0

    return {
        "initial_mse": float(mse_per_step[0]) if mse_per_step else 0.0,
        "final_mse": float(mse_per_step[-1]) if mse_per_step else 0.0,
        "mean_mse": float(np.mean(mse_per_step)),
        "max_mse": float(np.max(mse_per_step)),
        "divergence_rate": float(divergence_rate),
        "total_cumulative_error": float(np.sum(mse_per_step)),
    }


@click.command()
@click.option(
    "--checkpoint",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to world model checkpoint",
)
@click.option(
    "--horizon",
    "-h",
    type=int,
    default=10,
    help="Number of steps to imagine ahead (horizon length)",
)
@click.option(
    "--level",
    "-l",
    type=str,
    default="1,1",
    help="Level to test on (format: 'W,S')",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to save visualization",
)
@click.option(
    "--device",
    "-d",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="Device to run on",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed",
)
@click.option(
    "--num-trials",
    "-n",
    type=int,
    default=1,
    help="Number of rollouts to average over",
)
def main(
    checkpoint: Path,
    horizon: int,
    level: str,
    output: Path | None,
    device: str,
    seed: int | None,
    num_trials: int,
) -> None:
    """
    Visualize world model dynamics by comparing imagined trajectories
    with real environment rollouts.
    """

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Parse level
    if "," in level:
        world, stage = level.split(",")
        level_tuple = (int(world), int(stage))
    else:
        level_tuple = (1, 1)

    print(f"Loading world model from {checkpoint}...")
    world_model = load_world_model(checkpoint, device)

    print(f"Creating environment for level {level_tuple}...")
    env = create_mario_env(level=level_tuple, render_frames=False)

    # Collect multiple trials and show the last one
    all_metrics = []

    for trial in range(num_trials):
        print(f"\nTrial {trial + 1}/{num_trials}: Collecting {horizon}-step comparison...")
        real_frames, imagined_frames, actions = collect_comparison_rollout(
            env=env,
            world_model=world_model,
            horizon=horizon,
            device=device,
        )

        metrics = compute_dynamics_metrics(real_frames, imagined_frames)
        all_metrics.append(metrics)

        # Visualize last trial
        if trial == num_trials - 1:
            print(f"\nCollected {len(real_frames)} frames")

            print("\nDynamics Model Metrics:")
            print(f"  Initial MSE:       {metrics['initial_mse']:.6f}")
            print(f"  Final MSE:         {metrics['final_mse']:.6f}")
            print(f"  Mean MSE:          {metrics['mean_mse']:.6f}")
            print(f"  Max MSE:           {metrics['max_mse']:.6f}")
            print(f"  Divergence Rate:   {metrics['divergence_rate']:.6f} MSE/step")
            print(f"  Total Error:       {metrics['total_cumulative_error']:.6f}")

            visualize_dynamics_comparison(real_frames, imagined_frames, actions, save_path=output)

    # Print averaged metrics if multiple trials
    if num_trials > 1:
        print(f"\n=== Averaged over {num_trials} trials ===")
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            print(f"  {key:25s}: {np.mean(values):.6f} ± {np.std(values):.6f}")

    env.close()


if __name__ == "__main__":
    main()
