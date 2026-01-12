"""
Visualize world model reconstruction quality by comparing real environment
frames with world model reconstructions during a rollout.

Usage:
    uv run python scripts/visualize_world_model_reconstruction.py --checkpoint path/to/checkpoint.pt --steps 50
"""

import click
import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from pytorch_msssim import ssim

from mario_rl.agent.world_model import MarioWorldModel
from mario_rl.models.dreamer import DreamerModel
from mario_rl.environment.factory import create_mario_env


def load_world_model(checkpoint_path: Path, device: str):
    """Load trained world model from checkpoint.
    
    Returns:
        Either MarioWorldModel or DreamerModel depending on checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Try to detect model type from checkpoint keys
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
        # DreamerModel format (distributed training)
        input_shape = (4, 64, 64)  # (C, H, W)
        action_dim = 12
        latent_dim = 128
        
        world_model = DreamerModel(
            input_shape=input_shape,
            num_actions=action_dim,
            latent_dim=latent_dim,
        ).to(device)
        world_model.load_state_dict(checkpoint["model_state_dict"])
        
    else:
        # Direct state dict - try DreamerModel first
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


def collect_rollout(
    env,
    world_model,
    num_steps: int,
    device: str,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """
    Collect a rollout with real frames, reconstructed frames, and actions.
    
    Returns:
        real_frames: List of real environment frames (H, W)
        recon_frames: List of reconstructed frames from world model
        actions: List of actions taken
    """
    real_frames = []
    recon_frames = []
    actions_taken = []
    
    # Reset environment and get initial state
    state, _ = env.reset()
    
    # Detect model type
    is_dreamer = hasattr(world_model, "actor")  # DreamerModel has actor, MarioWorldModel doesn't
    
    with torch.no_grad():
        for _ in range(num_steps):
            # Store real frame (take last frame from stack for visualization)
            # State is (4, 64, 64) - stack of 4 grayscale frames
            real_frame = state[-1].copy()  # (64, 64)
            real_frames.append(real_frame)
            
            # Convert state to tensor
            state_tensor = torch.from_numpy(state).float().to(device)  # (4, 64, 64)
            state_tensor = state_tensor.unsqueeze(0)  # (1, 4, 64, 64)
            
            if is_dreamer:
                # DreamerModel expects (N, C, H, W) in [0, 255]
                # No need to normalize - model does it internally
                z = world_model.encode(state_tensor, deterministic=True)
                recon = world_model.decoder(z)  # (1, C, H, W) in [0, 1]
                
                # Extract last frame from reconstruction
                recon_frame = recon[0, -1, :, :].cpu().numpy()  # (64, 64) in [0, 1]
            else:
                # MarioWorldModel expects (N, F, H, W, C) in [0, 1]
                state_tensor = state_tensor.unsqueeze(-1) / 255.0  # (1, 4, 64, 64, 1)
                z = world_model.encode(state_tensor, deterministic=True)
                recon = world_model.decode(z)  # (1, 4, 64, 64, 1) in [0, 1]
                recon_frame = recon[0, -1, :, :, 0].cpu().numpy()  # (64, 64) in [0, 1]
            
            recon_frames.append(recon_frame)
            
            # Select action (simple epsilon-greedy with random policy for visualization)
            if np.random.rand() < 0.1:
                action = np.random.randint(0, 12)
            else:
                # Use world model to select action (basic policy)
                # For this visualization, we'll just use random actions
                action = np.random.randint(0, 12)
            
            actions_taken.append(action)
            
            # Step environment
            state, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
    
    return real_frames, recon_frames, actions_taken


def visualize_comparison(
    real_frames: list[np.ndarray],
    recon_frames: list[np.ndarray],
    actions: list[int],
    save_path: Path | None = None,
) -> None:
    """
    Create a grid visualization comparing real vs reconstructed frames.
    
    Args:
        real_frames: List of real frames (H, W) in [0, 255]
        recon_frames: List of reconstructed frames (H, W) in [0, 1]
        actions: List of actions taken
        save_path: Optional path to save the figure
    """
    num_frames = len(real_frames)
    
    # Select evenly spaced frames to display (max 8 columns)
    max_cols = 8
    if num_frames > max_cols:
        indices = np.linspace(0, num_frames - 1, max_cols, dtype=int)
    else:
        indices = range(num_frames)
    
    num_display = len(indices)
    
    # Create figure with 3 rows: real, reconstructed, difference
    fig, axes = plt.subplots(3, num_display, figsize=(2 * num_display, 6))
    
    if num_display == 1:
        axes = axes.reshape(3, 1)
    
    # Action meanings for labels
    action_names = [
        "NOOP", "RIGHT", "R+A", "R+B", "R+A+B", 
        "A", "LEFT", "L+A", "L+B", "L+A+B", "DOWN", "UP"
    ]
    
    for col, idx in enumerate(indices):
        real = real_frames[idx]
        recon = recon_frames[idx]
        action = actions[idx]
        
        # Normalize real frame to [0, 1] for display
        real_norm = real / 255.0
        
        # Compute SSIM for this frame
        real_tensor = torch.from_numpy(real_norm).unsqueeze(0).unsqueeze(0).float()
        recon_tensor = torch.from_numpy(recon).unsqueeze(0).unsqueeze(0).float()
        ssim_val = ssim(real_tensor, recon_tensor, data_range=1.0, size_average=True).item()
        
        # Compute absolute difference
        diff = np.abs(real_norm - recon)
        
        # Display real frame
        axes[0, col].imshow(real_norm, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"Real\nStep {idx}\n{action_names[action]}", fontsize=8)
        axes[0, col].axis("off")
        
        # Display reconstruction
        axes[1, col].imshow(recon, cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title(f"Recon", fontsize=8)
        axes[1, col].axis("off")
        
        # Display difference (use hot colormap for better visibility)
        im = axes[2, col].imshow(diff, cmap="hot", vmin=0, vmax=0.5)
        axes[2, col].set_title(f"Diff\nSSIM: {ssim_val:.4f}", fontsize=8)
        axes[2, col].axis("off")
    
    # Add row labels
    fig.text(0.02, 0.83, "Real", rotation=90, va="center", fontsize=12, weight="bold")
    fig.text(0.02, 0.50, "Reconstruction", rotation=90, va="center", fontsize=12, weight="bold")
    fig.text(0.02, 0.17, "Difference", rotation=90, va="center", fontsize=12, weight="bold")
    
    # Add colorbar for difference map
    cbar = fig.colorbar(im, ax=axes[2, :], orientation="horizontal", pad=0.05, fraction=0.05)
    cbar.set_label("Absolute Difference", fontsize=10)
    
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.97])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def compute_metrics(
    real_frames: list[np.ndarray],
    recon_frames: list[np.ndarray],
) -> dict[str, float]:
    """Compute reconstruction quality metrics using SSIM."""
    ssim_list = []
    
    for real, recon in zip(real_frames, recon_frames):
        real_norm = real / 255.0
        
        # Convert to tensors for SSIM computation
        real_tensor = torch.from_numpy(real_norm).unsqueeze(0).unsqueeze(0).float()
        recon_tensor = torch.from_numpy(recon).unsqueeze(0).unsqueeze(0).float()
        
        # Compute SSIM
        ssim_val = ssim(real_tensor, recon_tensor, data_range=1.0, size_average=True).item()
        ssim_list.append(ssim_val)
    
    return {
        "ssim_mean": float(np.mean(ssim_list)),
        "ssim_std": float(np.std(ssim_list)),
        "ssim_min": float(np.min(ssim_list)),
        "ssim_max": float(np.max(ssim_list)),
    }


@click.command()
@click.option(
    "--checkpoint",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to world model checkpoint (.pt file)",
)
@click.option(
    "--steps",
    "-s",
    type=int,
    default=50,
    help="Number of steps to rollout and visualize",
)
@click.option(
    "--level",
    "-l",
    type=str,
    default="1,1",
    help="Level to test on (format: 'W,S' like '1,1')",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to save visualization (optional)",
)
@click.option(
    "--device",
    "-d",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="Device to run on (cuda/cpu)",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility",
)
def main(
    checkpoint: Path,
    steps: int,
    level: str,
    output: Path | None,
    device: str,
    seed: int | None,
) -> None:
    """Visualize world model reconstruction quality on a real rollout."""
    
    # Set random seed if provided
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
    
    print(f"Collecting {steps}-step rollout...")
    real_frames, recon_frames, actions = collect_rollout(
        env=env,
        world_model=world_model,
        num_steps=steps,
        device=device,
    )
    
    print(f"Collected {len(real_frames)} frames")
    
    # Compute metrics
    metrics = compute_metrics(real_frames, recon_frames)
    print("\nReconstruction Metrics:")
    print(f"  SSIM Mean: {metrics['ssim_mean']:.6f}")
    print(f"  SSIM Std:  {metrics['ssim_std']:.6f}")
    print(f"  SSIM Min:  {metrics['ssim_min']:.6f}")
    print(f"  SSIM Max:  {metrics['ssim_max']:.6f}")
    
    # Create visualization
    print("\nCreating visualization...")
    visualize_comparison(real_frames, recon_frames, actions, save_path=output)
    
    # Clean up
    env.close()


if __name__ == "__main__":
    main()
