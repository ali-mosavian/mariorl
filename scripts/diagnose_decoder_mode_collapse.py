"""
Diagnose decoder mode collapse - when decoder outputs similar frames regardless of input.

This checks if the decoder is actually using the latent information or just
outputting a static "safe" image that minimizes average MSE.
"""

import click
import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from mario_rl.models.dreamer import DreamerModel
from mario_rl.environment.factory import create_mario_env


def load_world_model(checkpoint_path: Path, device: str):
    """Load trained world model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    input_shape = (4, 64, 64)
    action_dim = 12
    latent_dim = 128
    
    world_model = DreamerModel(
        input_shape=input_shape,
        num_actions=action_dim,
        latent_dim=latent_dim,
    ).to(device)
    
    if "model_state_dict" in checkpoint:
        world_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        world_model.load_state_dict(checkpoint)
    
    world_model.eval()
    return world_model


def collect_diverse_frames(env, num_frames: int = 100) -> list[np.ndarray]:
    """Collect diverse frames from random gameplay."""
    frames = []
    state, _ = env.reset()
    
    for _ in range(num_frames):
        frames.append(state.copy())
        action = np.random.randint(0, 12)
        state, _, terminated, truncated, _ = env.step(action)
        
        if terminated or truncated:
            state, _ = env.reset()
    
    return frames


def compute_reconstruction_diversity(
    world_model,
    frames: list[np.ndarray],
    device: str,
) -> dict:
    """
    Measure if reconstructions are diverse or all similar.
    
    Good decoder: High input variance → High reconstruction variance
    Collapsed decoder: High input variance → Low reconstruction variance
    """
    reconstructions = []
    
    with torch.no_grad():
        for frame in frames:
            state_tensor = torch.from_numpy(frame).float().to(device)
            state_tensor = state_tensor.unsqueeze(0)  # (1, 4, 64, 64)
            
            z = world_model.encode(state_tensor, deterministic=True)
            recon = world_model.decoder(z)  # (1, 4, 64, 64)
            
            # Take last frame
            recon_frame = recon[0, -1, :, :].cpu().numpy()
            reconstructions.append(recon_frame)
    
    # Convert to arrays
    real_frames = np.array([f[-1] for f in frames])  # (N, 64, 64)
    recon_frames = np.array(reconstructions)  # (N, 64, 64)
    
    # Compute statistics
    # 1. Variance across frames (measure diversity)
    real_var_per_pixel = np.var(real_frames, axis=0)  # (64, 64) - variance at each pixel
    recon_var_per_pixel = np.var(recon_frames, axis=0)
    
    real_total_var = np.mean(real_var_per_pixel)
    recon_total_var = np.mean(recon_var_per_pixel)
    
    # 2. Pairwise differences between frames
    real_diffs = []
    recon_diffs = []
    for i in range(min(50, len(frames))):
        for j in range(i + 1, min(i + 10, len(frames))):
            real_diff = np.mean(np.abs(real_frames[i] - real_frames[j]))
            recon_diff = np.mean(np.abs(recon_frames[i] - recon_frames[j]))
            real_diffs.append(real_diff)
            recon_diffs.append(recon_diff)
    
    mean_real_diff = np.mean(real_diffs)
    mean_recon_diff = np.mean(recon_diffs)
    
    # 3. Standard deviation of pixel values across frames
    real_std = np.std(real_frames)
    recon_std = np.std(recon_frames)
    
    # 4. Mean reconstruction
    mean_recon = np.mean(recon_frames, axis=0)
    
    # 5. Check if reconstructions are too similar to mean
    similarity_to_mean = []
    for recon in recon_frames:
        sim = np.mean(np.abs(recon - mean_recon))
        similarity_to_mean.append(sim)
    
    mean_similarity = np.mean(similarity_to_mean)
    
    return {
        "real_variance": float(real_total_var),
        "recon_variance": float(recon_total_var),
        "variance_ratio": float(recon_total_var / real_total_var) if real_total_var > 0 else 0,
        "real_pairwise_diff": float(mean_real_diff),
        "recon_pairwise_diff": float(mean_recon_diff),
        "diff_ratio": float(mean_recon_diff / mean_real_diff) if mean_real_diff > 0 else 0,
        "real_std": float(real_std),
        "recon_std": float(recon_std),
        "std_ratio": float(recon_std / real_std) if real_std > 0 else 0,
        "mean_similarity_to_average": float(mean_similarity),
        "mean_recon": mean_recon,
        "variance_map_real": real_var_per_pixel,
        "variance_map_recon": recon_var_per_pixel,
    }


def visualize_diversity_analysis(
    real_frames: list[np.ndarray],
    reconstructions: list[np.ndarray],
    stats: dict,
    save_path: Path | None = None,
) -> None:
    """Visualize whether decoder is collapsed."""
    
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    
    # Row 1: Sample real frames
    for i in range(5):
        idx = i * len(real_frames) // 5
        axes[0, i].imshow(real_frames[idx][-1], cmap="gray", vmin=0, vmax=255)
        axes[0, i].set_title(f"Real Frame {idx}", fontsize=9)
        axes[0, i].axis("off")
    
    # Row 2: Corresponding reconstructions
    for i in range(5):
        idx = i * len(reconstructions) // 5
        axes[1, i].imshow(reconstructions[idx], cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title(f"Recon {idx}", fontsize=9)
        axes[1, i].axis("off")
    
    # Row 3: Analysis
    # Mean reconstruction
    axes[2, 0].imshow(stats["mean_recon"], cmap="gray", vmin=0, vmax=1)
    axes[2, 0].set_title("Mean Recon\n(All Similar?)", fontsize=9)
    axes[2, 0].axis("off")
    
    # Variance map - Real
    axes[2, 1].imshow(stats["variance_map_real"], cmap="hot", vmin=0, vmax=0.1)
    axes[2, 1].set_title(f"Real Variance\nμ={stats['real_variance']:.6f}", fontsize=9)
    axes[2, 1].axis("off")
    
    # Variance map - Recon
    axes[2, 2].imshow(stats["variance_map_recon"], cmap="hot", vmin=0, vmax=0.1)
    axes[2, 2].set_title(f"Recon Variance\nμ={stats['recon_variance']:.6f}", fontsize=9)
    axes[2, 2].axis("off")
    
    # Diversity ratio
    axes[2, 3].text(0.5, 0.5, 
        f"Diversity Ratios:\n\n"
        f"Variance: {stats['variance_ratio']:.3f}\n"
        f"Pairwise Diff: {stats['diff_ratio']:.3f}\n"
        f"Std Dev: {stats['std_ratio']:.3f}\n\n"
        f"{'⚠️ COLLAPSED!' if stats['variance_ratio'] < 0.3 else '✅ OK'}",
        ha="center", va="center", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat" if stats['variance_ratio'] < 0.3 else "lightgreen"))
    axes[2, 3].set_xlim(0, 1)
    axes[2, 3].set_ylim(0, 1)
    axes[2, 3].axis("off")
    
    # Stats text
    axes[2, 4].text(0.1, 0.9, "Detailed Stats:", fontsize=10, weight="bold", va="top")
    axes[2, 4].text(0.1, 0.7,
        f"Real:\n"
        f"  Var: {stats['real_variance']:.6f}\n"
        f"  Diff: {stats['real_pairwise_diff']:.4f}\n"
        f"  Std: {stats['real_std']:.4f}\n\n"
        f"Recon:\n"
        f"  Var: {stats['recon_variance']:.6f}\n"
        f"  Diff: {stats['recon_pairwise_diff']:.4f}\n"
        f"  Std: {stats['recon_std']:.4f}\n\n"
        f"Similarity to Mean:\n"
        f"  {stats['mean_similarity_to_average']:.4f}",
        fontsize=8, va="top", family="monospace")
    axes[2, 4].set_xlim(0, 1)
    axes[2, 4].set_ylim(0, 1)
    axes[2, 4].axis("off")
    
    # Add row labels
    fig.text(0.02, 0.83, "Real\nFrames", rotation=0, va="center", fontsize=12, weight="bold")
    fig.text(0.02, 0.50, "Recon\nFrames", rotation=0, va="center", fontsize=12, weight="bold")
    fig.text(0.02, 0.17, "Analysis", rotation=0, va="center", fontsize=12, weight="bold")
    
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.97])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    
    plt.show()


@click.command()
@click.option(
    "--checkpoint",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to checkpoint",
)
@click.option(
    "--num-frames",
    "-n",
    type=int,
    default=100,
    help="Number of frames to analyze",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output path",
)
@click.option(
    "--device",
    "-d",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",
)
def main(
    checkpoint: Path,
    num_frames: int,
    output: Path | None,
    device: str,
) -> None:
    """Diagnose if decoder has collapsed to outputting static images."""
    
    print(f"Loading world model from {checkpoint}...")
    world_model = load_world_model(checkpoint, device)
    
    print("Creating environment...")
    env = create_mario_env(level=(1, 1), render_frames=False)
    
    print(f"Collecting {num_frames} diverse frames...")
    frames = collect_diverse_frames(env, num_frames)
    
    print("Computing reconstruction diversity...")
    stats = compute_reconstruction_diversity(world_model, frames, device)
    
    print("\n" + "="*60)
    print("DECODER DIVERSITY ANALYSIS")
    print("="*60)
    print(f"\nVariance Ratio (recon/real): {stats['variance_ratio']:.4f}")
    print(f"Pairwise Diff Ratio:         {stats['diff_ratio']:.4f}")
    print(f"Std Dev Ratio:               {stats['std_ratio']:.4f}")
    print(f"\nMean Similarity to Average:  {stats['mean_similarity_to_average']:.4f}")
    
    print("\n" + "-"*60)
    if stats['variance_ratio'] < 0.3:
        print("⚠️  DECODER COLLAPSED!")
        print("    Reconstructions are too similar - decoder ignoring latent info")
        print("\nRecommendations:")
        print("  1. Reduce KL weight (allow more info through bottleneck)")
        print("  2. Add perceptual loss (LPIPS or VGG features)")
        print("  3. Increase decoder capacity")
        print("  4. Add adversarial loss (discriminator)")
        print("  5. Weight loss toward dynamic regions")
    elif stats['variance_ratio'] < 0.6:
        print("⚠️  DECODER PARTIALLY COLLAPSED")
        print("    Some diversity loss - consider improvements")
    else:
        print("✅ DECODER OK")
        print("    Reconstructions show good diversity")
    print("="*60 + "\n")
    
    # Collect reconstructions for visualization
    reconstructions = []
    with torch.no_grad():
        for frame in frames:
            state_tensor = torch.from_numpy(frame).float().to(device).unsqueeze(0)
            z = world_model.encode(state_tensor, deterministic=True)
            recon = world_model.decoder(z)[0, -1, :, :].cpu().numpy()
            reconstructions.append(recon)
    
    visualize_diversity_analysis(frames, reconstructions, stats, save_path=output)
    
    env.close()


if __name__ == "__main__":
    main()
