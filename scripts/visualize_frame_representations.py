"""
Visualize different frame representation techniques for Mario.

Shows how various preprocessing methods capture (or fail to capture)
the important information in Mario frames.

Usage:
    uv run python scripts/visualize_frame_representations.py \
        --checkpoint checkpoints/ddqn_dist_*/weights.pt \
        --frames 50 \
        --output /tmp/frame_representations.png
"""

import click
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from mario_rl.models.ddqn import DoubleDQN
from mario_rl.environment.factory import create_mario_env


def load_ddqn_model(checkpoint_path: Path, device: str) -> DoubleDQN:
    """Load trained DDQN model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Determine if it's a full checkpoint or just weights
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    model = DoubleDQN(
        input_shape=(4, 64, 64),
        num_actions=7,  # SIMPLE_MOVEMENT
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def collect_frames_with_model(
    model: DoubleDQN,
    device: str,
    num_frames: int = 50,
    skip_frames: int = 4,
    epsilon: float = 0.05,
) -> tuple[list[np.ndarray], list[dict]]:
    """Collect frames by playing with trained model."""
    env = create_mario_env(level=(1, 1))
    
    frames = []
    infos = []
    
    state, info = env.reset()
    frame_count = 0
    step_count = 0
    
    while frame_count < num_frames:
        # Select action using trained model
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = model(state_tensor)
                action = q_values.argmax(dim=-1).item()
        
        state, reward, done, truncated, info = env.step(action)
        step_count += 1
        
        # Collect every skip_frames
        if step_count % skip_frames == 0:
            frames.append(state[-1].copy())  # Last frame of stack
            infos.append(info.copy())
            frame_count += 1
        
        if done or truncated:
            state, info = env.reset()
    
    env.close()
    return frames, infos


def compute_frame_difference(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Compute absolute difference between consecutive frames."""
    diffs = []
    for i in range(1, len(frames)):
        diff = np.abs(frames[i].astype(float) - frames[i-1].astype(float))
        diffs.append(diff.astype(np.uint8))
    return diffs


def compute_optical_flow(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Compute dense optical flow between consecutive frames."""
    flows = []
    for i in range(1, len(frames)):
        prev = frames[i-1]
        curr = frames[i]
        
        # Farneback optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev, curr, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Convert flow to magnitude for visualization
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flows.append(mag)
    
    return flows


def compute_edges(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Compute Canny edge detection."""
    edges = []
    for frame in frames:
        edge = cv2.Canny(frame, 30, 100)
        edges.append(edge)
    return edges


def compute_sobel(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Compute Sobel gradient magnitude."""
    sobels = []
    for frame in frames:
        sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobelx**2 + sobely**2)
        sobel = (sobel / sobel.max() * 255).astype(np.uint8)
        sobels.append(sobel)
    return sobels


def compute_variance_map(frames: list[np.ndarray]) -> np.ndarray:
    """Compute per-pixel variance across all frames."""
    stack = np.stack(frames).astype(float)
    variance = stack.var(axis=0)
    return variance


def compute_motion_history(frames: list[np.ndarray], decay: float = 0.9) -> list[np.ndarray]:
    """Compute motion history image (accumulated motion over time)."""
    history = np.zeros_like(frames[0], dtype=float)
    histories = []
    
    for i in range(1, len(frames)):
        diff = np.abs(frames[i].astype(float) - frames[i-1].astype(float))
        history = history * decay + diff
        histories.append((history / history.max() * 255).astype(np.uint8) if history.max() > 0 else history.astype(np.uint8))
    
    return histories


def compute_local_variance(frames: list[np.ndarray], window: int = 5) -> list[np.ndarray]:
    """Compute local variance within a sliding window (highlights texture/detail)."""
    local_vars = []
    for frame in frames:
        # Use box filter for local mean
        mean = cv2.blur(frame.astype(float), (window, window))
        sqr_mean = cv2.blur((frame.astype(float))**2, (window, window))
        variance = sqr_mean - mean**2
        variance = np.clip(variance, 0, None)
        local_vars.append((variance / (variance.max() + 1e-8) * 255).astype(np.uint8))
    return local_vars


def create_visualization(
    frames: list[np.ndarray],
    infos: list[dict],
    output_path: Path,
) -> None:
    """Create comprehensive visualization of different representations."""
    
    print("Computing representations...")
    
    # Compute all representations
    diffs = compute_frame_difference(frames)
    flows = compute_optical_flow(frames)
    edges = compute_edges(frames)
    sobels = compute_sobel(frames)
    variance_map = compute_variance_map(frames)
    motion_history = compute_motion_history(frames)
    local_vars = compute_local_variance(frames)
    
    # Select sample frames to display (every 10th)
    sample_indices = list(range(0, len(frames)-1, 10))[:5]
    
    # Create figure
    fig = plt.figure(figsize=(20, 24))
    
    # Define grid
    n_cols = len(sample_indices)
    n_rows = 9  # Different representations
    
    row_labels = [
        'Raw Frame',
        'Frame Diff (t vs t-1)',
        'Optical Flow Magnitude',
        'Canny Edges',
        'Sobel Gradient',
        'Motion History',
        'Local Variance',
        'Important Pixels\n(variance > thresh)',
        'Masked Frame\n(only important)'
    ]
    
    for row_idx, label in enumerate(row_labels):
        for col_idx, frame_idx in enumerate(sample_indices):
            ax = fig.add_subplot(n_rows, n_cols, row_idx * n_cols + col_idx + 1)
            
            if row_idx == 0:  # Raw frame
                ax.imshow(frames[frame_idx], cmap='gray')
                x_pos = infos[frame_idx].get('x_pos', 0)
                ax.set_title(f'Frame {frame_idx}\nx={x_pos}', fontsize=9)
            
            elif row_idx == 1:  # Frame diff
                ax.imshow(diffs[frame_idx], cmap='hot')
                diff_sum = diffs[frame_idx].sum()
                ax.set_title(f'Diff sum: {diff_sum:.0f}', fontsize=8)
            
            elif row_idx == 2:  # Optical flow
                ax.imshow(flows[frame_idx], cmap='hot')
                flow_mean = flows[frame_idx].mean()
                ax.set_title(f'Flow mean: {flow_mean:.2f}', fontsize=8)
            
            elif row_idx == 3:  # Edges
                ax.imshow(edges[frame_idx], cmap='gray')
                edge_count = (edges[frame_idx] > 0).sum()
                ax.set_title(f'Edge px: {edge_count}', fontsize=8)
            
            elif row_idx == 4:  # Sobel
                ax.imshow(sobels[frame_idx], cmap='gray')
            
            elif row_idx == 5:  # Motion history
                ax.imshow(motion_history[frame_idx], cmap='hot')
            
            elif row_idx == 6:  # Local variance
                ax.imshow(local_vars[frame_idx], cmap='hot')
            
            elif row_idx == 7:  # Important pixels mask
                important = (variance_map > np.percentile(variance_map, 70)).astype(np.uint8) * 255
                ax.imshow(important, cmap='gray')
                pct = 100 * (important > 0).sum() / important.size
                ax.set_title(f'{pct:.1f}% important', fontsize=8)
            
            elif row_idx == 8:  # Masked frame
                important_mask = variance_map > np.percentile(variance_map, 70)
                masked = frames[frame_idx].copy()
                masked[~important_mask] = 0
                ax.imshow(masked, cmap='gray')
            
            ax.axis('off')
            
            # Add row label on first column
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=10, rotation=0, ha='right', va='center')
    
    plt.suptitle('Mario Frame Representation Analysis\n(Using trained DDQN to play)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {output_path}")
    
    # Also create summary statistics plot
    create_summary_plot(frames, diffs, flows, variance_map, infos, output_path.parent / 'frame_stats.png')


def create_summary_plot(
    frames: list[np.ndarray],
    diffs: list[np.ndarray],
    flows: list[np.ndarray],
    variance_map: np.ndarray,
    infos: list[dict],
    output_path: Path,
) -> None:
    """Create summary statistics plot."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. X position over time
    x_positions = [info.get('x_pos', 0) for info in infos]
    axes[0, 0].plot(x_positions, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('X Position')
    axes[0, 0].set_title('Mario X Position Over Time')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Frame difference magnitude over time
    diff_magnitudes = [d.mean() for d in diffs]
    axes[0, 1].plot(diff_magnitudes, 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('Mean Pixel Difference')
    axes[0, 1].set_title('Frame-to-Frame Difference')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Optical flow magnitude over time
    flow_magnitudes = [f.mean() for f in flows]
    axes[0, 2].plot(flow_magnitudes, 'g-', linewidth=2)
    axes[0, 2].set_xlabel('Frame')
    axes[0, 2].set_ylabel('Mean Flow Magnitude')
    axes[0, 2].set_title('Optical Flow Over Time')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Variance map
    im = axes[1, 0].imshow(variance_map, cmap='hot')
    axes[1, 0].set_title(f'Pixel Variance Map\n(brighter = more change)')
    plt.colorbar(im, ax=axes[1, 0])
    
    # 5. Histogram of variance
    axes[1, 1].hist(variance_map.flatten(), bins=50, color='purple', alpha=0.7)
    axes[1, 1].axvline(np.percentile(variance_map, 70), color='r', linestyle='--', label='70th percentile')
    axes[1, 1].set_xlabel('Variance')
    axes[1, 1].set_ylabel('Pixel Count')
    axes[1, 1].set_title('Distribution of Pixel Variance')
    axes[1, 1].legend()
    axes[1, 1].set_yscale('log')
    
    # 6. Correlation between x_pos change and frame diff
    x_changes = np.abs(np.diff(x_positions))
    axes[1, 2].scatter(x_changes, diff_magnitudes[:len(x_changes)], alpha=0.5, s=20)
    axes[1, 2].set_xlabel('|Δx_pos|')
    axes[1, 2].set_ylabel('Frame Difference')
    axes[1, 2].set_title('Position Change vs Frame Difference')
    
    # Calculate correlation
    corr = np.corrcoef(x_changes, diff_magnitudes[:len(x_changes)])[0, 1]
    axes[1, 2].text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=axes[1, 2].transAxes, 
                    fontsize=10, verticalalignment='top')
    
    plt.suptitle('Frame Representation Statistics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved stats to {output_path}")


@click.command()
@click.option('-c', '--checkpoint', type=click.Path(exists=True), required=True,
              help='Path to DDQN checkpoint')
@click.option('-f', '--frames', default=50, help='Number of frames to collect')
@click.option('-s', '--skip', default=4, help='Frame skip interval')
@click.option('-o', '--output', type=click.Path(), default='/tmp/frame_representations.png',
              help='Output path for visualization')
@click.option('-d', '--device', default='cuda', help='Device (cuda/cpu)')
@click.option('-e', '--epsilon', default=0.05, help='Exploration epsilon')
def main(checkpoint: str, frames: int, skip: int, output: str, device: str, epsilon: float):
    """Visualize different frame representation techniques for Mario."""
    
    print(f"Loading DDQN model from {checkpoint}...")
    model = load_ddqn_model(Path(checkpoint), device)
    
    print(f"Collecting {frames} frames (skip={skip}) using trained model...")
    collected_frames, infos = collect_frames_with_model(
        model, device, num_frames=frames, skip_frames=skip, epsilon=epsilon
    )
    
    print(f"Collected {len(collected_frames)} frames")
    print(f"X position range: {min(i.get('x_pos', 0) for i in infos)} - {max(i.get('x_pos', 0) for i in infos)}")
    
    create_visualization(collected_frames, infos, Path(output))


if __name__ == '__main__':
    main()
