#!/usr/bin/env python3
"""Visualize what the DDQN network sees and attends to.

Techniques:
1. Saliency maps - gradient of Q-value w.r.t. input pixels
2. Feature maps - what each conv layer detects
3. Occlusion sensitivity - which regions matter for decisions

Usage:
    uv run python scripts/visualize_network_attention.py
"""

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mario_rl.environment.factory import create_mario_env
from mario_rl.agent.ddqn_net import DoubleDQN


ACTION_NAMES = ["NOOP", "→", "→+A", "→+B", "→+A+B", "A", "←"]


def load_model(checkpoint_path: Path, device: str = "cuda"):
    """Load DDQN model.
    
    Returns:
        (model, input_size, action_history_len)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    
    ln_key = "online.backbone.layer_norm.weight"
    ln_size = state_dict[ln_key].shape[0] if ln_key in state_dict else 1024
    input_size = 64 if ln_size == 1024 else 84
    
    # Detect action_history_len from FC layer input size
    fc_key = "online.backbone.fc.weight"
    action_history_len = 0
    if fc_key in state_dict:
        fc_input = state_dict[fc_key].shape[1]
        flat_size = 1024 if input_size == 64 else 3136
        extra = fc_input - flat_size
        
        # action_history_len=4 with 7 actions -> extra=28
        if extra >= 28:
            action_history_len = 4
        elif extra > 0 and extra % 7 == 0:
            action_history_len = extra // 7
    
    # Detect danger_prediction_bins from danger_head
    danger_key = "online.danger_head.2.weight"
    danger_prediction_bins = 16 if danger_key in state_dict else 0
    
    print(f"  Input size: {input_size}x{input_size}")
    print(f"  Action history len: {action_history_len}")
    print(f"  Danger prediction bins: {danger_prediction_bins}")
    
    model = DoubleDQN(
        input_shape=(4, input_size, input_size), 
        num_actions=7,
        action_history_len=action_history_len,
        danger_prediction_bins=danger_prediction_bins,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, input_size, action_history_len


def compute_saliency(model, state_tensor, action_idx, device, action_hist=None):
    """Compute saliency map: gradient of Q[action] w.r.t. input pixels."""
    state_tensor = state_tensor.clone().requires_grad_(True)
    
    q_values = model.online(state_tensor, action_hist)
    q_action = q_values[0, action_idx]
    
    q_action.backward()
    
    # Saliency = absolute gradient, max over frame stack
    saliency = state_tensor.grad.abs()
    saliency = saliency[0].max(dim=0)[0]  # Max over 4 frames
    
    return saliency.cpu().numpy()


def compute_saliency_all_actions(model, state_tensor, device, action_hist=None):
    """Compute saliency maps for all actions."""
    saliencies = []
    for action_idx in range(7):
        state_tensor = state_tensor.clone().detach().requires_grad_(True)
        q_values = model.online(state_tensor, action_hist)
        q_action = q_values[0, action_idx]
        q_action.backward()
        saliency = state_tensor.grad.abs()[0].max(dim=0)[0]
        saliencies.append(saliency.cpu().numpy())
    return saliencies


def get_conv_features(model, state_tensor):
    """Extract intermediate feature maps from conv layers."""
    features = []
    x = state_tensor
    
    # Access backbone conv layers
    backbone = model.online.backbone
    
    # Conv1
    x = backbone.conv1(x)
    features.append(("conv1", x.detach().cpu().numpy()[0]))
    x = F.relu(x)
    
    # Conv2
    x = backbone.conv2(x)
    features.append(("conv2", x.detach().cpu().numpy()[0]))
    x = F.relu(x)
    
    # Conv3
    x = backbone.conv3(x)
    features.append(("conv3", x.detach().cpu().numpy()[0]))
    
    return features


def occlusion_sensitivity(model, state_tensor, device, patch_size=8, stride=4, action_hist=None):
    """Compute how Q-values change when different regions are occluded."""
    with torch.no_grad():
        if action_hist is not None:
            base_q = model.online(state_tensor, action_hist).cpu().numpy()[0]
        else:
            base_q = model.online(state_tensor).cpu().numpy()[0]
    
    _, _, H, W = state_tensor.shape
    sensitivity = np.zeros((H, W))
    
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            # Create occluded version (set patch to 0)
            occluded = state_tensor.clone()
            occluded[:, :, y:y+patch_size, x:x+patch_size] = 0
            
            with torch.no_grad():
                if action_hist is not None:
                    occluded_q = model.online(occluded, action_hist).cpu().numpy()[0]
                else:
                    occluded_q = model.online(occluded).cpu().numpy()[0]
            
            # How much did Q-values change?
            q_change = np.abs(base_q - occluded_q).max()
            
            # Assign to patch region
            sensitivity[y:y+patch_size, x:x+patch_size] = np.maximum(
                sensitivity[y:y+patch_size, x:x+patch_size],
                q_change
            )
    
    return sensitivity, base_q


def create_attention_visualization(
    frames: np.ndarray,
    saliency: np.ndarray,
    saliencies_all: list,
    conv_features: list,
    occlusion_map: np.ndarray,
    q_values: np.ndarray,
    chosen_action: int,
    x_pos: int,
    output_path: Path,
    danger_pred: np.ndarray = None,
):
    """Create comprehensive visualization of network attention."""
    
    fig = plt.figure(figsize=(20, 18))
    gs = gridspec.GridSpec(5, 6, figure=fig, hspace=0.3, wspace=0.3, height_ratios=[1, 1, 1, 1, 0.8])
    
    fig.suptitle(
        f"Network Attention Analysis | X={x_pos} | "
        f"Chosen: {ACTION_NAMES[chosen_action]} (Q={q_values[chosen_action]:.2f})",
        fontsize=14, fontweight="bold"
    )
    
    # Row 1: Input frames (4 stacked)
    for i in range(4):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(frames[i], cmap="gray")
        ax.set_title(f"Frame t-{3-i}", fontsize=10)
        ax.axis("off")
    
    # Row 1, Col 4-5: Saliency map for chosen action
    ax = fig.add_subplot(gs[0, 4:6])
    im = ax.imshow(saliency, cmap="hot")
    ax.set_title(f"Saliency Map (Chosen: {ACTION_NAMES[chosen_action]})", fontsize=11)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Row 2: Saliency maps for all actions
    for i in range(6):
        ax = fig.add_subplot(gs[1, i])
        if i < 7:
            ax.imshow(saliencies_all[i] if i < len(saliencies_all) else np.zeros_like(saliency), cmap="hot")
            ax.set_title(f"{ACTION_NAMES[i]}\nQ={q_values[i]:.2f}", fontsize=9)
        ax.axis("off")
    
    # Row 3: Conv feature maps (sample channels)
    col = 0
    for name, feat_maps in conv_features:
        n_show = min(2, feat_maps.shape[0])
        for j in range(n_show):
            if col < 6:
                ax = fig.add_subplot(gs[2, col])
                ax.imshow(feat_maps[j], cmap="viridis")
                ax.set_title(f"{name} ch{j}", fontsize=9)
                ax.axis("off")
                col += 1
    
    # Row 4, Col 0-2: Occlusion sensitivity
    ax = fig.add_subplot(gs[3, 0:2])
    ax.imshow(frames[3], cmap="gray", alpha=0.5)
    im = ax.imshow(occlusion_map, cmap="hot", alpha=0.7)
    ax.set_title("Occlusion Sensitivity\n(brighter = more important)", fontsize=11)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Row 4, Col 2-4: Frame + Saliency overlay
    ax = fig.add_subplot(gs[3, 2:4])
    ax.imshow(frames[3], cmap="gray", alpha=0.6)
    ax.imshow(saliency, cmap="hot", alpha=0.5)
    ax.set_title("Saliency Overlay on Frame", fontsize=11)
    ax.axis("off")
    
    # Row 4, Col 4-5: Q-value bar chart
    ax = fig.add_subplot(gs[3, 4:6])
    colors = ["red" if i == chosen_action else "steelblue" for i in range(7)]
    ax.bar(ACTION_NAMES, q_values, color=colors)
    ax.set_ylabel("Q-Value")
    ax.set_title("Q-Values", fontsize=11)
    ax.tick_params(axis="x", rotation=45)
    
    # Row 5: Danger predictions
    if danger_pred is not None and len(danger_pred) > 0:
        ax = fig.add_subplot(gs[4, :])
        bins = len(danger_pred)
        behind_bins = bins // 4
        
        # Color bars based on danger level
        colors = plt.cm.Reds(danger_pred)
        bars = ax.bar(range(bins), danger_pred, color=colors, edgecolor="black", linewidth=0.5)
        
        # Mark the "current position" divider
        ax.axvline(x=behind_bins - 0.5, color="blue", linestyle="--", linewidth=2, alpha=0.7, label="Mario's position")
        
        ax.set_ylim(0, 1.1)
        ax.set_xlim(-0.5, bins - 0.5)
        ax.set_ylabel("Danger Probability", fontsize=10)
        ax.set_xlabel("Distance Bins (← Behind | Ahead →)", fontsize=10)
        ax.set_title(f"Danger Prediction (max={danger_pred.max():.2f})", fontsize=11, fontweight="bold")
        
        # Add bin labels
        bin_labels = [f"-{behind_bins - i}" for i in range(behind_bins)] + [f"+{i}" for i in range(bins - behind_bins)]
        ax.set_xticks(range(bins))
        ax.set_xticklabels(bin_labels, fontsize=8)
        ax.legend(loc="upper right")
    
    # Add Q-range annotation
    q_range = q_values.max() - q_values.min()
    ax.text(0.95, 0.95, f"Q-range: {q_range:.3f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=10, 
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.8))
    
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def run_analysis(
    checkpoint_path: Path,
    output_dir: Path,
    num_frames: int = 20,
    device: str = "cuda",
):
    """Run network attention analysis."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from {checkpoint_path}...")
    model, input_size, action_history_len = load_model(checkpoint_path, device)
    
    print("Creating environment...")
    mario_env = create_mario_env(
        level=(1, 1), 
        render_frames=False, 
        action_history_len=action_history_len,
    )
    env = mario_env.env
    
    state, info = env.reset()
    
    print(f"\nGenerating {num_frames} attention visualizations...")
    
    for step in range(num_frames):
        state_np = np.array(state) if hasattr(state, "__array__") else state
        state_tensor = torch.from_numpy(state_np.copy()).float().unsqueeze(0).to(device) / 255.0
        
        # Get action history if enabled
        action_hist_tensor = None
        if action_history_len > 0 and "action_history" in info:
            action_hist_tensor = torch.from_numpy(info["action_history"]).float().unsqueeze(0).to(device)
        
        # Get Q-values and danger predictions
        with torch.no_grad():
            result = model.online(state_tensor, action_hist_tensor, return_danger=True)
            if isinstance(result, tuple):
                q_values, danger_pred = result
                q_values = q_values.cpu().numpy()[0]
                danger_pred = danger_pred.cpu().numpy()[0]
            else:
                q_values = result.cpu().numpy()[0]
                danger_pred = np.zeros(16)
        
        action = q_values.argmax()
        x_pos = info.get("x_pos", 0)
        
        # Compute attention maps
        print(f"  Step {step}: X={x_pos}, computing attention maps...")
        
        # 1. Saliency for chosen action
        saliency = compute_saliency(model, state_tensor.clone(), action, device, action_hist_tensor)
        
        # 2. Saliency for all actions
        saliencies_all = compute_saliency_all_actions(model, state_tensor.clone(), device, action_hist_tensor)
        
        # 3. Conv feature maps
        conv_features = get_conv_features(model, state_tensor)
        
        # 4. Occlusion sensitivity
        occlusion_map, _ = occlusion_sensitivity(model, state_tensor, device, action_hist=action_hist_tensor)
        
        # Create visualization
        output_path = output_dir / f"attention_step{step:03d}_x{x_pos}.png"
        create_attention_visualization(
            frames=state_np,
            saliency=saliency,
            saliencies_all=saliencies_all,
            conv_features=conv_features,
            occlusion_map=occlusion_map,
            q_values=q_values,
            chosen_action=action,
            x_pos=x_pos,
            output_path=output_path,
            danger_pred=danger_pred,
        )
        
        # Step environment
        state, reward, done, truncated, info = env.step(action)
        if done:
            print(f"  Episode ended at step {step}")
            state, info = env.reset()
    
    env.close()
    print(f"\nDone! Output saved to {output_dir}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, 
                        default="checkpoints/ddqn_dist_2026-01-15T21-58-18/checkpoint_310000.pt")
    parser.add_argument("--output", type=str, default="attention_analysis")
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    run_analysis(
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.output),
        num_frames=args.frames,
        device=args.device,
    )


if __name__ == "__main__":
    main()
