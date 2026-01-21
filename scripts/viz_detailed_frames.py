#!/usr/bin/env python
"""Visualize frames with attention, Q-values, and danger predictions.

Creates a 4-column visualization for each frame:
1. Raw grayscale frame
2. Frame with attention overlay (jet colormap)
3. Q-values bar chart per action
4. Danger prediction distribution (16 bins)
"""

import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from mario_rl.models.ddqn import symexp
from mario_rl.agent.ddqn_net import DoubleDQN
from mario_rl.environment.factory import create_mario_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize frames with attention and predictions")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to weights.pt file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="detailed_frame_analysis.png",
        help="Output image path",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=12,
        help="Number of frames to collect",
    )
    parser.add_argument(
        "--spacing",
        type=int,
        default=200,
        help="X-position spacing between frames",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=200,
        help="Maximum episodes to run for collection",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.15,
        help="Exploration epsilon during collection",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--action-history-len",
        type=int,
        default=None,
        help="Action history length (auto-detected if not specified)",
    )
    parser.add_argument(
        "--danger-bins",
        type=int,
        default=16,
        help="Danger prediction bins",
    )
    return parser.parse_args()


def load_model(
    weights_path: str,
    device: str,
    action_history_len: int | None = None,
    danger_bins: int = 16,
) -> tuple[DoubleDQN, int]:
    """Load the DoubleDQN model from weights.

    Auto-detects action_history_len from checkpoint fc2 weight shape if not specified.

    Returns:
        Tuple of (model, action_history_len)
    """
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Auto-detect action_history_len from fc2 weight shape if not specified
    # fc2 input = cnn_out + action_history_len * num_actions
    # For 64x64 input: cnn_out = 512 (after attention pooling)
    # fc2.weight shape is [512, cnn_out + action_history_len * 7]
    if action_history_len is None:
        fc2_key = "online.backbone.fc2.weight"
        if fc2_key in state_dict:
            fc2_in = state_dict[fc2_key].shape[1]
            # cnn_out = 512, num_actions = 7
            # fc2_in = 512 + action_history_len * 7
            inferred = (fc2_in - 512) // 7
            action_history_len = inferred
            print(f"Auto-detected action_history_len={action_history_len} from checkpoint")
        else:
            action_history_len = 4  # fallback default

    model = DoubleDQN(
        input_shape=(4, 64, 64),
        num_actions=7,
        action_history_len=action_history_len,
        danger_prediction_bins=danger_bins,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, action_history_len


def collect_frames(
    model: DoubleDQN,
    device: str,
    num_frames: int,
    spacing: int,
    max_episodes: int,
    epsilon: float,
    action_history_len: int,
) -> list[dict]:
    """Collect frames at regular X-position intervals."""
    target_positions = list(range(100, 3200, spacing))
    collected = {}
    max_x = 0

    env = create_mario_env(level=(1, 1), action_history_len=action_history_len)

    for ep in range(max_episodes):
        obs, _ = env.reset()
        action_history = torch.zeros(1, action_history_len, 7, device=device)
        done = False
        step = 0

        while not done and step < 8000:
            # Network expects uint8 and normalizes internally
            state_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)

            with torch.no_grad():
                features = model.online.backbone(state_tensor, action_history)
                q_symlog = model.online(state_tensor, action_history)
                q_real = symexp(q_symlog)
                danger_logits = model.online.danger_head(features)
                danger_probs = torch.softmax(danger_logits, dim=1)

            if np.random.random() < epsilon:
                action = np.random.randint(7)
            else:
                action = q_symlog.argmax(dim=1).item()

            obs_next, _, terminated, truncated, info = env.step(action)
            x_pos = info.get("x_pos", 0)

            for target_x in target_positions:
                if abs(x_pos - target_x) < 25 and target_x not in collected:
                    # Run forward pass again to get attention
                    _ = model.online(state_tensor, action_history)
                    attn = model.online.backbone.get_attention_map()
                    collected[target_x] = {
                        "x": x_pos,
                        "frame": obs[-1],
                        "attn": attn[0, 0].cpu().numpy() if attn is not None else None,
                        "q_values": q_real[0].cpu().numpy(),
                        "danger_probs": danger_probs[0].cpu().numpy(),
                    }

            max_x = max(max_x, x_pos)

            # Update action history
            ah = torch.zeros(1, 1, 7, device=device)
            ah[0, 0, action] = 1.0
            action_history = torch.cat([action_history[:, 1:, :], ah], dim=1)

            obs = obs_next
            done = terminated or truncated
            step += 1

        if ep % 50 == 0:
            print(f"  Ep {ep}: max_x={max_x}, collected {len(collected)}")

        if len(collected) >= num_frames:
            break

    env.close()

    sorted_data = sorted(collected.items())[:num_frames]
    print(f"Max X: {max_x}, collected {len(sorted_data)} frames")

    return sorted_data, max_x


def create_visualization(
    sorted_data: list[tuple[int, dict]],
    max_x: int,
    output_path: str,
) -> None:
    """Create the 4-column visualization."""
    actions = ["NOOP", "Right", "R+Jump", "Right+B", "R+J+B", "Jump", "Left"]
    n = len(sorted_data)

    fig, axes = plt.subplots(n, 4, figsize=(14, n * 2.5))

    for i, (_, data) in enumerate(sorted_data):
        # Col 1: Raw frame
        axes[i, 0].imshow(data["frame"], cmap="gray")
        axes[i, 0].set_title(f"X={data['x']}", fontsize=9)
        axes[i, 0].axis("off")

        # Col 2: Frame + attention overlay
        axes[i, 1].imshow(data["frame"], cmap="gray")
        if data["attn"] is not None:
            attn_big = zoom(data["attn"], 8, order=1)
            axes[i, 1].imshow(attn_big, cmap="jet", alpha=0.5)
        axes[i, 1].set_title("Attention", fontsize=9)
        axes[i, 1].axis("off")

        # Col 3: Q-values bar chart
        q_vals = data["q_values"]
        colors = ["#d62728" if q < 0 else "#2ca02c" for q in q_vals]
        axes[i, 2].barh(range(7), q_vals, color=colors)
        axes[i, 2].set_yticks(range(7))
        axes[i, 2].set_yticklabels(actions, fontsize=7)
        axes[i, 2].axvline(0, color="black", linewidth=0.5)
        axes[i, 2].set_title(f"Q-values (max={q_vals.max():.1f})", fontsize=9)
        axes[i, 2].set_xlim(min(q_vals.min() * 1.1, -10), max(q_vals.max() * 1.1, 10))

        # Col 4: Danger prediction bar chart
        # Bins 0-3: Behind (-64px to 0), Bins 4-15: Ahead (0 to +256px)
        danger = data["danger_probs"]
        num_bins = len(danger)
        behind_bins = num_bins // 4  # 4 bins behind
        ahead_bins = num_bins - behind_bins  # 12 bins ahead

        # Color behind bins differently from ahead bins
        colors = ["#9467bd"] * behind_bins + ["#ff7f0e"] * ahead_bins
        x_bins = np.arange(num_bins)
        axes[i, 3].bar(x_bins, danger, color=colors, alpha=0.8)

        # Add vertical line separating behind/ahead
        axes[i, 3].axvline(behind_bins - 0.5, color="white", linewidth=1.5, linestyle="--")

        # Custom x-ticks showing distance in pixels
        # Behind: 4 bins cover -64 to 0 (16px each), Ahead: 12 bins cover 0 to 256 (~21px each)
        tick_positions = [0, behind_bins - 1, behind_bins, num_bins - 1]
        tick_labels = ["-64px", "0", "0", "+256px"]
        axes[i, 3].set_xticks(tick_positions)
        axes[i, 3].set_xticklabels(tick_labels, fontsize=6)

        # Annotate regions
        axes[i, 3].text(behind_bins / 2 - 0.5, 0.95, "Behind", ha="center", fontsize=6, color="#9467bd")
        axes[i, 3].text(behind_bins + ahead_bins / 2 - 0.5, 0.95, "Ahead", ha="center", fontsize=6, color="#ff7f0e")

        axes[i, 3].set_title(f"Danger (peak bin {danger.argmax()})", fontsize=9)
        axes[i, 3].set_xlim(-0.5, num_bins - 0.5)
        axes[i, 3].set_ylim(0, 1.05)

    plt.suptitle(
        f"Detailed Frame Analysis (Max X={max_x}): Frame | Attention | Q-values | Danger",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"Saved to {output_path}")


def main() -> None:
    args = parse_args()

    print(f"Loading model from {args.weights}...")
    model, action_history_len = load_model(
        args.weights,
        args.device,
        action_history_len=args.action_history_len,
        danger_bins=args.danger_bins,
    )

    print(f"Collecting {args.num_frames} frames (spacing={args.spacing})...")
    sorted_data, max_x = collect_frames(
        model=model,
        device=args.device,
        num_frames=args.num_frames,
        spacing=args.spacing,
        max_episodes=args.max_episodes,
        epsilon=args.epsilon,
        action_history_len=action_history_len,
    )

    print("Creating visualization...")
    create_visualization(sorted_data, max_x, args.output)


if __name__ == "__main__":
    main()
