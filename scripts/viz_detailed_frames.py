#!/usr/bin/env python
"""Visualize frames with attention, Q-values, and danger predictions.

Creates a 4-column visualization for each frame:
1. Raw grayscale frame
2. Frame with attention overlay (jet colormap)
3. Q-values bar chart per action (or policy logits for PPO)
4. Danger prediction distribution (or value estimate for PPO)

Supports both DDQN and PPO checkpoints (auto-detected).
"""

import argparse
from typing import Protocol

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from mario_rl.models.ddqn import symexp
from mario_rl.agent.ddqn_net import DoubleDQN
from mario_rl.environment.factory import create_mario_env


class VisualizableModel(Protocol):
    """Protocol for models that can be visualized."""

    def get_attention_map(self) -> torch.Tensor | None: ...


# Import PPONetwork if available
try:
    from scripts.ppo_vec import PPONetwork

    PPO_AVAILABLE = True
except ImportError:
    PPO_AVAILABLE = False
    PPONetwork = None  # type: ignore


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
        help="Exploration epsilon during collection (DDQN only; PPO uses policy sampling)",
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


def detect_model_type(state_dict: dict) -> str:
    """Detect model type from state_dict keys.

    Returns:
        'ppo' or 'ddqn'
    """
    keys = list(state_dict.keys())
    # PPO has backbone.*, policy.*, value.* (no online/target prefix)
    if any(k.startswith("backbone.") for k in keys) and any(k.startswith("policy.") for k in keys):
        return "ppo"
    # DDQN has online.* and target.*
    if any(k.startswith("online.") for k in keys):
        return "ddqn"
    # Fallback
    return "ddqn"


def load_model(
    weights_path: str,
    device: str,
    action_history_len: int | None = None,
    danger_bins: int = 16,
) -> tuple[nn.Module, int, str]:
    """Load model from weights (auto-detects DDQN or PPO).

    Auto-detects action_history_len from checkpoint fc2 weight shape if not specified.

    Returns:
        Tuple of (model, action_history_len, model_type)
    """
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    model_type = detect_model_type(state_dict)
    print(f"Detected model type: {model_type.upper()}")

    if model_type == "ppo":
        if not PPO_AVAILABLE:
            raise ImportError("PPONetwork not available. Run from project root.")

        # Auto-detect action_history_len from PPO backbone fc2 weight
        if action_history_len is None:
            fc2_key = "backbone.fc2.weight"
            if fc2_key in state_dict:
                fc2_in = state_dict[fc2_key].shape[1]
                inferred = (fc2_in - 512) // 7
                action_history_len = inferred
                print(f"Auto-detected action_history_len={action_history_len} from checkpoint")
            else:
                action_history_len = 0

        model = PPONetwork(
            input_shape=(4, 64, 64),
            num_actions=7,
            action_history_len=action_history_len,
        )
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, action_history_len, model_type

    # DDQN model
    # Auto-detect action_history_len from fc2 weight shape if not specified
    if action_history_len is None:
        fc2_key = "online.backbone.fc2.weight"
        if fc2_key in state_dict:
            fc2_in = state_dict[fc2_key].shape[1]
            inferred = (fc2_in - 512) // 7
            action_history_len = inferred
            print(f"Auto-detected action_history_len={action_history_len} from checkpoint")
        else:
            action_history_len = 4  # fallback default

    # Auto-detect danger_bins from checkpoint (0 if danger_head not present)
    if "online.danger_head.0.weight" not in state_dict:
        danger_bins = 0
        print("No danger_head in checkpoint, disabling danger prediction")

    model = DoubleDQN(
        input_shape=(4, 64, 64),
        num_actions=7,
        action_history_len=action_history_len,
        danger_prediction_bins=danger_bins,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, action_history_len, model_type


def collect_frames(
    model: nn.Module,
    device: str,
    num_frames: int,
    spacing: int,
    max_episodes: int,
    epsilon: float,
    action_history_len: int,
    model_type: str,
) -> list[dict]:
    """Collect frames from a single episode at regular X-position intervals.

    Keeps trying episodes until one reaches the flag, then uses frames from that episode.
    """
    target_positions = list(range(100, 3200, spacing))

    env = create_mario_env(level=(1, 1), action_history_len=action_history_len)

    for ep in range(max_episodes):
        obs, _ = env.reset()
        action_history = torch.zeros(1, action_history_len, 7, device=device) if action_history_len > 0 else None
        done = False
        step = 0
        max_x = 0
        flag_reached = False

        # Collect frames for this episode only
        episode_collected = {}

        while not done and step < 8000:
            # Network expects uint8 and normalizes internally
            state_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)

            with torch.no_grad():
                if model_type == "ppo":
                    # PPO model: get policy logits and value
                    policy_logits, value = model.forward(state_tensor, action_history)
                    probs = torch.softmax(policy_logits, dim=-1)
                    # Use policy logits as "q_values" for visualization
                    q_values = policy_logits[0].cpu().numpy()
                    value_scalar = value[0].item()
                    danger_probs = None  # PPO doesn't have danger head
                else:
                    # DDQN model: get Q-values and danger predictions
                    features = model.online.backbone(state_tensor, action_history)
                    q_symlog = model.online(state_tensor, action_history)
                    q_real = symexp(q_symlog)
                    q_values = q_real[0].cpu().numpy()
                    value_scalar = None
                    if model.online.danger_head is not None:
                        danger_logits = model.online.danger_head(features)
                        danger_probs = torch.softmax(danger_logits, dim=1)[0].cpu().numpy()
                    else:
                        danger_probs = None

            # Action selection
            if model_type == "ppo":
                # PPO: sample from policy distribution (natural stochastic exploration)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
            else:
                # DDQN: epsilon-greedy exploration
                if np.random.random() < epsilon:
                    action = np.random.randint(7)
                else:
                    action = q_symlog.argmax(dim=1).item()

            obs_next, _, terminated, truncated, info = env.step(action)
            x_pos = info.get("x_pos", 0)
            flag_get = info.get("flag_get", False)

            for target_x in target_positions:
                if abs(x_pos - target_x) < 25 and target_x not in episode_collected:
                    # Get attention map
                    if model_type == "ppo":
                        attn = model.get_attention_map()
                    else:
                        _ = model.online(state_tensor, action_history)
                        attn = model.online.backbone.get_attention_map()

                    episode_collected[target_x] = {
                        "x": x_pos,
                        "frame": obs[-1],
                        "attn": attn[0, 0].cpu().numpy() if attn is not None else None,
                        "q_values": q_values,
                        "danger_probs": danger_probs,
                        "value": value_scalar,  # PPO value estimate
                        "model_type": model_type,
                    }

            max_x = max(max_x, x_pos)

            # Check if flag reached
            if flag_get:
                flag_reached = True
                print(f"  Ep {ep}: FLAG REACHED at x={x_pos}! Collected {len(episode_collected)} frames")
                break

            # Update action history
            if action_history_len > 0:
                ah = torch.zeros(1, 1, 7, device=device)
                ah[0, 0, action] = 1.0
                action_history = torch.cat([action_history[:, 1:, :], ah], dim=1)

            obs = obs_next
            done = terminated or truncated
            step += 1

        if flag_reached:
            # Use frames from this successful episode
            env.close()
            sorted_data = sorted(episode_collected.items())[:num_frames]
            print(f"Using {len(sorted_data)} frames from flag-reaching episode (max_x={max_x})")
            return sorted_data, max_x

        if ep % 10 == 0:
            print(f"  Ep {ep}: max_x={max_x}, no flag (collected {len(episode_collected)} frames)")

    # No flag reached after max_episodes - use best episode we have
    env.close()
    print(f"Warning: No flag reached after {max_episodes} episodes. Using last episode.")
    sorted_data = sorted(episode_collected.items())[:num_frames]
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

    # Detect model type from first frame
    model_type = sorted_data[0][1].get("model_type", "ddqn") if sorted_data else "ddqn"

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

        # Col 3: Q-values (DDQN) or Policy logits (PPO)
        q_vals = data["q_values"]
        # Replace NaN/Inf with 0 for display
        q_vals = np.nan_to_num(q_vals, nan=0.0, posinf=100.0, neginf=-100.0)
        colors = ["#d62728" if q < 0 else "#2ca02c" for q in q_vals]
        axes[i, 2].barh(range(7), q_vals, color=colors)
        axes[i, 2].set_yticks(range(7))
        axes[i, 2].set_yticklabels(actions, fontsize=7)
        axes[i, 2].axvline(0, color="black", linewidth=0.5)

        if model_type == "ppo":
            # For PPO, show softmax probabilities
            probs = np.exp(q_vals - q_vals.max())  # stable softmax
            probs = probs / probs.sum()
            title = f"Policy (p={probs.max():.2f})"
        else:
            title = f"Q-values (max={q_vals.max():.1f})"
        axes[i, 2].set_title(title, fontsize=9)
        q_min, q_max = q_vals.min(), q_vals.max()
        axes[i, 2].set_xlim(min(q_min * 1.1, -10), max(q_max * 1.1, 10))

        # Col 4: Danger prediction (DDQN) or Value estimate (PPO)
        danger = data.get("danger_probs")
        value = data.get("value")

        if model_type == "ppo" and value is not None:
            # For PPO: show value estimate as a gauge
            axes[i, 3].barh([0], [value], color="#1f77b4", height=0.5)
            axes[i, 3].set_xlim(-50, 50)
            axes[i, 3].axvline(0, color="black", linewidth=0.5)
            axes[i, 3].set_yticks([])
            axes[i, 3].set_title(f"Value: {value:.2f}", fontsize=9)
        elif danger is not None:
            # DDQN: show danger distribution
            num_bins = len(danger)
            behind_bins = num_bins // 4  # 4 bins behind
            ahead_bins = num_bins - behind_bins  # 12 bins ahead

            colors = ["#9467bd"] * behind_bins + ["#ff7f0e"] * ahead_bins
            x_bins = np.arange(num_bins)
            axes[i, 3].bar(x_bins, danger, color=colors, alpha=0.8)
            axes[i, 3].axvline(behind_bins - 0.5, color="white", linewidth=1.5, linestyle="--")

            tick_positions = [0, behind_bins - 1, behind_bins, num_bins - 1]
            tick_labels = ["-64px", "0", "0", "+256px"]
            axes[i, 3].set_xticks(tick_positions)
            axes[i, 3].set_xticklabels(tick_labels, fontsize=6)

            axes[i, 3].text(behind_bins / 2 - 0.5, 0.95, "Behind", ha="center", fontsize=6, color="#9467bd")
            axes[i, 3].text(behind_bins + ahead_bins / 2 - 0.5, 0.95, "Ahead", ha="center", fontsize=6, color="#ff7f0e")

            axes[i, 3].set_title(f"Danger (peak bin {danger.argmax()})", fontsize=9)
            axes[i, 3].set_xlim(-0.5, num_bins - 0.5)
            axes[i, 3].set_ylim(0, 1.05)
        else:
            axes[i, 3].text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=9, color="gray")
            axes[i, 3].set_title("(no aux output)", fontsize=9)
            axes[i, 3].axis("off")

    # Dynamic title based on model type
    if model_type == "ppo":
        title = f"PPO Frame Analysis (Max X={max_x}): Frame | Attention | Policy | Value"
    else:
        title = f"DDQN Frame Analysis (Max X={max_x}): Frame | Attention | Q-values | Danger"

    plt.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"Saved to {output_path}")


def main() -> None:
    args = parse_args()

    print(f"Loading model from {args.weights}...")
    model, action_history_len, model_type = load_model(
        args.weights,
        args.device,
        action_history_len=args.action_history_len,
        danger_bins=args.danger_bins,
    )
    print(f"Action history length: {action_history_len}")

    print(f"Collecting {args.num_frames} frames (spacing={args.spacing})...")
    sorted_data, max_x = collect_frames(
        model=model,
        device=args.device,
        num_frames=args.num_frames,
        spacing=args.spacing,
        max_episodes=args.max_episodes,
        epsilon=args.epsilon,
        action_history_len=action_history_len,
        model_type=model_type,
    )

    print("Creating visualization...")
    create_visualization(sorted_data, max_x, args.output)


if __name__ == "__main__":
    main()
