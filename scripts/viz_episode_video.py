#!/usr/bin/env python
"""Generate episode video with attention overlay and prediction charts.

Creates a video showing:
1. Color game frame with attention map overlay (alpha 0.35)
2. Q-values bar chart overlay (top right)
3. Danger prediction bar chart overlay (below Q-values)
"""

import argparse
from io import BytesIO
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import zoom

from mario_rl.agent.ddqn_net import DoubleDQN
from mario_rl.environment.factory import create_mario_env
from mario_rl.models.ddqn import symexp

matplotlib.use("Agg")  # Non-interactive backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate episode video with overlays")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to weights.pt or checkpoint file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="episode_analysis.mp4",
        help="Output video path",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=4000,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video frame rate",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="Exploration epsilon",
    )
    parser.add_argument(
        "--attention-alpha",
        type=float,
        default=0.35,
        help="Attention overlay alpha",
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
    parser.add_argument(
        "--level",
        type=str,
        default="1-1",
        help="Level to play (e.g., 1-1, 1-2)",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=3,
        help="Scale factor for output video (base is 256x240)",
    )
    return parser.parse_args()


def load_model(
    weights_path: str,
    device: str,
    action_history_len: int | None = None,
    danger_bins: int = 16,
) -> tuple[DoubleDQN, int]:
    """Load the DoubleDQN model from weights."""
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Auto-detect action_history_len from fc2 weight shape
    if action_history_len is None:
        fc2_key = "online.backbone.fc2.weight"
        if fc2_key in state_dict:
            fc2_in = state_dict[fc2_key].shape[1]
            inferred = (fc2_in - 512) // 7
            action_history_len = inferred
            print(f"Auto-detected action_history_len={action_history_len}")
        else:
            action_history_len = 4

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


def render_chart_to_image(
    fig: plt.Figure,
    width: int,
    height: int,
) -> np.ndarray:
    """Render matplotlib figure to RGBA numpy array."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, transparent=True, bbox_inches="tight", pad_inches=0.05)
    buf.seek(0)
    img = Image.open(buf).convert("RGBA")
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    return np.array(img)


def create_qvalue_chart(q_values: np.ndarray, chart_width: int, chart_height: int) -> np.ndarray:
    """Create Q-values bar chart as RGBA image."""
    actions = ["NOOP", "Right", "R+J", "R+B", "R+J+B", "Jump", "Left"]
    
    fig, ax = plt.subplots(figsize=(2.5, 2.0))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0.6)
    ax.patch.set_facecolor("black")
    
    colors = ["#d62728" if q < 0 else "#2ca02c" for q in q_values]
    best_action = q_values.argmax()
    colors[best_action] = "#ffd700"  # Gold for best action
    
    ax.barh(range(7), q_values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(7))
    ax.set_yticklabels(actions, fontsize=7, color="white")
    ax.axvline(0, color="white", linewidth=0.5)
    ax.set_title(f"Q (max={q_values.max():.1f})", fontsize=8, color="white", pad=2)
    ax.tick_params(axis="x", colors="white", labelsize=6)
    ax.set_xlim(min(q_values.min() * 1.1, -10), max(q_values.max() * 1.1, 10))
    
    for spine in ax.spines.values():
        spine.set_color("white")
        spine.set_linewidth(0.5)
    
    img = render_chart_to_image(fig, chart_width, chart_height)
    plt.close(fig)
    return img


def create_danger_chart(danger_probs: np.ndarray, chart_width: int, chart_height: int) -> np.ndarray:
    """Create danger prediction bar chart as RGBA image."""
    fig, ax = plt.subplots(figsize=(2.5, 1.5))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0.6)
    ax.patch.set_facecolor("black")
    
    num_bins = len(danger_probs)
    behind_bins = num_bins // 4
    ahead_bins = num_bins - behind_bins
    
    colors = ["#9467bd"] * behind_bins + ["#ff7f0e"] * ahead_bins
    peak_bin = danger_probs.argmax()
    colors[peak_bin] = "#ff0000"  # Red for peak danger
    
    ax.bar(range(num_bins), danger_probs, color=colors, edgecolor="white", linewidth=0.3)
    ax.axvline(behind_bins - 0.5, color="white", linewidth=1, linestyle="--", alpha=0.7)
    
    ax.set_title(f"Danger (peak={peak_bin})", fontsize=8, color="white", pad=2)
    ax.set_xlim(-0.5, num_bins - 0.5)
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="both", colors="white", labelsize=5)
    ax.set_xticks([0, behind_bins, num_bins - 1])
    ax.set_xticklabels(["-64", "0", "+256"], fontsize=6, color="white")
    
    for spine in ax.spines.values():
        spine.set_color("white")
        spine.set_linewidth(0.5)
    
    img = render_chart_to_image(fig, chart_width, chart_height)
    plt.close(fig)
    return img


def overlay_rgba_on_bgr(base: np.ndarray, overlay: np.ndarray, x: int, y: int) -> np.ndarray:
    """Overlay RGBA image on BGR image at position (x, y)."""
    h, w = overlay.shape[:2]
    
    # Clamp to image bounds
    if x + w > base.shape[1]:
        w = base.shape[1] - x
        overlay = overlay[:, :w]
    if y + h > base.shape[0]:
        h = base.shape[0] - y
        overlay = overlay[:h]
    
    if w <= 0 or h <= 0:
        return base
    
    # Extract alpha channel and normalize
    alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
    rgb = overlay[:, :, :3]
    
    # Convert RGB to BGR for OpenCV
    bgr = rgb[:, :, ::-1]
    
    # Blend
    roi = base[y : y + h, x : x + w].astype(np.float32)
    blended = roi * (1 - alpha) + bgr.astype(np.float32) * alpha
    base[y : y + h, x : x + w] = blended.astype(np.uint8)
    
    return base


def apply_attention_overlay(
    frame: np.ndarray,
    attention: np.ndarray | None,
    alpha: float,
) -> np.ndarray:
    """Apply attention heatmap overlay to frame."""
    if attention is None:
        return frame
    
    # Resize attention to match frame size
    scale_y = frame.shape[0] / attention.shape[0]
    scale_x = frame.shape[1] / attention.shape[1]
    attn_resized = zoom(attention, (scale_y, scale_x), order=1)
    
    # Normalize attention
    attn_norm = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
    
    # Apply grayscale colormap (white = high attention)
    heatmap = plt.cm.gray(attn_norm)[:, :, :3]  # Drop alpha
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_bgr = heatmap[:, :, ::-1]  # RGB to BGR
    
    # Blend with original frame
    blended = cv2.addWeighted(frame, 1 - alpha, heatmap_bgr, alpha, 0)
    return blended


def generate_episode_video(
    model: DoubleDQN,
    device: str,
    output_path: str,
    max_steps: int,
    fps: int,
    epsilon: float,
    attention_alpha: float,
    action_history_len: int,
    level: tuple[int, int],
    scale: int,
) -> None:
    """Generate video of a single episode with overlays."""
    # Create environment with RGB rendering
    env = create_mario_env(level=level, action_history_len=action_history_len)
    
    # Video dimensions
    base_width, base_height = 256, 240
    out_width, out_height = base_width * scale, base_height * scale
    
    # Chart dimensions (relative to output size)
    chart_width = out_width // 4
    chart_height_q = out_height // 4
    chart_height_d = out_height // 5
    chart_margin = 10
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
    
    obs, _ = env.reset()
    action_history = torch.zeros(1, action_history_len, 7, device=device)
    
    frames_written = 0
    total_reward = 0
    
    print(f"Recording episode (max {max_steps} steps)...")
    
    for step in range(max_steps):
        # Get RGB frame from environment
        rgb_frame = env.render()
        if rgb_frame is None:
            print("Warning: render() returned None")
            break
        
        # Resize to output dimensions
        frame_resized = cv2.resize(rgb_frame, (out_width, out_height), interpolation=cv2.INTER_NEAREST)
        frame_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
        
        # Get model predictions
        state_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = model.online.backbone(state_tensor, action_history)
            q_symlog = model.online(state_tensor, action_history)
            q_real = symexp(q_symlog)
            danger_logits = model.online.danger_head(features)
            danger_probs = torch.softmax(danger_logits, dim=1)
            
            # Get attention map
            _ = model.online(state_tensor, action_history)
            attn = model.online.backbone.get_attention_map()
        
        # Apply attention overlay
        if attn is not None:
            attn_np = attn[0, 0].cpu().numpy()
            frame_bgr = apply_attention_overlay(frame_bgr, attn_np, attention_alpha)
        
        # Create and overlay Q-value chart
        q_chart = create_qvalue_chart(q_real[0].cpu().numpy(), chart_width, chart_height_q)
        frame_bgr = overlay_rgba_on_bgr(
            frame_bgr,
            q_chart,
            out_width - chart_width - chart_margin,
            chart_margin,
        )
        
        # Create and overlay danger chart
        d_chart = create_danger_chart(danger_probs[0].cpu().numpy(), chart_width, chart_height_d)
        frame_bgr = overlay_rgba_on_bgr(
            frame_bgr,
            d_chart,
            out_width - chart_width - chart_margin,
            chart_margin + chart_height_q + 5,
        )
        
        # Add info text overlay
        info_text = f"Step: {step} | Reward: {total_reward:.0f}"
        cv2.putText(
            frame_bgr,
            info_text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            info_text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        
        writer.write(frame_bgr)
        frames_written += 1
        
        # Select action
        if np.random.random() < epsilon:
            action = np.random.randint(7)
        else:
            action = q_symlog.argmax(dim=1).item()
        
        # Step environment
        obs_next, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Update action history
        ah = torch.zeros(1, 1, 7, device=device)
        ah[0, 0, action] = 1.0
        action_history = torch.cat([action_history[:, 1:, :], ah], dim=1)
        
        obs = obs_next
        
        if terminated or truncated:
            x_pos = info.get("x_pos", 0)
            flag = info.get("flag_get", False)
            print(f"Episode ended: steps={step+1}, reward={total_reward:.0f}, x={x_pos}, flag={flag}")
            break
        
        if step % 500 == 0:
            print(f"  Step {step}, x={info.get('x_pos', 0)}, reward={total_reward:.0f}")
    
    writer.release()
    env.close()
    
    print(f"Saved {frames_written} frames to {output_path}")


def main() -> None:
    args = parse_args()
    
    # Parse level
    world, stage = map(int, args.level.split("-"))
    level = (world, stage)
    
    print(f"Loading model from {args.weights}...")
    model, action_history_len = load_model(
        args.weights,
        args.device,
        action_history_len=args.action_history_len,
        danger_bins=args.danger_bins,
    )
    
    print(f"Generating episode video for level {args.level}...")
    generate_episode_video(
        model=model,
        device=args.device,
        output_path=args.output,
        max_steps=args.max_steps,
        fps=args.fps,
        epsilon=args.epsilon,
        attention_alpha=args.attention_alpha,
        action_history_len=action_history_len,
        level=level,
        scale=args.scale,
    )


if __name__ == "__main__":
    main()
