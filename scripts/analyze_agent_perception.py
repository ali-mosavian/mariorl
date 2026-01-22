#!/usr/bin/env python3
"""Analyze what the DDQN agent sees and believes at critical moments.

This script:
1. Loads the latest checkpoint
2. Runs the agent through levels, recording frames and Q-values
3. Identifies critical scenarios (near death, obstacles, decisions)
4. Saves diagnostic PNGs showing frames + network beliefs

Usage:
    uv run python scripts/analyze_agent_perception.py
"""

import argparse
from pathlib import Path
from dataclasses import dataclass

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mario_rl.agent.ddqn_net import DDQNNet
from mario_rl.agent.ddqn_net import DoubleDQN
from mario_rl.environment.factory import create_mario_env

# Action names for visualization
ACTION_NAMES = ["NOOP", "→", "→+A", "→+B", "→+A+B", "A", "←"]
ACTION_COLORS = ["gray", "blue", "cyan", "orange", "red", "green", "purple"]


@dataclass
class FrameRecord:
    """Record of a single frame with network analysis."""

    step: int
    raw_frame: np.ndarray  # Original 240x256 RGB frame
    processed_frames: np.ndarray  # 4x84x84 grayscale stack
    q_values: np.ndarray  # Q-values for each action
    chosen_action: int
    reward: float
    x_pos: int
    y_pos: int
    info: dict
    is_death: bool = False
    scenario: str = "normal"


def load_model(checkpoint_path: Path, device: str = "cuda") -> DDQNNet:
    """Load DDQN model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract model state dict from different checkpoint formats
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Check if this is a DoubleDQN checkpoint (has online.* keys)
    is_double_dqn = any(k.startswith("online.") for k in state_dict.keys())

    # Infer dimensions from state dict
    first_conv_weight = None
    for key in ["online.backbone.conv1.weight", "backbone.conv1.weight", "features.0.weight", "conv.0.weight"]:
        if key in state_dict:
            first_conv_weight = state_dict[key]
            break

    if first_conv_weight is not None:
        in_channels = first_conv_weight.shape[1]
    else:
        in_channels = 4

    # Infer input size from layer norm weight shape
    # layer_norm_size = 1024 -> 64x64 input, 3136 -> 84x84 input
    ln_key = "online.backbone.layer_norm.weight" if is_double_dqn else "backbone.layer_norm.weight"
    if ln_key in state_dict:
        ln_size = state_dict[ln_key].shape[0]
        if ln_size == 1024:
            input_size = 64
        elif ln_size == 3136:
            input_size = 84
        else:
            input_size = 64  # default
    else:
        input_size = 64

    print(
        f"  Detected {'DoubleDQN' if is_double_dqn else 'DDQNNet'} with {in_channels} channels, {input_size}x{input_size} input"
    )

    if is_double_dqn:
        # Load full DoubleDQN model and return the online network
        model = DoubleDQN(
            input_shape=(in_channels, input_size, input_size),
            num_actions=7,
        )
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        # Return just the online network for inference
        return model.online
    else:
        # Load single DDQNNet
        model = DDQNNet(
            input_shape=(in_channels, input_size, input_size),
            num_actions=7,
        )
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model


def get_q_values(model: DDQNNet, frames, device: str) -> np.ndarray:
    """Get Q-values for a frame stack."""
    with torch.no_grad():
        # Convert LazyFrames to numpy if needed
        if hasattr(frames, "__array__"):
            frames = np.array(frames)
        # Normalize and convert to tensor
        state = torch.from_numpy(frames).float().unsqueeze(0).to(device) / 255.0
        q_values = model(state).cpu().numpy()[0]
    return q_values


def classify_scenario(
    record: FrameRecord,
    prev_records: list[FrameRecord],
) -> str:
    """Classify what scenario this frame represents."""

    # Death frame
    if record.is_death:
        return "DEATH"

    # Check for sudden x_pos drop (might be falling)
    if prev_records and len(prev_records) >= 2:
        prev_x = prev_records[-1].x_pos
        prev_prev_x = prev_records[-2].x_pos if len(prev_records) >= 2 else prev_x

        # No forward progress (stuck at obstacle?)
        if record.x_pos == prev_x == prev_prev_x:
            return "STUCK"

        # Moving backward (retreating or bounced off)
        if record.x_pos < prev_x:
            return "BACKWARD"

    # High Q-value variance = uncertain decision
    q_range = np.max(record.q_values) - np.min(record.q_values)
    if q_range < 2.0:
        return "LOW_CONFIDENCE"

    # Check if jump action has notably different value
    jump_actions = [2, 4, 5]  # →+A, →+A+B, A
    non_jump_actions = [0, 1, 3, 6]  # NOOP, →, →+B, ←

    jump_q = np.mean([record.q_values[a] for a in jump_actions])
    non_jump_q = np.mean([record.q_values[a] for a in non_jump_actions])

    if jump_q > non_jump_q + 5:
        return "JUMP_PREFERRED"
    elif non_jump_q > jump_q + 5:
        return "NO_JUMP_PREFERRED"

    return "NORMAL"


def create_diagnostic_image(
    record: FrameRecord,
    output_path: Path,
    episode: int,
    prev_records: list[FrameRecord] = None,
):
    """Create a comprehensive diagnostic PNG for a frame."""

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    # Title with metadata
    scenario_colors = {
        "DEATH": "red",
        "STUCK": "orange",
        "BACKWARD": "yellow",
        "LOW_CONFIDENCE": "purple",
        "JUMP_PREFERRED": "cyan",
        "NO_JUMP_PREFERRED": "blue",
        "NORMAL": "green",
    }

    fig.suptitle(
        f"Episode {episode} | Step {record.step} | X={record.x_pos} | "
        f"Scenario: {record.scenario} | Reward: {record.reward:.1f}",
        fontsize=14,
        fontweight="bold",
        color=scenario_colors.get(record.scenario, "black"),
    )

    # 1. Raw game frame (large)
    ax_raw = fig.add_subplot(gs[0:2, 0:2])
    ax_raw.imshow(record.raw_frame)
    ax_raw.set_title("Raw Game Frame (240x256)", fontsize=12)
    ax_raw.axis("off")

    # Add position indicator
    ax_raw.text(
        10,
        230,
        f"X: {record.x_pos}",
        fontsize=10,
        color="white",
        bbox={"boxstyle": "round", "facecolor": "black", "alpha": 0.7},
    )

    # 2. Q-value bar chart
    ax_q = fig.add_subplot(gs[0, 2:4])
    bars = ax_q.bar(ACTION_NAMES, record.q_values, color=ACTION_COLORS)

    # Highlight chosen action
    bars[record.chosen_action].set_edgecolor("black")
    bars[record.chosen_action].set_linewidth(3)

    ax_q.axhline(y=np.mean(record.q_values), color="gray", linestyle="--", alpha=0.5, label="Mean Q")
    ax_q.set_ylabel("Q-Value", fontsize=10)
    ax_q.set_title(f"Q-Values (Chosen: {ACTION_NAMES[record.chosen_action]})", fontsize=12)
    ax_q.tick_params(axis="x", rotation=45)

    # Add value labels on bars
    for bar, val in zip(bars, record.q_values, strict=False):
        ax_q.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # 3. Processed frames (4 stacked frames)
    ax_frames = fig.add_subplot(gs[1, 2:4])

    # Concatenate the 4 frames horizontally
    combined = np.concatenate([record.processed_frames[i] for i in range(4)], axis=1)
    ax_frames.imshow(combined, cmap="gray")
    ax_frames.set_title("Processed Frames (4 × 84x84, oldest → newest)", fontsize=12)
    ax_frames.axis("off")

    # Add frame numbers
    for i in range(4):
        ax_frames.text(
            i * 84 + 42,
            5,
            f"t-{3 - i}",
            fontsize=9,
            color="red",
            ha="center",
            va="top",
            fontweight="bold",
        )

    # 4. Q-value analysis
    ax_analysis = fig.add_subplot(gs[2, 0:2])
    ax_analysis.axis("off")

    # Compute statistics
    best_action = np.argmax(record.q_values)
    worst_action = np.argmin(record.q_values)
    q_std = np.std(record.q_values)
    q_range = record.q_values.max() - record.q_values.min()

    # Jump vs non-jump analysis
    jump_actions = [2, 4, 5]
    non_jump = [0, 1, 3, 6]
    jump_q_avg = np.mean([record.q_values[a] for a in jump_actions])
    non_jump_q_avg = np.mean([record.q_values[a] for a in non_jump])

    analysis_text = f"""
    Q-VALUE ANALYSIS
    ================
    Best Action:  {ACTION_NAMES[best_action]} (Q={record.q_values[best_action]:.2f})
    Worst Action: {ACTION_NAMES[worst_action]} (Q={record.q_values[worst_action]:.2f})
    Q Range:      {q_range:.2f}
    Q Std Dev:    {q_std:.2f}
    
    JUMP ANALYSIS
    =============
    Avg Q (jump actions):     {jump_q_avg:.2f}
    Avg Q (non-jump actions): {non_jump_q_avg:.2f}
    Jump Preference:          {jump_q_avg - non_jump_q_avg:+.2f}
    
    POSITION
    ========
    X Position: {record.x_pos}
    Y Position: {record.y_pos}
    """

    ax_analysis.text(
        0.05,
        0.95,
        analysis_text,
        transform=ax_analysis.transAxes,
        fontsize=10,
        fontfamily="monospace",
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "lightgray", "alpha": 0.8},
    )

    # 5. Q-value history (if we have previous records)
    ax_history = fig.add_subplot(gs[2, 2:4])

    if prev_records and len(prev_records) > 1:
        steps = [r.step for r in prev_records[-30:]] + [record.step]

        # Plot Q-values for each action over time
        for action_idx in range(7):
            q_history = [r.q_values[action_idx] for r in prev_records[-30:]] + [record.q_values[action_idx]]
            ax_history.plot(
                steps,
                q_history,
                color=ACTION_COLORS[action_idx],
                label=ACTION_NAMES[action_idx],
                alpha=0.7,
                linewidth=1.5,
            )

        ax_history.set_xlabel("Step", fontsize=10)
        ax_history.set_ylabel("Q-Value", fontsize=10)
        ax_history.set_title("Q-Value History (last 30 steps)", fontsize=12)
        ax_history.legend(loc="upper left", fontsize=8, ncol=4)
        ax_history.grid(True, alpha=0.3)
    else:
        ax_history.text(
            0.5,
            0.5,
            "Q-value history\n(accumulating...)",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax_history.axis("off")

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_analysis(
    checkpoint_path: Path,
    output_dir: Path,
    num_episodes: int = 3,
    level: tuple[int, int] = (1, 1),
    device: str = "cuda",
    save_every_n: int = 10,
    max_steps_per_episode: int = 2000,
):
    """Run comprehensive perception analysis."""

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {checkpoint_path}...")
    model = load_model(checkpoint_path, device)

    print(f"Creating environment for level {level[0]}-{level[1]}...")
    mario_env = create_mario_env(level=level, render_frames=False)
    env = mario_env.env
    base_env = mario_env.base_env  # For getting raw RGB frames

    all_records = []

    for episode in range(num_episodes):
        print(f"\n{'=' * 60}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'=' * 60}")

        state, info = env.reset()
        episode_records = []

        # Get initial raw frame from base environment's screen
        raw_frame = base_env.screen.copy() if hasattr(base_env, "screen") else np.zeros((240, 256, 3), dtype=np.uint8)

        step = 0
        total_reward = 0
        done = False

        while not done and step < max_steps_per_episode:
            # Get Q-values
            q_values = get_q_values(model, state, device)

            # Choose action (greedy for analysis)
            action = np.argmax(q_values)

            # Convert LazyFrames to numpy if needed
            state_np = np.array(state) if hasattr(state, "__array__") else state

            # Create record
            record = FrameRecord(
                step=step,
                raw_frame=raw_frame.copy() if raw_frame is not None else np.zeros((240, 256, 3), dtype=np.uint8),
                processed_frames=state_np.copy(),
                q_values=q_values.copy(),
                chosen_action=action,
                reward=0,  # Will update after step
                x_pos=info.get("x_pos", 0),
                y_pos=info.get("y_pos", 0),
                info=info.copy(),
            )

            # Classify scenario
            record.scenario = classify_scenario(record, episode_records)

            episode_records.append(record)

            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Update record with reward
            record.reward = reward
            total_reward += reward

            # Get next raw frame from base environment's screen
            raw_frame = (
                base_env.screen.copy() if hasattr(base_env, "screen") else np.zeros((240, 256, 3), dtype=np.uint8)
            )

            # Check for death
            if done and info.get("life", 3) < 3:
                record.is_death = True
                record.scenario = "DEATH"

            # Save images at intervals or interesting scenarios
            should_save = (
                step % save_every_n == 0
                or record.scenario in ["DEATH", "STUCK", "BACKWARD", "LOW_CONFIDENCE"]
                or (step > 0 and step % 50 == 0)
            )

            if should_save:
                img_path = output_dir / f"ep{episode:02d}_step{step:05d}_{record.scenario}.png"
                create_diagnostic_image(
                    record,
                    img_path,
                    episode,
                    prev_records=episode_records[:-1] if len(episode_records) > 1 else None,
                )
                print(
                    f"  Step {step:5d} | X={record.x_pos:5d} | {record.scenario:15s} | Q-range={np.ptp(q_values):.1f}"
                )

            state = next_state
            step += 1

        # Always save the death frame
        if episode_records and episode_records[-1].is_death:
            img_path = output_dir / f"ep{episode:02d}_DEATH_step{step:05d}.png"
            create_diagnostic_image(
                episode_records[-1],
                img_path,
                episode,
                prev_records=episode_records[:-1],
            )

        # Save frames leading up to death (last 10 before death)
        if episode_records and len(episode_records) > 10:
            print("\n  Saving 10 frames before death...")
            for i, rec in enumerate(episode_records[-10:]):
                idx = len(episode_records) - 10 + i
                img_path = output_dir / f"ep{episode:02d}_predeath_{i:02d}_step{rec.step:05d}.png"
                create_diagnostic_image(
                    rec,
                    img_path,
                    episode,
                    prev_records=episode_records[:idx] if idx > 0 else None,
                )

        print(
            f"\n  Episode {episode + 1} finished: {step} steps, reward={total_reward:.1f}, x_pos={info.get('x_pos', 0)}"
        )
        all_records.extend(episode_records)

    env.close()

    # Create summary statistics
    create_summary(all_records, output_dir)

    print(f"\nAnalysis complete! Output saved to {output_dir}")
    return all_records


def create_summary(records: list[FrameRecord], output_dir: Path):
    """Create summary visualizations."""

    if not records:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Q-value distribution by scenario
    ax = axes[0, 0]
    scenarios = {}
    for r in records:
        if r.scenario not in scenarios:
            scenarios[r.scenario] = []
        scenarios[r.scenario].append(np.mean(r.q_values))

    scenario_names = list(scenarios.keys())
    scenario_means = [np.mean(v) for v in scenarios.values()]
    scenario_stds = [np.std(v) for v in scenarios.values()]

    ax.bar(scenario_names, scenario_means, yerr=scenario_stds, capsize=5)
    ax.set_ylabel("Mean Q-Value")
    ax.set_title("Average Q-Value by Scenario")
    ax.tick_params(axis="x", rotation=45)

    # 2. Action distribution
    ax = axes[0, 1]
    action_counts = np.zeros(7)
    for r in records:
        action_counts[r.chosen_action] += 1
    action_pcts = action_counts / len(records) * 100

    ax.bar(ACTION_NAMES, action_pcts, color=ACTION_COLORS)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Action Distribution")
    ax.tick_params(axis="x", rotation=45)

    # 3. Q-value over x_pos
    ax = axes[1, 0]
    x_positions = [r.x_pos for r in records]
    q_means = [np.mean(r.q_values) for r in records]
    q_stds = [np.std(r.q_values) for r in records]

    ax.scatter(x_positions, q_means, c=q_stds, cmap="viridis", alpha=0.5, s=10)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Mean Q-Value")
    ax.set_title("Q-Value vs Position (color = Q std)")
    ax.colorbar = plt.colorbar(ax.collections[0], ax=ax, label="Q Std")

    # 4. Jump preference over x_pos
    ax = axes[1, 1]
    jump_prefs = []
    for r in records:
        jump_q = np.mean([r.q_values[a] for a in [2, 4, 5]])
        non_jump_q = np.mean([r.q_values[a] for a in [0, 1, 3, 6]])
        jump_prefs.append(jump_q - non_jump_q)

    ax.scatter(x_positions, jump_prefs, alpha=0.5, s=10, c="blue")
    ax.axhline(y=0, color="red", linestyle="--", label="No preference")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Jump Preference (Q_jump - Q_no_jump)")
    ax.set_title("Jump Preference by Position")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Save Q-value statistics
    with open(output_dir / "statistics.txt", "w") as f:
        f.write("PERCEPTION ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Total frames analyzed: {len(records)}\n")
        f.write(f"Unique x positions: {len(set(x_positions))}\n")
        f.write(f"Max x reached: {max(x_positions)}\n\n")

        f.write("SCENARIO COUNTS:\n")
        for scenario, count in sorted(scenarios.items(), key=lambda x: -len(x[1])):
            f.write(f"  {scenario}: {len(count)} frames ({len(count) / len(records) * 100:.1f}%)\n")

        f.write("\nACTION DISTRIBUTION:\n")
        for i, name in enumerate(ACTION_NAMES):
            f.write(f"  {name}: {action_pcts[i]:.1f}%\n")

        f.write("\nQ-VALUE STATISTICS:\n")
        all_q = np.array([r.q_values for r in records])
        f.write(f"  Overall mean: {np.mean(all_q):.2f}\n")
        f.write(f"  Overall std: {np.std(all_q):.2f}\n")
        f.write(f"  Min Q seen: {np.min(all_q):.2f}\n")
        f.write(f"  Max Q seen: {np.max(all_q):.2f}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze DDQN agent perception")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/ddqn_dist_2026-01-15T21-58-18/weights.pt",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="analysis_output",
        help="Output directory for images",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to analyze",
    )
    parser.add_argument(
        "--world",
        type=int,
        default=1,
        help="World number",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=1,
        help="Stage number",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=20,
        help="Save image every N steps",
    )

    args = parser.parse_args()

    run_analysis(
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.output),
        num_episodes=args.episodes,
        level=(args.world, args.stage),
        device=args.device,
        save_every_n=args.save_every,
    )


if __name__ == "__main__":
    main()
