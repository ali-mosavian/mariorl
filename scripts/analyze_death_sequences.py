#!/usr/bin/env python3
"""Analyze death and stuck sequences in DDQN agent.

Detects when agent dies or gets stuck, then saves a visualization
showing the sequence of frames and Q-values leading up to the event.

Usage:
    uv run python scripts/analyze_death_sequences.py
"""

from pathlib import Path
from dataclasses import dataclass, field
from collections import deque

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mario_rl.environment.factory import create_mario_env
from mario_rl.agent.ddqn_net import DoubleDQN


ACTION_NAMES = ["NOOP", "→", "→+A", "→+B", "→+A+B", "A", "←"]
ACTION_COLORS = ["#808080", "#0066CC", "#00CCCC", "#FF9900", "#CC0000", "#00CC00", "#9900CC"]


@dataclass
class FrameData:
    """Data for a single frame."""
    step: int
    x_pos: int
    y_pos: int
    q_values: np.ndarray
    chosen_action: int
    reward: float
    processed_frame: np.ndarray  # Single frame (64x64)
    danger_pred: np.ndarray = field(default_factory=lambda: np.zeros(16))  # Danger predictions
    raw_frame: np.ndarray = field(default_factory=lambda: np.zeros((240, 256, 3), dtype=np.uint8))


def load_model(checkpoint_path: Path, device: str = "cuda") -> tuple[torch.nn.Module, int, int]:
    """Load DDQN model from checkpoint.
    
    Returns:
        (model, input_size, action_history_len)
    """
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    # Detect input size from layer norm
    ln_key = "online.backbone.layer_norm.weight"
    if ln_key in state_dict:
        ln_size = state_dict[ln_key].shape[0]
        input_size = 64 if ln_size == 1024 else 84
    else:
        input_size = 64
    
    # Detect action_history_len from FC layer input size
    fc_key = "online.backbone.fc.weight"
    action_history_len = 0
    if fc_key in state_dict:
        fc_input = state_dict[fc_key].shape[1]
        # Expected flat_size for 64x64 input is 1024
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


def detect_stuck(x_history: list[int], threshold: int = 15) -> bool:
    """Detect if agent is stuck (no x progress for threshold steps)."""
    if len(x_history) < threshold:
        return False
    recent = x_history[-threshold:]
    return max(recent) - min(recent) < 5


def create_sequence_image(
    frames: list[FrameData],
    event_type: str,
    output_path: Path,
    episode: int,
):
    """Create visualization of frame sequence leading to event."""
    n_frames = len(frames)
    if n_frames == 0:
        return
    
    # Create figure - show up to 10 frames
    n_show = min(n_frames, 10)
    frames_to_show = frames[-n_show:]
    
    fig = plt.figure(figsize=(20, 16))
    
    # Title
    last = frames_to_show[-1]
    color = "red" if event_type == "DEATH" else "orange"
    fig.suptitle(
        f"Episode {episode} | {event_type} at X={last.x_pos} | "
        f"Last {n_show} frames before event",
        fontsize=16, fontweight="bold", color=color
    )
    
    # Grid: 5 rows
    # Row 1: Processed frames
    # Row 2: Q-value bars for each frame
    # Row 3: Danger predictions for each frame
    # Row 4: Action timeline
    # Row 5: Q-value timeseries
    
    gs = gridspec.GridSpec(5, n_show, figure=fig, height_ratios=[1, 1.2, 0.8, 0.3, 1], hspace=0.4, wspace=0.1)
    
    # Row 1: Processed frames
    for i, frame in enumerate(frames_to_show):
        ax = fig.add_subplot(gs[0, i])
        # Show the most recent frame from the stack (index 3)
        ax.imshow(frame.processed_frame[3], cmap="gray")
        ax.set_title(f"t-{n_show - 1 - i}\nX={frame.x_pos}", fontsize=9)
        ax.axis("off")
        
        # Highlight chosen action
        action_color = ACTION_COLORS[frame.chosen_action]
        for spine in ax.spines.values():
            spine.set_edgecolor(action_color)
            spine.set_linewidth(3)
            spine.set_visible(True)
    
    # Row 2: Q-value bars for each frame
    for i, frame in enumerate(frames_to_show):
        ax = fig.add_subplot(gs[1, i])
        bars = ax.barh(ACTION_NAMES, frame.q_values, color=ACTION_COLORS)
        
        # Highlight chosen action
        bars[frame.chosen_action].set_edgecolor("black")
        bars[frame.chosen_action].set_linewidth(2)
        
        ax.axvline(x=np.mean(frame.q_values), color="gray", linestyle="--", alpha=0.5)
        
        if i == 0:
            ax.set_ylabel("Action", fontsize=9)
        else:
            ax.set_yticklabels([])
        
        ax.tick_params(axis="both", labelsize=7)
        
        # Show Q-range
        q_range = frame.q_values.max() - frame.q_values.min()
        ax.set_xlabel(f"Q-range: {q_range:.2f}", fontsize=8)
    
    # Row 3: Danger predictions for each frame
    for i, frame in enumerate(frames_to_show):
        ax = fig.add_subplot(gs[2, i])
        
        # Danger predictions: 4 bins behind (left), 12 bins ahead (right)
        # Create bar plot with color based on danger level
        bins = len(frame.danger_pred)
        behind_bins = bins // 4
        
        colors = plt.cm.Reds(frame.danger_pred)
        bars = ax.bar(range(bins), frame.danger_pred, color=colors, edgecolor="black", linewidth=0.5)
        
        # Mark the "current position" divider
        ax.axvline(x=behind_bins - 0.5, color="blue", linestyle="--", linewidth=2, alpha=0.7)
        
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.5, bins - 0.5)
        
        if i == 0:
            ax.set_ylabel("Danger", fontsize=9)
            ax.set_yticks([0, 0.5, 1])
        else:
            ax.set_yticklabels([])
            ax.set_yticks([])
        
        ax.set_xticks([])
        
        # Show max danger value
        max_danger = frame.danger_pred.max()
        ax.set_xlabel(f"max={max_danger:.2f}", fontsize=8)
        
        if i == n_show // 2:
            ax.set_title("← Behind | Ahead →", fontsize=8)
    
    # Row 4: Action timeline
    ax_timeline = fig.add_subplot(gs[3, :])
    for i, frame in enumerate(frames_to_show):
        ax_timeline.scatter(i, 0, c=[ACTION_COLORS[frame.chosen_action]], s=200, marker="s")
        ax_timeline.annotate(
            ACTION_NAMES[frame.chosen_action],
            (i, 0), textcoords="offset points", xytext=(0, 10),
            ha="center", fontsize=8, fontweight="bold"
        )
    ax_timeline.set_xlim(-0.5, n_show - 0.5)
    ax_timeline.set_ylim(-0.5, 0.5)
    ax_timeline.set_xticks(range(n_show))
    ax_timeline.set_xticklabels([f"t-{n_show - 1 - i}" for i in range(n_show)])
    ax_timeline.set_yticks([])
    ax_timeline.set_title("Action Sequence", fontsize=10, fontweight="bold")
    
    # Row 5: Q-value time series for all actions
    ax_ts = fig.add_subplot(gs[4, :])
    steps = list(range(n_show))
    
    for action_idx in range(7):
        q_vals = [f.q_values[action_idx] for f in frames_to_show]
        ax_ts.plot(steps, q_vals, color=ACTION_COLORS[action_idx], 
                   label=ACTION_NAMES[action_idx], linewidth=2, marker="o", markersize=4)
    
    # Mark chosen actions
    for i, frame in enumerate(frames_to_show):
        ax_ts.scatter(i, frame.q_values[frame.chosen_action], 
                      c="black", s=100, zorder=5, marker="*")
    
    ax_ts.set_xlabel("Frame (relative to event)", fontsize=10)
    ax_ts.set_ylabel("Q-Value", fontsize=10)
    ax_ts.set_title("Q-Values Over Time (★ = chosen action)", fontsize=11, fontweight="bold")
    ax_ts.legend(loc="upper left", fontsize=8, ncol=4)
    ax_ts.grid(True, alpha=0.3)
    ax_ts.set_xticks(steps)
    ax_ts.set_xticklabels([f"t-{n_show - 1 - i}" for i in range(n_show)])
    
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def run_analysis(
    checkpoint_path: Path,
    output_dir: Path,
    num_episodes: int = 5,
    level: tuple[int, int] = (1, 1),
    device: str = "cuda",
    history_size: int = 30,
    stuck_threshold: int = 20,
    max_steps: int = 2000,
    escape_epsilon: float = 0.8,
    escape_steps: int = 50,
):
    """Run death/stuck sequence analysis.
    
    When stuck, increases epsilon to escape the obstacle and continue exploring.
    """
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model, input_size, action_history_len = load_model(checkpoint_path, device)
    
    print(f"Creating environment for level {level[0]}-{level[1]}...")
    mario_env = create_mario_env(
        level=level, 
        render_frames=False, 
        action_history_len=action_history_len,
    )
    env = mario_env.env
    
    all_events = []
    
    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*60}")
        
        state, info = env.reset()
        
        frame_history: deque[FrameData] = deque(maxlen=history_size)
        x_history: list[int] = []
        
        step = 0
        event_count = 0
        prev_x = 0
        
        # Escape mode tracking
        escape_mode = False
        escape_steps_remaining = 0
        stuck_x_pos = 0
        visited_stuck_positions: set[int] = set()  # Track where we've been stuck (rounded to 50)
        
        while step < max_steps:
            # Get Q-values and danger predictions
            state_np = np.array(state) if hasattr(state, "__array__") else state
            with torch.no_grad():
                state_t = torch.from_numpy(state_np.copy()).float().unsqueeze(0).to(device) / 255.0
                
                action_hist = None
                
                if action_history_len > 0 and "action_history" in info:
                    action_hist = torch.from_numpy(info["action_history"]).float().unsqueeze(0).to(device)
                
                # Get Q-values and danger predictions
                result = model.online(state_t, action_hist, return_danger=True)
                if isinstance(result, tuple):
                    q_values, danger_pred = result
                    q_values = q_values.cpu().numpy()[0]
                    danger_pred = danger_pred.cpu().numpy()[0]
                else:
                    q_values = result.cpu().numpy()[0]
                    danger_pred = np.zeros(16)
            
            x_pos = info.get("x_pos", 0)
            y_pos = info.get("y_pos", 0)
            
            # Choose action - use epsilon-greedy in escape mode
            if escape_mode and escape_steps_remaining > 0:
                if np.random.random() < escape_epsilon:
                    # Random action, biased toward jump+right
                    action = np.random.choice([2, 4, 5, 1], p=[0.35, 0.35, 0.2, 0.1])
                else:
                    action = np.argmax(q_values)
                escape_steps_remaining -= 1
                
                # Check if we've escaped (moved past stuck position)
                if x_pos > stuck_x_pos + 50:
                    print(f"  ✓ Escaped! Now at X={x_pos}")
                    escape_mode = False
                    escape_steps_remaining = 0
                    # Clear history to start fresh tracking
                    x_history = [x_pos]
                    frame_history.clear()
            else:
                action = np.argmax(q_values)
                escape_mode = False
            
            # Store frame data (only when not in escape mode)
            if not escape_mode:
                frame_data = FrameData(
                    step=step,
                    x_pos=x_pos,
                    y_pos=y_pos,
                    q_values=q_values.copy(),
                    chosen_action=action,
                    reward=0,
                    processed_frame=state_np.copy(),
                    danger_pred=danger_pred.copy(),
                )
                frame_history.append(frame_data)
                x_history.append(x_pos)
            
            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            if not escape_mode and frame_history:
                frame_history[-1].reward = reward
            
            done = terminated or truncated
            
            # Check for DEATH
            if done:
                print(f"  💀 DEATH at step {step}, X={x_pos}")
                if frame_history:
                    output_path = output_dir / f"ep{episode:02d}_event{event_count:02d}_DEATH_x{x_pos}.png"
                    create_sequence_image(list(frame_history), "DEATH", output_path, episode)
                    all_events.append(("DEATH", episode, step, x_pos))
                    event_count += 1
                break
            
            # Check for STUCK (only when not in escape mode)
            if not escape_mode and detect_stuck(x_history, stuck_threshold):
                # Round position to avoid capturing same obstacle multiple times
                rounded_pos = (x_pos // 50) * 50
                
                if rounded_pos not in visited_stuck_positions:
                    print(f"  🚧 STUCK at step {step}, X={x_pos}")
                    output_path = output_dir / f"ep{episode:02d}_event{event_count:02d}_STUCK_x{x_pos}.png"
                    create_sequence_image(list(frame_history), "STUCK", output_path, episode)
                    all_events.append(("STUCK", episode, step, x_pos))
                    event_count += 1
                    visited_stuck_positions.add(rounded_pos)
                
                # Enter escape mode
                print(f"  🔄 Entering escape mode (epsilon={escape_epsilon}) for {escape_steps} steps...")
                escape_mode = True
                escape_steps_remaining = escape_steps
                stuck_x_pos = x_pos
                x_history = [x_pos]
            
            # Progress indicator
            if step % 200 == 0:
                q_range = q_values.max() - q_values.min()
                mode_str = "ESCAPE" if escape_mode else "GREEDY"
                print(f"  Step {step:5d} | X={x_pos:5d} | {mode_str:6s} | Action={ACTION_NAMES[action]:6s} | Q-range={q_range:.2f}")
            
            state = next_state
            prev_x = x_pos
            step += 1
        
        print(f"\n  Episode {episode + 1} summary: {event_count} events, max X={info.get('x_pos', 0)}")
    
    env.close()
    
    # Create summary
    create_summary(all_events, output_dir)
    
    print(f"\n{'='*60}")
    print(f"Analysis complete! {len(all_events)} events captured.")
    print(f"Output saved to: {output_dir}")
    print(f"{'='*60}")


def create_summary(events: list[tuple], output_dir: Path):
    """Create summary of all events."""
    if not events:
        return
    
    with open(output_dir / "events_summary.txt", "w") as f:
        f.write("DEATH/STUCK SEQUENCE ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        deaths = [e for e in events if e[0] == "DEATH"]
        stucks = [e for e in events if e[0] == "STUCK"]
        
        f.write(f"Total events: {len(events)}\n")
        f.write(f"  Deaths: {len(deaths)}\n")
        f.write(f"  Stuck:  {len(stucks)}\n\n")
        
        if deaths:
            death_x = [e[3] for e in deaths]
            f.write(f"Death positions:\n")
            f.write(f"  Min X: {min(death_x)}\n")
            f.write(f"  Max X: {max(death_x)}\n")
            f.write(f"  Mean X: {np.mean(death_x):.0f}\n\n")
        
        if stucks:
            stuck_x = [e[3] for e in stucks]
            f.write(f"Stuck positions:\n")
            f.write(f"  Min X: {min(stuck_x)}\n")
            f.write(f"  Max X: {max(stuck_x)}\n")
            f.write(f"  Mean X: {np.mean(stuck_x):.0f}\n\n")
        
        f.write("\nAll events:\n")
        for event_type, episode, step, x_pos in events:
            f.write(f"  Episode {episode}, Step {step}: {event_type} at X={x_pos}\n")
    
    # Create position histogram
    if len(events) >= 2:
        fig, ax = plt.subplots(figsize=(10, 5))
        
        x_positions = [e[3] for e in events]
        colors = ["red" if e[0] == "DEATH" else "orange" for e in events]
        
        ax.scatter(x_positions, range(len(events)), c=colors, s=100)
        ax.set_xlabel("X Position", fontsize=12)
        ax.set_ylabel("Event #", fontsize=12)
        ax.set_title("Event Positions (Red=Death, Orange=Stuck)", fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Mark common stuck points (pipes, holes)
        pipe_positions = [434, 914, 1230, 1486]  # Approximate pipe positions in 1-1
        for pipe_x in pipe_positions:
            ax.axvline(x=pipe_x, color="gray", linestyle="--", alpha=0.5)
            ax.text(pipe_x, len(events) - 0.5, "pipe", rotation=90, va="top", fontsize=8)
        
        plt.savefig(output_dir / "event_positions.png", dpi=150, bbox_inches="tight")
        plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze death/stuck sequences")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/ddqn_dist_2026-01-15T21-58-18/checkpoint_310000.pt",
    )
    parser.add_argument("--output", type=str, default="death_analysis")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--history", type=int, default=30)
    parser.add_argument("--stuck-threshold", type=int, default=20)
    
    args = parser.parse_args()
    
    run_analysis(
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.output),
        num_episodes=args.episodes,
        level=(args.world, args.stage),
        device=args.device,
        history_size=args.history,
        stuck_threshold=args.stuck_threshold,
    )


if __name__ == "__main__":
    main()
