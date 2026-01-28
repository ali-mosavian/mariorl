#!/usr/bin/env python3
"""
Play Mario using a trained checkpoint (DDQN or PPO).

Loads the latest checkpoint from a checkpoint directory and plays Mario
with visual rendering, using the trained policy for action selection.
Automatically detects model type (DDQN or PPO) from checkpoint.

Usage:
    uv run python scripts/play_mario.py
    uv run python scripts/play_mario.py --checkpoint-dir checkpoints/ddqn_dist_2026-01-15T21-58-18
    uv run python scripts/play_mario.py --checkpoint-dir checkpoints/vec_ppo_2026-01-26T16-00-00
    uv run python scripts/play_mario.py --level 1,2 --episodes 5
    uv run python scripts/play_mario.py --epsilon 0.05  # Add some exploration (DDQN only)
    uv run python scripts/play_mario.py --track-entities  # Enable entity tracking overlay
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import click
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from mario_rl.agent.ddqn_net import DoubleDQN
from mario_rl.environment.factory import create_mario_env
from mario_rl.environment.motion_tracker import render_danger_hud
from mario_rl.environment.motion_tracker import MotionEntityTracker
from mario_rl.environment.motion_tracker import render_entity_overlay

# Import PPONetwork if available
try:
    from scripts.ppo_vec import PPONetwork

    PPO_AVAILABLE = True
except ImportError:
    PPO_AVAILABLE = False
    PPONetwork = None  # type: ignore


def find_latest_checkpoint_dir(base_dir: Path) -> Path | None:
    """Find the most recently modified checkpoint directory.

    Args:
        base_dir: Base directory containing checkpoint subdirectories

    Returns:
        Path to the latest checkpoint directory, or None if not found
    """
    if not base_dir.exists():
        return None

    # Find directories that look like checkpoint runs (contain .pt files)
    checkpoint_dirs = []
    for d in base_dir.iterdir():
        if d.is_dir() and list(d.glob("checkpoint_*.pt")):
            checkpoint_dirs.append(d)

    if not checkpoint_dirs:
        return None

    # Sort by modification time (most recent first)
    checkpoint_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoint_dirs[0]


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Find the checkpoint with the highest update count.

    Args:
        checkpoint_dir: Directory containing checkpoint_*.pt files

    Returns:
        Path to the latest checkpoint file, or None if not found
    """
    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
        reverse=True,
    )
    return checkpoints[0] if checkpoints else None


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


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, int, str]:
    """Load a model from a checkpoint (auto-detects DDQN or PPO).

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on

    Returns:
        Tuple of (model, action_history_len, model_type)
    """
    # Load checkpoint first to detect architecture
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle both full checkpoint and weights-only formats
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Detect model type
    model_type = detect_model_type(state_dict)
    print(f"Detected model type: {model_type.upper()}")

    if model_type == "ppo":
        if not PPO_AVAILABLE:
            raise ImportError("PPONetwork not available. Run from project root.")

        # Auto-detect action_history_len from PPO backbone fc2 weight
        fc2_key = "backbone.fc2.weight"
        action_history_len = 0
        if fc2_key in state_dict:
            fc2_in = state_dict[fc2_key].shape[1]
            # fc2_in = 512 + action_history_len * 7
            action_history_len = (fc2_in - 512) // 7
            print(f"Detected: action_history_len={action_history_len}")

        model = PPONetwork(
            input_shape=(4, 64, 64),
            num_actions=7,
            feature_dim=512,
            dropout=0.1,
            action_history_len=action_history_len,
        )
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

    else:  # DDQN
        # Auto-detect action_history_len from fc2 weight shape
        fc2_key = "online.backbone.fc2.weight"
        action_history_len = 0
        if fc2_key in state_dict:
            fc2_in = state_dict[fc2_key].shape[1]
            # fc2_in = 512 + action_history_len * 7
            action_history_len = (fc2_in - 512) // 7

        # Auto-detect danger_prediction_bins from danger_head
        danger_key = "online.danger_head.2.weight"
        danger_prediction_bins = 0
        if danger_key in state_dict:
            danger_prediction_bins = state_dict[danger_key].shape[0]

        print(f"Detected: action_history_len={action_history_len}, danger_bins={danger_prediction_bins}")

        # Create model with detected architecture
        model = DoubleDQN(  # type: ignore[assignment]
            input_shape=(4, 64, 64),
            num_actions=7,
            feature_dim=512,
            hidden_dim=256,
            dropout=0.1,
            action_history_len=action_history_len,
            danger_prediction_bins=danger_prediction_bins,
        )
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

    # Print checkpoint info
    if "model_state_dict" in checkpoint:
        update_count = checkpoint.get("update_count", "unknown")
        total_steps = checkpoint.get("total_steps", "unknown")
        print(f"Loaded checkpoint: update={update_count}, steps={total_steps}")
    else:
        print("Loaded weights file")

    return model, action_history_len, model_type


def preprocess_state(state: np.ndarray) -> np.ndarray:
    """Preprocess state: squeeze extra channel dimension if present.

    Converts (4, 64, 64, 1) -> (4, 64, 64) for frame-stacked grayscale.
    This matches the preprocessing in TrainingWorker._preprocess_state.
    """
    if state.ndim == 4 and state.shape[-1] == 1:
        state = np.squeeze(state, axis=-1)
    return state


def parse_level(level_str: str) -> tuple[int, int] | str:
    """Parse level string to tuple or special mode.

    Args:
        level_str: 'random', 'sequential', or 'W,S' (e.g. '1,1')

    Returns:
        (world, stage) tuple or 'random'/'sequential' string
    """
    if level_str in ("random", "sequential"):
        return level_str
    try:
        parts = level_str.split(",")
        return (int(parts[0]), int(parts[1]))
    except (ValueError, IndexError):
        return (1, 1)


@click.command()
@click.option(
    "--checkpoint-dir",
    "-c",
    type=click.Path(exists=False, path_type=Path),
    default=None,
    help="Checkpoint directory. If not specified, uses latest in checkpoints/",
)
@click.option(
    "--checkpoint-file",
    "-f",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Specific checkpoint file to load (overrides --checkpoint-dir)",
)
@click.option(
    "--level",
    "-l",
    default="1,1",
    help="Level: 'random', 'sequential', or 'W,S' (e.g. '1,1')",
)
@click.option(
    "--episodes",
    "-n",
    default=3,
    help="Number of episodes to play",
)
@click.option(
    "--epsilon",
    "-e",
    default=0.1,
    help="Exploration rate for DDQN (default 0.1). Ignored for PPO (uses stochastic policy).",
)
@click.option(
    "--fps",
    default=30,
    help="Target frames per second for playback",
)
@click.option(
    "--device",
    "-d",
    default=None,
    help="Device to use (cpu, cuda, mps). Auto-detected if not specified.",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Print debug info (observation shape, Q-values, etc.)",
)
@click.option(
    "--no-render",
    is_flag=True,
    help="Disable rendering (for performance testing)",
)
@click.option(
    "--track-entities",
    is_flag=True,
    help="Enable motion-based entity tracking overlay (shows detected enemies)",
)
@click.option(
    "--entity-window-scale",
    default=2,
    help="Scale factor for entity tracking window (default 2x)",
)
def main(
    checkpoint_dir: Path | None,
    checkpoint_file: Path | None,
    level: str,
    episodes: int,
    epsilon: float,
    fps: int,
    device: str | None,
    debug: bool,
    no_render: bool,
    track_entities: bool,
    entity_window_scale: int,
) -> None:
    """Play Mario using a trained checkpoint (DDQN or PPO)."""

    # Determine checkpoint to load
    if checkpoint_file is not None:
        ckpt_path = checkpoint_file
    else:
        # Find checkpoint directory
        if checkpoint_dir is None:
            base_dir = Path("checkpoints")
            checkpoint_dir = find_latest_checkpoint_dir(base_dir)
            if checkpoint_dir is None:
                print("Error: No checkpoint directories found in checkpoints/")
                print("Please specify a checkpoint with --checkpoint-dir or --checkpoint-file")
                sys.exit(1)
            print(f"Using latest checkpoint directory: {checkpoint_dir}")

        if not checkpoint_dir.exists():
            print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
            sys.exit(1)

        # Find latest checkpoint in directory
        ckpt_path = find_latest_checkpoint(checkpoint_dir)  # type: ignore[assignment]
        if ckpt_path is None:
            print(f"Error: No checkpoint files found in {checkpoint_dir}")
            sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")

    # Setup device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    torch_device = torch.device(device)
    print(f"Using device: {torch_device}")

    # Load model
    model, action_history_len, model_type = load_model(ckpt_path, torch_device)

    # Parse level
    level_spec = parse_level(level)
    print(f"Level: {level_spec}")
    if model_type == "ddqn":
        print(f"Epsilon: {epsilon}")
    else:
        print("Policy: Stochastic (PPO sampling)")
        if epsilon > 0:
            print("  Note: --epsilon is ignored for PPO models")
    print(f"Episodes: {episodes}")
    print()

    # Create environment
    render = not no_render
    env = create_mario_env(level=level_spec, render_frames=render, lz4_compress=False, action_history_len=action_history_len)  # type: ignore[arg-type]
    if no_render:
        print("Rendering disabled (performance test mode)")

    # Create entity tracker if enabled
    entity_tracker: MotionEntityTracker | None = None
    if track_entities:
        # Uses phase correlation to measure TRUE background scroll.
        # Only objects moving differently than background are detected.
        entity_tracker = MotionEntityTracker(
            history_len=4,
            motion_threshold=25,
            min_blob_area=30,  # Enemies are at least ~6x6 pixels
            max_blob_area=800,  # Large enemies like Bowser ~25x32
            mario_exclusion_width=40,
            mario_screen_x=80,  # Mario is typically slightly left of center
            edge_margin=8,  # Small edge margin for scroll boundary
            max_aspect_ratio=4.0,  # Filter noise artifacts
            tracker_type="kalman",  # Particle filter with acceleration model
            n_particles=50,  # Number of particles per tracked entity
        )
        cv2.namedWindow("Entity Tracking", cv2.WINDOW_NORMAL)
        print("Entity tracking enabled:")
        print("  'p' = pause/resume, 'q' = quit")
        print("  Using phase correlation + particle filter with acceleration model")

    frame_delay = 1.0 / fps

    try:
        for episode in range(1, episodes + 1):
            obs, info = env.reset()

            # Initialize action history for models that use it
            action_history = None
            if action_history_len > 0:
                action_history = torch.zeros(1, action_history_len, 7, device=torch_device)

            # Reset entity tracker on new episode
            if entity_tracker is not None:
                entity_tracker.reset()

            # Debug first observation
            if debug and episode == 1:
                print(f"  [DEBUG] Observation type: {type(obs)}")
                print(f"  [DEBUG] Observation shape: {obs.shape}")
                print(f"  [DEBUG] Observation dtype: {obs.dtype}")
                print(f"  [DEBUG] Observation range: [{obs.min()}, {obs.max()}]")

            done = False
            total_reward = 0.0
            steps = 0
            start_time = time.time()

            print(f"Episode {episode}/{episodes}")

            while not done:
                # Preprocess state (same as training)
                state = preprocess_state(np.asarray(obs))

                # Convert to tensor - model expects uint8 [0, 255]
                state_t = torch.from_numpy(state).unsqueeze(0).to(torch_device)

                # Select action based on model type
                if model_type == "ppo":
                    # PPO: Sample from policy distribution
                    with torch.no_grad():
                        policy_logits, value = model.forward(state_t, action_history)
                        probs = torch.softmax(policy_logits/2.5, dim=-1)
                        dist = torch.distributions.Categorical(probs)
                        action = int(dist.sample().item())

                        # Debug policy periodically
                        if debug and steps % 100 == 0:
                            probs_np = probs.cpu().numpy()[0]
                            print(f"  [DEBUG] Step {steps}: Policy={probs_np}, Value={value.item():.2f}, action={action}")
                else:
                    # DDQN: Epsilon-greedy
                    if np.random.random() < epsilon:
                        # Random action (exploration)
                        action = int(np.random.randint(0, model.num_actions))  # type: ignore[arg-type]
                    else:
                        # Greedy action (same as training: direct forward + argmax)
                        with torch.no_grad():
                            q_values = model(state_t)  # Uses online network
                            action = int(q_values.argmax(dim=1).item())

                            # Debug Q-values periodically
                            if debug and steps % 100 == 0:
                                q_np = q_values.cpu().numpy()[0]
                                print(f"  [DEBUG] Step {steps}: Q={q_np}, action={action}")

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1

                # Update action history if enabled
                if action_history is not None and action_history_len > 0:
                    # Shift left and add new action as one-hot
                    action_onehot = torch.zeros(1, 1, 7, device=torch_device)
                    action_onehot[0, 0, action] = 1.0
                    action_history = torch.cat([action_history[:, 1:, :], action_onehot], dim=1)

                # Entity tracking overlay
                if entity_tracker is not None:
                    # Get raw RGB frame from the underlying NES environment
                    # base_env is SuperMarioBrosMultiLevel, which wraps a MarioBrosLevel
                    raw_frame = env.base_env.env.screen  # type: ignore[union-attr]

                    # Update tracker with frame and Mario's position
                    mario_x = info.get("x_pos", 0)
                    mario_y = info.get("y_pos", 150)  # Screen Y position
                    entities = entity_tracker.update(raw_frame, mario_x, mario_y)

                    # Create display frame (scale up for visibility)
                    display_frame = cv2.resize(
                        raw_frame,
                        (raw_frame.shape[1] * entity_window_scale, raw_frame.shape[0] * entity_window_scale),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    # Convert RGB to BGR for OpenCV
                    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)

                    # Draw entity overlays (with particle spread for uncertainty)
                    render_entity_overlay(
                        display_frame,
                        entities,
                        mario_screen_x=80,
                        mario_screen_y=mario_y,
                        scale=entity_window_scale,
                        tracker_pool=entity_tracker.tracker_pool,
                        show_particle_spread=True,
                    )
                    render_danger_hud(
                        display_frame,
                        entities,
                        mario_screen_x=80,
                        hud_y=25,
                    )

                    # Show entity count
                    cv2.putText(
                        display_frame,
                        f"Entities: {len(entities)} | x={mario_x}",
                        (10, display_frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )

                    cv2.imshow("Entity Tracking", display_frame)
                    key = cv2.waitKey(1)
                    if key == ord("q"):
                        raise KeyboardInterrupt
                    elif key == ord("p"):
                        # Pause - wait for 'p' again to resume
                        cv2.putText(
                            display_frame,
                            "PAUSED - Press 'p' to resume",
                            (10, display_frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2,
                        )
                        cv2.imshow("Entity Tracking", display_frame)
                        while True:
                            pause_key = cv2.waitKey(100)
                            if pause_key == ord("p"):
                                break
                            elif pause_key == ord("q"):
                                raise KeyboardInterrupt

                # Render at target FPS (only if rendering enabled)
                if render:
                    time.sleep(max(0, frame_delay - (time.time() - start_time) % frame_delay))

            # Episode stats
            elapsed = time.time() - start_time
            x_pos = info.get("x_pos", 0)
            flag_get = info.get("flag_get", False)
            world = info.get("world", 1)
            stage = info.get("stage", 1)

            status = "🎉 FLAG!" if flag_get else "💀 DIED" if not info.get("is_timeout", False) else "⏰ TIMEOUT"
            print(f"  {status} | x={x_pos:4d} | reward={total_reward:7.1f} | steps={steps:4d} | time={elapsed:.1f}s")
            print(f"  Level: {world}-{stage}")
            print()

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        env.close()
        if track_entities:
            cv2.destroyAllWindows()

    print("Done!")


if __name__ == "__main__":
    main()
