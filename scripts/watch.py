#!/usr/bin/env python
"""
Watch a trained agent play Super Mario Bros.

Usage:
    # Watch world model agent:
    uv run python watch.py agents/test/agent.pt --world-model

    # Watch standard DQN agent:
    uv run python watch.py agents/dueling_ddqn/agent.pt

    # Watch on a specific level:
    uv run python watch.py agents/test/agent.pt --world-model --level 1-2
"""

import os
import argparse
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Fix missing 'key' import in nes_py._image_viewer (needs display)
try:
    from pyglet.window import key
    import nes_py._image_viewer as _iv

    _iv.key = key
except Exception as e:
    print("ERROR: Cannot initialize display for rendering.")
    print("This script requires a display (X server) to show the game.")
    print(f"Error: {e}")
    print("\nIf running on a headless server:")
    print("  - Use Xvfb: xvfb-run -a uv run mario-watch <args>")
    print("  - Or set up VNC/X11 forwarding")
    import sys

    sys.exit(1)

import torch
import numpy as np
from gymnasium.spaces import Box
from nes_py.wrappers import JoypadSpace
from gymnasium.wrappers import GrayscaleObservation
from gymnasium.wrappers import TransformObservation
from gymnasium.wrappers import FrameStackObservation
from gym_super_mario_bros import actions as smb_actions

from mario_rl.agent.neural import DuelingDDQNNet
from mario_rl.agent.world_model import LatentDDQN
from mario_rl.environment.wrappers import SkipFrame
from mario_rl.agent.world_model import MarioWorldModel
from mario_rl.environment.wrappers import ResizeObservation
from mario_rl.environment.mariogym import SuperMarioBrosMultiLevel


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def create_env(level: tuple[int, int] = (1, 1), render: bool = True):
    """Create wrapped Mario environment."""
    base_env = SuperMarioBrosMultiLevel(level=level)  # type: ignore[arg-type]
    env = JoypadSpace(base_env, actions=smb_actions.COMPLEX_MOVEMENT)
    env = SkipFrame(env, skip=4, render_frames=render)
    env = GrayscaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, shape=64)
    env = TransformObservation(
        env,
        func=lambda x: x / 255.0,
        observation_space=Box(low=0.0, high=1.0, shape=(64, 64, 1), dtype=np.float32),
    )
    fstack = FrameStackObservation(env, stack_size=4)
    return fstack, base_env


def load_world_model_agent(
    weights_path: Path, device: str, latent_dim: int = 128
) -> tuple[MarioWorldModel, LatentDDQN]:
    """Load world model and latent Q-network from checkpoint."""
    state_dim = (4, 64, 64, 1)
    action_dim = 12

    world_model = MarioWorldModel(
        frame_shape=state_dim,
        num_actions=action_dim,
        latent_dim=latent_dim,
    ).to(device)

    q_network = LatentDDQN(
        latent_dim=latent_dim,
        num_actions=action_dim,
    ).to(device)

    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    world_model.load_state_dict(checkpoint["world_model"])
    q_network.load_state_dict(checkpoint["q_network"])

    world_model.eval()
    q_network.eval()

    return world_model, q_network


def load_dqn_agent(weights_path: Path, device: str) -> DuelingDDQNNet:
    """Load standard DQN agent from checkpoint."""
    state_dim = (4, 64, 64, 1)
    action_dim = 12
    hidden_dim = 512

    net = DuelingDDQNNet(state_dim, action_dim, hidden_dim).to(device)

    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        net.load_state_dict(checkpoint["model"])
    else:
        net.load_state_dict(checkpoint)

    net.eval()
    return net  # type: ignore[no-any-return]


@torch.no_grad()
def select_action_world_model(
    state: np.ndarray,
    world_model: MarioWorldModel,
    q_network: LatentDDQN,
    device: str,
) -> tuple[int, float, float]:
    """Select action using world model encoder + latent Q-network."""
    state_tensor = torch.from_numpy(np.expand_dims(state, 0)).to(device)
    z = world_model.encode(state_tensor, deterministic=True)
    q_values = q_network.online(z)
    action = q_values.argmax(dim=-1).item()
    return action, float(q_values.mean()), float(q_values.max())


@torch.no_grad()
def select_action_dqn(state: np.ndarray, net: DuelingDDQNNet, device: str) -> tuple[int, float, float]:
    """Select action using standard DQN."""
    state_tensor = torch.from_numpy(np.expand_dims(state, 0)).to(device)
    q_values = net.online(state_tensor)
    action = q_values.argmax(dim=-1).item()
    return action, float(q_values.mean()), float(q_values.max())


def watch(
    weights_path: Path,
    level: tuple[int, int] = (1, 1),
    use_world_model: bool = False,
    latent_dim: int = 128,
    num_episodes: int = 5,
):
    """Watch the agent play."""
    device = best_device()
    print(f"Using device: {device}")
    print(f"Loading weights from: {weights_path}")

    # Load agent
    if use_world_model:
        print("Loading world model agent...")
        world_model, q_network = load_world_model_agent(weights_path, device, latent_dim)

        def select_fn(s):
            return select_action_world_model(s, world_model, q_network, device)
    else:
        print("Loading standard DQN agent...")
        net = load_dqn_agent(weights_path, device)

        def select_fn(s):
            return select_action_dqn(s, net, device)

    # Create environment
    print(f"Creating environment for level {level[0]}-{level[1]}...")
    env, base_env = create_env(level=level, render=True)

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        step = 0
        max_x = 0

        print(f"\n{'='*50}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*50}")

        while True:
            # Render
            base_env.render()

            # Select action
            action, q_mean, q_max = select_fn(state)

            # Debug: print first few actions
            if step < 10:
                print(f"  [DEBUG] Step {step}: action={action}, Q_mean={q_mean:.3f}, Q_max={q_max:.3f}")

            # Step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            step += 1
            max_x = max(max_x, info.get("x_pos", 0))

            # Print progress occasionally
            if step % 100 == 0:
                print(
                    f"  Step {step:4d} | x={info.get('x_pos', 0):4d} | "
                    f"Q={q_mean:6.2f}/{q_max:6.2f} | reward={total_reward:.0f}"
                )

            state = next_state

            if done:
                flag_get = info.get("flag_get", False)
                status = "🏁 FLAG!" if flag_get else "💀 DIED"
                print(f"\n{status} | Steps: {step} | Max X: {max_x} | " f"Total Reward: {total_reward:.0f}")
                break

    try:
        env.close()
    except Exception:
        pass  # Ignore close errors from nes_py
    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description="Watch trained Mario agent play")
    parser.add_argument("weights", type=Path, help="Path to weights file (agent.pt)")
    parser.add_argument("--world-model", action="store_true", help="Use world model agent")
    parser.add_argument(
        "--level",
        type=str,
        default="1-1",
        help="Level to play (e.g., 1-1, 1-2, 2-1)",
    )
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent dimension (for world model)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to watch")

    args = parser.parse_args()

    # Parse level
    world, stage = args.level.split("-")
    level = (int(world), int(stage))

    watch(
        weights_path=args.weights,
        level=level,
        use_world_model=args.world_model,
        latent_dim=args.latent_dim,
        num_episodes=args.episodes,
    )


if __name__ == "__main__":
    main()
