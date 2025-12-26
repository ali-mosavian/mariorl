"""
Simple synchronous PPO implementation.

Single-process, no distributed training, no async updates.
Just the core PPO algorithm:

    ┌─────────────────────────────────────────────┐
    │               Training Loop                 │
    │  ┌─────────┐   ┌─────────┐   ┌──────────┐  │
    │  │ Collect │ → │ Compute │ → │  Update  │  │
    │  │ Rollout │   │   GAE   │   │  Policy  │  │
    │  └─────────┘   └─────────┘   └──────────┘  │
    │       ↑                            │        │
    │       └────────────────────────────┘        │
    └─────────────────────────────────────────────┘

Usage:
    uv run mario-train-simple --level 1,1 --total-steps 1000000
"""

import time
import argparse
from typing import Tuple

import torch
import numpy as np
from torch.optim import Adam

from mario_rl.agent.ppo_net import ActorCritic
from mario_rl.environment.env_factory import LevelType
from mario_rl.environment.env_factory import make_mario_env


def parse_level(level_str: str) -> LevelType:
    """Parse level string like '1,1' or 'random'."""
    if level_str == "random":
        return "random"
    if level_str == "sequential":
        return "sequential"
    parts = level_str.split(",")
    if len(parts) == 2:
        world = int(parts[0])
        stage = int(parts[1])
        # Validate ranges
        if 1 <= world <= 8 and 1 <= stage <= 4:
            return (world, stage)  # type: ignore[return-value]
    raise ValueError(f"Invalid level: {level_str}")


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Generalized Advantage Estimation.

    Args:
        rewards: Rewards (n_steps,)
        values: Value estimates (n_steps,)
        dones: Episode done flags (n_steps,)
        last_value: Bootstrap value for last state
        gamma: Discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        advantages: GAE advantages (n_steps,)
        returns: Returns = advantages + values (n_steps,)
    """
    n_steps = len(rewards)
    advantages = np.zeros(n_steps, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            next_value = last_value
            next_non_terminal = 1.0 - float(dones[t])
        else:
            next_value = values[t + 1]
            next_non_terminal = 1.0 - float(dones[t])

        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple synchronous PPO")
    parser.add_argument("--level", "-l", type=str, default="1,1", help="Level (e.g., '1,1' or 'random')")
    parser.add_argument("--total-steps", type=int, default=1_000_000, help="Total training steps")
    parser.add_argument("--n-steps", type=int, default=128, help="Steps per rollout")
    parser.add_argument("--n-epochs", type=int, default=4, help="PPO epochs per rollout")
    parser.add_argument("--minibatch-size", type=int, default=32, help="Minibatch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument(
        "--ent-coef", type=float, default=0.05, help="Entropy coefficient (try 0.1 if entropy collapses)"
    )
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Max gradient norm")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate (0 to disable)")
    args = parser.parse_args()

    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Create environment
    level = parse_level(args.level)
    env = make_mario_env(level=level, render_mode=None)
    obs_shape = env.observation_space.shape  # (4, 64, 64)
    n_actions = env.action_space.n
    print(f"Observation shape: {obs_shape}, Actions: {n_actions}")

    # Create network
    model = ActorCritic(
        input_shape=obs_shape,
        num_actions=n_actions,
        feature_dim=512,
        dropout=args.dropout,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Rollout storage
    obs_buffer = np.zeros((args.n_steps, *obs_shape), dtype=np.float32)
    actions_buffer = np.zeros(args.n_steps, dtype=np.int64)
    rewards_buffer = np.zeros(args.n_steps, dtype=np.float32)
    dones_buffer = np.zeros(args.n_steps, dtype=np.float32)
    values_buffer = np.zeros(args.n_steps, dtype=np.float32)
    log_probs_buffer = np.zeros(args.n_steps, dtype=np.float32)

    # Training state
    obs, _ = env.reset()
    obs = np.array(obs, dtype=np.float32)
    total_steps = 0
    episode_count = 0
    episode_reward = 0.0
    episode_length = 0
    best_x = 0
    current_x = 0
    recent_rewards: list[float] = []
    start_time = time.time()

    # Episode stats
    total_deaths = 0
    total_flags = 0
    recent_x_at_death: list[int] = []  # X position when dying (to track progress)
    recent_times_to_flag: list[int] = []  # Game time remaining when getting flag

    print("=" * 70)
    print("Starting training...")
    print(f"  Level: {args.level}")
    print(f"  Device: {device}")
    print(f"  LR: {args.lr}, Entropy: {args.ent_coef}, Clip: {args.clip_range}")
    print("=" * 70)

    # Debug: test reward function with a few random actions
    print("Testing reward function (10 random steps):")
    test_obs, _ = env.reset()
    for i in range(10):
        action = env.action_space.sample()
        _, reward, terminated, truncated, info = env.step(action)
        x_pos = info.get("x_pos", 0)
        print(f"  Step {i}: action={action}, reward={reward:.4f}, x_pos={x_pos}")
        if terminated or truncated:
            test_obs, _ = env.reset()
    # Reset for real training
    obs, _ = env.reset()
    obs = np.array(obs, dtype=np.float32)
    print("=" * 70)

    while total_steps < args.total_steps:
        # Collect rollout
        model.eval()
        for step in range(args.n_steps):
            obs_buffer[step] = obs

            with torch.no_grad():
                obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
                action, log_prob, _, value = model.get_action_and_value(obs_t)
                action = action.cpu().numpy()[0]
                log_prob = log_prob.cpu().numpy()[0]
                value = value.cpu().numpy()[0]

            actions_buffer[step] = action
            log_probs_buffer[step] = log_prob
            values_buffer[step] = value

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_obs = np.array(next_obs, dtype=np.float32)

            rewards_buffer[step] = reward
            dones_buffer[step] = float(done)

            episode_reward += reward
            episode_length += 1

            # Track positions
            x_pos = info.get("x_pos", 0)
            current_x = x_pos
            if x_pos > best_x:
                best_x = x_pos

            if done:
                recent_rewards.append(episode_reward)
                if len(recent_rewards) > 100:
                    recent_rewards.pop(0)

                # Track deaths and flags
                flag_get = info.get("flag_get", False)
                is_dead = info.get("is_dead", False) or info.get("is_dying", False)
                game_time = info.get("time", 0)

                if flag_get:
                    total_flags += 1
                    recent_times_to_flag.append(game_time)
                    if len(recent_times_to_flag) > 20:
                        recent_times_to_flag.pop(0)
                elif is_dead:
                    total_deaths += 1
                    recent_x_at_death.append(x_pos)
                    if len(recent_x_at_death) > 20:
                        recent_x_at_death.pop(0)

                episode_count += 1
                episode_reward = 0.0
                episode_length = 0
                next_obs, _ = env.reset()
                next_obs = np.array(next_obs, dtype=np.float32)

            obs = next_obs
            total_steps += 1

        # Bootstrap value for last state
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
            last_value = model.get_value(obs_t).cpu().numpy()[0]

        # Compute advantages
        advantages, returns = compute_gae(
            rewards_buffer,
            values_buffer,
            dones_buffer,
            last_value,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        obs_t = torch.from_numpy(obs_buffer).to(device)
        actions_t = torch.from_numpy(actions_buffer).to(device)
        old_log_probs_t = torch.from_numpy(log_probs_buffer).to(device)
        advantages_t = torch.from_numpy(advantages).to(device)
        returns_t = torch.from_numpy(returns).to(device)

        # PPO update
        model.train()
        indices = np.arange(args.n_steps)

        for _ in range(args.n_epochs):
            np.random.shuffle(indices)

            for start in range(0, args.n_steps, args.minibatch_size):
                end = start + args.minibatch_size
                mb_indices = indices[start:end]

                mb_obs = obs_t[mb_indices]
                mb_actions = actions_t[mb_indices]
                mb_old_log_probs = old_log_probs_t[mb_indices]
                mb_advantages = advantages_t[mb_indices]
                mb_returns = returns_t[mb_indices]

                # Get current policy outputs
                _, new_log_probs, entropy, new_values = model.get_action_and_value(mb_obs, mb_actions)

                # Compute ratio
                log_ratio = new_log_probs - mb_old_log_probs
                ratio = torch.exp(log_ratio)

                # Clipped surrogate loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_range, 1 + args.clip_range)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (clipped to prevent explosion)
                v_loss_unclipped = (new_values - mb_returns) ** 2
                v_loss = 0.5 * torch.clamp(v_loss_unclipped, max=100.0).mean()

                # Entropy loss (negative because we want to maximize entropy)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = pg_loss + args.vf_coef * v_loss + args.ent_coef * entropy_loss

                # Early stop this epoch if entropy is too low (prevents collapse)
                if entropy.mean().item() < 0.01:
                    break

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

        # Logging
        elapsed = time.time() - start_time
        steps_per_sec = total_steps / elapsed if elapsed > 0 else 0
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
        avg_x_at_death = np.mean(recent_x_at_death) if recent_x_at_death else 0.0
        avg_time_to_flag = np.mean(recent_times_to_flag) if recent_times_to_flag else 0.0

        # Compute final metrics for logging
        with torch.no_grad():
            _, new_log_probs, entropy, new_values = model.get_action_and_value(obs_t, actions_t)
            log_ratio = new_log_probs - old_log_probs_t
            ratio = torch.exp(log_ratio)
            kl = ((ratio - 1) - log_ratio).mean().item()
            clip_frac = ((ratio - 1).abs() > args.clip_range).float().mean().item()

        # Two-line output for readability
        print(
            f"Step {total_steps:>7,} | "
            f"Ep: {episode_count:>4} | "
            f"π: {pg_loss.item():>7.4f} | "
            f"v: {v_loss.item():>6.4f} | "
            f"H: {entropy.mean().item():>5.3f} | "
            f"KL: {kl:>6.4f} | "
            f"Clip: {clip_frac:>4.2f}"
        )
        print(
            f"         Avg R: {avg_reward:>6.1f} | "
            f"X: {current_x:>4} | "
            f"Best: {best_x:>4} | "
            f"Deaths: {total_deaths:>4} (avg X: {avg_x_at_death:>4.0f}) | "
            f"Flags: {total_flags:>3} (avg T: {avg_time_to_flag:>3.0f}) | "
            f"SPS: {steps_per_sec:>4.0f}"
        )
        print("-" * 70)

    print("=" * 70)
    print("Training complete!")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Episodes: {episode_count}")
    print(f"  Best X position: {best_x}")
    print(f"  Total deaths: {total_deaths}")
    print(f"  Total flags: {total_flags}")
    if total_flags > 0:
        print(f"  Flag rate: {total_flags / episode_count * 100:.1f}%")
    print("=" * 70)

    env.close()


if __name__ == "__main__":
    main()
