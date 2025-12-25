#!/usr/bin/env python3
"""
Plot training metrics from CSV logs.

Usage:
    uv run python scripts/plot_training.py checkpoints/distributed_YYYY-MM-DD/
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_worker_episodes(checkpoint_dir: Path) -> pd.DataFrame:
    """Load and combine episode data from all workers."""
    episode_files = list(checkpoint_dir.glob("worker_*_episodes.csv"))
    if not episode_files:
        print(f"No worker episode CSVs found in {checkpoint_dir}")
        return pd.DataFrame()

    dfs = []
    for f in episode_files:
        df = pd.read_csv(f)
        # Extract worker ID from filename
        worker_id = int(f.stem.split("_")[1])
        df["worker_id"] = worker_id
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values("timestamp")

    # Convert timestamp to relative time
    if len(combined) > 0:
        combined["time_minutes"] = (combined["timestamp"] - combined["timestamp"].min()) / 60

    return combined


def load_learner_metrics(checkpoint_dir: Path) -> pd.DataFrame:
    """Load learner metrics."""
    metrics_file = checkpoint_dir / "learner_metrics.csv"
    if not metrics_file.exists():
        print(f"No learner metrics CSV found: {metrics_file}")
        return pd.DataFrame()

    df = pd.read_csv(metrics_file)

    # Convert timestamp to relative time
    if len(df) > 0:
        df["time_minutes"] = (df["timestamp"] - df["timestamp"].min()) / 60

    return df


def plot_training(checkpoint_dir: Path, output_file: Path | None = None):
    """Generate training plots."""
    episodes_df = load_worker_episodes(checkpoint_dir)
    learner_df = load_learner_metrics(checkpoint_dir)

    if episodes_df.empty and learner_df.empty:
        print("No data to plot!")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Training Progress: {checkpoint_dir.name}", fontsize=14)

    # Plot 1: Average Reward Over Time (rolling average)
    ax1 = axes[0, 0]
    if not episodes_df.empty:
        # Group by time bins (1 minute) and calculate mean
        episodes_df["time_bin"] = (episodes_df["time_minutes"] // 1).astype(int)
        episodes_df.groupby("time_bin")["reward"].mean()

        # Also calculate rolling average
        rolling_reward = episodes_df["reward"].rolling(window=100, min_periods=1).mean()

        ax1.plot(episodes_df["time_minutes"], episodes_df["reward"], alpha=0.3, label="Raw", color="blue")
        ax1.plot(episodes_df["time_minutes"], rolling_reward, linewidth=2, label="Rolling Avg (100)", color="red")

        ax1.axhline(y=0, color="green", linestyle="--", alpha=0.5, label="Break-even")
        ax1.set_xlabel("Time (minutes)")
        ax1.set_ylabel("Episode Reward")
        ax1.set_title("Episode Reward Over Time")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # Plot 2: X Position Progress
    ax2 = axes[0, 1]
    if not episodes_df.empty:
        ax2.plot(episodes_df["time_minutes"], episodes_df["x_pos"], alpha=0.3, label="Episode X", color="blue")
        rolling_x = episodes_df["x_pos"].rolling(window=50, min_periods=1).mean()
        ax2.plot(episodes_df["time_minutes"], rolling_x, linewidth=2, label="Rolling Avg (50)", color="orange")
        ax2.plot(episodes_df["time_minutes"], episodes_df["best_x"], linewidth=1, label="Best X", color="green")

        ax2.set_xlabel("Time (minutes)")
        ax2.set_ylabel("X Position")
        ax2.set_title("X Position Progress")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # Plot 3: Loss and Q-values (from learner)
    ax3 = axes[1, 0]
    if not learner_df.empty:
        ax3.plot(learner_df["time_minutes"], learner_df["loss"], alpha=0.5, label="Loss", color="red")
        rolling_loss = learner_df["loss"].rolling(window=50, min_periods=1).mean()
        ax3.plot(learner_df["time_minutes"], rolling_loss, linewidth=2, label="Rolling Avg", color="darkred")

        ax3.set_xlabel("Time (minutes)")
        ax3.set_ylabel("Loss")
        ax3.set_title("Training Loss")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Plot 4: Steps to Flag / Flag Timeline
    ax4 = axes[1, 1]
    if not episodes_df.empty:
        # Find flag events (where flags increased)
        flag_events = episodes_df[episodes_df["flags"] > episodes_df["flags"].shift(1).fillna(0)]

        if len(flag_events) > 0:
            # Plot cumulative flags over time
            ax4.step(episodes_df["time_minutes"], episodes_df["flags"], where="post", linewidth=2, color="green")
            ax4.scatter(flag_events["time_minutes"], flag_events["flags"], color="gold", s=100, zorder=5, marker="★")

            # Annotate first flag time
            first_flag = flag_events.iloc[0] if len(flag_events) > 0 else None
            if first_flag is not None:
                ax4.annotate(
                    f"First flag: {first_flag['time_minutes']:.1f} min",
                    xy=(first_flag["time_minutes"], first_flag["flags"]),
                    xytext=(first_flag["time_minutes"] + 2, first_flag["flags"] + 0.5),
                    fontsize=10,
                    arrowprops={"arrowstyle": "->"},
                )

            ax4.set_xlabel("Time (minutes)")
            ax4.set_ylabel("Total Flags Captured")
            ax4.set_title("Flag Captures Over Time")
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(
                0.5,
                0.5,
                "No flags captured yet",
                ha="center",
                va="center",
                transform=ax4.transAxes,
                fontsize=14,
            )
            ax4.set_title("Flag Captures (none yet)")

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_file}")
    else:
        plt.show()


def print_summary(checkpoint_dir: Path):
    """Print training summary statistics."""
    episodes_df = load_worker_episodes(checkpoint_dir)
    learner_df = load_learner_metrics(checkpoint_dir)

    print("\n" + "=" * 60)
    print(f"Training Summary: {checkpoint_dir.name}")
    print("=" * 60)

    if not episodes_df.empty:
        total_episodes = len(episodes_df)
        total_time = episodes_df["time_minutes"].max() if len(episodes_df) > 0 else 0
        total_flags = episodes_df["flags"].max() if len(episodes_df) > 0 else 0

        # Recent performance (last 100 episodes)
        recent = episodes_df.tail(100)
        recent_avg_reward = recent["reward"].mean()
        recent_avg_x = recent["x_pos"].mean()
        best_x_ever = episodes_df["x_pos"].max()

        # First flag time
        first_flag_time = episodes_df[episodes_df["first_flag_time"] > 0]["first_flag_time"].min()

        print("\n📊 Episode Stats:")
        print(f"   Total Episodes:     {total_episodes:,}")
        print(f"   Training Time:      {total_time:.1f} minutes")
        print(f"   Total Flags:        {total_flags}")

        print("\n📈 Recent Performance (last 100 eps):")
        print(f"   Avg Reward:         {recent_avg_reward:,.0f}")
        print(f"   Avg X Position:     {recent_avg_x:,.0f}")

        print("\n🏆 Best Results:")
        print(f"   Best X Position:    {best_x_ever:,}")
        if not np.isnan(first_flag_time) and first_flag_time > 0:
            print(f"   First Flag Time:    {first_flag_time:.1f} seconds")
        else:
            print("   First Flag Time:    Not achieved yet")

    if not learner_df.empty:
        print("\n🧠 Learner Stats:")
        print(f"   Training Steps:     {learner_df['step'].max():,}")
        print(f"   Final Loss:         {learner_df['loss'].iloc[-1]:.2f}")
        print(f"   Avg Q-mean:         {learner_df['q_mean'].mean():.2f}")

    print("=" * 60 + "\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/plot_training.py <checkpoint_dir> [--save]")
        print("Example: python scripts/plot_training.py checkpoints/distributed_2025-12-25/")
        sys.exit(1)

    checkpoint_dir = Path(sys.argv[1])
    if not checkpoint_dir.exists():
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)

    # Print summary
    print_summary(checkpoint_dir)

    # Determine output file
    save_plot = "--save" in sys.argv
    output_file = checkpoint_dir / "training_plot.png" if save_plot else None

    # Generate plot
    plot_training(checkpoint_dir, output_file)


if __name__ == "__main__":
    main()
