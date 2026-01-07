#!/usr/bin/env python3
"""
Real-time training dashboard for DDQN distributed training.

Run with:
    uv run streamlit run scripts/training_dashboard.py -- --checkpoint-dir checkpoints/ddqn_dist_YYYY-MM-DDTHH-MM-SS

Or auto-detect latest:
    uv run streamlit run scripts/training_dashboard.py
"""

import time
from pathlib import Path
from datetime import datetime

import click
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Page config
st.set_page_config(
    page_title="Mario RL Training Dashboard",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark theme styling
st.markdown("""
<style>
    .stMetric {
        background-color: #1e1e2e;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #313244;
    }
    .stMetric label {
        color: #cdd6f4 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #89b4fa !important;
    }
    .stMetric [data-testid="stMetricDelta"] svg {
        display: none;
    }
    div[data-testid="stSidebarContent"] {
        background-color: #181825;
    }
    .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        color: #cba6f7 !important;
    }
    .good { color: #a6e3a1 !important; }
    .warn { color: #f9e2af !important; }
    .bad { color: #f38ba8 !important; }
</style>
""", unsafe_allow_html=True)


def find_latest_checkpoint(base_dir: Path = Path("checkpoints")) -> Path | None:
    """Find the most recent checkpoint directory."""
    if not base_dir.exists():
        return None

    ddqn_dirs = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("ddqn_dist_")],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    return ddqn_dirs[0] if ddqn_dirs else None


def load_learner_metrics(checkpoint_dir: Path) -> pd.DataFrame | None:
    """Load learner metrics CSV."""
    csv_path = checkpoint_dir / "ddqn_metrics.csv"
    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path)
        if "timestamp" in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
            df["elapsed_min"] = (df["timestamp"] - df["timestamp"].iloc[0]) / 60
        return df
    except Exception as e:
        st.error(f"Error loading learner metrics: {e}")
        return None


def load_worker_episodes(checkpoint_dir: Path) -> dict[int, pd.DataFrame]:
    """Load all worker episode CSVs."""
    workers = {}
    for csv_path in checkpoint_dir.glob("ddqn_worker_*_episodes.csv"):
        try:
            worker_id = int(csv_path.stem.split("_")[2])
            df = pd.read_csv(csv_path)
            if "timestamp" in df.columns and len(df) > 0:
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
                df["elapsed_min"] = (df["timestamp"] - df["timestamp"].iloc[0]) / 60
            workers[worker_id] = df
        except Exception as e:
            st.warning(f"Error loading {csv_path.name}: {e}")
    return workers


def create_learner_charts(df: pd.DataFrame) -> None:
    """Create charts for learner metrics."""
    if df is None or len(df) < 2:
        st.info("Waiting for more learner data...")
        return

    # Loss and Q-values chart
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Loss", "Q-Values", "TD Error & Grad Norm", "Learning Rate"),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # Loss
    fig.add_trace(
        go.Scatter(x=df["elapsed_min"], y=df["loss"], name="Loss", line=dict(color="#f38ba8")),
        row=1, col=1
    )

    # Q-values
    fig.add_trace(
        go.Scatter(x=df["elapsed_min"], y=df["q_mean"], name="Q Mean", line=dict(color="#89b4fa")),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=df["elapsed_min"], y=df["q_max"], name="Q Max", line=dict(color="#74c7ec")),
        row=1, col=2
    )

    # TD Error and Grad Norm
    fig.add_trace(
        go.Scatter(x=df["elapsed_min"], y=df["td_error"], name="TD Error", line=dict(color="#fab387")),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df["elapsed_min"], y=df["grad_norm"], name="Grad Norm", line=dict(color="#f9e2af")),
        row=2, col=1
    )

    # Learning Rate
    fig.add_trace(
        go.Scatter(x=df["elapsed_min"], y=df["lr"], name="LR", line=dict(color="#a6e3a1")),
        row=2, col=2
    )

    fig.update_layout(
        height=500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_dark",
        paper_bgcolor="#1e1e2e",
        plot_bgcolor="#181825",
    )
    fig.update_xaxes(title_text="Time (min)", gridcolor="#313244")
    fig.update_yaxes(gridcolor="#313244")

    st.plotly_chart(fig, use_container_width=True)


def create_progress_charts(df: pd.DataFrame) -> None:
    """Create charts for training progress."""
    if df is None or len(df) < 2:
        return

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Reward (avg across workers)", "Best X Position", "Throughput"),
        horizontal_spacing=0.08,
    )

    # Average Reward
    if "avg_reward" in df.columns:
        fig.add_trace(
            go.Scatter(x=df["elapsed_min"], y=df["avg_reward"], name="Avg Reward",
                      line=dict(color="#cba6f7"), fill="tozeroy"),
            row=1, col=1
        )

    # Best X
    if "global_best_x" in df.columns:
        fig.add_trace(
            go.Scatter(x=df["elapsed_min"], y=df["global_best_x"], name="Best X",
                      line=dict(color="#a6e3a1")),
            row=1, col=2
        )

    # Throughput
    if "grads_per_sec" in df.columns:
        fig.add_trace(
            go.Scatter(x=df["elapsed_min"], y=df["grads_per_sec"], name="Grads/sec",
                      line=dict(color="#89b4fa")),
            row=1, col=3
        )

    fig.update_layout(
        height=300,
        showlegend=False,
        template="plotly_dark",
        paper_bgcolor="#1e1e2e",
        plot_bgcolor="#181825",
    )
    fig.update_xaxes(title_text="Time (min)", gridcolor="#313244")
    fig.update_yaxes(gridcolor="#313244")

    st.plotly_chart(fig, use_container_width=True)


def create_worker_comparison(workers: dict[int, pd.DataFrame]) -> None:
    """Create worker comparison charts."""
    if not workers:
        st.info("No worker data available yet...")
        return

    # Combine all worker data for comparison
    all_episodes = []
    for worker_id, df in workers.items():
        if len(df) > 0:
            df_copy = df.copy()
            df_copy["worker_id"] = worker_id
            all_episodes.append(df_copy)

    if not all_episodes:
        return

    combined = pd.concat(all_episodes, ignore_index=True)

    # Worker reward comparison
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Episode Reward by Worker", "X Position by Worker", "Epsilon Decay"),
        horizontal_spacing=0.08,
    )

    colors = px.colors.qualitative.Set2
    for i, (worker_id, df) in enumerate(sorted(workers.items())):
        if len(df) == 0:
            continue
        color = colors[i % len(colors)]

        fig.add_trace(
            go.Scatter(x=df["episode"], y=df["reward"], name=f"W{worker_id}",
                      line=dict(color=color), opacity=0.7),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=df["episode"], y=df["x_pos"], name=f"W{worker_id}",
                      line=dict(color=color), opacity=0.7, showlegend=False),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(x=df["episode"], y=df["epsilon"], name=f"W{worker_id}",
                      line=dict(color=color), opacity=0.7, showlegend=False),
            row=1, col=3
        )

    fig.update_layout(
        height=350,
        template="plotly_dark",
        paper_bgcolor="#1e1e2e",
        plot_bgcolor="#181825",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title_text="Episode", gridcolor="#313244")
    fig.update_yaxes(gridcolor="#313244")

    st.plotly_chart(fig, use_container_width=True)


def create_worker_detail(workers: dict[int, pd.DataFrame], worker_id: int) -> None:
    """Create detailed view for a single worker."""
    if worker_id not in workers or len(workers[worker_id]) == 0:
        st.info(f"No data for Worker {worker_id}")
        return

    df = workers[worker_id]

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            "Reward & Rolling Avg", "X Position & Best", "Speed & Steps/sec",
            "Buffer Fill %", "Deaths & Flags", "Entropy"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.08,
    )

    # Row 1
    fig.add_trace(go.Scatter(x=df["episode"], y=df["reward"], name="Reward", line=dict(color="#f38ba8")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["episode"], y=df["rolling_avg_reward"], name="Rolling Avg", line=dict(color="#a6e3a1")), row=1, col=1)

    fig.add_trace(go.Scatter(x=df["episode"], y=df["x_pos"], name="X Pos", line=dict(color="#89b4fa")), row=1, col=2)
    fig.add_trace(go.Scatter(x=df["episode"], y=df["best_x_ever"], name="Best Ever", line=dict(color="#74c7ec")), row=1, col=2)

    fig.add_trace(go.Scatter(x=df["episode"], y=df["avg_speed"], name="Avg Speed", line=dict(color="#fab387")), row=1, col=3)
    fig.add_trace(go.Scatter(x=df["episode"], y=df["steps_per_sec"], name="Steps/sec", line=dict(color="#f9e2af")), row=1, col=3)

    # Row 2
    fig.add_trace(go.Scatter(x=df["episode"], y=df["buffer_fill_pct"], name="Buffer %", line=dict(color="#cba6f7"), fill="tozeroy"), row=2, col=1)

    fig.add_trace(go.Scatter(x=df["episode"], y=df["deaths"], name="Deaths", line=dict(color="#f38ba8")), row=2, col=2)
    fig.add_trace(go.Scatter(x=df["episode"], y=df["flags"], name="Flags", line=dict(color="#a6e3a1")), row=2, col=2)

    fig.add_trace(go.Scatter(x=df["episode"], y=df["entropy"], name="Entropy", line=dict(color="#89dceb")), row=2, col=3)

    fig.update_layout(
        height=500,
        showlegend=True,
        template="plotly_dark",
        paper_bgcolor="#1e1e2e",
        plot_bgcolor="#181825",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title_text="Episode", gridcolor="#313244")
    fig.update_yaxes(gridcolor="#313244")

    st.plotly_chart(fig, use_container_width=True)


def display_learner_summary(df: pd.DataFrame) -> None:
    """Display summary metrics for learner."""
    if df is None or len(df) == 0:
        st.info("Waiting for learner data...")
        return

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    cols = st.columns(6)

    with cols[0]:
        st.metric("Updates", f"{int(latest['update']):,}",
                 delta=f"+{int(latest['update'] - prev['update'])}" if len(df) > 1 else None)

    with cols[1]:
        st.metric("Timesteps", f"{int(latest['timesteps']):,}",
                 delta=f"+{int(latest['timesteps'] - prev['timesteps']):,}" if len(df) > 1 else None)

    with cols[2]:
        st.metric("Episodes", f"{int(latest.get('total_episodes', 0)):,}")

    with cols[3]:
        loss_delta = latest["loss"] - prev["loss"] if len(df) > 1 else 0
        st.metric("Loss", f"{latest['loss']:.4f}",
                 delta=f"{loss_delta:+.4f}" if len(df) > 1 else None,
                 delta_color="inverse")

    with cols[4]:
        st.metric("Q Mean", f"{latest['q_mean']:.2f}")

    with cols[5]:
        st.metric("Grads/sec", f"{latest['grads_per_sec']:.1f}")

    # Second row
    cols2 = st.columns(6)

    with cols2[0]:
        st.metric("Weight Version", f"{int(latest['weight_version'])}")

    with cols2[1]:
        st.metric("Gradients Received", f"{int(latest['gradients_received']):,}")

    with cols2[2]:
        st.metric("Best X", f"{int(latest.get('global_best_x', 0)):,}")

    with cols2[3]:
        st.metric("Deaths", f"{int(latest.get('total_deaths', 0)):,}")

    with cols2[4]:
        st.metric("Flags 🏁", f"{int(latest.get('total_flags', 0))}")

    with cols2[5]:
        st.metric("LR", f"{latest['lr']:.2e}")


def display_worker_summary(workers: dict[int, pd.DataFrame]) -> None:
    """Display summary for all workers."""
    if not workers:
        st.info("Waiting for worker data...")
        return

    cols = st.columns(min(len(workers), 4))

    for i, (worker_id, df) in enumerate(sorted(workers.items())):
        if len(df) == 0:
            continue

        latest = df.iloc[-1]
        col_idx = i % len(cols)

        with cols[col_idx]:
            st.markdown(f"**Worker {worker_id}** ({latest.get('current_level', '?')})")

            inner_cols = st.columns(2)
            with inner_cols[0]:
                st.metric("Episodes", int(latest["episode"]), label_visibility="collapsed")
                st.caption("Episodes")
            with inner_cols[1]:
                st.metric("Best X", int(latest["best_x_ever"]), label_visibility="collapsed")
                st.caption("Best X")

            inner_cols2 = st.columns(2)
            with inner_cols2[0]:
                reward_color = "good" if latest["rolling_avg_reward"] > 0 else "bad"
                st.markdown(f'<span class="{reward_color}">r̄ = {latest["rolling_avg_reward"]:.0f}</span>', unsafe_allow_html=True)
            with inner_cols2[1]:
                st.markdown(f"ε = {latest['epsilon']:.3f}")

            st.progress(min(latest["buffer_fill_pct"] / 100, 1.0), text=f"Buffer: {latest['buffer_fill_pct']:.1f}%")


def main():
    """Main dashboard function."""
    st.title("🍄 Mario RL Training Dashboard")

    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")

        # Checkpoint directory selection
        default_checkpoint = find_latest_checkpoint()
        checkpoint_input = st.text_input(
            "Checkpoint Directory",
            value=str(default_checkpoint) if default_checkpoint else "checkpoints/",
            help="Path to the training checkpoint directory"
        )
        checkpoint_dir = Path(checkpoint_input)

        # Refresh settings
        auto_refresh = st.toggle("Auto Refresh", value=True)
        refresh_interval = st.slider("Refresh Interval (sec)", 1, 30, 5)

        st.divider()

        # Manual refresh button
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.rerun()

        st.divider()

        # Status
        if checkpoint_dir.exists():
            learner_csv = checkpoint_dir / "ddqn_metrics.csv"
            if learner_csv.exists():
                mod_time = datetime.fromtimestamp(learner_csv.stat().st_mtime)
                st.success(f"📁 Connected")
                st.caption(f"Last update: {mod_time.strftime('%H:%M:%S')}")
            else:
                st.warning("Waiting for training data...")
        else:
            st.error(f"Directory not found: {checkpoint_dir}")

    # Check if checkpoint exists
    if not checkpoint_dir.exists():
        st.error(f"Checkpoint directory not found: {checkpoint_dir}")
        st.info("Start training with: `uv run mario-train-ddqn-dist`")
        return

    # Load data
    learner_df = load_learner_metrics(checkpoint_dir)
    workers = load_worker_episodes(checkpoint_dir)

    # Main content
    st.header("📊 Learner Metrics")
    display_learner_summary(learner_df)
    create_learner_charts(learner_df)

    st.divider()

    st.header("📈 Training Progress")
    create_progress_charts(learner_df)

    st.divider()

    st.header("👷 Workers")
    display_worker_summary(workers)

    if workers:
        st.subheader("Worker Comparison")
        create_worker_comparison(workers)

        # Individual worker detail
        st.subheader("Worker Detail")
        worker_ids = sorted(workers.keys())
        selected_worker = st.selectbox("Select Worker", worker_ids, format_func=lambda x: f"Worker {x}")
        if selected_worker is not None:
            create_worker_detail(workers, selected_worker)

    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()

