#!/usr/bin/env python3
"""
Real-time training dashboard for DDQN distributed training.

Run with:
    uv run streamlit run scripts/training_dashboard.py
"""

from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Page config
st.set_page_config(
    page_title="Mario RL Training",
    page_icon="🍄",
    layout="wide",
)

# Minimal custom CSS
st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    [data-testid="stMetricValue"] { font-size: 1.8rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 16px; }
</style>
""", unsafe_allow_html=True)

# Color palette (Catppuccin Mocha)
COLORS = {
    "red": "#f38ba8",
    "green": "#a6e3a1",
    "blue": "#89b4fa",
    "yellow": "#f9e2af",
    "peach": "#fab387",
    "mauve": "#cba6f7",
    "teal": "#94e2d5",
    "sky": "#89dceb",
}


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
        if "timestamp" in df.columns and len(df) > 0:
            df["elapsed_min"] = (df["timestamp"] - df["timestamp"].iloc[0]) / 60
        return df
    except Exception:
        return None


def load_worker_episodes(checkpoint_dir: Path) -> dict[int, pd.DataFrame]:
    """Load all worker episode CSVs."""
    workers = {}
    for csv_path in checkpoint_dir.glob("ddqn_worker_*_episodes.csv"):
        try:
            worker_id = int(csv_path.stem.split("_")[2])
            df = pd.read_csv(csv_path)
            if "timestamp" in df.columns and len(df) > 0:
                df["elapsed_min"] = (df["timestamp"] - df["timestamp"].iloc[0]) / 60
            workers[worker_id] = df
        except Exception:
            pass
    return workers


def make_chart(df: pd.DataFrame, metrics: list[tuple[str, str, str]], title: str, height: int = 280) -> go.Figure:
    """Create a simple line chart with multiple metrics."""
    fig = go.Figure()
    for col, name, color in metrics:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["elapsed_min"], y=df[col],
                name=name, line=dict(color=COLORS.get(color, color), width=2),
                mode="lines",
            ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        height=height,
        margin=dict(l=0, r=0, t=30, b=0),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=10)),
        xaxis=dict(title="Time (min)", gridcolor="#313244", showgrid=True),
        yaxis=dict(gridcolor="#313244", showgrid=True),
    )
    return fig


def render_learner_tab(df: pd.DataFrame) -> None:
    """Render the learner metrics tab."""
    if df is None or len(df) == 0:
        st.info("⏳ Waiting for training data...")
        return

    latest = df.iloc[-1]

    # Key metrics row
    cols = st.columns(6)
    cols[0].metric("Updates", f"{int(latest['update']):,}")
    cols[1].metric("Timesteps", f"{int(latest['timesteps']):,}")
    cols[2].metric("Episodes", f"{int(latest.get('total_episodes', 0)):,}")
    cols[3].metric("Best X", f"{int(latest.get('global_best_x', 0)):,}")
    cols[4].metric("🏁 Flags", f"{int(latest.get('total_flags', 0))}")
    cols[5].metric("💀 Deaths", f"{int(latest.get('total_deaths', 0)):,}")

    st.divider()

    # Charts in 2x2 grid
    col1, col2 = st.columns(2)

    with col1:
        fig = make_chart(df, [
            ("loss", "Loss", "red"),
            ("td_error", "TD Error", "peach"),
        ], "Loss & TD Error")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = make_chart(df, [
            ("q_mean", "Q Mean", "blue"),
            ("q_max", "Q Max", "sky"),
        ], "Q-Values")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig = make_chart(df, [
            ("grad_norm", "Grad Norm", "yellow"),
        ], "Gradient Norm")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = make_chart(df, [
            ("grads_per_sec", "Grads/sec", "green"),
        ], "Throughput")
        st.plotly_chart(fig, use_container_width=True)

    # Additional stats in expander
    with st.expander("📊 More Details"):
        dcols = st.columns(4)
        dcols[0].metric("Weight Version", f"{int(latest['weight_version'])}")
        dcols[1].metric("Gradients Received", f"{int(latest['gradients_received']):,}")
        dcols[2].metric("Learning Rate", f"{latest['lr']:.2e}")
        dcols[3].metric("Avg Reward", f"{latest.get('avg_reward', 0):.1f}")


def render_workers_tab(workers: dict[int, pd.DataFrame]) -> None:
    """Render the workers tab."""
    if not workers:
        st.info("⏳ Waiting for worker data...")
        return

    # Summary table
    rows = []
    for wid, df in sorted(workers.items()):
        if len(df) == 0:
            continue
        latest = df.iloc[-1]
        rows.append({
            "Worker": f"W{wid}",
            "Level": latest.get("current_level", "?"),
            "Episodes": int(latest["episode"]),
            "Best X": int(latest["best_x_ever"]),
            "Avg Reward": f"{latest['rolling_avg_reward']:.0f}",
            "ε": f"{latest['epsilon']:.3f}",
            "Buffer %": f"{latest['buffer_fill_pct']:.0f}%",
            "Deaths": int(latest["deaths"]),
            "Flags": int(latest["flags"]),
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.divider()

    # Comparison charts
    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        colors = list(COLORS.values())
        for i, (wid, df) in enumerate(sorted(workers.items())):
            if len(df) > 0:
                fig.add_trace(go.Scatter(
                    x=df["episode"], y=df["reward"],
                    name=f"W{wid}", line=dict(color=colors[i % len(colors)], width=1.5),
                    opacity=0.8,
                ))
        fig.update_layout(
            title="Episode Reward by Worker",
            height=300,
            margin=dict(l=0, r=0, t=30, b=0),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Episode", gridcolor="#313244"),
            yaxis=dict(gridcolor="#313244"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        for i, (wid, df) in enumerate(sorted(workers.items())):
            if len(df) > 0:
                fig.add_trace(go.Scatter(
                    x=df["episode"], y=df["best_x_ever"],
                    name=f"W{wid}", line=dict(color=colors[i % len(colors)], width=1.5),
                    opacity=0.8,
                ))
        fig.update_layout(
            title="Best X Position by Worker",
            height=300,
            margin=dict(l=0, r=0, t=30, b=0),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Episode", gridcolor="#313244"),
            yaxis=dict(gridcolor="#313244"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)


def render_analysis_tab(df: pd.DataFrame, workers: dict[int, pd.DataFrame]) -> None:
    """Render analysis/debugging tab."""
    if df is None or len(df) < 2:
        st.info("⏳ Need more data for analysis...")
        return

    st.subheader("Training Health")

    # Calculate health indicators
    latest = df.iloc[-1]
    recent = df.tail(10)

    loss_trend = recent["loss"].diff().mean()
    q_trend = recent["q_mean"].diff().mean()
    grad_norm_avg = recent["grad_norm"].mean()

    col1, col2, col3, col4 = st.columns(4)

    # Loss trend
    if loss_trend > 1:
        col1.error(f"📈 Loss Rising (+{loss_trend:.2f}/update)")
    elif loss_trend < -0.5:
        col1.success(f"📉 Loss Falling ({loss_trend:.2f}/update)")
    else:
        col1.info(f"➡️ Loss Stable ({loss_trend:+.2f}/update)")

    # Q-value health
    if abs(latest["q_mean"]) > 1000:
        col2.error(f"⚠️ Q-values large ({latest['q_mean']:.0f})")
    elif latest["q_mean"] < -100:
        col2.warning(f"📉 Q-values negative ({latest['q_mean']:.1f})")
    else:
        col2.success(f"✓ Q-values OK ({latest['q_mean']:.1f})")

    # Gradient norm
    if grad_norm_avg > 50:
        col3.error(f"⚠️ High grad norm ({grad_norm_avg:.1f})")
    elif grad_norm_avg > 10:
        col3.warning(f"📊 Elevated grad norm ({grad_norm_avg:.1f})")
    else:
        col3.success(f"✓ Grad norm OK ({grad_norm_avg:.1f})")

    # Throughput
    throughput = latest.get("grads_per_sec", 0)
    if throughput < 5:
        col4.warning(f"🐢 Low throughput ({throughput:.1f} g/s)")
    else:
        col4.success(f"⚡ Good throughput ({throughput:.1f} g/s)")

    st.divider()

    # Progress over time
    st.subheader("Progress Timeline")

    if "global_best_x" in df.columns:
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Best X Progress", "Learning Rate Schedule"))

        fig.add_trace(go.Scatter(
            x=df["elapsed_min"], y=df["global_best_x"],
            fill="tozeroy", line=dict(color=COLORS["green"]),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df["elapsed_min"], y=df["lr"],
            line=dict(color=COLORS["mauve"]),
        ), row=1, col=2)

        fig.update_layout(
            height=250,
            showlegend=False,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=30, b=0),
        )
        fig.update_xaxes(title_text="Time (min)", gridcolor="#313244")
        fig.update_yaxes(gridcolor="#313244")

        st.plotly_chart(fig, use_container_width=True)


def main():
    """Main dashboard."""
    # Header
    st.title("🍄 Mario RL Training")

    # Sidebar
    with st.sidebar:
        st.header("Settings")

        default_checkpoint = find_latest_checkpoint()
        checkpoint_input = st.text_input(
            "Checkpoint",
            value=str(default_checkpoint) if default_checkpoint else "checkpoints/",
        )
        checkpoint_dir = Path(checkpoint_input)

        auto_refresh = st.toggle("Auto Refresh", value=True)
        if auto_refresh:
            refresh_sec = st.slider("Interval (sec)", 2, 30, 5)

        st.divider()

        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

        # Status
        if checkpoint_dir.exists():
            learner_csv = checkpoint_dir / "ddqn_metrics.csv"
            if learner_csv.exists():
                mod_time = datetime.fromtimestamp(learner_csv.stat().st_mtime)
                st.caption(f"📁 {checkpoint_dir.name}")
                st.caption(f"🕐 {mod_time.strftime('%H:%M:%S')}")

    # Check checkpoint
    if not checkpoint_dir.exists():
        st.error(f"Directory not found: {checkpoint_dir}")
        return

    # Load data
    learner_df = load_learner_metrics(checkpoint_dir)
    workers = load_worker_episodes(checkpoint_dir)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Learner", "👷 Workers", "🔍 Analysis"])

    with tab1:
        render_learner_tab(learner_df)

    with tab2:
        render_workers_tab(workers)

    with tab3:
        render_analysis_tab(learner_df, workers)

    # Auto-refresh
    if auto_refresh:
        import time
        time.sleep(refresh_sec)
        st.rerun()


if __name__ == "__main__":
    main()
