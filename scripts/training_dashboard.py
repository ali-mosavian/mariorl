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


@st.cache_data(ttl=2)
def find_latest_checkpoint(base_dir: str = "checkpoints") -> str | None:
    """Find the most recent checkpoint directory."""
    base = Path(base_dir)
    if not base.exists():
        return None
    ddqn_dirs = sorted(
        [d for d in base.iterdir() if d.is_dir() and d.name.startswith("ddqn_dist_")],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    return str(ddqn_dirs[0]) if ddqn_dirs else None


@st.cache_data(ttl=2)
def load_learner_metrics(checkpoint_dir: str) -> pd.DataFrame | None:
    """Load learner metrics CSV."""
    MAX_VALID_X = 10000  # No Mario level is longer than this
    csv_path = Path(checkpoint_dir) / "ddqn_metrics.csv"
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
        if "timestamp" in df.columns and len(df) > 0:
            df["elapsed_min"] = (df["timestamp"] - df["timestamp"].iloc[0]) / 60
        # Replace invalid x positions with NaN (65535 is a RAM read glitch)
        if "global_best_x" in df.columns:
            df.loc[df["global_best_x"] > MAX_VALID_X, "global_best_x"] = pd.NA
        return df
    except Exception:
        return None


@st.cache_data(ttl=2)
def load_worker_episodes(checkpoint_dir: str) -> dict[int, pd.DataFrame]:
    """Load all worker episode CSVs."""
    MAX_VALID_X = 10000  # No Mario level is longer than this
    workers = {}
    for csv_path in Path(checkpoint_dir).glob("ddqn_worker_*_episodes.csv"):
        try:
            worker_id = int(csv_path.stem.split("_")[2])
            df = pd.read_csv(csv_path)
            if "timestamp" in df.columns and len(df) > 0:
                df["elapsed_min"] = (df["timestamp"] - df["timestamp"].iloc[0]) / 60
            # Replace invalid x positions with NaN (65535 is a RAM read glitch)
            for col in ["best_x", "best_x_ever", "x_pos"]:
                if col in df.columns:
                    df.loc[df[col] > MAX_VALID_X, col] = pd.NA
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

    # Key metrics - row 1: Progress
    st.caption("PROGRESS")
    cols = st.columns(6)
    cols[0].metric("Updates", f"{int(latest['update']):,}")
    cols[1].metric("Timesteps", f"{int(latest['timesteps']):,}")
    cols[2].metric("Episodes", f"{int(latest.get('total_episodes', 0)):,}")
    best_x = latest.get('global_best_x', 0)
    cols[3].metric("Best X", f"{int(best_x):,}" if pd.notna(best_x) else "N/A")
    cols[4].metric("🏁 Flags", f"{int(latest.get('total_flags', 0))}")
    cols[5].metric("💀 Deaths", f"{int(latest.get('total_deaths', 0)):,}")

    # Key metrics - row 2: Performance
    st.caption("PERFORMANCE (rolling averages)")
    cols2 = st.columns(4)
    avg_reward = latest.get('avg_reward', 0)
    reward_color = "normal" if avg_reward >= 0 else "inverse"
    cols2[0].metric("Avg Reward", f"{avg_reward:,.0f}")
    cols2[1].metric("Avg Speed", f"{latest.get('avg_speed', 0):.2f}")
    cols2[2].metric("Entropy", f"{latest.get('avg_entropy', 0):.3f}")
    cols2[3].metric("LR", f"{latest['lr']:.2e}")

    st.divider()

    # Charts - row 1: Training health
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

    # Charts - row 2: Performance trends
    col3, col4 = st.columns(2)

    with col3:
        fig = make_chart(df, [
            ("avg_reward", "Avg Reward", "green"),
        ], "Average Reward (across workers)")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = make_chart(df, [
            ("avg_speed", "Avg Speed", "teal"),
        ], "Average Speed (x-pos / game-time)")
        st.plotly_chart(fig, use_container_width=True)

    # Charts - row 3: Diagnostics
    col5, col6 = st.columns(2)

    with col5:
        fig = make_chart(df, [
            ("grad_norm", "Grad Norm", "yellow"),
        ], "Gradient Norm")
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        fig = make_chart(df, [
            ("grads_per_sec", "Grads/sec", "mauve"),
        ], "Throughput")
        st.plotly_chart(fig, use_container_width=True)

    # Additional stats in expander
    with st.expander("📊 More Details"):
        dcols = st.columns(4)
        dcols[0].metric("Weight Version", f"{int(latest['weight_version'])}")
        dcols[1].metric("Gradients Received", f"{int(latest['gradients_received']):,}")
        dcols[2].metric("Grad Norm", f"{latest.get('grad_norm', 0):.2f}")
        dcols[3].metric("Packets/Update", f"{int(latest.get('num_packets', 0))}")


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
        best_x = latest["best_x_ever"]
        rows.append({
            "Worker": f"W{wid}",
            "Level": latest.get("current_level", "?"),
            "Episodes": int(latest["episode"]),
            "Best X": int(best_x) if pd.notna(best_x) else 0,
            "Avg Reward": f"{latest['rolling_avg_reward']:.0f}",
            "Avg Speed": f"{latest.get('avg_speed', 0):.2f}",
            "ε": f"{latest['epsilon']:.3f}",
            "Buffer %": f"{latest['buffer_fill_pct']:.0f}%",
            "Deaths": int(latest["deaths"]),
            "Flags": int(latest["flags"]),
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.divider()

    # Comparison charts - row 1
    col1, col2 = st.columns(2)
    colors = list(COLORS.values())

    with col1:
        fig = go.Figure()
        for i, (wid, df) in enumerate(sorted(workers.items())):
            if len(df) > 0:
                fig.add_trace(go.Scatter(
                    x=df["episode"], y=df["rolling_avg_reward"],
                    name=f"W{wid}", line=dict(color=colors[i % len(colors)], width=1.5),
                    opacity=0.8,
                ))
        fig.update_layout(
            title="Rolling Avg Reward by Worker",
            height=280,
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
                    x=df["episode"], y=df["avg_speed"],
                    name=f"W{wid}", line=dict(color=colors[i % len(colors)], width=1.5),
                    opacity=0.8,
                ))
        fig.update_layout(
            title="Rolling Avg Speed by Worker",
            height=280,
            margin=dict(l=0, r=0, t=30, b=0),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Episode", gridcolor="#313244"),
            yaxis=dict(gridcolor="#313244"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Comparison charts - row 2
    col3, col4 = st.columns(2)

    with col3:
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
            height=280,
            margin=dict(l=0, r=0, t=30, b=0),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Episode", gridcolor="#313244"),
            yaxis=dict(gridcolor="#313244"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = go.Figure()
        for i, (wid, df) in enumerate(sorted(workers.items())):
            if len(df) > 0:
                fig.add_trace(go.Scatter(
                    x=df["episode"], y=df["epsilon"],
                    name=f"W{wid}", line=dict(color=colors[i % len(colors)], width=1.5),
                    opacity=0.8,
                ))
        fig.update_layout(
            title="Epsilon Decay by Worker",
            height=280,
            margin=dict(l=0, r=0, t=30, b=0),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Episode", gridcolor="#313244"),
            yaxis=dict(gridcolor="#313244"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)


def render_levels_tab(workers: dict[int, pd.DataFrame]) -> None:
    """Render per-level breakdown tab."""
    if not workers:
        st.info("⏳ Waiting for worker data...")
        return

    # Combine all worker episodes
    all_episodes = []
    for wid, df in workers.items():
        if len(df) > 0:
            df_copy = df.copy()
            df_copy["worker_id"] = wid
            all_episodes.append(df_copy)
    
    if not all_episodes:
        st.info("⏳ No episode data yet...")
        return
    
    all_df = pd.concat(all_episodes, ignore_index=True)
    
    # Check if we have level data
    if "current_level" not in all_df.columns:
        st.warning("No level information in data (single-level training?)")
        return
    
    # Aggregate by level
    level_stats = all_df.groupby("current_level").agg({
        "episode": "count",  # Number of episodes
        "reward": ["mean", "std"],
        "avg_speed": "mean",
        "best_x": "max",
        "x_pos": "mean",
        "flag_get": "sum" if "flag_get" in all_df.columns else "count",
        "steps": "mean",
    }).round(2)
    
    # Flatten column names
    level_stats.columns = ["Episodes", "Avg Reward", "Reward Std", "Avg Speed", "Best X", "Avg X", "Flags", "Avg Steps"]
    level_stats = level_stats.reset_index().rename(columns={"current_level": "Level"})
    
    # Calculate success rate
    if "flag_get" in all_df.columns:
        flag_counts = all_df.groupby("current_level")["flag_get"].sum()
        episode_counts = all_df.groupby("current_level")["episode"].count()
        level_stats["Success %"] = (flag_counts.values / episode_counts.values * 100).round(1)
    
    # Sort by level name (world-stage order)
    def level_sort_key(level: str) -> tuple:
        """Sort levels by world then stage (e.g., 1-1, 1-2, 2-1, etc.)"""
        try:
            parts = level.split("-")
            return (int(parts[0]), int(parts[1]))
        except (ValueError, IndexError):
            return (99, 99)  # Unknown levels go last
    
    level_stats["_sort"] = level_stats["Level"].apply(level_sort_key)
    level_stats = level_stats.sort_values("_sort").drop(columns=["_sort"])
    
    # Summary metrics
    unique_levels = all_df["current_level"].nunique()
    total_episodes = len(all_df)
    total_flags = all_df["flag_get"].sum() if "flag_get" in all_df.columns else 0
    
    st.caption("LEVEL OVERVIEW")
    cols = st.columns(4)
    cols[0].metric("Unique Levels", unique_levels)
    cols[1].metric("Total Episodes", f"{total_episodes:,}")
    cols[2].metric("Total Flags", int(total_flags))
    cols[3].metric("Overall Success", f"{total_flags/total_episodes*100:.1f}%" if total_episodes > 0 else "0%")
    
    st.divider()
    
    # Per-level table
    st.subheader("📊 Per-Level Statistics")
    st.dataframe(
        level_stats.style.format({
            "Avg Reward": "{:.1f}",
            "Reward Std": "{:.1f}",
            "Avg Speed": "{:.2f}",
            "Avg X": "{:.0f}",
            "Avg Steps": "{:.0f}",
            "Success %": "{:.1f}%",
        }),
        hide_index=True,
        use_container_width=True,
    )
    
    st.divider()
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Average reward by level
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=level_stats["Level"],
            y=level_stats["Avg Reward"],
            marker_color=COLORS["green"],
            error_y=dict(type="data", array=level_stats["Reward Std"], visible=True),
        ))
        fig.update_layout(
            title="Average Reward by Level",
            height=300,
            margin=dict(l=0, r=0, t=40, b=0),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Level", gridcolor="#313244", categoryorder="array", categoryarray=level_stats["Level"].tolist()),
            yaxis=dict(title="Avg Reward", gridcolor="#313244"),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Average speed by level
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=level_stats["Level"],
            y=level_stats["Avg Speed"],
            marker_color=COLORS["teal"],
        ))
        fig.update_layout(
            title="Average Speed by Level",
            height=300,
            margin=dict(l=0, r=0, t=40, b=0),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Level", gridcolor="#313244", categoryorder="array", categoryarray=level_stats["Level"].tolist()),
            yaxis=dict(title="Avg Speed", gridcolor="#313244"),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Best X by level
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=level_stats["Level"],
            y=level_stats["Best X"],
            marker_color=COLORS["blue"],
        ))
        fig.update_layout(
            title="Best X Position by Level",
            height=300,
            margin=dict(l=0, r=0, t=40, b=0),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Level", gridcolor="#313244", categoryorder="array", categoryarray=level_stats["Level"].tolist()),
            yaxis=dict(title="Best X", gridcolor="#313244"),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        # Episode count by level (training distribution)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=level_stats["Level"],
            y=level_stats["Episodes"],
            marker_color=COLORS["mauve"],
        ))
        fig.update_layout(
            title="Episodes per Level (Training Distribution)",
            height=300,
            margin=dict(l=0, r=0, t=40, b=0),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Level", gridcolor="#313244", categoryorder="array", categoryarray=level_stats["Level"].tolist()),
            yaxis=dict(title="Episodes", gridcolor="#313244"),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed level selector
    st.divider()
    st.subheader("🔍 Level Deep Dive")
    
    selected_level = st.selectbox(
        "Select a level to analyze",
        options=sorted(all_df["current_level"].unique()),
    )
    
    if selected_level:
        level_df = all_df[all_df["current_level"] == selected_level].copy()
        level_df = level_df.sort_values("timestamp")
        
        # Add episode index within level
        level_df["level_episode"] = range(1, len(level_df) + 1)
        
        lcol1, lcol2, lcol3, lcol4 = st.columns(4)
        lcol1.metric("Episodes", len(level_df))
        lcol2.metric("Avg Reward", f"{level_df['reward'].mean():.1f}")
        best_x = level_df['best_x'].max()
        lcol3.metric("Best X", f"{best_x:.0f}" if pd.notna(best_x) else "N/A")
        flags = level_df["flag_get"].sum() if "flag_get" in level_df.columns else 0
        lcol4.metric("Flags", int(flags))
        
        # Rolling reward over episodes for this level
        if len(level_df) >= 3:
            level_df["rolling_reward"] = level_df["reward"].rolling(window=min(10, len(level_df)), min_periods=1).mean()
            level_df["rolling_speed"] = level_df["avg_speed"].rolling(window=min(10, len(level_df)), min_periods=1).mean()
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=level_df["level_episode"],
                    y=level_df["rolling_reward"],
                    mode="lines",
                    line=dict(color=COLORS["green"], width=2),
                    name="Rolling Avg",
                ))
                fig.add_trace(go.Scatter(
                    x=level_df["level_episode"],
                    y=level_df["reward"],
                    mode="markers",
                    marker=dict(color=COLORS["green"], size=4, opacity=0.3),
                    name="Episode",
                ))
                fig.update_layout(
                    title=f"Reward on {selected_level} (over time)",
                    height=280,
                    margin=dict(l=0, r=0, t=40, b=0),
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(title="Episode on Level", gridcolor="#313244"),
                    yaxis=dict(title="Reward", gridcolor="#313244"),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col_b:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=level_df["level_episode"],
                    y=level_df["rolling_speed"],
                    mode="lines",
                    line=dict(color=COLORS["teal"], width=2),
                    name="Rolling Avg",
                ))
                fig.add_trace(go.Scatter(
                    x=level_df["level_episode"],
                    y=level_df["avg_speed"],
                    mode="markers",
                    marker=dict(color=COLORS["teal"], size=4, opacity=0.3),
                    name="Episode",
                ))
                fig.update_layout(
                    title=f"Speed on {selected_level} (over time)",
                    height=280,
                    margin=dict(l=0, r=0, t=40, b=0),
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(title="Episode on Level", gridcolor="#313244"),
                    yaxis=dict(title="Avg Speed", gridcolor="#313244"),
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


def render_dashboard_content(checkpoint_dir: str, refresh_sec: int) -> None:
    """Render the main dashboard content (tabs and charts)."""
    
    # Tabs (created outside fragments so they persist)
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Learner", "👷 Workers", "🗺️ Levels", "🔍 Analysis"])

    with tab1:
        @st.fragment(run_every=refresh_sec)
        def learner_fragment():
            load_learner_metrics.clear()
            df = load_learner_metrics(checkpoint_dir)
            if df is not None and len(df) > 0:
                st.caption(f"🔄 {datetime.now().strftime('%H:%M:%S')} • {len(df)} updates")
            render_learner_tab(df)
        learner_fragment()

    with tab2:
        @st.fragment(run_every=refresh_sec)
        def workers_fragment():
            load_worker_episodes.clear()
            workers = load_worker_episodes(checkpoint_dir)
            st.caption(f"🔄 {datetime.now().strftime('%H:%M:%S')}")
            render_workers_tab(workers)
        workers_fragment()

    with tab3:
        @st.fragment(run_every=refresh_sec)
        def levels_fragment():
            load_worker_episodes.clear()
            workers = load_worker_episodes(checkpoint_dir)
            st.caption(f"🔄 {datetime.now().strftime('%H:%M:%S')}")
            render_levels_tab(workers)
        levels_fragment()

    with tab4:
        @st.fragment(run_every=refresh_sec)
        def analysis_fragment():
            load_learner_metrics.clear()
            load_worker_episodes.clear()
            df = load_learner_metrics(checkpoint_dir)
            workers = load_worker_episodes(checkpoint_dir)
            st.caption(f"🔄 {datetime.now().strftime('%H:%M:%S')}")
            render_analysis_tab(df, workers)
        analysis_fragment()


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
            value=default_checkpoint if default_checkpoint else "checkpoints/",
        )
        checkpoint_dir = checkpoint_input

        refresh_sec = st.slider("Refresh interval (sec)", 2, 30, 5)

        st.divider()

        if st.button("🔄 Manual Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        # Status
        checkpoint_path = Path(checkpoint_dir)
        if checkpoint_path.exists():
            learner_csv = checkpoint_path / "ddqn_metrics.csv"
            if learner_csv.exists():
                mod_time = datetime.fromtimestamp(learner_csv.stat().st_mtime)
                st.caption(f"📁 {checkpoint_path.name}")
                st.caption(f"🕐 {mod_time.strftime('%H:%M:%S')}")

    # Check checkpoint
    if not Path(checkpoint_dir).exists():
        st.error(f"Directory not found: {checkpoint_dir}")
        return

    # Render content - each tab has its own fragment for independent refresh
    render_dashboard_content(checkpoint_dir, refresh_sec)


if __name__ == "__main__":
    main()
