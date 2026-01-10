#!/usr/bin/env python3
"""
Real-time training dashboard for distributed Mario RL training.

Run with:
    uv run streamlit run scripts/training_dashboard.py
"""

import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.express as px
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
    # Look for any dist_ directories (ddqn_dist_ or dreamer_dist_)
    dirs = sorted(
        [d for d in base.iterdir() if d.is_dir() and "_dist_" in d.name],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    return str(dirs[0]) if dirs else None


@st.cache_data(ttl=2)
def load_coordinator_metrics(checkpoint_dir: str) -> pd.DataFrame | None:
    """Load coordinator metrics CSV."""
    csv_path = Path(checkpoint_dir) / "coordinator.csv"
    if not csv_path.exists():
        return None
    
    try:
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            return None
            
        if "timestamp" in df.columns:
            df["elapsed_min"] = (df["timestamp"] - df["timestamp"].iloc[0]) / 60
            
        return df
    except Exception as e:
        st.warning(f"Error loading coordinator metrics: {e}")
        return None


@st.cache_data(ttl=2)
def load_death_hotspots(checkpoint_dir: str) -> dict[str, dict[int, int]] | None:
    """Load death hotspot data from JSON file."""
    json_path = Path(checkpoint_dir) / "death_hotspots.json"
    if not json_path.exists():
        return None
    
    try:
        with open(json_path) as f:
            data = json.load(f)
        
        # Handle both old format (flat) and new format (with "levels" key)
        levels_data = data.get("levels", data)
        
        # Convert string keys to int for bucket positions
        result = {}
        for level_id, buckets in levels_data.items():
            if isinstance(buckets, dict):
                result[level_id] = {int(k): v for k, v in buckets.items()}
        return result if result else None
    except Exception:
        return None


@st.cache_data(ttl=2)
def load_worker_metrics(checkpoint_dir: str) -> dict[int, pd.DataFrame]:
    """Load all worker CSV files."""
    MAX_VALID_X = 10000  # No Mario level is longer than this
    workers = {}
    
    checkpoint_path = Path(checkpoint_dir)
    
    for csv_path in checkpoint_path.glob("worker_*.csv"):
        try:
            # Extract worker ID from filename (worker_0.csv -> 0)
            worker_id = int(csv_path.stem.split("_")[1])
            df = pd.read_csv(csv_path)
            if len(df) == 0:
                continue
            
            # Compute current_level from world/stage if not present
            if "current_level" not in df.columns and "world" in df.columns and "stage" in df.columns:
                df["current_level"] = df["world"].astype(int).astype(str) + "-" + df["stage"].astype(int).astype(str)
            
            if "timestamp" in df.columns:
                df["elapsed_min"] = (df["timestamp"] - df["timestamp"].iloc[0]) / 60
                
            # Replace invalid x positions with NaN
            for col in ["best_x", "best_x_ever", "x_pos"]:
                if col in df.columns:
                    df.loc[df[col] > MAX_VALID_X, col] = pd.NA
                    
            workers[worker_id] = df
        except Exception:
            pass  # Skip malformed files
                
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
        margin=dict(l=0, r=0, t=40, b=0),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, font=dict(size=10)),
        xaxis=dict(title="Time (min)", gridcolor="#313244", showgrid=True),
        yaxis=dict(gridcolor="#313244", showgrid=True),
    )
    return fig


def render_coordinator_tab(df: pd.DataFrame) -> None:
    """Render the coordinator metrics tab."""
    if df is None or len(df) == 0:
        st.info("⏳ Waiting for coordinator data...")
        return

    latest = df.iloc[-1]

    # Key metrics - row 1: Progress
    st.caption("PROGRESS")
    cols = st.columns(6)
    cols[0].metric("Updates", f"{int(latest.get('update_count', 0)):,}")
    cols[1].metric("Total Steps", f"{int(latest.get('total_steps', 0)):,}")
    cols[2].metric("Episodes", f"{int(latest.get('total_episodes', 0)):,}")
    cols[3].metric("Weight Ver", f"{int(latest.get('weight_version', 0)):,}")
    cols[4].metric("Grads/sec", f"{latest.get('grads_per_sec', 0):.1f}")
    cols[5].metric("LR", f"{latest.get('learning_rate', 0):.2e}")

    # Key metrics - row 2: Training metrics (aggregated from workers)
    st.caption("TRAINING METRICS (aggregated)")
    cols2 = st.columns(7)
    cols2[0].metric("Avg Reward", f"{latest.get('avg_reward', 0):.1f}")
    cols2[1].metric("Avg Speed", f"{latest.get('avg_speed', 0):.2f}")
    cols2[2].metric("Avg Loss", f"{latest.get('avg_loss', 0):.4f}")
    cols2[3].metric("Q Mean", f"{latest.get('q_mean', 0):.2f}")
    cols2[4].metric("TD Error", f"{latest.get('td_error', 0):.4f}")
    cols2[5].metric("Grad Norm", f"{latest.get('grad_norm', 0):.2f}")
    cols2[6].metric("Total SPS", f"{latest.get('total_sps', 0):.0f}")

    st.divider()

    # Charts - row 1: Training health
    col1, col2 = st.columns(2)

    with col1:
        fig = make_chart(df, [
            ("avg_loss", "Avg Loss", "red"),
            ("loss", "Loss", "peach"),
        ], "Loss")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Dual y-axis for Q Mean and TD Error (different scales)
        fig = go.Figure()
        
        if "q_mean" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["elapsed_min"], y=df["q_mean"],
                name="Q Mean", line=dict(color=COLORS["blue"], width=2),
                mode="lines", yaxis="y",
            ))
        
        if "td_error" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["elapsed_min"], y=df["td_error"],
                name="TD Error", line=dict(color=COLORS["sky"], width=2),
                mode="lines", yaxis="y2",
            ))
        
        fig.update_layout(
            title=dict(text="Q-Values & TD Error", font=dict(size=14)),
            height=280,
            margin=dict(l=0, r=40, t=40, b=0),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, font=dict(size=10)),
            xaxis=dict(title="Time (min)", gridcolor="#313244", showgrid=True),
            yaxis=dict(title=dict(text="Q Mean", font=dict(color=COLORS["blue"])), gridcolor="#313244", showgrid=True, side="left"),
            yaxis2=dict(title=dict(text="TD Error", font=dict(color=COLORS["sky"])), gridcolor="#313244", overlaying="y", side="right"),
        )
        
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
        # Use dual y-axis for throughput since scales are very different
        fig = go.Figure()
        
        if "total_sps" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["elapsed_min"], y=df["total_sps"],
                name="Steps/sec", line=dict(color=COLORS["green"], width=2),
                mode="lines", yaxis="y",
            ))
        
        if "grads_per_sec" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["elapsed_min"], y=df["grads_per_sec"],
                name="Grads/sec", line=dict(color=COLORS["mauve"], width=2),
                mode="lines", yaxis="y2",
            ))
        
        fig.update_layout(
            title=dict(text="Throughput", font=dict(size=14)),
            height=280,
            margin=dict(l=0, r=40, t=40, b=0),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, font=dict(size=10)),
            xaxis=dict(title="Time (min)", gridcolor="#313244", showgrid=True),
            yaxis=dict(title=dict(text="SPS", font=dict(color=COLORS["green"])), gridcolor="#313244", showgrid=True, side="left"),
            yaxis2=dict(title=dict(text="Grads/s", font=dict(color=COLORS["mauve"])), gridcolor="#313244", overlaying="y", side="right"),
        )
        
        st.plotly_chart(fig, use_container_width=True)


def render_workers_tab(workers: dict[int, pd.DataFrame]) -> None:
    """Render the workers tab."""
    import time
    
    if not workers:
        st.info("⏳ Waiting for worker data...")
        return

    # Summary table
    rows = []
    current_time = time.time()
    
    for wid, df in sorted(workers.items()):
        if len(df) == 0:
            continue
        latest = df.iloc[-1]
        best_x = latest.get("best_x_ever", latest.get("best_x", 0))
        
        # Calculate heartbeat status based on last timestamp
        last_timestamp = latest.get("timestamp", 0)
        if last_timestamp > 0:
            seconds_since = int(current_time - last_timestamp)
            if seconds_since < 60:
                heartbeat = "💚"  # Healthy
            elif seconds_since < 120:
                heartbeat = "💛"  # Warning
            else:
                heartbeat = "💔"  # Stale/crashed
        else:
            heartbeat = "⚪"  # No data
        
        # Get level info
        level = latest.get("current_level", "?")
        if level == "?" and "world" in latest and "stage" in latest:
            level = f"{int(latest['world'])}-{int(latest['stage'])}"
        
        rows.append({
            "Status": heartbeat,
            "Worker": f"W{wid}",
            "Level": level,
            "Episodes": int(latest.get("episodes", 0)),
            "Steps": int(latest.get("steps", 0)),
            "Best X": int(best_x) if pd.notna(best_x) else 0,
            "Reward": f"{latest.get('reward', 0):.1f}",
            "Ep Reward": f"{latest.get('episode_reward', 0):.1f}",
            "Speed": f"{latest.get('speed', 0):.2f}",
            "ε": f"{latest.get('epsilon', 1.0):.3f}",
            "Loss": f"{latest.get('loss', 0):.3f}",
            "Q Mean": f"{latest.get('q_mean', 0):.1f}",
            "TD Err": f"{latest.get('td_error', 0):.3f}",
            "Deaths": int(latest.get("deaths", 0)),
            "Flags": int(latest.get("flags", 0)),
            "Saves": int(latest.get("snapshot_saves", 0)),
            "Restores": int(latest.get("snapshot_restores", 0)),
            "Grads": int(latest.get("grads_sent", 0)),
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, 
                     column_config={
                         "Status": st.column_config.TextColumn(
                             "Status",
                             help="💚 Healthy (<60s) | 💛 Stale (60-120s) | 💔 Crashed (>120s)",
                             width="small",
                         ),
                     })

    st.divider()

    # Comparison charts - row 1
    col1, col2 = st.columns(2)
    colors = list(COLORS.values())

    with col1:
        fig = go.Figure()
        for i, (wid, df) in enumerate(sorted(workers.items())):
            if len(df) > 0 and "reward" in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.get("episodes", range(len(df))), y=df["reward"],
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
            if len(df) > 0 and "speed" in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.get("episodes", range(len(df))), y=df["speed"],
                    name=f"W{wid}", line=dict(color=colors[i % len(colors)], width=1.5),
                    opacity=0.8,
                ))
        fig.update_layout(
            title="Speed by Worker",
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
            if len(df) > 0 and "best_x_ever" in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.get("episodes", range(len(df))), y=df["best_x_ever"],
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
            if len(df) > 0 and "loss" in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.get("episodes", range(len(df))), y=df["loss"],
                    name=f"W{wid}", line=dict(color=colors[i % len(colors)], width=1.5),
                    opacity=0.8,
                ))
        fig.update_layout(
            title="Loss by Worker",
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

    # Row 3: Q-values and TD error
    col5, col6 = st.columns(2)

    with col5:
        fig = go.Figure()
        for i, (wid, df) in enumerate(sorted(workers.items())):
            if len(df) > 0 and "q_mean" in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.get("episodes", range(len(df))), y=df["q_mean"],
                    name=f"W{wid}", line=dict(color=colors[i % len(colors)], width=1.5),
                    opacity=0.8,
                ))
        fig.update_layout(
            title="Q Mean by Worker",
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

    with col6:
        fig = go.Figure()
        for i, (wid, df) in enumerate(sorted(workers.items())):
            if len(df) > 0 and "td_error" in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.get("episodes", range(len(df))), y=df["td_error"],
                    name=f"W{wid}", line=dict(color=colors[i % len(colors)], width=1.5),
                    opacity=0.8,
                ))
        fig.update_layout(
            title="TD Error by Worker",
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

    # Row 4: Epsilon decay tracking
    col7, col8 = st.columns(2)

    with col7:
        fig = go.Figure()
        for i, (wid, df) in enumerate(sorted(workers.items())):
            if len(df) > 0 and "epsilon" in df.columns:
                # Use steps if available, else episodes
                x_axis = df["steps"] if "steps" in df.columns else df.get("episodes", range(len(df)))
                fig.add_trace(go.Scatter(
                    x=x_axis, y=df["epsilon"],
                    name=f"W{wid}", line=dict(color=colors[i % len(colors)], width=1.5),
                    opacity=0.8,
                ))
        fig.update_layout(
            title="Epsilon by Worker (vs Steps)",
            height=280,
            margin=dict(l=0, r=0, t=30, b=0),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Steps", gridcolor="#313244"),
            yaxis=dict(title="ε", gridcolor="#313244", range=[0, 1.05]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col8:
        # Buffer size / PER beta if available
        fig = go.Figure()
        has_beta = False
        for i, (wid, df) in enumerate(sorted(workers.items())):
            if len(df) > 0 and "per_beta" in df.columns:
                has_beta = True
                x_axis = df["steps"] if "steps" in df.columns else df.get("episodes", range(len(df)))
                fig.add_trace(go.Scatter(
                    x=x_axis, y=df["per_beta"],
                    name=f"W{wid}", line=dict(color=colors[i % len(colors)], width=1.5),
                    opacity=0.8,
                ))
        
        if has_beta:
            fig.update_layout(
                title="PER Beta by Worker (vs Steps)",
                height=280,
                margin=dict(l=0, r=0, t=30, b=0),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Steps", gridcolor="#313244"),
                yaxis=dict(title="β", gridcolor="#313244", range=[0, 1.05]),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback to buffer size if no beta
            fig = go.Figure()
            for i, (wid, df) in enumerate(sorted(workers.items())):
                if len(df) > 0 and "buffer_size" in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.get("episodes", range(len(df))), y=df["buffer_size"],
                        name=f"W{wid}", line=dict(color=colors[i % len(colors)], width=1.5),
                        opacity=0.8,
                    ))
            fig.update_layout(
                title="Buffer Size by Worker",
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

    # Row 5: Action entropy and distribution
    col9, col10 = st.columns(2)

    with col9:
        # Action entropy over time
        fig = go.Figure()
        has_action_entropy = False
        for i, (wid, df) in enumerate(sorted(workers.items())):
            if len(df) > 0 and "action_entropy" in df.columns:
                has_action_entropy = True
                x_axis = df["steps"] if "steps" in df.columns else df.get("episodes", range(len(df)))
                fig.add_trace(go.Scatter(
                    x=x_axis, y=df["action_entropy"],
                    name=f"W{wid}", line=dict(color=colors[i % len(colors)], width=1.5),
                    opacity=0.8,
                ))
        
        if has_action_entropy:
            fig.update_layout(
                title="Action Entropy by Worker (0=deterministic, 1=uniform)",
                height=280,
                margin=dict(l=0, r=0, t=30, b=0),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Steps", gridcolor="#313244"),
                yaxis=dict(title="Entropy", gridcolor="#313244", range=[0, 1.05]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("⏳ Action entropy tracking not yet available...")

    with col10:
        # Action distribution heatmap over time (aggregated across workers)
        # Actions: NOOP, right, right+A, right+B, right+A+B, A, left, left+A, left+B, left+A+B, down, up
        action_names = ["NOOP", "→", "→A", "→B", "→AB", "A", "←", "←A", "←B", "←AB", "↓", "↑"]
        
        # Collect all action_dist data with timestamps
        all_action_data = []
        for wid, df in sorted(workers.items()):
            if len(df) > 0 and "action_dist" in df.columns:
                for idx, row in df.iterrows():
                    dist_str = row.get("action_dist", "")
                    if dist_str and isinstance(dist_str, str):
                        try:
                            pcts = [float(p) for p in dist_str.split(",")]
                            if len(pcts) == 12:
                                steps = row.get("steps", idx)
                                all_action_data.append((steps, pcts))
                        except (ValueError, TypeError):
                            pass
        
        if all_action_data:
            # Sort by steps and sample to ~50 time points for clean heatmap
            all_action_data.sort(key=lambda x: x[0])
            sample_size = min(50, len(all_action_data))
            sample_indices = [int(i * len(all_action_data) / sample_size) for i in range(sample_size)]
            sampled_data = [all_action_data[i] for i in sample_indices]
            
            # Build heatmap matrix: rows = actions, cols = time
            x_labels = [f"{int(d[0]//1000)}k" for d in sampled_data]  # Steps in k
            z_data = [[d[1][action_idx] for d in sampled_data] for action_idx in range(12)]
            
            fig = go.Figure(data=go.Heatmap(
                z=z_data,
                x=x_labels,
                y=action_names,
                colorscale="Viridis",
                hovertemplate="Action: %{y}<br>Steps: %{x}<br>%{z:.1f}%<extra></extra>",
            ))
            
            fig.update_layout(
                title="Action Distribution Over Time (%)",
                height=280,
                margin=dict(l=0, r=0, t=30, b=0),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Steps", gridcolor="#313244"),
                yaxis=dict(gridcolor="#313244"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("⏳ Action distribution tracking not yet available...")


def render_levels_tab(workers: dict[int, pd.DataFrame], death_hotspots: dict[str, dict[int, int]] | None) -> None:
    """Render levels tab with death hotspot visualization."""
    
    # Collect level data from workers with full reward tracking
    level_stats: dict[str, dict] = {}
    
    for wid, df in workers.items():
        if len(df) == 0:
            continue
        
        # Get level from each row
        for _, row in df.iterrows():
            level = row.get("current_level", "?")
            if level == "?" and "world" in row and "stage" in row:
                try:
                    level = f"{int(row['world'])}-{int(row['stage'])}"
                except (ValueError, TypeError):
                    continue
            
            if level == "?" or pd.isna(level):
                continue
                
            if level not in level_stats:
                level_stats[level] = {
                    "episodes": 0,
                    "deaths": 0,
                    "flags": 0,
                    "best_x": 0,
                    "rewards": [],  # Track all rewards for min/max/avg
                    "speeds": [],   # Track speeds
                    "x_positions": [],  # Track x positions for progress analysis
                }
            
            level_stats[level]["episodes"] += 1
            level_stats[level]["deaths"] += int(row.get("deaths", 0))
            level_stats[level]["flags"] += int(row.get("flags", 0))
            
            best_x = row.get("best_x_ever", row.get("best_x", 0))
            if pd.notna(best_x):
                level_stats[level]["best_x"] = max(level_stats[level]["best_x"], int(best_x))
                level_stats[level]["x_positions"].append(int(best_x))
            
            # Track rewards for range calculation
            reward = row.get("reward", row.get("episode_reward", 0))
            if pd.notna(reward):
                level_stats[level]["rewards"].append(float(reward))
            
            # Track speed
            speed = row.get("speed", 0)
            if pd.notna(speed) and speed > 0:
                level_stats[level]["speeds"].append(float(speed))
    
    if not level_stats:
        st.info("⏳ Waiting for level data...")
        return
    
    # Summary table with reward range
    st.subheader("📊 Level Statistics")
    
    rows = []
    for level, stats in sorted(level_stats.items()):
        rewards = stats["rewards"]
        speeds = stats["speeds"]
        
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        min_reward = min(rewards) if rewards else 0
        max_reward = max(rewards) if rewards else 0
        avg_speed = sum(speeds) / len(speeds) if speeds else 0
        
        rows.append({
            "Level": level,
            "Episodes": stats["episodes"],
            "Best X": stats["best_x"],
            "Deaths": stats["deaths"],
            "Flags": stats["flags"],
            "Avg Reward": f"{avg_reward:.1f}",
            "Min R": f"{min_reward:.1f}",
            "Max R": f"{max_reward:.1f}",
            "Avg Speed": f"{avg_speed:.2f}",
        })
    
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    
    # Reward and episodes charts
    st.divider()
    st.subheader("📈 Level Performance")
    
    # Create box plot for rewards per level
    levels_with_data = [(l, s) for l, s in sorted(level_stats.items()) if len(s["rewards"]) > 0]
    
    if levels_with_data:
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Episodes per level bar chart
            level_names = [l for l, _ in levels_with_data]
            episode_counts = [s["episodes"] for _, s in levels_with_data]
            
        fig = go.Figure()
        fig.add_trace(go.Bar(
                x=level_names,
                y=episode_counts,
                marker_color=COLORS["mauve"],
                hovertemplate="Level %{x}<br>Episodes: %{y}<extra></extra>",
            ))
            
        fig.update_layout(
                title="Episodes per Level",
                yaxis_title="Episodes",
                height=350,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="#313244"),
                yaxis=dict(gridcolor="#313244"),
                margin=dict(l=0, r=0, t=50, b=0),
        )
            
        st.plotly_chart(fig, use_container_width=True)
    
        with chart_col2:
            # Reward distribution box plot
            fig = go.Figure()
            
            for level, stats in levels_with_data:
                rewards = stats["rewards"]
                fig.add_trace(go.Box(
                    y=rewards,
                    name=level,
                    boxpoints="outliers",
                    marker_color=COLORS["green"],
                    line_color=COLORS["teal"],
                ))
            
            fig.update_layout(
                title="Reward Distribution per Level",
                yaxis_title="Reward",
                height=350,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="#313244"),
                yaxis=dict(gridcolor="#313244"),
                margin=dict(l=0, r=0, t=50, b=0),
                showlegend=False,
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Progress chart (best X over time per level)
    col1, col2 = st.columns(2)
    
    with col1:
        if levels_with_data:
            fig = go.Figure()
            
            for i, (level, stats) in enumerate(levels_with_data):
                x_positions = stats["x_positions"]
                if x_positions:
                    # Show cumulative max (best X progression)
                    cummax = []
                    current_max = 0
                    for x in x_positions:
                        current_max = max(current_max, x)
                        cummax.append(current_max)
                    
                    colors = list(COLORS.values())
                    fig.add_trace(go.Scatter(
                        x=list(range(len(cummax))),
                        y=cummax,
                        name=level,
                        mode="lines",
                        line=dict(color=colors[i % len(colors)], width=2),
                    ))
            
            fig.update_layout(
                title="Best X Progression by Level",
                xaxis_title="Episode",
                yaxis_title="Best X Position",
                height=300,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="#313244"),
                yaxis=dict(gridcolor="#313244"),
                margin=dict(l=0, r=0, t=50, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Speed distribution per level (box plot)
        if any(len(s["speeds"]) > 0 for _, s in levels_with_data):
            fig = go.Figure()
            
            for level, stats in levels_with_data:
                speeds = stats["speeds"]
                if speeds:
                    fig.add_trace(go.Box(
                        y=speeds,
                        name=level,
                        boxpoints="outliers",
                        marker_color=COLORS["sky"],
                        line_color=COLORS["blue"],
                    ))
            
            fig.update_layout(
                title="Speed Distribution by Level",
                yaxis_title="Speed (x/time)",
                height=300,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="#313244"),
                yaxis=dict(gridcolor="#313244"),
                margin=dict(l=0, r=0, t=50, b=0),
                showlegend=False,
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Combined view: Action Distribution + Death Heatmap per level (side by side)
    st.subheader("🎮 Per-Level Analysis: Actions & Deaths")
    
    action_names = ["NOOP", "→", "→A", "→B", "→AB", "A", "←", "←A", "←B", "←AB", "↓", "↑"]
    
    # ===== Collect Action Distribution Data =====
    # Format: {level: [(steps, [pcts...]), ...]}
    level_action_data: dict[str, list[tuple[int, list[float]]]] = {}
    
    for wid, df in workers.items():
        if len(df) == 0 or "action_dist" not in df.columns:
            continue
        
        for _, row in df.iterrows():
            level = row.get("current_level", "?")
            if level == "?" and "world" in row and "stage" in row:
                try:
                    level = f"{int(row['world'])}-{int(row['stage'])}"
                except (ValueError, TypeError):
                    continue
            
            if level == "?" or pd.isna(level):
                continue
            
            dist_str = row.get("action_dist", "")
            steps = row.get("steps", 0)
            if dist_str and isinstance(dist_str, str):
                try:
                    pcts = [float(p) for p in dist_str.split(",")]
                    if len(pcts) == 12:
                        if level not in level_action_data:
                            level_action_data[level] = []
                        level_action_data[level].append((steps, pcts))
                except (ValueError, TypeError):
                    pass
    
    # ===== Collect Death Hotspot Data =====
    if death_hotspots is None or len(death_hotspots) == 0:
        csv_hotspots: dict[str, dict[int, int]] = {}
        
        for wid, df in workers.items():
            if "death_positions" not in df.columns:
                continue
            
            for _, row in df.iterrows():
                death_str = row.get("death_positions", "")
                if not death_str or not isinstance(death_str, str) or ":" not in death_str:
                    continue
                
                try:
                    level, positions_str = death_str.split(":", 1)
                    if not positions_str:
                        continue
                    
                    positions = [int(p.strip()) for p in positions_str.split(",") if p.strip()]
                    
                    if level not in csv_hotspots:
                        csv_hotspots[level] = {}
                    
                    for pos in positions:
                        bucket = (pos // 25) * 25
                        csv_hotspots[level][bucket] = csv_hotspots[level].get(bucket, 0) + 1
                except (ValueError, TypeError):
                    continue
        
        if csv_hotspots:
            death_hotspots = csv_hotspots
    
    # ===== Get All Levels (union of both data sources) =====
    all_levels: set[str] = set()
    all_levels.update(level_action_data.keys())
    if death_hotspots:
        all_levels.update(death_hotspots.keys())
    
    if not all_levels:
        st.info("⏳ No level data available yet...")
        return
    
    # Sort levels by world-stage (e.g., 1-1, 1-2, 1-3, 2-1, ...)
    def level_sort_key(lvl: str) -> tuple:
        try:
            parts = lvl.split("-")
            return (int(parts[0]), int(parts[1]))
        except (ValueError, IndexError):
            return (999, 999)
    
    sorted_levels = sorted(all_levels, key=level_sort_key)
    
    # Summary stats
    total_deaths = sum(sum(b.values()) for b in (death_hotspots or {}).values())
    stats_cols = st.columns(3)
    stats_cols[0].metric("Levels", len(sorted_levels))
    stats_cols[1].metric("Total Deaths", total_deaths)
    if death_hotspots:
        deadliest = max(death_hotspots.items(), key=lambda x: sum(x[1].values()), default=("?", {}))
        stats_cols[2].metric("Deadliest Level", deadliest[0])
    
    st.divider()
    
    # ===== Render Each Level as a Row =====
    for level in sorted_levels:
        st.markdown(f"### Level {level}")
        col_actions, col_deaths = st.columns(2)
        
        # ----- Left Column: Action Distribution Over Time -----
        with col_actions:
            if level in level_action_data and len(level_action_data[level]) >= 2:
                level_data = sorted(level_action_data[level], key=lambda x: x[0])
                
                # Sample to ~30 time points for compact heatmap
                sample_size = min(30, len(level_data))
                sample_indices = [int(i * len(level_data) / sample_size) for i in range(sample_size)]
                sampled_data = [level_data[i] for i in sample_indices]
                
                x_labels = [f"{int(d[0]//1000)}k" for d in sampled_data]
                z_data = [[d[1][action_idx] for d in sampled_data] for action_idx in range(12)]
                
                fig = go.Figure(data=go.Heatmap(
                    z=z_data,
                    x=x_labels,
                    y=action_names,
                    colorscale="Viridis",
                    showscale=False,
                    hovertemplate="Action: %{y}<br>Steps: %{x}<br>%{z:.1f}%<extra></extra>",
                ))
                
                fig.update_layout(
                    title=f"🎮 Actions Over Time",
                    height=250,
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(title="Steps", gridcolor="#313244"),
                    yaxis=dict(gridcolor="#313244"),
                    margin=dict(l=0, r=0, t=30, b=0),
                )
                
                st.plotly_chart(fig, use_container_width=True, key=f"actions_{level}")
                
                # Current top actions
                latest = level_data[-1][1]
                top_idx = sorted(range(12), key=lambda i: latest[i], reverse=True)[:3]
                top_actions = ", ".join(f"{action_names[i]} ({latest[i]:.0f}%)" for i in top_idx)
                st.caption(f"🏆 Current: {top_actions}")
            else:
                st.info("⏳ Not enough action data")
        
        # ----- Right Column: Death Heatmap (horizontal bar) -----
        with col_deaths:
            if death_hotspots and level in death_hotspots and death_hotspots[level]:
                buckets = death_hotspots[level]
                sorted_buckets = sorted(buckets.items(), key=lambda x: x[0])
                
                positions = [f"{b[0]}" for b in sorted_buckets]
                counts = [b[1] for b in sorted_buckets]
                
                fig = go.Figure(data=go.Bar(
                    x=counts,
                    y=positions,
                    orientation="h",
                    marker_color="crimson",
                    hovertemplate="Position: %{y}<br>Deaths: %{x}<extra></extra>",
                ))
                
                fig.update_layout(
                    title=f"💀 Deaths by Position",
                    height=250,
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(title="Deaths", gridcolor="#313244"),
                    yaxis=dict(title="X Position", gridcolor="#313244"),
                    margin=dict(l=0, r=0, t=30, b=0),
                )
                
                st.plotly_chart(fig, use_container_width=True, key=f"deaths_{level}")
                
                # Death stats
                total_level_deaths = sum(counts)
                hottest_pos = max(buckets.items(), key=lambda x: x[1])
                st.caption(f"💀 Total: {total_level_deaths} | Hotspot: x={hottest_pos[0]}")
            else:
                st.info("⏳ No death data")
        
        st.divider()
    
    # ===== Global Death Hotspots Table =====
    if death_hotspots:
        st.caption("🔥 TOP DEATH ZONES (ALL LEVELS)")
        
        all_hotspots = []
        for level, buckets in death_hotspots.items():
            for pos, count in buckets.items():
                all_hotspots.append((level, pos, count))
        
        top_spots = sorted(all_hotspots, key=lambda x: x[2], reverse=True)[:15]
        spot_rows = [
            {
                "Level": level,
                "Position": f"x={pos}-{pos+25}",
                "Deaths": count,
                "% of Total": f"{count/total_deaths*100:.1f}%" if total_deaths > 0 else "0%",
            }
            for level, pos, count in top_spots
        ]
        st.dataframe(pd.DataFrame(spot_rows), hide_index=True, use_container_width=True)
        
        # Deaths per level bar chart (compact)
        st.divider()
        st.caption("📊 TOTAL DEATHS PER LEVEL")
        
        level_deaths = {
            level: sum(buckets.values()) 
            for level, buckets in death_hotspots.items()
        }
        sorted_level_deaths = sorted(level_deaths.items(), key=lambda x: x[0])
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[l[0] for l in sorted_level_deaths],
            y=[l[1] for l in sorted_level_deaths],
            marker_color=COLORS["red"],
            hovertemplate="Level %{x}<br>Deaths: %{y}<extra></extra>",
        ))
        
        fig.update_layout(
            height=200,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#313244"),
            yaxis=dict(gridcolor="#313244"),
            margin=dict(l=0, r=0, t=10, b=0),
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

    loss_col = "avg_loss" if "avg_loss" in recent.columns else "loss"
    loss_trend = recent[loss_col].diff().mean() if loss_col in recent.columns else 0
    q_trend = recent["q_mean"].diff().mean() if "q_mean" in recent.columns else 0
    grad_norm_avg = recent["grad_norm"].mean() if "grad_norm" in recent.columns else 0

    col1, col2, col3, col4 = st.columns(4)

    # Loss trend
    if loss_trend > 1:
        col1.error(f"📈 Loss Rising (+{loss_trend:.2f}/update)")
    elif loss_trend < -0.5:
        col1.success(f"📉 Loss Falling ({loss_trend:.2f}/update)")
    else:
        col1.info(f"➡️ Loss Stable ({loss_trend:+.2f}/update)")

    # Q-value health
    q_mean = latest.get("q_mean", 0)
    if abs(q_mean) > 1000:
        col2.error(f"⚠️ Q-values large ({q_mean:.0f})")
    elif q_mean < -100:
        col2.warning(f"📉 Q-values negative ({q_mean:.1f})")
    else:
        col2.success(f"✓ Q-values OK ({q_mean:.1f})")

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

    # Combined worker analysis
    st.subheader("Worker Comparison")
    
    if workers:
        # Aggregate worker stats
        worker_stats = []
        for wid, wdf in sorted(workers.items()):
            if len(wdf) > 0:
                latest_w = wdf.iloc[-1]
                worker_stats.append({
                    "Worker": f"W{wid}",
                    "Episodes": int(latest_w.get("episodes", 0)),
                    "Best X": int(latest_w.get("best_x_ever", 0)),
                    "Avg Reward": wdf["reward"].mean() if "reward" in wdf.columns else 0,
                    "Avg Loss": wdf["loss"].mean() if "loss" in wdf.columns else 0,
                    "Avg Q": wdf["q_mean"].mean() if "q_mean" in wdf.columns else 0,
                    "TD Error": wdf["td_error"].mean() if "td_error" in wdf.columns else 0,
                })
        
        if worker_stats:
            st.dataframe(pd.DataFrame(worker_stats), hide_index=True, use_container_width=True)

    st.divider()

    # Progress over time
    st.subheader("Progress Timeline")

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Average Reward", "Learning Rate"))

    if "avg_reward" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["elapsed_min"], y=df["avg_reward"],
            fill="tozeroy", line=dict(color=COLORS["green"]),
        ), row=1, col=1)

    if "learning_rate" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["elapsed_min"], y=df["learning_rate"],
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
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Coordinator", "👷 Workers", "🗺️ Levels", "🔍 Analysis"])

    with tab1:
        @st.fragment(run_every=refresh_sec)
        def coordinator_fragment():
            load_coordinator_metrics.clear()
            df = load_coordinator_metrics(checkpoint_dir)
            if df is not None and len(df) > 0:
                st.caption(f"🔄 {datetime.now().strftime('%H:%M:%S')} • {len(df)} updates")
            render_coordinator_tab(df)
        coordinator_fragment()

    with tab2:
        @st.fragment(run_every=refresh_sec)
        def workers_fragment():
            load_worker_metrics.clear()
            workers = load_worker_metrics(checkpoint_dir)
            st.caption(f"🔄 {datetime.now().strftime('%H:%M:%S')}")
            render_workers_tab(workers)
        workers_fragment()

    with tab3:
        @st.fragment(run_every=refresh_sec)
        def levels_fragment():
            load_worker_metrics.clear()
            load_death_hotspots.clear()
            workers = load_worker_metrics(checkpoint_dir)
            death_hotspots = load_death_hotspots(checkpoint_dir)
            st.caption(f"🔄 {datetime.now().strftime('%H:%M:%S')}")
            render_levels_tab(workers, death_hotspots)
        levels_fragment()

    with tab4:
        @st.fragment(run_every=refresh_sec)
        def analysis_fragment():
            load_coordinator_metrics.clear()
            load_worker_metrics.clear()
            df = load_coordinator_metrics(checkpoint_dir)
            workers = load_worker_metrics(checkpoint_dir)
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
            coord_csv = checkpoint_path / "coordinator.csv"
            if coord_csv.exists():
                mod_time = datetime.fromtimestamp(coord_csv.stat().st_mtime)
                st.caption(f"📁 {checkpoint_path.name}")
                st.caption(f"🕐 {mod_time.strftime('%H:%M:%S')}")
            
            # Worker health summary
            import time
            workers = load_worker_metrics(checkpoint_dir)
            if workers:
                current_time = time.time()
                healthy = 0
                stale = 0
                crashed = 0
                
                for wid, df in workers.items():
                    if len(df) > 0:
                        last_timestamp = df.iloc[-1].get("timestamp", 0)
                        if last_timestamp > 0:
                            seconds_since = int(current_time - last_timestamp)
                            if seconds_since < 60:
                                healthy += 1
                            elif seconds_since < 120:
                                stale += 1
                            else:
                                crashed += 1
                
                st.divider()
                st.caption("WORKER HEALTH")
                health_cols = st.columns(3)
                health_cols[0].metric("💚", healthy, help="Healthy (<60s)")
                health_cols[1].metric("💛", stale, help="Stale (60-120s)")
                health_cols[2].metric("💔", crashed, help="Crashed (>120s)")

    # Check checkpoint
    if not Path(checkpoint_dir).exists():
        st.error(f"Directory not found: {checkpoint_dir}")
        return

    # Render content
    render_dashboard_content(checkpoint_dir, refresh_sec)


if __name__ == "__main__":
    main()
