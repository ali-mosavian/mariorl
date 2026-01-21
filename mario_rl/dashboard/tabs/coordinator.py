"""Coordinator metrics tab for the training dashboard."""

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from mario_rl.dashboard.chart_helpers import COLORS
from mario_rl.dashboard.chart_helpers import make_metric_chart


def _detect_model_type(df: pd.DataFrame) -> str:
    """Detect model type from coordinator CSV columns."""
    cols = set(df.columns)
    # Dreamer-specific columns
    if "wm_loss" in cols or "actor_loss" in cols or "world_loss" in cols:
        return "dreamer"
    # DDQN-specific columns
    if "q_mean" in cols or "td_error" in cols:
        return "ddqn"
    return "unknown"


def render_coordinator_tab(df: pd.DataFrame) -> None:
    """Render the coordinator metrics tab."""
    if df is None or len(df) == 0:
        st.info("⏳ Waiting for coordinator data...")
        return

    latest = df.iloc[-1]
    model_type = _detect_model_type(df)

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

    if model_type == "dreamer":
        cols2[3].metric("WM Loss", f"{latest.get('wm_loss', 0):.4f}")
        cols2[4].metric("Actor Loss", f"{latest.get('actor_loss', 0):.4f}")
    else:
        cols2[3].metric("Q Mean", f"{latest.get('q_mean', 0):.2f}")
        cols2[4].metric("TD Error", f"{latest.get('td_error', 0):.4f}")

    cols2[5].metric("Grad Norm", f"{latest.get('grad_norm', 0):.2f}")
    cols2[6].metric("Total SPS", f"{latest.get('total_sps', 0):.0f}")

    st.divider()

    # Charts - row 1: Training health
    col1, col2 = st.columns(2)

    with col1:
        fig = make_metric_chart(
            df,
            [
                ("avg_loss", "Avg Loss", COLORS["red"]),
                ("loss", "Loss", COLORS["peach"]),
            ],
            "Loss",
            x_col="elapsed_min",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if model_type == "dreamer":
            fig = _make_dreamer_loss_chart(df)
        else:
            fig = _make_q_td_chart(df)
        st.plotly_chart(fig, use_container_width=True)

    # Charts - row 2: Performance trends
    col3, col4 = st.columns(2)

    with col3:
        fig = make_metric_chart(
            df,
            [
                ("avg_reward", "Avg Reward", COLORS["green"]),
            ],
            "Average Reward (across workers)",
            x_col="elapsed_min",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = make_metric_chart(
            df,
            [
                ("avg_speed", "Avg Speed", COLORS["teal"]),
            ],
            "Average Speed (x-pos / game-time)",
            x_col="elapsed_min",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Charts - row 3: Diagnostics
    col5, col6 = st.columns(2)

    with col5:
        fig = make_metric_chart(
            df,
            [
                ("grad_norm", "Grad Norm", COLORS["yellow"]),
            ],
            "Gradient Norm",
            x_col="elapsed_min",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        fig = _make_throughput_chart(df)
        st.plotly_chart(fig, use_container_width=True)


def _make_q_td_chart(df: pd.DataFrame) -> go.Figure:
    """Create dual y-axis chart for Q Mean and TD Error (DDQN)."""
    fig = go.Figure()

    if "q_mean" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["elapsed_min"],
                y=df["q_mean"],
                name="Q Mean",
                line={"color": COLORS["blue"], "width": 2},
                mode="lines",
                yaxis="y",
            )
        )

    if "td_error" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["elapsed_min"],
                y=df["td_error"],
                name="TD Error",
                line={"color": COLORS["sky"], "width": 2},
                mode="lines",
                yaxis="y2",
            )
        )

    fig.update_layout(
        title={"text": "Q-Values & TD Error", "font": {"size": 14}},
        height=280,
        margin={"l": 0, "r": 40, "t": 40, "b": 0},
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1, "font": {"size": 10}},
        xaxis={"title": "Time (min)", "gridcolor": "#313244", "showgrid": True},
        yaxis={
            "title": {"text": "Q Mean", "font": {"color": COLORS["blue"]}},
            "gridcolor": "#313244",
            "showgrid": True,
            "side": "left",
        },
        yaxis2={
            "title": {"text": "TD Error", "font": {"color": COLORS["sky"]}},
            "gridcolor": "#313244",
            "overlaying": "y",
            "side": "right",
        },
    )

    return fig


def _make_dreamer_loss_chart(df: pd.DataFrame) -> go.Figure:
    """Create multi-line chart for Dreamer component losses."""
    fig = go.Figure()

    if "wm_loss" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["elapsed_min"],
                y=df["wm_loss"],
                name="World Model",
                line={"color": COLORS["blue"], "width": 2},
                mode="lines",
            )
        )

    if "actor_loss" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["elapsed_min"],
                y=df["actor_loss"],
                name="Actor",
                line={"color": COLORS["green"], "width": 2},
                mode="lines",
            )
        )

    if "critic_loss" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["elapsed_min"],
                y=df["critic_loss"],
                name="Critic",
                line={"color": COLORS["yellow"], "width": 2},
                mode="lines",
            )
        )

    fig.update_layout(
        title={"text": "Dreamer Component Losses", "font": {"size": 14}},
        height=280,
        margin={"l": 0, "r": 40, "t": 40, "b": 0},
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1, "font": {"size": 10}},
        xaxis={"title": "Time (min)", "gridcolor": "#313244", "showgrid": True},
        yaxis={"title": "Loss", "gridcolor": "#313244", "showgrid": True},
    )

    return fig


def _make_throughput_chart(df: pd.DataFrame) -> go.Figure:
    """Create dual y-axis chart for throughput metrics."""
    fig = go.Figure()

    if "total_sps" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["elapsed_min"],
                y=df["total_sps"],
                name="Steps/sec",
                line={"color": COLORS["green"], "width": 2},
                mode="lines",
                yaxis="y",
            )
        )

    if "grads_per_sec" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["elapsed_min"],
                y=df["grads_per_sec"],
                name="Grads/sec",
                line={"color": COLORS["mauve"], "width": 2},
                mode="lines",
                yaxis="y2",
            )
        )

    fig.update_layout(
        title={"text": "Throughput", "font": {"size": 14}},
        height=280,
        margin={"l": 0, "r": 40, "t": 40, "b": 0},
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1, "font": {"size": 10}},
        xaxis={"title": "Time (min)", "gridcolor": "#313244", "showgrid": True},
        yaxis={
            "title": {"text": "SPS", "font": {"color": COLORS["green"]}},
            "gridcolor": "#313244",
            "showgrid": True,
            "side": "left",
        },
        yaxis2={
            "title": {"text": "Grads/s", "font": {"color": COLORS["mauve"]}},
            "gridcolor": "#313244",
            "overlaying": "y",
            "side": "right",
        },
    )

    return fig
