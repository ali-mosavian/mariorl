"""Coordinator metrics tab for the training dashboard."""

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from mario_rl.dashboard.chart_helpers import COLORS, make_metric_chart


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
        fig = make_metric_chart(df, [
            ("avg_loss", "Avg Loss", COLORS["red"]),
            ("loss", "Loss", COLORS["peach"]),
        ], "Loss", x_col="elapsed_min")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = _make_q_td_chart(df)
        st.plotly_chart(fig, use_container_width=True)

    # Charts - row 2: Performance trends
    col3, col4 = st.columns(2)

    with col3:
        fig = make_metric_chart(df, [
            ("avg_reward", "Avg Reward", COLORS["green"]),
        ], "Average Reward (across workers)", x_col="elapsed_min")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = make_metric_chart(df, [
            ("avg_speed", "Avg Speed", COLORS["teal"]),
        ], "Average Speed (x-pos / game-time)", x_col="elapsed_min")
        st.plotly_chart(fig, use_container_width=True)

    # Charts - row 3: Diagnostics
    col5, col6 = st.columns(2)

    with col5:
        fig = make_metric_chart(df, [
            ("grad_norm", "Grad Norm", COLORS["yellow"]),
        ], "Gradient Norm", x_col="elapsed_min")
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        fig = _make_throughput_chart(df)
        st.plotly_chart(fig, use_container_width=True)


def _make_q_td_chart(df: pd.DataFrame) -> go.Figure:
    """Create dual y-axis chart for Q Mean and TD Error."""
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
    
    return fig


def _make_throughput_chart(df: pd.DataFrame) -> go.Figure:
    """Create dual y-axis chart for throughput metrics."""
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
    
    return fig
