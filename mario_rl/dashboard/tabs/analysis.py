"""Analysis/debugging tab for the training dashboard."""

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mario_rl.dashboard.chart_helpers import COLORS, DARK_LAYOUT, GRID_STYLE


def render_analysis_tab(df: pd.DataFrame, workers: dict[int, pd.DataFrame]) -> None:
    """Render analysis/debugging tab."""
    if df is None or len(df) < 2:
        st.info("⏳ Need more data for analysis...")
        return

    st.subheader("Training Health")
    _render_health_indicators(df)
    
    st.divider()
    st.subheader("Worker Comparison")
    _render_worker_comparison(workers)
    
    st.divider()
    st.subheader("Progress Timeline")
    _render_progress_timeline(df)


def _render_health_indicators(df: pd.DataFrame) -> None:
    """Render training health indicator cards."""
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


def _render_worker_comparison(workers: dict[int, pd.DataFrame]) -> None:
    """Render worker comparison table."""
    if not workers:
        st.info("No worker data available")
        return
    
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


def _render_progress_timeline(df: pd.DataFrame) -> None:
    """Render progress timeline charts."""
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
        **DARK_LAYOUT,
        margin=dict(l=0, r=0, t=30, b=0),
    )
    fig.update_xaxes(title_text="Time (min)", gridcolor="#313244")
    fig.update_yaxes(gridcolor="#313244")

    st.plotly_chart(fig, use_container_width=True)
