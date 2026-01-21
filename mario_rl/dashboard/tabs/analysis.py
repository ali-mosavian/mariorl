"""Analysis/debugging tab for the training dashboard."""

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mario_rl.dashboard.chart_helpers import COLORS
from mario_rl.dashboard.chart_helpers import DARK_LAYOUT


def _detect_model_type(df: pd.DataFrame) -> str:
    """Detect model type from dataframe columns."""
    cols = set(df.columns)
    # Dreamer-specific columns
    if "wm_loss" in cols or "actor_loss" in cols or "world_loss" in cols:
        return "dreamer"
    # DDQN-specific columns
    if "q_mean" in cols or "td_error" in cols:
        return "ddqn"
    return "unknown"


def render_analysis_tab(df: pd.DataFrame, workers: dict[int, pd.DataFrame]) -> None:
    """Render analysis/debugging tab."""
    if df is None or len(df) < 2:
        st.info("⏳ Need more data for analysis...")
        return

    # Detect model type from worker data (more metrics available there)
    model_type = "unknown"
    for wdf in workers.values():
        if len(wdf) > 0:
            model_type = _detect_model_type(wdf)
            if model_type != "unknown":
                break

    # Fall back to coordinator data
    if model_type == "unknown":
        model_type = _detect_model_type(df)

    st.subheader("Training Health")
    _render_health_indicators(df, model_type)

    st.divider()
    st.subheader("Worker Comparison")
    _render_worker_comparison(workers, model_type)

    st.divider()
    st.subheader("Progress Timeline")
    _render_progress_timeline(df)


def _render_health_indicators(df: pd.DataFrame, model_type: str = "ddqn") -> None:
    """Render training health indicator cards."""
    latest = df.iloc[-1]
    recent = df.tail(10)

    loss_col = "avg_loss" if "avg_loss" in recent.columns else "loss"
    loss_trend = recent[loss_col].diff().mean() if loss_col in recent.columns else 0
    grad_norm_avg = recent["grad_norm"].mean() if "grad_norm" in recent.columns else 0

    col1, col2, col3, col4 = st.columns(4)

    # Loss trend
    if loss_trend > 1:
        col1.error(f"📈 Loss Rising (+{loss_trend:.2f}/update)")
    elif loss_trend < -0.5:
        col1.success(f"📉 Loss Falling ({loss_trend:.2f}/update)")
    else:
        col1.info(f"➡️ Loss Stable ({loss_trend:+.2f}/update)")

    # Model-specific health indicator
    if model_type == "dreamer":
        # Dreamer: World model loss health
        wm_loss = latest.get("wm_loss", latest.get("world_loss", 0))
        if wm_loss > 10:
            col2.error(f"⚠️ High WM loss ({wm_loss:.2f})")
        elif wm_loss > 1:
            col2.warning(f"📊 Elevated WM loss ({wm_loss:.2f})")
        else:
            col2.success(f"✓ WM loss OK ({wm_loss:.3f})")
    else:
        # DDQN: Q-value health
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


def _render_worker_comparison(workers: dict[int, pd.DataFrame], model_type: str = "ddqn") -> None:
    """Render worker comparison table."""
    if not workers:
        st.info("No worker data available")
        return

    worker_stats = []
    for wid, wdf in sorted(workers.items()):
        if len(wdf) > 0:
            latest_w = wdf.iloc[-1]
            row = {
                "Worker": f"W{wid}",
                "Episodes": int(latest_w.get("episodes", 0)),
                "Best X": int(latest_w.get("best_x_ever", 0)),
                "Avg Reward": wdf["reward"].mean() if "reward" in wdf.columns else 0,
                "Avg Loss": wdf["loss"].mean() if "loss" in wdf.columns else 0,
            }

            if model_type == "dreamer":
                row.update(
                    {
                        "WM Loss": wdf["wm_loss"].mean() if "wm_loss" in wdf.columns else 0,
                        "Actor Loss": wdf["actor_loss"].mean() if "actor_loss" in wdf.columns else 0,
                        "Critic Loss": wdf["critic_loss"].mean() if "critic_loss" in wdf.columns else 0,
                    }
                )
            else:
                row.update(
                    {
                        "Avg Q": wdf["q_mean"].mean() if "q_mean" in wdf.columns else 0,
                        "TD Error": wdf["td_error"].mean() if "td_error" in wdf.columns else 0,
                    }
                )

            worker_stats.append(row)

    if worker_stats:
        st.dataframe(pd.DataFrame(worker_stats), hide_index=True, use_container_width=True)


def _render_progress_timeline(df: pd.DataFrame) -> None:
    """Render progress timeline charts."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Average Reward", "Learning Rate"))

    if "avg_reward" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["elapsed_min"],
                y=df["avg_reward"],
                fill="tozeroy",
                line={"color": COLORS["green"]},
            ),
            row=1,
            col=1,
        )

    if "learning_rate" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["elapsed_min"],
                y=df["learning_rate"],
                line={"color": COLORS["mauve"]},
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        height=250,
        showlegend=False,
        **DARK_LAYOUT,
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
    )
    fig.update_xaxes(title_text="Time (min)", gridcolor="#313244")
    fig.update_yaxes(gridcolor="#313244")

    st.plotly_chart(fig, use_container_width=True)
