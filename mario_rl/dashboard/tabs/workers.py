"""Workers metrics tab for the training dashboard."""

import time

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from mario_rl.dashboard.chart_helpers import COLORS, DARK_LAYOUT, GRID_STYLE, make_heatmap
from mario_rl.dashboard.aggregators import sample_data


# Action names for Mario
ACTION_NAMES = ["NOOP", "→", "→A", "→B", "→AB", "A", "←", "←A", "←B", "←AB", "↓", "↑"]


def render_workers_tab(workers: dict[int, pd.DataFrame]) -> None:
    """Render the workers tab."""
    if not workers:
        st.info("⏳ Waiting for worker data...")
        return

    # Summary table
    _render_summary_table(workers)
    st.divider()

    # Comparison charts
    colors = list(COLORS.values())
    
    # Row 1: Reward and Speed
    col1, col2 = st.columns(2)
    with col1:
        _render_worker_chart(workers, "reward", "Rolling Avg Reward by Worker", colors, show_legend=True)
    with col2:
        _render_worker_chart(workers, "speed", "Speed by Worker", colors)

    # Row 2: Best X and Loss
    col3, col4 = st.columns(2)
    with col3:
        _render_worker_chart(workers, "best_x_ever", "Best X Position by Worker", colors)
    with col4:
        _render_worker_chart(workers, "loss", "Loss by Worker", colors)

    # Row 3: Q-values and TD error
    col5, col6 = st.columns(2)
    with col5:
        _render_worker_chart(workers, "q_mean", "Q Mean by Worker", colors)
    with col6:
        _render_worker_chart(workers, "td_error", "TD Error by Worker", colors)

    # Row 4: Epsilon and PER Beta
    col7, col8 = st.columns(2)
    with col7:
        _render_epsilon_chart(workers, colors)
    with col8:
        _render_beta_or_buffer_chart(workers, colors)

    # Row 5: Action entropy and distribution
    col9, col10 = st.columns(2)
    with col9:
        _render_action_entropy_chart(workers, colors)
    with col10:
        _render_action_distribution_heatmap(workers)


def _render_summary_table(workers: dict[int, pd.DataFrame]) -> None:
    """Render the worker summary table."""
    rows = []
    current_time = time.time()
    
    for wid, df in sorted(workers.items()):
        if len(df) == 0:
            continue
        latest = df.iloc[-1]
        best_x = latest.get("best_x_ever", latest.get("best_x", 0))
        
        # Calculate heartbeat status
        last_timestamp = latest.get("timestamp", 0)
        if last_timestamp > 0:
            seconds_since = int(current_time - last_timestamp)
            if seconds_since < 60:
                heartbeat = "💚"
            elif seconds_since < 120:
                heartbeat = "💛"
            else:
                heartbeat = "💔"
        else:
            heartbeat = "⚪"
        
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
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Status": st.column_config.TextColumn(
                    "Status",
                    help="💚 Healthy (<60s) | 💛 Stale (60-120s) | 💔 Crashed (>120s)",
                    width="small",
                ),
            },
        )


def _render_worker_chart(
    workers: dict[int, pd.DataFrame],
    column: str,
    title: str,
    colors: list[str],
    x_column: str = "episodes",
    show_legend: bool = False,
) -> None:
    """Render a comparison chart for a single metric across workers."""
    fig = go.Figure()
    
    for i, (wid, df) in enumerate(sorted(workers.items())):
        if len(df) > 0 and column in df.columns:
            x = df.get(x_column, range(len(df)))
            fig.add_trace(go.Scatter(
                x=x, y=df[column],
                name=f"W{wid}",
                line=dict(color=colors[i % len(colors)], width=1.5),
                opacity=0.8,
            ))
    
    fig.update_layout(
        title=title,
        height=280,
        margin=dict(l=0, r=0, t=30, b=0),
        **DARK_LAYOUT,
        xaxis=dict(title="Episode" if x_column == "episodes" else "Steps", **GRID_STYLE),
        yaxis=dict(**GRID_STYLE),
        showlegend=show_legend,
    )
    
    if show_legend:
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
    
    st.plotly_chart(fig, use_container_width=True)


def _render_epsilon_chart(workers: dict[int, pd.DataFrame], colors: list[str]) -> None:
    """Render epsilon decay chart by worker."""
    fig = go.Figure()
    
    for i, (wid, df) in enumerate(sorted(workers.items())):
        if len(df) > 0 and "epsilon" in df.columns:
            x_axis = df["steps"] if "steps" in df.columns else df.get("episodes", range(len(df)))
            fig.add_trace(go.Scatter(
                x=x_axis, y=df["epsilon"],
                name=f"W{wid}",
                line=dict(color=colors[i % len(colors)], width=1.5),
                opacity=0.8,
            ))
    
    fig.update_layout(
        title="Epsilon by Worker (vs Steps)",
        height=280,
        margin=dict(l=0, r=0, t=30, b=0),
        **DARK_LAYOUT,
        xaxis=dict(title="Steps", **GRID_STYLE),
        yaxis=dict(title="ε", range=[0, 1.05], **GRID_STYLE),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_beta_or_buffer_chart(workers: dict[int, pd.DataFrame], colors: list[str]) -> None:
    """Render PER beta chart (or fallback to buffer size)."""
    fig = go.Figure()
    has_beta = False
    
    for i, (wid, df) in enumerate(sorted(workers.items())):
        if len(df) > 0 and "per_beta" in df.columns:
            has_beta = True
            x_axis = df["steps"] if "steps" in df.columns else df.get("episodes", range(len(df)))
            fig.add_trace(go.Scatter(
                x=x_axis, y=df["per_beta"],
                name=f"W{wid}",
                line=dict(color=colors[i % len(colors)], width=1.5),
                opacity=0.8,
            ))
    
    if has_beta:
        fig.update_layout(
            title="PER Beta by Worker (vs Steps)",
            height=280,
            margin=dict(l=0, r=0, t=30, b=0),
            **DARK_LAYOUT,
            xaxis=dict(title="Steps", **GRID_STYLE),
            yaxis=dict(title="β", range=[0, 1.05], **GRID_STYLE),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Fallback to buffer size
        fig = go.Figure()
        for i, (wid, df) in enumerate(sorted(workers.items())):
            if len(df) > 0 and "buffer_size" in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.get("episodes", range(len(df))), y=df["buffer_size"],
                    name=f"W{wid}",
                    line=dict(color=colors[i % len(colors)], width=1.5),
                    opacity=0.8,
                ))
        
        fig.update_layout(
            title="Buffer Size by Worker",
            height=280,
            margin=dict(l=0, r=0, t=30, b=0),
            **DARK_LAYOUT,
            xaxis=dict(title="Episode", **GRID_STYLE),
            yaxis=dict(**GRID_STYLE),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_action_entropy_chart(workers: dict[int, pd.DataFrame], colors: list[str]) -> None:
    """Render action entropy chart by worker."""
    fig = go.Figure()
    has_data = False
    
    for i, (wid, df) in enumerate(sorted(workers.items())):
        if len(df) > 0 and "action_entropy" in df.columns:
            has_data = True
            x_axis = df["steps"] if "steps" in df.columns else df.get("episodes", range(len(df)))
            fig.add_trace(go.Scatter(
                x=x_axis, y=df["action_entropy"],
                name=f"W{wid}",
                line=dict(color=colors[i % len(colors)], width=1.5),
                opacity=0.8,
            ))
    
    if has_data:
        fig.update_layout(
            title="Action Entropy by Worker (0=deterministic, 1=uniform)",
            height=280,
            margin=dict(l=0, r=0, t=30, b=0),
            **DARK_LAYOUT,
            xaxis=dict(title="Steps", **GRID_STYLE),
            yaxis=dict(title="Entropy", range=[0, 1.05], **GRID_STYLE),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("⏳ Action entropy tracking not yet available...")


def _render_action_distribution_heatmap(workers: dict[int, pd.DataFrame]) -> None:
    """Render action distribution heatmap over time."""
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
        # Sort by steps and sample
        all_action_data.sort(key=lambda x: x[0])
        sampled_data = sample_data(all_action_data, max_points=50)
        
        # Build heatmap matrix
        x_labels = [f"{int(d[0]//1000)}k" for d in sampled_data]
        z_data = [[d[1][action_idx] for d in sampled_data] for action_idx in range(12)]
        
        fig = make_heatmap(
            z_data=z_data,
            x_labels=x_labels,
            y_labels=ACTION_NAMES,
            title="Action Distribution Over Time (%)",
            height=280,
        )
        fig.update_layout(xaxis=dict(title="Steps"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("⏳ Action distribution tracking not yet available...")
