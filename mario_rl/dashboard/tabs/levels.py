"""Levels metrics tab for the training dashboard."""

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from mario_rl.dashboard.chart_helpers import (
    COLORS,
    DARK_LAYOUT,
    GRID_STYLE,
    make_bar_chart,
    make_box_plot,
    make_heatmap,
    make_dual_axis_chart,
)
from mario_rl.dashboard.aggregators import (
    LevelStats,
    aggregate_level_stats,
    aggregate_action_distribution,
    aggregate_rate_data,
    aggregate_death_hotspots_from_csv,
    level_sort_key,
    sample_data,
)


# Action names for Mario
ACTION_NAMES = ["NOOP", "→", "→A", "→B", "→AB", "A", "←", "←A", "←B", "←AB", "↓", "↑"]


def render_levels_tab(
    workers: dict[int, pd.DataFrame],
    death_hotspots: dict[str, dict[int, int]] | None,
) -> None:
    """Render levels tab with death hotspot visualization."""
    
    # Aggregate all data
    level_stats = aggregate_level_stats(workers)
    action_data = aggregate_action_distribution(workers)
    rate_data = aggregate_rate_data(workers)
    
    # Get death hotspots from CSV if not provided
    if death_hotspots is None or len(death_hotspots) == 0:
        death_hotspots = aggregate_death_hotspots_from_csv(workers)
    
    if not level_stats:
        st.info("⏳ Waiting for level data...")
        return
    
    # Level statistics table
    _render_level_stats_table(level_stats)
    
    # Level performance charts
    st.divider()
    st.subheader("📈 Level Performance")
    _render_performance_charts(level_stats)
    
    # Per-level analysis section
    st.divider()
    st.subheader("🎮 Per-Level Analysis: Actions & Deaths")
    _render_per_level_analysis(level_stats, action_data, rate_data, death_hotspots)
    
    # Global death hotspots table
    if death_hotspots:
        _render_death_hotspots_summary(death_hotspots)


def _render_level_stats_table(level_stats: dict[str, LevelStats]) -> None:
    """Render the level statistics summary table."""
    st.subheader("📊 Level Statistics")
    
    rows = []
    for level, stats in sorted(level_stats.items(), key=lambda x: level_sort_key(x[0])):
        rewards = stats.rewards
        speeds = stats.speeds
        
        rows.append({
            "Level": level,
            "Episodes": stats.episodes,
            "Best X": stats.best_x,
            "Deaths": stats.deaths,
            "Flags": stats.flags,
            "Avg Reward": f"{stats.avg_reward:.1f}",
            "Min R": f"{min(rewards):.1f}" if rewards else "0.0",
            "Max R": f"{max(rewards):.1f}" if rewards else "0.0",
            "Avg Speed": f"{stats.avg_speed:.2f}",
        })
    
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _render_performance_charts(level_stats: dict[str, LevelStats]) -> None:
    """Render level performance comparison charts."""
    levels_with_data = [
        (l, s) for l, s in sorted(level_stats.items(), key=lambda x: level_sort_key(x[0]))
        if len(s.rewards) > 0
    ]
    
    if not levels_with_data:
        return
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Episodes per level bar chart
        level_names = [l for l, _ in levels_with_data]
        episode_counts = [s.episodes for _, s in levels_with_data]
        
        fig = make_bar_chart(
            x=level_names,
            y=episode_counts,
            title="Episodes per Level",
            y_title="Episodes",
            height=350,
            color=COLORS["mauve"],
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with chart_col2:
        # Reward distribution box plot
        data = {l: s.rewards for l, s in levels_with_data if s.rewards}
        fig = make_box_plot(
            data_by_category=data,
            title="Reward Distribution per Level",
            y_title="Reward",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Second row: Best X progression and Speed distribution
    col1, col2 = st.columns(2)
    
    with col1:
        _render_best_x_progression(levels_with_data)
    
    with col2:
        # Speed distribution
        data = {l: s.speeds for l, s in levels_with_data if s.speeds}
        if data:
            fig = make_box_plot(
                data_by_category=data,
                title="Speed Distribution by Level",
                y_title="Speed (x/time)",
                height=300,
                color=COLORS["sky"],
                line_color=COLORS["blue"],
            )
            st.plotly_chart(fig, use_container_width=True)


def _render_best_x_progression(levels_with_data: list[tuple[str, LevelStats]]) -> None:
    """Render best X position progression chart."""
    fig = go.Figure()
    colors = list(COLORS.values())
    
    for i, (level, stats) in enumerate(levels_with_data):
        x_positions = stats.x_positions
        if x_positions:
            # Show cumulative max
            cummax = []
            current_max = 0
            for x in x_positions:
                current_max = max(current_max, x)
                cummax.append(current_max)
            
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
        **DARK_LAYOUT,
        xaxis=dict(**GRID_STYLE),
        yaxis=dict(**GRID_STYLE),
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_per_level_analysis(
    level_stats: dict[str, LevelStats],
    action_data: dict[str, list],
    rate_data: dict[str, list],
    death_hotspots: dict[str, dict[int, int]] | None,
) -> None:
    """Render per-level analysis with actions, deaths, and rates side by side."""
    
    # Get all levels from all data sources
    all_levels: set[str] = set()
    all_levels.update(level_stats.keys())
    all_levels.update(action_data.keys())
    all_levels.update(rate_data.keys())
    if death_hotspots:
        all_levels.update(death_hotspots.keys())
    
    if not all_levels:
        st.info("⏳ No level data available yet...")
        return
    
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
    
    # Render each level as a row
    for level in sorted_levels:
        st.markdown(f"### Level {level}")
        col_actions, col_deaths, col_rates = st.columns(3)
        
        with col_actions:
            _render_level_action_heatmap(level, action_data)
        
        with col_deaths:
            _render_level_death_chart(level, death_hotspots)
        
        with col_rates:
            _render_level_rate_chart(level, rate_data)
        
        st.divider()


def _render_level_action_heatmap(level: str, action_data: dict[str, list]) -> None:
    """Render action distribution heatmap for a single level."""
    if level in action_data and len(action_data[level]) >= 2:
        level_data = sorted(action_data[level], key=lambda x: x.steps)
        sampled = sample_data(level_data, max_points=30)
        
        x_labels = [f"{int(d.steps // 1000)}k" for d in sampled]
        z_data = [[d.percentages[action_idx] for d in sampled] for action_idx in range(12)]
        
        fig = make_heatmap(
            z_data=z_data,
            x_labels=x_labels,
            y_labels=ACTION_NAMES,
            title="🎮 Actions Over Time",
            height=250,
        )
        fig.update_layout(xaxis=dict(title="Steps"))
        st.plotly_chart(fig, use_container_width=True, key=f"actions_{level}")
        
        # Current top actions
        latest = level_data[-1].percentages
        top_idx = sorted(range(12), key=lambda i: latest[i], reverse=True)[:3]
        top_actions = ", ".join(f"{ACTION_NAMES[i]} ({latest[i]:.0f}%)" for i in top_idx)
        st.caption(f"🏆 Current: {top_actions}")
    else:
        st.info("⏳ Not enough action data")


def _render_level_death_chart(level: str, death_hotspots: dict[str, dict[int, int]] | None) -> None:
    """Render death position chart for a single level."""
    if death_hotspots and level in death_hotspots and death_hotspots[level]:
        buckets = death_hotspots[level]
        sorted_buckets = sorted(buckets.items(), key=lambda x: x[0])
        
        positions = [f"{b[0]}" for b in sorted_buckets]
        counts = [b[1] for b in sorted_buckets]
        
        fig = make_bar_chart(
            x=positions,
            y=counts,
            title="💀 Deaths by Position",
            height=250,
            color="crimson",
            orientation="h",
            x_title="Deaths",
            y_title="X Position",
        )
        st.plotly_chart(fig, use_container_width=True, key=f"deaths_{level}")
        
        # Death stats
        total_level_deaths = sum(counts)
        hottest_pos = max(buckets.items(), key=lambda x: x[1])
        st.caption(f"💀 Total: {total_level_deaths} | Hotspot: x={hottest_pos[0]}")
    else:
        st.info("⏳ No death position data")


def _render_level_rate_chart(level: str, rate_data: dict[str, list]) -> None:
    """Render death/completion rate chart for a single level."""
    if level in rate_data and len(rate_data[level]) >= 2:
        level_data = sorted(rate_data[level], key=lambda x: x.steps)
        sampled = sample_data(level_data, max_points=30)
        
        x_labels = [f"{int(d.steps // 1000)}k" for d in sampled]
        deaths_per_ep = [d.deaths_per_episode for d in sampled]
        completion_rates = [d.completion_rate for d in sampled]
        
        fig = make_dual_axis_chart(
            x=x_labels,
            y1=deaths_per_ep,
            y2=completion_rates,
            name1="💀 Deaths/Ep",
            name2="🏁 Complete",
            title="📈 Death & Completion Rate",
            y1_title="Deaths/Ep",
            y2_title="Complete %",
            height=250,
        )
        st.plotly_chart(fig, use_container_width=True, key=f"rates_{level}")
        
        # Current rates
        st.caption(f"💀 Deaths/Ep: {deaths_per_ep[-1]:.2f} | 🏁 Complete: {completion_rates[-1]:.1f}%")
    else:
        st.info("⏳ Not enough rate data")


def _render_death_hotspots_summary(death_hotspots: dict[str, dict[int, int]]) -> None:
    """Render global death hotspots summary."""
    total_deaths = sum(sum(b.values()) for b in death_hotspots.values())
    
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
    
    # Deaths per level bar chart
    st.divider()
    st.caption("📊 TOTAL DEATHS PER LEVEL")
    
    level_deaths = {
        level: sum(buckets.values()) 
        for level, buckets in death_hotspots.items()
    }
    sorted_level_deaths = sorted(level_deaths.items(), key=lambda x: x[0])
    
    fig = make_bar_chart(
        x=[l[0] for l in sorted_level_deaths],
        y=[l[1] for l in sorted_level_deaths],
        title="",
        height=200,
        color=COLORS["red"],
    )
    st.plotly_chart(fig, use_container_width=True)
