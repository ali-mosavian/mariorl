"""Levels metrics tab for the training dashboard."""

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from mario_rl.dashboard.chart_helpers import (
    COLORS,
    DARK_LAYOUT,
    GRID_STYLE,
    make_bar_chart,
    make_heatmap,
    make_dual_axis_chart,
)
from mario_rl.dashboard.aggregators import (
    DeathDistPoint,
    LevelStats,
    aggregate_level_stats_direct,
    aggregate_action_distribution_direct,
    aggregate_rate_data_direct,
    aggregate_death_distribution_direct,
    aggregate_death_hotspots_direct,
    level_sort_key,
    sample_data,
)


# Action names for Mario (SIMPLE_MOVEMENT: 7 actions)
ACTION_NAMES = ["NOOP", "→", "→A", "→B", "→AB", "A", "←"]


# Cached aggregation functions - query files directly via DuckDB
@st.cache_data(ttl=5)
def _cached_level_stats(checkpoint_dir: str) -> dict[str, LevelStats]:
    return aggregate_level_stats_direct(checkpoint_dir)


@st.cache_data(ttl=5)
def _cached_action_distribution(checkpoint_dir: str) -> dict[str, list]:
    return aggregate_action_distribution_direct(checkpoint_dir)


@st.cache_data(ttl=5)
def _cached_rate_data(checkpoint_dir: str) -> dict[str, list]:
    return aggregate_rate_data_direct(checkpoint_dir)


@st.cache_data(ttl=5)
def _cached_death_distribution(checkpoint_dir: str) -> dict[str, list[DeathDistPoint]]:
    return aggregate_death_distribution_direct(checkpoint_dir)


@st.cache_data(ttl=5)
def _cached_death_hotspots(checkpoint_dir: str) -> dict[str, dict[int, int]]:
    return aggregate_death_hotspots_direct(checkpoint_dir)


def render_levels_tab(
    checkpoint_dir: str,
    death_hotspots: dict[str, dict[int, int]] | None,
) -> None:
    """Render levels tab with death hotspot visualization.
    
    Uses cached direct queries to DuckDB for optimal performance.
    """
    # Aggregate all data using cached direct queries
    level_stats = _cached_level_stats(checkpoint_dir)
    action_data = _cached_action_distribution(checkpoint_dir)
    rate_data = _cached_rate_data(checkpoint_dir)
    death_dist_data = _cached_death_distribution(checkpoint_dir)
    
    # Get death hotspots from CSV if not provided (for summary table)
    if death_hotspots is None or len(death_hotspots) == 0:
        death_hotspots = _cached_death_hotspots(checkpoint_dir)
    
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
    _render_per_level_analysis(level_stats, action_data, rate_data, death_dist_data, death_hotspots)
    
    # Global death hotspots table
    if death_hotspots:
        _render_death_hotspots_summary(death_hotspots)


def _render_level_stats_table(level_stats: dict[str, LevelStats]) -> None:
    """Render the level statistics summary table."""
    st.subheader("📊 Level Statistics")
    
    rows = []
    for level, stats in sorted(level_stats.items(), key=lambda x: level_sort_key(x[0])):
        rows.append({
            "Level": level,
            "Episodes": stats.episodes,
            "Best X": stats.best_x,
            "Deaths": stats.deaths,
            "Flags": stats.flags,
            "Avg Reward": f"{stats.avg_reward:.1f}",
            "Min R": f"{stats.min_reward:.1f}",
            "Max R": f"{stats.max_reward:.1f}",
            "Avg Speed": f"{stats.avg_speed:.2f}",
        })
    
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _render_performance_charts(level_stats: dict[str, LevelStats]) -> None:
    """Render level performance comparison charts."""
    levels_with_data = [
        (l, s) for l, s in sorted(level_stats.items(), key=lambda x: level_sort_key(x[0]))
        if s.episodes > 0
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
        # Reward range per level (avg with min/max indicators)
        _render_reward_range_chart(levels_with_data)
    
    # Second row: Best X and Speed
    col1, col2 = st.columns(2)
    
    with col1:
        # Best X per level bar chart
        level_names = [l for l, _ in levels_with_data]
        best_x_values = [s.best_x for _, s in levels_with_data]
        
        fig = make_bar_chart(
            x=level_names,
            y=best_x_values,
            title="Best X Position per Level",
            y_title="X Position",
            height=300,
            color=COLORS["green"],
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Speed range per level
        _render_speed_range_chart(levels_with_data)


def _render_reward_range_chart(levels_with_data: list[tuple[str, LevelStats]]) -> None:
    """Render reward range chart (avg with min/max error bars)."""
    level_names = [l for l, _ in levels_with_data]
    avg_rewards = [s.avg_reward for _, s in levels_with_data]
    min_rewards = [s.min_reward for _, s in levels_with_data]
    max_rewards = [s.max_reward for _, s in levels_with_data]
    
    # Calculate error bar values (distance from avg)
    error_minus = [avg - min_r for avg, min_r in zip(avg_rewards, min_rewards)]
    error_plus = [max_r - avg for avg, max_r in zip(avg_rewards, max_rewards)]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=level_names,
        y=avg_rewards,
        error_y=dict(
            type="data",
            symmetric=False,
            array=error_plus,
            arrayminus=error_minus,
            color=COLORS["red"],
        ),
        marker_color=COLORS["peach"],
        name="Avg Reward",
    ))
    
    fig.update_layout(
        title="Reward Range per Level",
        yaxis_title="Reward",
        height=350,
        **DARK_LAYOUT,
        xaxis=dict(**GRID_STYLE),
        yaxis=dict(**GRID_STYLE),
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=False,
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_speed_range_chart(levels_with_data: list[tuple[str, LevelStats]]) -> None:
    """Render speed range chart (avg with min/max error bars)."""
    level_names = [l for l, _ in levels_with_data]
    avg_speeds = [s.avg_speed for _, s in levels_with_data]
    min_speeds = [s.min_speed for _, s in levels_with_data]
    max_speeds = [s.max_speed for _, s in levels_with_data]
    
    # Calculate error bar values (distance from avg)
    error_minus = [avg - min_s for avg, min_s in zip(avg_speeds, min_speeds)]
    error_plus = [max_s - avg for avg, max_s in zip(avg_speeds, max_speeds)]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=level_names,
        y=avg_speeds,
        error_y=dict(
            type="data",
            symmetric=False,
            array=error_plus,
            arrayminus=error_minus,
            color=COLORS["blue"],
        ),
        marker_color=COLORS["sky"],
        name="Avg Speed",
    ))
    
    fig.update_layout(
        title="Speed Range per Level",
        yaxis_title="Speed (x/time)",
        height=300,
        **DARK_LAYOUT,
        xaxis=dict(**GRID_STYLE),
        yaxis=dict(**GRID_STYLE),
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=False,
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_per_level_analysis(
    level_stats: dict[str, LevelStats],
    action_data: dict[str, list],
    rate_data: dict[str, list],
    death_dist_data: dict[str, list[DeathDistPoint]],
    death_hotspots: dict[str, dict[int, int]] | None,
) -> None:
    """Render per-level analysis with actions, deaths, and rates side by side."""
    
    # Get all levels from all data sources
    all_levels: set[str] = set()
    all_levels.update(level_stats.keys())
    all_levels.update(action_data.keys())
    all_levels.update(rate_data.keys())
    all_levels.update(death_dist_data.keys())
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
            _render_level_death_heatmap(level, death_dist_data, death_hotspots)
        
        with col_rates:
            _render_level_rate_chart(level, rate_data)
        
        st.divider()


def _render_level_action_heatmap(level: str, action_data: dict[str, list]) -> None:
    """Render action distribution heatmap for a single level."""
    if level in action_data and len(action_data[level]) >= 2:
        level_data = sorted(action_data[level], key=lambda x: x.steps)
        sampled = sample_data(level_data, max_points=30)
        
        x_labels = [f"{int(d.steps // 1000)}k" for d in sampled]
        num_actions = len(sampled[0].percentages) if sampled else len(ACTION_NAMES)
        z_data = [[d.percentages[action_idx] for d in sampled] for action_idx in range(num_actions)]
        y_labels = ACTION_NAMES[:num_actions]
        
        fig = make_heatmap(
            z_data=z_data,
            x_labels=x_labels,
            y_labels=y_labels,
            title="🎮 Actions Over Time",
            height=250,
        )
        fig.update_layout(xaxis=dict(title="Steps"))
        st.plotly_chart(fig, use_container_width=True, key=f"actions_{level}")
        
        # Current top actions
        latest = level_data[-1].percentages
        top_idx = sorted(range(len(ACTION_NAMES)), key=lambda i: latest[i], reverse=True)[:3]
        top_actions = ", ".join(f"{ACTION_NAMES[i]} ({latest[i]:.0f}%)" for i in top_idx)
        st.caption(f"🏆 Current: {top_actions}")
    else:
        st.info("⏳ Not enough action data")


def _render_level_death_heatmap(
    level: str,
    death_dist_data: dict[str, list[DeathDistPoint]],
    death_hotspots: dict[str, dict[int, int]] | None,
) -> None:
    """Render death position heatmap over time for a single level."""
    if level in death_dist_data and len(death_dist_data[level]) >= 2:
        level_data = sorted(death_dist_data[level], key=lambda x: x.steps)
        sampled = sample_data(level_data, max_points=30)
        
        # Collect all position buckets across all time points
        all_positions: set[int] = set()
        for point in sampled:
            all_positions.update(point.position_counts.keys())
        
        if not all_positions:
            st.info("⏳ No death position data")
            return
        
        # Sort positions and create labels
        sorted_positions = sorted(all_positions)
        y_labels = [f"{pos}" for pos in sorted_positions]
        x_labels = [f"{int(d.steps // 1000)}k" for d in sampled]
        
        # Build heatmap matrix: rows=positions, cols=time points
        z_data = []
        for pos in sorted_positions:
            row = [point.position_counts.get(pos, 0) for point in sampled]
            z_data.append(row)
        
        fig = make_heatmap(
            z_data=z_data,
            x_labels=x_labels,
            y_labels=y_labels,
            title="💀 Deaths Over Time",
            height=250,
            colorscale="Reds",
        )
        fig.update_layout(
            xaxis=dict(title="Steps"),
            yaxis=dict(title="X Position"),
        )
        st.plotly_chart(fig, use_container_width=True, key=f"deaths_{level}")
        
        # Death stats from hotspots (total counts)
        if death_hotspots and level in death_hotspots:
            buckets = death_hotspots[level]
            total_level_deaths = sum(buckets.values())
            if buckets:
                hottest_pos = max(buckets.items(), key=lambda x: x[1])
                st.caption(f"💀 Total: {total_level_deaths} | Hotspot: x={hottest_pos[0]}")
            else:
                st.caption(f"💀 Total: {total_level_deaths}")
        else:
            # Count from distribution data
            total = sum(sum(p.position_counts.values()) for p in level_data)
            st.caption(f"💀 Total: {total}")
    elif death_hotspots and level in death_hotspots and death_hotspots[level]:
        # Fallback to bar chart if not enough time-series data
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
        
        total_level_deaths = sum(counts)
        hottest_pos = max(buckets.items(), key=lambda x: x[1])
        st.caption(f"💀 Total: {total_level_deaths} | Hotspot: x={hottest_pos[0]}")
    else:
        st.info("⏳ No death position data")


def _render_level_rate_chart(level: str, rate_data: dict[str, list]) -> None:
    """Render death/timeout/completion rate chart for a single level."""
    if level in rate_data and len(rate_data[level]) >= 2:
        level_data = sorted(rate_data[level], key=lambda x: x.steps)
        sampled = sample_data(level_data, max_points=30)
        
        x_labels = [f"{int(d.steps // 1000)}k" for d in sampled]
        deaths_per_ep = [d.deaths_per_episode for d in sampled]
        timeouts_per_ep = [d.timeouts_per_episode for d in sampled]
        completion_rates = [d.completion_rate for d in sampled]
        
        fig = make_dual_axis_chart(
            x=x_labels,
            y1=deaths_per_ep,
            y2=completion_rates,
            name1="💀 Deaths/Ep",
            name2="🏁 Complete",
            title="📈 Death, Timeout & Completion Rate",
            y1_title="Per Episode",
            y2_title="Complete %",
            height=250,
            y1_extra=timeouts_per_ep,
            name1_extra="⏰ Timeouts/Ep",
        )
        st.plotly_chart(fig, use_container_width=True, key=f"rates_{level}")
        
        # Current rates
        st.caption(
            f"💀 Deaths/Ep: {deaths_per_ep[-1]:.2f} | "
            f"⏰ Timeouts/Ep: {timeouts_per_ep[-1]:.2f} | "
            f"🏁 Complete: {completion_rates[-1]:.1f}%"
        )
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
