"""Levels metrics tab for the training dashboard."""

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from mario_rl.dashboard.aggregators import DeathDistPoint
from mario_rl.dashboard.aggregators import LevelStats
from mario_rl.dashboard.aggregators import aggregate_action_distribution_direct
from mario_rl.dashboard.aggregators import aggregate_death_distribution_direct
from mario_rl.dashboard.aggregators import aggregate_death_hotspots_direct
from mario_rl.dashboard.aggregators import aggregate_level_stats_direct
from mario_rl.dashboard.aggregators import aggregate_rate_data_direct
from mario_rl.dashboard.aggregators import level_sort_key
from mario_rl.dashboard.aggregators import load_difficulty_ranges
from mario_rl.dashboard.aggregators import sample_data
from mario_rl.dashboard.chart_helpers import COLORS
from mario_rl.dashboard.chart_helpers import DARK_LAYOUT
from mario_rl.dashboard.chart_helpers import GRID_STYLE
from mario_rl.dashboard.chart_helpers import make_bar_chart
from mario_rl.dashboard.chart_helpers import make_dual_axis_chart
from mario_rl.dashboard.chart_helpers import make_heatmap

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


@st.cache_data(ttl=5)
def _cached_difficulty_ranges(checkpoint_dir: str) -> dict[str, list[tuple[int, int]]]:
    return load_difficulty_ranges(checkpoint_dir)


def render_levels_tab(
    checkpoint_dir: str,
    death_hotspots: dict[str, dict[int, int]] | None,
) -> None:
    """Render levels tab with death hotspot and difficulty visualization.

    Uses cached direct queries to DuckDB for optimal performance.
    """
    # Aggregate all data using cached direct queries
    level_stats = _cached_level_stats(checkpoint_dir)
    action_data = _cached_action_distribution(checkpoint_dir)
    rate_data = _cached_rate_data(checkpoint_dir)
    death_dist_data = _cached_death_distribution(checkpoint_dir)
    difficulty_ranges = _cached_difficulty_ranges(checkpoint_dir)

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
    st.subheader("🎮 Per-Level Analysis: Actions, Deaths & Difficulty")
    _render_per_level_analysis(level_stats, action_data, rate_data, death_dist_data, death_hotspots, difficulty_ranges)

    # Global hotspots summary (deaths and difficulty)
    if death_hotspots or difficulty_ranges:
        _render_hotspots_summary(death_hotspots or {}, difficulty_ranges)


def _render_level_stats_table(level_stats: dict[str, LevelStats]) -> None:
    """Render the level statistics summary table."""
    st.subheader("📊 Level Statistics")

    rows = []
    for level, stats in sorted(level_stats.items(), key=lambda x: level_sort_key(x[0])):
        rows.append(
            {
                "Level": level,
                "Episodes": stats.episodes,
                "Best X": stats.best_x,
                "Deaths": stats.deaths,
                "Flags": stats.flags,
                "Avg Reward": f"{stats.avg_reward:.1f}",
                "Min R": f"{stats.min_reward:.1f}",
                "Max R": f"{stats.max_reward:.1f}",
                "Avg Speed": f"{stats.avg_speed:.2f}",
            }
        )

    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _render_performance_charts(level_stats: dict[str, LevelStats]) -> None:
    """Render level performance comparison charts."""
    levels_with_data = [
        (lvl, s) for lvl, s in sorted(level_stats.items(), key=lambda x: level_sort_key(x[0])) if s.episodes > 0
    ]

    if not levels_with_data:
        return

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Episodes per level bar chart
        level_names = [lvl for lvl, _ in levels_with_data]
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
        level_names = [lvl for lvl, _ in levels_with_data]
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
    level_names = [lvl for lvl, _ in levels_with_data]
    avg_rewards = [s.avg_reward for _, s in levels_with_data]
    min_rewards = [s.min_reward for _, s in levels_with_data]
    max_rewards = [s.max_reward for _, s in levels_with_data]

    # Calculate error bar values (distance from avg)
    error_minus = [avg - min_r for avg, min_r in zip(avg_rewards, min_rewards, strict=False)]
    error_plus = [max_r - avg for avg, max_r in zip(avg_rewards, max_rewards, strict=False)]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=level_names,
            y=avg_rewards,
            error_y={
                "type": "data",
                "symmetric": False,
                "array": error_plus,
                "arrayminus": error_minus,
                "color": COLORS["red"],
            },
            marker_color=COLORS["peach"],
            name="Avg Reward",
        )
    )

    fig.update_layout(
        title="Reward Range per Level",
        yaxis_title="Reward",
        height=350,
        **DARK_LAYOUT,
        xaxis=dict(**GRID_STYLE),
        yaxis=dict(**GRID_STYLE),
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_speed_range_chart(levels_with_data: list[tuple[str, LevelStats]]) -> None:
    """Render speed range chart (avg with min/max error bars)."""
    level_names = [lvl for lvl, _ in levels_with_data]
    avg_speeds = [s.avg_speed for _, s in levels_with_data]
    min_speeds = [s.min_speed for _, s in levels_with_data]
    max_speeds = [s.max_speed for _, s in levels_with_data]

    # Calculate error bar values (distance from avg)
    error_minus = [avg - min_s for avg, min_s in zip(avg_speeds, min_speeds, strict=False)]
    error_plus = [max_s - avg for avg, max_s in zip(avg_speeds, max_speeds, strict=False)]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=level_names,
            y=avg_speeds,
            error_y={
                "type": "data",
                "symmetric": False,
                "array": error_plus,
                "arrayminus": error_minus,
                "color": COLORS["blue"],
            },
            marker_color=COLORS["sky"],
            name="Avg Speed",
        )
    )

    fig.update_layout(
        title="Speed Range per Level",
        yaxis_title="Speed (x/time)",
        height=300,
        **DARK_LAYOUT,
        xaxis=dict(**GRID_STYLE),
        yaxis=dict(**GRID_STYLE),
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_per_level_analysis(
    level_stats: dict[str, LevelStats],
    action_data: dict[str, list],
    rate_data: dict[str, list],
    death_dist_data: dict[str, list[DeathDistPoint]],
    death_hotspots: dict[str, dict[int, int]] | None,
    difficulty_ranges: dict[str, list[tuple[int, int]]] | None = None,
) -> None:
    """Render per-level analysis with actions, deaths, difficulty, and rates side by side."""

    # Get all levels from all data sources
    all_levels: set[str] = set()
    all_levels.update(level_stats.keys())
    all_levels.update(action_data.keys())
    all_levels.update(rate_data.keys())
    all_levels.update(death_dist_data.keys())
    if death_hotspots:
        all_levels.update(death_hotspots.keys())
    if difficulty_ranges:
        all_levels.update(difficulty_ranges.keys())

    if not all_levels:
        st.info("⏳ No level data available yet...")
        return

    sorted_levels = sorted(all_levels, key=level_sort_key)

    # Summary stats
    total_deaths = sum(sum(b.values()) for b in (death_hotspots or {}).values())
    total_diff_zones = sum(len(r) for r in (difficulty_ranges or {}).values())
    stats_cols = st.columns(4)
    stats_cols[0].metric("Levels", len(sorted_levels))
    stats_cols[1].metric("💀 Total Deaths", total_deaths)
    stats_cols[2].metric("⚠️ Difficult Zones", total_diff_zones)
    if death_hotspots:
        deadliest = max(death_hotspots.items(), key=lambda x: sum(x[1].values()), default=("?", {}))
        stats_cols[3].metric("Deadliest Level", deadliest[0])

    st.divider()

    # Render each level as a row with 4 columns: actions, deaths barchart, difficulty, rates
    for level in sorted_levels:
        st.markdown(f"### Level {level}")
        col_actions, col_deaths, col_difficulty, col_rates = st.columns(4)

        with col_actions:
            _render_level_action_heatmap(level, action_data)

        with col_deaths:
            _render_level_death_barchart(level, death_hotspots)

        with col_difficulty:
            _render_level_difficulty_chart(level, difficulty_ranges)

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
        fig.update_layout(xaxis={"title": "Steps"})
        st.plotly_chart(fig, use_container_width=True, key=f"actions_{level}")

        # Current top actions
        latest = level_data[-1].percentages
        top_idx = sorted(range(len(ACTION_NAMES)), key=lambda i: latest[i], reverse=True)[:3]
        top_actions = ", ".join(f"{ACTION_NAMES[i]} ({latest[i]:.0f}%)" for i in top_idx)
        st.caption(f"🏆 Current: {top_actions}")
    else:
        st.info("⏳ Not enough action data")


def _render_level_death_barchart(
    level: str,
    death_hotspots: dict[str, dict[int, int]] | None,
) -> None:
    """Render death position barchart for a single level."""
    if death_hotspots and level in death_hotspots and death_hotspots[level]:
        buckets = death_hotspots[level]
        sorted_buckets = sorted(buckets.items(), key=lambda x: x[0])

        positions = [f"{b[0]}" for b in sorted_buckets]
        counts = [b[1] for b in sorted_buckets]

        fig = make_bar_chart(
            x=counts,
            y=positions,
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


def _render_level_difficulty_chart(
    level: str,
    difficulty_ranges: dict[str, list[tuple[int, int]]] | None,
) -> None:
    """Render difficulty zones chart for a single level.

    Shows horizontal bars for each difficult section identified from death hotspots.
    """
    if difficulty_ranges and level in difficulty_ranges and difficulty_ranges[level]:
        ranges = difficulty_ranges[level]

        # Create horizontal bar chart showing difficult zones
        # Each bar goes from start_x to end_x
        fig = go.Figure()

        for i, (start, end) in enumerate(ranges):
            fig.add_trace(go.Bar(
                y=[f"Zone {i+1}"],
                x=[end - start],
                base=[start],
                orientation="h",
                marker_color=COLORS["red"],
                text=[f"x={start}-{end}"],
                textposition="inside",
                hovertemplate=f"Zone {i+1}: x={start} to x={end} ({end-start}px)<extra></extra>",
            ))

        fig.update_layout(
            title="⚠️ Difficult Zones",
            xaxis_title="X Position",
            yaxis_title="",
            height=250,
            showlegend=False,
            barmode="overlay",
            **DARK_LAYOUT,
            xaxis=dict(**GRID_STYLE, range=[0, 3500]),
            yaxis=dict(**GRID_STYLE),
            margin=dict(l=0, r=0, t=50, b=0),
        )
        st.plotly_chart(fig, use_container_width=True, key=f"difficulty_{level}")

        total_coverage = sum(end - start for start, end in ranges)
        st.caption(f"⚠️ {len(ranges)} zones | {total_coverage}px total")
    else:
        st.info("⏳ No difficulty data")


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


def _render_hotspots_summary(
    death_hotspots: dict[str, dict[int, int]],
    difficulty_ranges: dict[str, list[tuple[int, int]]] | None,
) -> None:
    """Render global death hotspots and difficulty zones summary."""
    total_deaths = sum(sum(b.values()) for b in death_hotspots.values())

    col1, col2 = st.columns(2)

    with col1:
        st.caption("🔥 TOP DEATH ZONES (ALL LEVELS)")

        all_death_hotspots = []
        for level, buckets in death_hotspots.items():
            for pos, count in buckets.items():
                all_death_hotspots.append((level, pos, count))

        if all_death_hotspots:
            top_death_spots = sorted(all_death_hotspots, key=lambda x: x[2], reverse=True)[:15]
            death_rows = [
                {
                    "Level": level,
                    "Position": f"x={pos}-{pos + 25}",
                    "Deaths": count,
                    "% of Total": f"{count / total_deaths * 100:.1f}%" if total_deaths > 0 else "0%",
                }
                for level, pos, count in top_death_spots
            ]
            st.dataframe(pd.DataFrame(death_rows), hide_index=True, use_container_width=True)
        else:
            st.info("⏳ No death data yet")

    with col2:
        st.caption("⚠️ DIFFICULT ZONES (ALL LEVELS)")

        all_difficulty_zones = []
        for level, ranges in (difficulty_ranges or {}).items():
            for start, end in ranges:
                all_difficulty_zones.append((level, start, end, end - start))

        if all_difficulty_zones:
            # Sort by size (largest zones first)
            sorted_zones = sorted(all_difficulty_zones, key=lambda x: x[3], reverse=True)[:15]
            diff_rows = [
                {
                    "Level": level,
                    "Range": f"x={start}-{end}",
                    "Width": f"{width}px",
                }
                for level, start, end, width in sorted_zones
            ]
            st.dataframe(pd.DataFrame(diff_rows), hide_index=True, use_container_width=True)
        else:
            st.info("⏳ No difficulty data yet")

    # Per-level totals bar charts (deaths and difficulty coverage)
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.caption("📊 TOTAL DEATHS PER LEVEL")

        level_deaths = {level: sum(buckets.values()) for level, buckets in death_hotspots.items()}
        if level_deaths:
            sorted_level_deaths = sorted(level_deaths.items(), key=lambda x: level_sort_key(x[0]))

            fig = make_bar_chart(
                x=[item[0] for item in sorted_level_deaths],
                y=[item[1] for item in sorted_level_deaths],
                title="",
                height=200,
                color=COLORS["red"],
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("⏳ No death data yet")

    with col2:
        st.caption("📊 DIFFICULTY COVERAGE PER LEVEL")

        level_coverage = {
            level: sum(end - start for start, end in ranges)
            for level, ranges in (difficulty_ranges or {}).items()
        }
        if level_coverage:
            sorted_level_coverage = sorted(level_coverage.items(), key=lambda x: level_sort_key(x[0]))

            fig = make_bar_chart(
                x=[l[0] for l in sorted_level_coverage],
                y=[l[1] for l in sorted_level_coverage],
                title="",
                height=200,
                color=COLORS["peach"],
                y_title="Pixels",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("⏳ No difficulty data yet")
