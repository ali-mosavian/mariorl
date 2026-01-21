"""Dashboard tab components."""

from mario_rl.dashboard.tabs.levels import render_levels_tab
from mario_rl.dashboard.tabs.workers import render_workers_tab
from mario_rl.dashboard.tabs.analysis import render_analysis_tab
from mario_rl.dashboard.tabs.coordinator import render_coordinator_tab

__all__ = [
    "render_coordinator_tab",
    "render_workers_tab",
    "render_levels_tab",
    "render_analysis_tab",
]
