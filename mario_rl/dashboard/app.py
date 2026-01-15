"""
Main Streamlit dashboard application for Mario RL training.

Run with:
    uv run streamlit run -m mario_rl.dashboard.app
"""

import time
from datetime import datetime
from pathlib import Path

import streamlit as st

from mario_rl.dashboard.data_loaders import (
    list_checkpoints,
    load_coordinator_metrics,
    load_death_hotspots,
    load_worker_metrics,
)
from mario_rl.dashboard.tabs import (
    render_coordinator_tab,
    render_workers_tab,
    render_levels_tab,
    render_analysis_tab,
)


def setup_page() -> None:
    """Configure page settings and styling."""
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


def render_sidebar() -> tuple[str, int]:
    """Render sidebar and return checkpoint_dir and refresh_sec."""
    with st.sidebar:
        st.header("Settings")

        checkpoints = list_checkpoints()
        if checkpoints:
            # Format display names: show just the directory name for cleaner UI
            display_names = [Path(cp).name for cp in checkpoints]
            selected_idx = st.selectbox(
                "Checkpoint",
                range(len(checkpoints)),
                index=0,  # Default to latest (list is sorted newest first)
                format_func=lambda i: display_names[i],
                help="Select a checkpoint directory (sorted by newest first)",
            )
            checkpoint_dir = checkpoints[selected_idx]
        else:
            # Fallback to text input if no checkpoints found
            checkpoint_dir = st.text_input(
                "Checkpoint",
                value="checkpoints/",
                help="No checkpoints found. Enter path manually.",
            )

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
            _render_worker_health_summary(checkpoint_dir)

    return checkpoint_dir, refresh_sec


def _render_worker_health_summary(checkpoint_dir: str) -> None:
    """Render worker health summary in sidebar."""
    workers = load_worker_metrics(checkpoint_dir)
    if not workers:
        return
    
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


def render_dashboard_content(checkpoint_dir: str, refresh_sec: int) -> None:
    """Render the main dashboard content (tabs and charts)."""
    
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


def run_dashboard() -> None:
    """Main entry point for the dashboard."""
    setup_page()
    
    st.title("🍄 Mario RL Training")
    
    checkpoint_dir, refresh_sec = render_sidebar()
    
    if not Path(checkpoint_dir).exists():
        st.error(f"Directory not found: {checkpoint_dir}")
        return
    
    render_dashboard_content(checkpoint_dir, refresh_sec)


# Allow running as module: python -m mario_rl.dashboard.app
if __name__ == "__main__":
    run_dashboard()
