"""
Dashboard package for Mario RL training visualization.

This package provides a Streamlit-based dashboard for monitoring
distributed training progress in real-time.
"""

from mario_rl.dashboard.app import run_dashboard

__all__ = ["run_dashboard"]
