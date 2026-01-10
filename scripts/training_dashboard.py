#!/usr/bin/env python3
"""
Real-time training dashboard for distributed Mario RL training.

Run with:
    uv run streamlit run scripts/training_dashboard.py
"""

from mario_rl.dashboard.app import run_dashboard

if __name__ == "__main__":
    run_dashboard()
