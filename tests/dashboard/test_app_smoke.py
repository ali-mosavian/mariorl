"""Smoke tests for the Streamlit dashboard using AppTest.

These tests verify the dashboard loads without errors and
basic rendering works correctly.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

# Import AppTest - skip tests if not available
pytest.importorskip("streamlit.testing.v1")

from streamlit.testing.v1 import AppTest


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_worker_data() -> dict[int, pd.DataFrame]:
    """Create mock worker data for testing."""
    return {
        0: pd.DataFrame({
            "timestamp": [1000.0, 1001.0, 1002.0],
            "world": [1, 1, 1],
            "stage": [1, 1, 1],
            "episodes": [10, 20, 30],
            "steps": [1000, 2000, 3000],
            "deaths": [5, 10, 15],
            "timeouts": [1, 2, 3],
            "flags": [1, 2, 3],
            "reward": [10.0, 20.0, 30.0],
            "episode_reward": [100.0, 200.0, 300.0],
            "speed": [1.0, 1.5, 2.0],
            "epsilon": [0.5, 0.4, 0.3],
            "loss": [0.1, 0.08, 0.05],
            "q_mean": [5.0, 6.0, 7.0],
            "q_max": [10.0, 12.0, 14.0],
            "td_error": [0.5, 0.4, 0.3],
            "best_x": [100, 200, 300],
            "best_x_ever": [100, 200, 300],
            "snapshot_saves": [1, 2, 3],
            "snapshot_restores": [0, 1, 2],
            "grads_sent": [10, 20, 30],
        })
    }


@pytest.fixture
def mock_coordinator_data() -> pd.DataFrame:
    """Create mock coordinator data for testing."""
    return pd.DataFrame({
        "timestamp": [1000.0, 1001.0, 1002.0],
        "update_count": [100, 200, 300],
        "total_steps": [10000, 20000, 30000],
        "grads_per_sec": [50.0, 55.0, 60.0],
        "learning_rate": [0.001, 0.001, 0.001],
        "loss": [0.1, 0.08, 0.05],
        "q_mean": [5.0, 6.0, 7.0],
        "td_error": [0.5, 0.4, 0.3],
    })


# =============================================================================
# Smoke Tests - Dashboard Imports
# =============================================================================


class TestDashboardImports:
    """Test that dashboard modules import without errors."""

    def test_import_dashboard_app(self) -> None:
        """Dashboard app module should import successfully."""
        from mario_rl.dashboard.app import run_dashboard
        assert run_dashboard is not None

    def test_import_data_loaders(self) -> None:
        """Data loaders module should import successfully."""
        from mario_rl.dashboard.data_loaders import (
            find_latest_checkpoint,
            load_coordinator_metrics,
            load_worker_metrics,
        )
        assert find_latest_checkpoint is not None
        assert load_coordinator_metrics is not None
        assert load_worker_metrics is not None

    def test_import_aggregators(self) -> None:
        """Aggregators module should import successfully."""
        from mario_rl.dashboard.aggregators import (
            aggregate_level_stats,
            aggregate_rate_data,
            aggregate_action_distribution,
        )
        assert aggregate_level_stats is not None
        assert aggregate_rate_data is not None
        assert aggregate_action_distribution is not None

    def test_import_chart_helpers(self) -> None:
        """Chart helpers module should import successfully."""
        from mario_rl.dashboard.chart_helpers import (
            COLORS,
            make_metric_chart,
            make_bar_chart,
            make_dual_axis_chart,
        )
        assert COLORS is not None
        assert make_metric_chart is not None
        assert make_bar_chart is not None
        assert make_dual_axis_chart is not None


# =============================================================================
# Smoke Tests - AppTest
# =============================================================================


class TestDashboardAppTest:
    """Smoke tests using Streamlit's AppTest framework."""

    def test_dashboard_loads_without_checkpoint(self) -> None:
        """Dashboard should load and show 'no checkpoint' message."""
        with patch(
            "mario_rl.dashboard.data_loaders.find_latest_checkpoint",
            return_value=None,
        ):
            at = AppTest.from_file("scripts/training_dashboard.py", default_timeout=10)
            at.run()
            
            # Should not crash
            assert not at.exception, f"Dashboard crashed: {at.exception}"

    def test_dashboard_loads_with_mock_data(
        self,
        mock_worker_data: dict[int, pd.DataFrame],
        mock_coordinator_data: pd.DataFrame,
    ) -> None:
        """Dashboard should load successfully with mock data."""
        with patch(
            "mario_rl.dashboard.data_loaders.find_latest_checkpoint",
            return_value="/mock/checkpoint",
        ), patch(
            "mario_rl.dashboard.data_loaders.load_worker_metrics",
            return_value=mock_worker_data,
        ), patch(
            "mario_rl.dashboard.data_loaders.load_coordinator_metrics",
            return_value=mock_coordinator_data,
        ), patch(
            "mario_rl.dashboard.data_loaders.load_death_hotspots",
            return_value=None,
        ):
            at = AppTest.from_file("scripts/training_dashboard.py", default_timeout=10)
            at.run()
            
            # Should not crash
            assert not at.exception, f"Dashboard crashed: {at.exception}"

    def test_dashboard_has_tabs(
        self,
        mock_worker_data: dict[int, pd.DataFrame],
        mock_coordinator_data: pd.DataFrame,
    ) -> None:
        """Dashboard should have multiple tabs."""
        with patch(
            "mario_rl.dashboard.data_loaders.find_latest_checkpoint",
            return_value="/mock/checkpoint",
        ), patch(
            "mario_rl.dashboard.data_loaders.load_worker_metrics",
            return_value=mock_worker_data,
        ), patch(
            "mario_rl.dashboard.data_loaders.load_coordinator_metrics",
            return_value=mock_coordinator_data,
        ), patch(
            "mario_rl.dashboard.data_loaders.load_death_hotspots",
            return_value=None,
        ):
            at = AppTest.from_file("scripts/training_dashboard.py", default_timeout=10)
            at.run()
            
            assert not at.exception
            # Check tabs exist
            assert len(at.tabs) > 0


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test dashboard handles errors gracefully."""

    def test_handles_empty_worker_data(self) -> None:
        """Dashboard should handle empty worker data without crashing."""
        with patch(
            "mario_rl.dashboard.data_loaders.find_latest_checkpoint",
            return_value="/mock/checkpoint",
        ), patch(
            "mario_rl.dashboard.data_loaders.load_worker_metrics",
            return_value={},
        ), patch(
            "mario_rl.dashboard.data_loaders.load_coordinator_metrics",
            return_value=None,
        ), patch(
            "mario_rl.dashboard.data_loaders.load_death_hotspots",
            return_value=None,
        ):
            at = AppTest.from_file("scripts/training_dashboard.py", default_timeout=10)
            at.run()
            
            assert not at.exception, f"Dashboard crashed with empty data: {at.exception}"

    def test_handles_missing_columns(self) -> None:
        """Dashboard should handle data with missing columns."""
        # Minimal data with only essential columns
        minimal_data = {
            0: pd.DataFrame({
                "timestamp": [1000.0],
                "episodes": [10],
                "steps": [1000],
            })
        }
        
        with patch(
            "mario_rl.dashboard.data_loaders.find_latest_checkpoint",
            return_value="/mock/checkpoint",
        ), patch(
            "mario_rl.dashboard.data_loaders.load_worker_metrics",
            return_value=minimal_data,
        ), patch(
            "mario_rl.dashboard.data_loaders.load_coordinator_metrics",
            return_value=None,
        ), patch(
            "mario_rl.dashboard.data_loaders.load_death_hotspots",
            return_value=None,
        ):
            at = AppTest.from_file("scripts/training_dashboard.py", default_timeout=10)
            at.run()
            
            assert not at.exception, f"Dashboard crashed with minimal data: {at.exception}"
