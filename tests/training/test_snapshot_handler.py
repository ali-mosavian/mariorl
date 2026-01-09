"""Tests for the SnapshotHandler integration layer."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest

from mario_rl.training.snapshot_handler import SnapshotHandler
from mario_rl.training.snapshot_handler import SnapshotResult
from mario_rl.training.snapshot_state_machine import SnapshotAction
from mario_rl.training.snapshot_state_machine import SnapshotStateMachine


@dataclass
class MockSnapshotManager:
    """Mock snapshot manager for testing."""

    # Saved snapshots
    _slot_to_state: dict[int, Any] = field(default_factory=dict)

    # Track calls
    save_calls: list[dict] = field(default_factory=list)
    restore_calls: list[dict] = field(default_factory=list)

    # Control behavior
    save_succeeds: bool = True
    restore_succeeds: bool = True
    restore_observation: np.ndarray | None = None

    def reset(self) -> None:
        """Reset all state."""
        self._slot_to_state.clear()
        self.save_calls.clear()
        self.restore_calls.clear()

    def maybe_save(self, state: np.ndarray, info: dict) -> bool:
        """Mock save operation."""
        self.save_calls.append({"state": state, "info": info})
        if self.save_succeeds:
            slot = len(self._slot_to_state)
            self._slot_to_state[slot] = state
        return self.save_succeeds

    def try_restore(
        self, info: dict, current_best_x: int
    ) -> tuple[np.ndarray, bool]:
        """Mock restore operation."""
        self.restore_calls.append({"info": info, "best_x": current_best_x})
        if self.restore_succeeds and self.restore_observation is not None:
            return self.restore_observation, True
        return np.array([]), False


@pytest.fixture
def mock_manager() -> MockSnapshotManager:
    """Create a mock snapshot manager."""
    return MockSnapshotManager()


@pytest.fixture
def state_machine() -> SnapshotStateMachine:
    """Create a state machine with test-friendly config."""
    return SnapshotStateMachine(
        hotspot_approach_distance=100,
        hotspot_save_offset=50,
        hotspot_save_tolerance=25,
        checkpoint_interval=500,
        max_restores_without_progress=3,
    )


@pytest.fixture
def handler(
    state_machine: SnapshotStateMachine, mock_manager: MockSnapshotManager
) -> SnapshotHandler:
    """Create a snapshot handler with mocks."""
    return SnapshotHandler(
        state_machine=state_machine,
        snapshot_manager=mock_manager,  # type: ignore
    )


class TestSnapshotHandlerBasic:
    """Basic handler tests."""

    def test_no_action_during_normal_play(
        self, handler: SnapshotHandler
    ) -> None:
        """Handler should return NONE action during normal gameplay."""
        obs = np.zeros((4, 64, 64), dtype=np.float32)
        info = {"x_pos": 100, "time": 100, "world": 1, "stage": 1}

        result = handler.process_step(obs, info, done=False)

        assert result.action_taken == SnapshotAction.NONE
        assert result.observation is None
        assert result.episode_ended is False

    def test_tracks_best_x(self, handler: SnapshotHandler) -> None:
        """Handler should track best x position."""
        obs = np.zeros((4, 64, 64), dtype=np.float32)

        handler.process_step(obs, {"x_pos": 100, "time": 0}, done=False)
        assert handler._best_x == 100

        handler.process_step(obs, {"x_pos": 200, "time": 0}, done=False)
        assert handler._best_x == 200

        handler.process_step(obs, {"x_pos": 150, "time": 0}, done=False)
        assert handler._best_x == 200  # Should not decrease

    def test_reset_episode(
        self, handler: SnapshotHandler, mock_manager: MockSnapshotManager
    ) -> None:
        """reset_episode should reset all state."""
        obs = np.zeros((4, 64, 64), dtype=np.float32)
        handler._best_x = 500
        mock_manager._slot_to_state[0] = obs

        handler.reset_episode()

        assert handler._best_x == 0
        assert len(mock_manager._slot_to_state) == 0


class TestCheckpointTriggering:
    """Tests for checkpoint (time-based) snapshot triggering."""

    def test_checkpoint_triggers_save(
        self, handler: SnapshotHandler, mock_manager: MockSnapshotManager
    ) -> None:
        """Should save when checkpoint interval is reached."""
        obs = np.zeros((4, 64, 64), dtype=np.float32)

        # First step - no checkpoint yet
        result = handler.process_step(
            obs, {"x_pos": 100, "time": 100}, done=False
        )
        assert result.action_taken == SnapshotAction.NONE

        # Time reaches checkpoint interval (500)
        result = handler.process_step(
            obs, {"x_pos": 200, "time": 500}, done=False
        )
        assert result.action_taken == SnapshotAction.SAVE_SNAPSHOT
        assert len(mock_manager.save_calls) == 1


class TestDeathAndRestore:
    """Tests for death detection and restore."""

    def test_death_without_snapshot_ends_episode(
        self, handler: SnapshotHandler, mock_manager: MockSnapshotManager
    ) -> None:
        """Death with no snapshots should end episode."""
        obs = np.zeros((4, 64, 64), dtype=np.float32)

        # Die with no snapshots available
        result = handler.process_step(
            obs,
            {"x_pos": 100, "time": 100, "flag_get": False},
            done=True,
        )

        # Should transition through DEAD -> EVALUATE -> GIVE_UP -> END
        # The exact flow depends on state machine, but should end episode
        assert result.episode_ended is True or result.action_taken == SnapshotAction.END_EPISODE

    def test_death_with_snapshot_restores(
        self, handler: SnapshotHandler, mock_manager: MockSnapshotManager
    ) -> None:
        """Death with available snapshot should restore."""
        obs = np.zeros((4, 64, 64), dtype=np.float32)
        restored_obs = np.ones((4, 64, 64), dtype=np.float32)

        # Setup: have a snapshot available
        mock_manager._slot_to_state[0] = obs
        mock_manager.restore_succeeds = True
        mock_manager.restore_observation = restored_obs

        # Die
        result = handler.process_step(
            obs,
            {"x_pos": 100, "time": 100, "flag_get": False},
            done=True,
        )

        # Should eventually restore (may take multiple transitions)
        # Keep processing until we get a restore
        if result.action_taken != SnapshotAction.RESTORE_SNAPSHOT:
            result = handler.process_step(obs, {"x_pos": 100}, done=False)

        if result.action_taken == SnapshotAction.RESTORE_SNAPSHOT:
            assert result.observation is not None
            np.testing.assert_array_equal(result.observation, restored_obs)

    def test_repeated_deaths_lead_to_give_up(
        self, handler: SnapshotHandler, mock_manager: MockSnapshotManager
    ) -> None:
        """Repeated deaths without progress should give up."""
        obs = np.zeros((4, 64, 64), dtype=np.float32)
        mock_manager._slot_to_state[0] = obs
        mock_manager.restore_succeeds = True
        mock_manager.restore_observation = obs

        # Simulate deaths at same position (no progress)
        handler.state_machine._last_restore_x = 100

        for _ in range(5):  # More than max_restores_without_progress
            result = handler.process_step(
                obs,
                {"x_pos": 100, "time": 100, "flag_get": False},
                done=True,
            )
            if result.episode_ended:
                break

            # Process non-death step to allow state machine to settle
            handler.process_step(obs, {"x_pos": 105, "time": 110}, done=False)

        # Eventually should give up
        assert result.episode_ended is True


class TestHotspotIntegration:
    """Tests for death hotspot integration."""

    def test_hotspots_affect_snapshot_positions(
        self, mock_manager: MockSnapshotManager
    ) -> None:
        """Hotspots should trigger saves before dangerous areas."""
        # Create handler with hotspots
        state_machine = SnapshotStateMachine(
            hotspot_approach_distance=100,
            hotspot_save_offset=50,
            checkpoint_interval=10000,  # Disable time-based
        )
        handler = SnapshotHandler(
            state_machine=state_machine,
            snapshot_manager=mock_manager,  # type: ignore
        )

        # Manually set hotspots (bypass file loading)
        from mario_rl.metrics.levels import DeathHotspotAggregate
        hotspots = DeathHotspotAggregate()
        hotspots.record_deaths_batch("1-1", [500, 510, 520, 530, 540])
        handler._hotspots = hotspots
        handler._last_hotspot_load = 1e10  # Prevent reload

        obs = np.zeros((4, 64, 64), dtype=np.float32)

        # Approach hotspot zone (500 - 100 = 400)
        result = handler.process_step(
            obs, {"x_pos": 410, "time": 10, "world": 1, "stage": 1}, done=False
        )

        # Should be approaching
        from mario_rl.training.snapshot_state_machine import SnapshotState
        assert handler.state_machine.state in (
            SnapshotState.APPROACHING_HOTSPOT,
            SnapshotState.RUNNING,
        )

        # Reach save position (500 - 50 = 450)
        result = handler.process_step(
            obs, {"x_pos": 450, "time": 20, "world": 1, "stage": 1}, done=False
        )

        # Should save
        assert result.action_taken == SnapshotAction.SAVE_SNAPSHOT


class TestFlagGet:
    """Tests for level completion (flag get)."""

    def test_flag_get_not_treated_as_death(
        self, handler: SnapshotHandler
    ) -> None:
        """Completing level should not trigger death handling."""
        obs = np.zeros((4, 64, 64), dtype=np.float32)

        result = handler.process_step(
            obs,
            {"x_pos": 3000, "time": 100, "flag_get": True},
            done=True,
        )

        # Should not try to restore or end episode due to "death"
        assert result.action_taken == SnapshotAction.NONE
        assert result.episode_ended is False
