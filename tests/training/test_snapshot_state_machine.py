"""Tests for the SnapshotStateMachine.

These tests follow TDD principles - they define the expected behavior
of the state machine before implementation details are finalized.
"""

from __future__ import annotations

import pytest

from mario_rl.training.snapshot_state_machine import SnapshotAction
from mario_rl.training.snapshot_state_machine import SnapshotContext
from mario_rl.training.snapshot_state_machine import SnapshotState
from mario_rl.training.snapshot_state_machine import SnapshotStateMachine


def make_context(
    x_pos: int = 100,
    level_id: str = "1-1",
    game_time: int = 0,
    is_dead: bool = False,
    flag_get: bool = False,
    best_x: int = 100,
    hotspot_positions: tuple[int, ...] = (),
    snapshot_available: bool = False,
    suggested_restore_x: int | None = None,
) -> SnapshotContext:
    """Helper to create SnapshotContext with defaults."""
    return SnapshotContext(
        x_pos=x_pos,
        level_id=level_id,
        game_time=game_time,
        is_dead=is_dead,
        flag_get=flag_get,
        best_x=best_x,
        hotspot_positions=hotspot_positions,
        snapshot_available=snapshot_available,
        suggested_restore_x=suggested_restore_x,
    )


class TestSnapshotStateMachineInit:
    """Tests for state machine initialization."""

    def test_initial_state_is_running(self) -> None:
        """State machine should start in RUNNING state."""
        sm = SnapshotStateMachine()
        assert sm.state == SnapshotState.RUNNING

    def test_default_configuration(self) -> None:
        """State machine should have sensible defaults."""
        sm = SnapshotStateMachine()
        assert sm.hotspot_approach_distance == 100
        assert sm.hotspot_save_offset == 50
        assert sm.hotspot_save_tolerance == 25
        assert sm.checkpoint_interval == 500
        assert sm.max_restores_without_progress == 3
        assert sm.progress_threshold == 50

    def test_custom_configuration(self) -> None:
        """State machine should accept custom configuration."""
        sm = SnapshotStateMachine(
            hotspot_approach_distance=200,
            checkpoint_interval=1000,
            max_restores_without_progress=5,
        )
        assert sm.hotspot_approach_distance == 200
        assert sm.checkpoint_interval == 1000
        assert sm.max_restores_without_progress == 5


class TestRunningState:
    """Tests for RUNNING state transitions."""

    def test_stays_running_when_no_events(self) -> None:
        """Should stay in RUNNING when nothing special happens."""
        sm = SnapshotStateMachine()
        ctx = make_context(x_pos=100, game_time=100)

        state, action = sm.transition(ctx)

        assert state == SnapshotState.RUNNING
        assert action == SnapshotAction.NONE
        assert sm.state == SnapshotState.RUNNING

    def test_transitions_to_dead_on_death(self) -> None:
        """Should transition to DEAD when death is detected."""
        sm = SnapshotStateMachine()
        ctx = make_context(is_dead=True)

        state, action = sm.transition(ctx)

        assert state == SnapshotState.DEAD
        assert action == SnapshotAction.NONE
        assert sm.state == SnapshotState.DEAD

    def test_transitions_to_approaching_hotspot(self) -> None:
        """Should transition to APPROACHING_HOTSPOT when entering approach zone."""
        sm = SnapshotStateMachine(hotspot_approach_distance=100)
        # Hotspot at x=500, approach starts at x=400
        ctx = make_context(x_pos=420, hotspot_positions=(500,))

        state, action = sm.transition(ctx)

        assert state == SnapshotState.APPROACHING_HOTSPOT
        assert action == SnapshotAction.NONE

    def test_no_transition_when_before_approach_zone(self) -> None:
        """Should stay RUNNING when before hotspot approach zone."""
        sm = SnapshotStateMachine(hotspot_approach_distance=100)
        # Hotspot at x=500, approach starts at x=400
        ctx = make_context(x_pos=300, hotspot_positions=(500,))

        state, action = sm.transition(ctx)

        assert state == SnapshotState.RUNNING
        assert action == SnapshotAction.NONE

    def test_no_transition_when_past_hotspot(self) -> None:
        """Should stay RUNNING when already past hotspot."""
        sm = SnapshotStateMachine(hotspot_approach_distance=100)
        ctx = make_context(x_pos=600, hotspot_positions=(500,))

        state, action = sm.transition(ctx)

        assert state == SnapshotState.RUNNING
        assert action == SnapshotAction.NONE

    def test_transitions_to_checkpoint_due(self) -> None:
        """Should transition to CHECKPOINT_DUE when interval reached."""
        sm = SnapshotStateMachine(checkpoint_interval=500)
        ctx = make_context(game_time=500)

        state, action = sm.transition(ctx)

        assert state == SnapshotState.CHECKPOINT_DUE
        assert action == SnapshotAction.NONE

    def test_no_checkpoint_before_interval(self) -> None:
        """Should stay RUNNING when checkpoint interval not reached."""
        sm = SnapshotStateMachine(checkpoint_interval=500)
        ctx = make_context(game_time=499)

        state, action = sm.transition(ctx)

        assert state == SnapshotState.RUNNING
        assert action == SnapshotAction.NONE

    def test_death_has_priority_over_hotspot(self) -> None:
        """Death should take priority over entering hotspot zone."""
        sm = SnapshotStateMachine(hotspot_approach_distance=100)
        ctx = make_context(x_pos=420, hotspot_positions=(500,), is_dead=True)

        state, action = sm.transition(ctx)

        assert state == SnapshotState.DEAD

    def test_death_has_priority_over_checkpoint(self) -> None:
        """Death should take priority over checkpoint due."""
        sm = SnapshotStateMachine(checkpoint_interval=500)
        ctx = make_context(game_time=600, is_dead=True)

        state, action = sm.transition(ctx)

        assert state == SnapshotState.DEAD


class TestApproachingHotspotState:
    """Tests for APPROACHING_HOTSPOT state transitions."""

    def test_transitions_to_save_at_optimal_position(self) -> None:
        """Should transition to SAVE_SNAPSHOT at optimal position."""
        sm = SnapshotStateMachine(
            hotspot_approach_distance=100,
            hotspot_save_offset=50,
            hotspot_save_tolerance=25,
        )
        # First, enter approach zone
        ctx1 = make_context(x_pos=420, hotspot_positions=(500,))
        sm.transition(ctx1)
        assert sm.state == SnapshotState.APPROACHING_HOTSPOT

        # Now reach optimal save position (500 - 50 = 450)
        ctx2 = make_context(x_pos=450, hotspot_positions=(500,))
        state, action = sm.transition(ctx2)

        assert state == SnapshotState.SAVE_SNAPSHOT
        assert action == SnapshotAction.SAVE_SNAPSHOT

    def test_save_within_tolerance(self) -> None:
        """Should trigger save within tolerance range."""
        sm = SnapshotStateMachine(
            hotspot_approach_distance=100,
            hotspot_save_offset=50,
            hotspot_save_tolerance=25,
        )
        # Enter approach zone
        ctx1 = make_context(x_pos=420, hotspot_positions=(500,))
        sm.transition(ctx1)

        # Within tolerance (450 ± 25)
        ctx2 = make_context(x_pos=460, hotspot_positions=(500,))
        state, action = sm.transition(ctx2)

        assert state == SnapshotState.SAVE_SNAPSHOT
        assert action == SnapshotAction.SAVE_SNAPSHOT

    def test_stays_approaching_before_optimal(self) -> None:
        """Should stay in APPROACHING_HOTSPOT before optimal position."""
        sm = SnapshotStateMachine(
            hotspot_approach_distance=100,
            hotspot_save_offset=50,
            hotspot_save_tolerance=25,
        )
        # Enter approach zone
        ctx1 = make_context(x_pos=410, hotspot_positions=(500,))
        sm.transition(ctx1)

        # Still before optimal (450 - 25 = 425)
        ctx2 = make_context(x_pos=420, hotspot_positions=(500,))
        state, action = sm.transition(ctx2)

        assert state == SnapshotState.APPROACHING_HOTSPOT
        assert action == SnapshotAction.NONE

    def test_transitions_to_running_when_passed_hotspot(self) -> None:
        """Should return to RUNNING if hotspot is passed without saving."""
        sm = SnapshotStateMachine(
            hotspot_approach_distance=100,
            hotspot_save_offset=50,
            hotspot_save_tolerance=25,
        )
        # Enter approach zone
        ctx1 = make_context(x_pos=410, hotspot_positions=(500,))
        sm.transition(ctx1)

        # Skip past hotspot (e.g., warp or fast movement)
        ctx2 = make_context(x_pos=550, hotspot_positions=(500,))
        state, action = sm.transition(ctx2)

        assert state == SnapshotState.RUNNING
        assert action == SnapshotAction.NONE

    def test_death_while_approaching(self) -> None:
        """Should transition to DEAD if death occurs while approaching."""
        sm = SnapshotStateMachine(hotspot_approach_distance=100)
        # Enter approach zone
        ctx1 = make_context(x_pos=420, hotspot_positions=(500,))
        sm.transition(ctx1)

        # Die while approaching
        ctx2 = make_context(x_pos=440, hotspot_positions=(500,), is_dead=True)
        state, action = sm.transition(ctx2)

        assert state == SnapshotState.DEAD
        assert action == SnapshotAction.NONE


class TestCheckpointDueState:
    """Tests for CHECKPOINT_DUE state transitions."""

    def test_transitions_to_save_snapshot(self) -> None:
        """Should transition to SAVE_SNAPSHOT immediately."""
        sm = SnapshotStateMachine(checkpoint_interval=500)
        # Trigger checkpoint
        ctx1 = make_context(game_time=500)
        sm.transition(ctx1)
        assert sm.state == SnapshotState.CHECKPOINT_DUE

        # Process checkpoint
        ctx2 = make_context(game_time=500)
        state, action = sm.transition(ctx2)

        assert state == SnapshotState.SAVE_SNAPSHOT
        assert action == SnapshotAction.SAVE_SNAPSHOT

    def test_updates_last_checkpoint_time(self) -> None:
        """Should update last checkpoint time."""
        sm = SnapshotStateMachine(checkpoint_interval=500)
        assert sm._last_checkpoint_time == 0

        ctx1 = make_context(game_time=500)
        sm.transition(ctx1)  # -> CHECKPOINT_DUE

        ctx2 = make_context(game_time=500)
        sm.transition(ctx2)  # -> SAVE_SNAPSHOT

        assert sm._last_checkpoint_time == 500

    def test_death_in_checkpoint_due(self) -> None:
        """Should transition to DEAD if death occurs during checkpoint."""
        sm = SnapshotStateMachine(checkpoint_interval=500)
        ctx1 = make_context(game_time=500)
        sm.transition(ctx1)  # -> CHECKPOINT_DUE

        ctx2 = make_context(game_time=500, is_dead=True)
        state, action = sm.transition(ctx2)

        assert state == SnapshotState.DEAD


class TestSaveSnapshotState:
    """Tests for SAVE_SNAPSHOT state transitions."""

    def test_transitions_to_snapshot_saved(self) -> None:
        """Should transition to SNAPSHOT_SAVED after processing."""
        sm = SnapshotStateMachine()
        sm.state = SnapshotState.SAVE_SNAPSHOT

        ctx = make_context()
        state, action = sm.transition(ctx)

        assert state == SnapshotState.SNAPSHOT_SAVED
        assert action == SnapshotAction.NONE

    def test_clears_hotspot_target(self) -> None:
        """Should clear current hotspot target after save."""
        sm = SnapshotStateMachine()
        sm._current_hotspot_target = 500
        sm.state = SnapshotState.SAVE_SNAPSHOT

        ctx = make_context()
        sm.transition(ctx)

        assert sm._current_hotspot_target is None


class TestSnapshotSavedState:
    """Tests for SNAPSHOT_SAVED state transitions."""

    def test_transitions_to_running(self) -> None:
        """Should immediately transition to RUNNING."""
        sm = SnapshotStateMachine()
        sm.state = SnapshotState.SNAPSHOT_SAVED

        ctx = make_context()
        state, action = sm.transition(ctx)

        assert state == SnapshotState.RUNNING
        assert action == SnapshotAction.NONE


class TestDeadState:
    """Tests for DEAD state transitions."""

    def test_transitions_to_evaluate_restore(self) -> None:
        """Should transition to EVALUATE_RESTORE."""
        sm = SnapshotStateMachine()
        sm.state = SnapshotState.DEAD

        ctx = make_context()
        state, action = sm.transition(ctx)

        assert state == SnapshotState.EVALUATE_RESTORE
        assert action == SnapshotAction.NONE


class TestEvaluateRestoreState:
    """Tests for EVALUATE_RESTORE state transitions."""

    def test_transitions_to_restoring_when_snapshot_available(self) -> None:
        """Should transition to RESTORING when snapshot is available."""
        sm = SnapshotStateMachine()
        sm.state = SnapshotState.EVALUATE_RESTORE

        ctx = make_context(snapshot_available=True)
        state, action = sm.transition(ctx)

        assert state == SnapshotState.RESTORING
        assert action == SnapshotAction.RESTORE_SNAPSHOT

    def test_transitions_to_give_up_no_snapshot(self) -> None:
        """Should transition to GIVE_UP when no snapshot available."""
        sm = SnapshotStateMachine()
        sm.state = SnapshotState.EVALUATE_RESTORE

        ctx = make_context(snapshot_available=False)
        state, action = sm.transition(ctx)

        assert state == SnapshotState.GIVE_UP
        assert action == SnapshotAction.NONE

    def test_transitions_to_give_up_max_restores(self) -> None:
        """Should transition to GIVE_UP when max restores exceeded."""
        sm = SnapshotStateMachine(max_restores_without_progress=3)
        sm.state = SnapshotState.EVALUATE_RESTORE
        sm._restores_without_progress = 3

        ctx = make_context(snapshot_available=True)
        state, action = sm.transition(ctx)

        assert state == SnapshotState.GIVE_UP
        assert action == SnapshotAction.NONE


class TestRestoringState:
    """Tests for RESTORING state transitions."""

    def test_transitions_to_running(self) -> None:
        """Should transition to RUNNING after restore."""
        sm = SnapshotStateMachine()
        sm.state = SnapshotState.RESTORING
        sm._last_restore_x = 100

        ctx = make_context(x_pos=200)  # Progress made
        state, action = sm.transition(ctx)

        assert state == SnapshotState.RUNNING
        assert action == SnapshotAction.NONE

    def test_resets_counter_on_progress(self) -> None:
        """Should reset restore counter when progress is made."""
        sm = SnapshotStateMachine(progress_threshold=50)
        sm.state = SnapshotState.RESTORING
        sm._last_restore_x = 100
        sm._restores_without_progress = 2

        ctx = make_context(x_pos=200)  # Progress > 50
        sm.transition(ctx)

        assert sm._restores_without_progress == 0

    def test_increments_counter_no_progress(self) -> None:
        """Should increment restore counter when no progress."""
        sm = SnapshotStateMachine(progress_threshold=50)
        sm.state = SnapshotState.RESTORING
        sm._last_restore_x = 100
        sm._restores_without_progress = 1

        ctx = make_context(x_pos=120)  # Progress < 50
        sm.transition(ctx)

        assert sm._restores_without_progress == 2

    def test_updates_last_restore_x(self) -> None:
        """Should update last restore x position."""
        sm = SnapshotStateMachine()
        sm.state = SnapshotState.RESTORING
        sm._last_restore_x = 100

        ctx = make_context(x_pos=150)
        sm.transition(ctx)

        assert sm._last_restore_x == 150


class TestGiveUpState:
    """Tests for GIVE_UP state transitions."""

    def test_returns_end_episode_action(self) -> None:
        """Should return END_EPISODE action."""
        sm = SnapshotStateMachine()
        sm.state = SnapshotState.GIVE_UP

        ctx = make_context()
        state, action = sm.transition(ctx)

        assert state == SnapshotState.GIVE_UP
        assert action == SnapshotAction.END_EPISODE

    def test_transitions_to_running(self) -> None:
        """Should transition to RUNNING after give up."""
        sm = SnapshotStateMachine()
        sm.state = SnapshotState.GIVE_UP

        ctx = make_context()
        sm.transition(ctx)

        assert sm.state == SnapshotState.RUNNING

    def test_resets_restore_counter(self) -> None:
        """Should reset restore counter on give up."""
        sm = SnapshotStateMachine()
        sm.state = SnapshotState.GIVE_UP
        sm._restores_without_progress = 5

        ctx = make_context()
        sm.transition(ctx)

        assert sm._restores_without_progress == 0


class TestReset:
    """Tests for state machine reset."""

    def test_reset_restores_initial_state(self) -> None:
        """Reset should restore initial state."""
        sm = SnapshotStateMachine()
        sm.state = SnapshotState.DEAD
        sm._last_checkpoint_time = 1000
        sm._restores_without_progress = 5
        sm._last_restore_x = 500
        sm._current_hotspot_target = 600

        sm.reset()

        assert sm.state == SnapshotState.RUNNING
        assert sm._last_checkpoint_time == 0
        assert sm._restores_without_progress == 0
        assert sm._last_restore_x == 0
        assert sm._current_hotspot_target is None


class TestNotifyMethods:
    """Tests for notification methods."""

    def test_notify_save_complete(self) -> None:
        """notify_save_complete should transition from SAVE_SNAPSHOT to SNAPSHOT_SAVED."""
        sm = SnapshotStateMachine()
        sm.state = SnapshotState.SAVE_SNAPSHOT

        sm.notify_save_complete()

        assert sm.state == SnapshotState.SNAPSHOT_SAVED

    def test_notify_save_complete_ignored_in_other_states(self) -> None:
        """notify_save_complete should be ignored in other states."""
        sm = SnapshotStateMachine()
        sm.state = SnapshotState.RUNNING

        sm.notify_save_complete()

        assert sm.state == SnapshotState.RUNNING

    def test_notify_restore_complete_with_progress(self) -> None:
        """notify_restore_complete should reset counter on progress."""
        sm = SnapshotStateMachine(progress_threshold=50)
        sm.state = SnapshotState.RESTORING
        sm._last_restore_x = 100
        sm._restores_without_progress = 2

        sm.notify_restore_complete(restored_x=200)

        assert sm.state == SnapshotState.RUNNING
        assert sm._restores_without_progress == 0
        assert sm._last_restore_x == 200

    def test_notify_restore_complete_without_progress(self) -> None:
        """notify_restore_complete should increment counter without progress."""
        sm = SnapshotStateMachine(progress_threshold=50)
        sm.state = SnapshotState.RESTORING
        sm._last_restore_x = 100
        sm._restores_without_progress = 1

        sm.notify_restore_complete(restored_x=120)

        assert sm.state == SnapshotState.RUNNING
        assert sm._restores_without_progress == 2
        assert sm._last_restore_x == 120


class TestFullWorkflow:
    """Integration tests for complete workflows."""

    def test_hotspot_save_and_restore_workflow(self) -> None:
        """Test complete workflow: approach hotspot, save, die, restore."""
        sm = SnapshotStateMachine(
            hotspot_approach_distance=100,
            hotspot_save_offset=50,
            hotspot_save_tolerance=25,
        )

        # 1. Normal running
        state, action = sm.transition(make_context(x_pos=300, hotspot_positions=(500,)))
        assert state == SnapshotState.RUNNING

        # 2. Enter approach zone
        state, action = sm.transition(make_context(x_pos=420, hotspot_positions=(500,)))
        assert state == SnapshotState.APPROACHING_HOTSPOT

        # 3. Reach save position
        state, action = sm.transition(make_context(x_pos=450, hotspot_positions=(500,)))
        assert state == SnapshotState.SAVE_SNAPSHOT
        assert action == SnapshotAction.SAVE_SNAPSHOT

        # 4. Save complete
        state, action = sm.transition(make_context(x_pos=450))
        assert state == SnapshotState.SNAPSHOT_SAVED

        # 5. Back to running
        state, action = sm.transition(make_context(x_pos=460))
        assert state == SnapshotState.RUNNING

        # 6. Die in hotspot
        state, action = sm.transition(make_context(x_pos=480, is_dead=True))
        assert state == SnapshotState.DEAD

        # 7. Evaluate restore
        state, action = sm.transition(make_context(snapshot_available=True))
        assert state == SnapshotState.EVALUATE_RESTORE

        # 8. Decide to restore
        state, action = sm.transition(make_context(snapshot_available=True))
        assert state == SnapshotState.RESTORING
        assert action == SnapshotAction.RESTORE_SNAPSHOT

        # 9. Restore complete (back at save position)
        sm.notify_restore_complete(restored_x=450)
        assert sm.state == SnapshotState.RUNNING

    def test_repeated_deaths_leads_to_give_up(self) -> None:
        """Test that repeated deaths without progress leads to give up.

        The state machine tracks progress by comparing restore positions.
        If we keep restoring to the same position (no forward progress),
        the counter increments until we give up.
        """
        sm = SnapshotStateMachine(
            max_restores_without_progress=2,
            progress_threshold=50,
        )

        # Set up: simulate having restored before at x=500
        # This establishes a baseline for progress detection
        sm._last_restore_x = 500

        # First death at the same spot - restore to x=500 (no progress)
        sm.state = SnapshotState.DEAD
        sm.transition(make_context())  # -> EVALUATE_RESTORE
        sm.transition(make_context(snapshot_available=True))  # -> RESTORING
        sm.notify_restore_complete(restored_x=500)  # Same spot, no progress
        assert sm._restores_without_progress == 1

        # Second death at same spot - restore to x=500 (no progress)
        sm.state = SnapshotState.DEAD
        sm.transition(make_context())  # -> EVALUATE_RESTORE
        sm.transition(make_context(snapshot_available=True))  # -> RESTORING
        sm.notify_restore_complete(restored_x=510)  # Barely moved, no real progress
        assert sm._restores_without_progress == 2

        # Third death - should give up (counter >= max)
        sm.state = SnapshotState.DEAD
        sm.transition(make_context())  # -> EVALUATE_RESTORE
        state, action = sm.transition(make_context(snapshot_available=True))
        assert state == SnapshotState.GIVE_UP
        assert action == SnapshotAction.NONE

        # Process give up
        state, action = sm.transition(make_context())
        assert action == SnapshotAction.END_EPISODE

    def test_checkpoint_based_save_workflow(self) -> None:
        """Test time-based checkpoint workflow."""
        sm = SnapshotStateMachine(checkpoint_interval=500)

        # Running with time passing
        state, action = sm.transition(make_context(game_time=100))
        assert state == SnapshotState.RUNNING

        state, action = sm.transition(make_context(game_time=400))
        assert state == SnapshotState.RUNNING

        # Checkpoint due
        state, action = sm.transition(make_context(game_time=500))
        assert state == SnapshotState.CHECKPOINT_DUE

        # Save triggered
        state, action = sm.transition(make_context(game_time=500))
        assert state == SnapshotState.SAVE_SNAPSHOT
        assert action == SnapshotAction.SAVE_SNAPSHOT

        # Complete save
        state, action = sm.transition(make_context(game_time=510))
        assert state == SnapshotState.SNAPSHOT_SAVED

        # Back to running
        state, action = sm.transition(make_context(game_time=510))
        assert state == SnapshotState.RUNNING

        # Next checkpoint
        state, action = sm.transition(make_context(game_time=1000))
        assert state == SnapshotState.CHECKPOINT_DUE
