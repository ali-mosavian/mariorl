"""
Snapshot/Restore State Machine for Mario RL Training.

This module implements a state machine to manage emulator snapshotting and
restoring based on death hotspots. The goal is to improve learning efficiency
by allowing the agent to practice difficult sections identified by frequent deaths.

State Machine Diagram
=====================

                                        ┌─────────────────┐
                                        │                 │
                                        │    RUNNING      │◄──────────────────────┐
                                        │                 │                       │
                                        └────────┬────────┘                       │
                                                 │                                │
                        ┌────────────────────────┼────────────────────────┐       │
                        │                        │                        │       │
                        ▼                        ▼                        ▼       │
            ┌───────────────────┐    ┌───────────────────┐    ┌──────────────┐    │
            │                   │    │                   │    │              │    │
            │ APPROACHING_      │    │  CHECKPOINT_DUE   │    │    DEAD      │    │
            │ HOTSPOT           │    │  (time-based)     │    │              │    │
            │                   │    │                   │    └──────┬───────┘    │
            └─────────┬─────────┘    └─────────┬─────────┘           │            │
                      │                        │                     │            │
                      │                        │           ┌─────────┴─────────┐  │
                      ▼                        ▼           ▼                   ▼  │
            ┌───────────────────┐    ┌───────────────────────────┐   ┌─────────────────┐
            │                   │    │                           │   │                 │
            │  SAVE_SNAPSHOT    │───►│      SNAPSHOT_SAVED       │   │  EVALUATE_      │
            │  (near hotspot)   │    │                           │   │  RESTORE        │
            │                   │    └─────────────┬─────────────┘   │                 │
            └───────────────────┘                  │                 └────────┬────────┘
                                                   │                          │
                                                   │              ┌───────────┴───────────┐
                                                   │              │                       │
                                                   │              ▼                       ▼
                                                   │    ┌─────────────────┐     ┌─────────────────┐
                                                   │    │                 │     │                 │
                                                   │    │   RESTORING     │     │   GIVE_UP       │
                                                   │    │                 │     │                 │
                                                   │    └────────┬────────┘     └────────┬────────┘
                                                   │             │                       │
                                                   │             │ (success)             │
                                                   └─────────────┴───────────────────────┘
                                                                 │
                                                                 ▼
                                                          (back to RUNNING
                                                           or episode ends)

State Descriptions
==================

RUNNING:
    Normal gameplay state. Mario is alive and moving through the level.
    Transitions to:
    - APPROACHING_HOTSPOT: When x_pos enters approach zone before a death hotspot
    - CHECKPOINT_DUE: When time-based checkpoint interval is reached
    - DEAD: When death is detected

APPROACHING_HOTSPOT:
    Mario is approaching a known death hotspot. We're looking for the optimal
    position to save a snapshot.
    Transitions to:
    - SAVE_SNAPSHOT: When optimal snapshot position is reached
    - DEAD: When death is detected before reaching save point
    - RUNNING: When hotspot zone is passed without needing to save

CHECKPOINT_DUE:
    Time-based checkpoint is due. Used as fallback in unexplored areas.
    Transitions to:
    - SAVE_SNAPSHOT: Immediately, to save the checkpoint
    - DEAD: If death occurs before save

SAVE_SNAPSHOT:
    Saving emulator state (NES state + frame stack).
    Transitions to:
    - SNAPSHOT_SAVED: After save completes

SNAPSHOT_SAVED:
    Transient state indicating save completed.
    Transitions to:
    - RUNNING: Immediately

DEAD:
    Mario just died. Need to decide whether to restore.
    Transitions to:
    - EVALUATE_RESTORE: Immediately

EVALUATE_RESTORE:
    Evaluating whether and where to restore from a snapshot.
    Transitions to:
    - RESTORING: If snapshot available and restore attempts not exhausted
    - GIVE_UP: If no snapshot or too many failed restore attempts

RESTORING:
    Loading a snapshot and restoring emulator state.
    Transitions to:
    - RUNNING: After successful restore

GIVE_UP:
    No more restore attempts. Let the episode end naturally.
    Transitions to:
    - RUNNING: After episode reset (externally triggered)
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from enum import auto


class SnapshotState(Enum):
    """States for the snapshot/restore state machine."""

    RUNNING = auto()
    APPROACHING_HOTSPOT = auto()
    CHECKPOINT_DUE = auto()
    SAVE_SNAPSHOT = auto()
    SNAPSHOT_SAVED = auto()
    DEAD = auto()
    EVALUATE_RESTORE = auto()
    RESTORING = auto()
    GIVE_UP = auto()


class SnapshotAction(Enum):
    """Actions that can be triggered by state transitions."""

    NONE = auto()
    SAVE_SNAPSHOT = auto()
    RESTORE_SNAPSHOT = auto()
    END_EPISODE = auto()


@dataclass(frozen=True)
class SnapshotContext:
    """Context data for state transitions.

    Attributes:
        x_pos: Mario's current x position in pixels.
        level_id: Current level identifier (e.g., "1-1").
        game_time: Current game time in ticks.
        is_dead: Whether Mario just died.
        flag_get: Whether Mario reached the flag (level complete).
        best_x: Best x position reached this episode.
        hotspot_positions: Suggested snapshot positions from DeathHotspotAggregate.
        snapshot_available: Whether any snapshots are available to restore.
        suggested_restore_x: Suggested x position to restore to (from hotspot analysis).
    """

    x_pos: int
    level_id: str
    game_time: int
    is_dead: bool
    flag_get: bool
    best_x: int
    hotspot_positions: tuple[int, ...]
    snapshot_available: bool
    suggested_restore_x: int | None = None


@dataclass
class SnapshotStateMachine:
    """State machine for managing emulator snapshots.

    This state machine decides when to save snapshots and when/where to restore
    from them based on death hotspots and other heuristics.

    Attributes:
        hotspot_approach_distance: Pixels before hotspot to start tracking approach.
        hotspot_save_offset: Pixels before hotspot to save snapshot.
        hotspot_save_tolerance: Tolerance in pixels for save position matching.
        checkpoint_interval: Game ticks between time-based checkpoints.
        max_restores_without_progress: Max restore attempts without progress before giving up.
        progress_threshold: Minimum x progress to reset restore counter.
    """

    hotspot_approach_distance: int = 100
    hotspot_save_offset: int = 50
    hotspot_save_tolerance: int = 25
    checkpoint_interval: int = 500
    max_restores_without_progress: int = 3
    progress_threshold: int = 50

    # Current state
    state: SnapshotState = field(default=SnapshotState.RUNNING, init=False)

    # Internal tracking
    _last_checkpoint_time: int = field(default=0, init=False)
    _restores_without_progress: int = field(default=0, init=False)
    _last_restore_x: int = field(default=0, init=False)
    _current_hotspot_target: int | None = field(default=None, init=False)

    def transition(self, ctx: SnapshotContext) -> tuple[SnapshotState, SnapshotAction]:
        """Process a state transition based on current context.

        Args:
            ctx: Current game context.

        Returns:
            Tuple of (new_state, action) where action indicates what the caller should do.
        """
        match self.state:
            case SnapshotState.RUNNING:
                return self._handle_running(ctx)

            case SnapshotState.APPROACHING_HOTSPOT:
                return self._handle_approaching_hotspot(ctx)

            case SnapshotState.CHECKPOINT_DUE:
                return self._handle_checkpoint_due(ctx)

            case SnapshotState.SAVE_SNAPSHOT:
                return self._handle_save_snapshot(ctx)

            case SnapshotState.SNAPSHOT_SAVED:
                self.state = SnapshotState.RUNNING
                return (SnapshotState.RUNNING, SnapshotAction.NONE)

            case SnapshotState.DEAD:
                return self._handle_dead(ctx)

            case SnapshotState.EVALUATE_RESTORE:
                return self._handle_evaluate_restore(ctx)

            case SnapshotState.RESTORING:
                return self._handle_restoring(ctx)

            case SnapshotState.GIVE_UP:
                self.state = SnapshotState.RUNNING
                self._restores_without_progress = 0
                return (SnapshotState.GIVE_UP, SnapshotAction.END_EPISODE)

        return (self.state, SnapshotAction.NONE)

    def _handle_running(self, ctx: SnapshotContext) -> tuple[SnapshotState, SnapshotAction]:
        """Handle RUNNING state transitions."""
        # Check for death first (highest priority)
        if ctx.is_dead:
            self.state = SnapshotState.DEAD
            return (SnapshotState.DEAD, SnapshotAction.NONE)

        # Check if approaching a hotspot
        for hotspot_x in ctx.hotspot_positions:
            approach_start = hotspot_x - self.hotspot_approach_distance
            if approach_start <= ctx.x_pos < hotspot_x:
                self.state = SnapshotState.APPROACHING_HOTSPOT
                self._current_hotspot_target = hotspot_x
                return (SnapshotState.APPROACHING_HOTSPOT, SnapshotAction.NONE)

        # Check for time-based checkpoint
        if ctx.game_time - self._last_checkpoint_time >= self.checkpoint_interval:
            self.state = SnapshotState.CHECKPOINT_DUE
            return (SnapshotState.CHECKPOINT_DUE, SnapshotAction.NONE)

        return (SnapshotState.RUNNING, SnapshotAction.NONE)

    def _handle_approaching_hotspot(
        self, ctx: SnapshotContext
    ) -> tuple[SnapshotState, SnapshotAction]:
        """Handle APPROACHING_HOTSPOT state transitions."""
        if ctx.is_dead:
            self.state = SnapshotState.DEAD
            self._current_hotspot_target = None
            return (SnapshotState.DEAD, SnapshotAction.NONE)

        # Check if we've reached optimal snapshot position
        if self._current_hotspot_target is not None:
            snapshot_x = self._current_hotspot_target - self.hotspot_save_offset
            if abs(ctx.x_pos - snapshot_x) <= self.hotspot_save_tolerance:
                self.state = SnapshotState.SAVE_SNAPSHOT
                return (SnapshotState.SAVE_SNAPSHOT, SnapshotAction.SAVE_SNAPSHOT)

            # If we passed the hotspot, go back to running
            if ctx.x_pos > self._current_hotspot_target:
                self.state = SnapshotState.RUNNING
                self._current_hotspot_target = None
                return (SnapshotState.RUNNING, SnapshotAction.NONE)

        return (SnapshotState.APPROACHING_HOTSPOT, SnapshotAction.NONE)

    def _handle_checkpoint_due(
        self, ctx: SnapshotContext
    ) -> tuple[SnapshotState, SnapshotAction]:
        """Handle CHECKPOINT_DUE state transitions."""
        if ctx.is_dead:
            self.state = SnapshotState.DEAD
            return (SnapshotState.DEAD, SnapshotAction.NONE)

        self._last_checkpoint_time = ctx.game_time
        self.state = SnapshotState.SAVE_SNAPSHOT
        return (SnapshotState.SAVE_SNAPSHOT, SnapshotAction.SAVE_SNAPSHOT)

    def _handle_save_snapshot(
        self, ctx: SnapshotContext
    ) -> tuple[SnapshotState, SnapshotAction]:
        """Handle SAVE_SNAPSHOT state transitions."""
        # After save action is taken externally, mark as saved
        self.state = SnapshotState.SNAPSHOT_SAVED
        self._current_hotspot_target = None
        return (SnapshotState.SNAPSHOT_SAVED, SnapshotAction.NONE)

    def _handle_dead(self, ctx: SnapshotContext) -> tuple[SnapshotState, SnapshotAction]:
        """Handle DEAD state transitions."""
        self.state = SnapshotState.EVALUATE_RESTORE
        return (SnapshotState.EVALUATE_RESTORE, SnapshotAction.NONE)

    def _handle_evaluate_restore(
        self, ctx: SnapshotContext
    ) -> tuple[SnapshotState, SnapshotAction]:
        """Handle EVALUATE_RESTORE state transitions."""
        # Check if we should give up
        if self._restores_without_progress >= self.max_restores_without_progress:
            self.state = SnapshotState.GIVE_UP
            return (SnapshotState.GIVE_UP, SnapshotAction.NONE)

        # Check if we have a snapshot to restore
        if not ctx.snapshot_available:
            self.state = SnapshotState.GIVE_UP
            return (SnapshotState.GIVE_UP, SnapshotAction.NONE)

        # Attempt restore
        self.state = SnapshotState.RESTORING
        return (SnapshotState.RESTORING, SnapshotAction.RESTORE_SNAPSHOT)

    def _handle_restoring(
        self, ctx: SnapshotContext
    ) -> tuple[SnapshotState, SnapshotAction]:
        """Handle RESTORING state transitions."""
        # Check if we made progress since last restore
        if ctx.x_pos > self._last_restore_x + self.progress_threshold:
            self._restores_without_progress = 0
        else:
            self._restores_without_progress += 1

        self._last_restore_x = ctx.x_pos
        self.state = SnapshotState.RUNNING
        return (SnapshotState.RUNNING, SnapshotAction.NONE)

    def reset(self) -> None:
        """Reset state machine for a new episode."""
        self.state = SnapshotState.RUNNING
        self._last_checkpoint_time = 0
        self._restores_without_progress = 0
        self._last_restore_x = 0
        self._current_hotspot_target = None

    def notify_save_complete(self) -> None:
        """Notify the state machine that a save operation completed.

        Call this after successfully saving a snapshot.
        """
        if self.state == SnapshotState.SAVE_SNAPSHOT:
            self.state = SnapshotState.SNAPSHOT_SAVED

    def notify_restore_complete(self, restored_x: int) -> None:
        """Notify the state machine that a restore operation completed.

        Call this after successfully restoring from a snapshot.

        Args:
            restored_x: The x position after restoration.
        """
        if self.state == SnapshotState.RESTORING:
            # Check progress
            if restored_x > self._last_restore_x + self.progress_threshold:
                self._restores_without_progress = 0
            else:
                self._restores_without_progress += 1

            self._last_restore_x = restored_x
            self.state = SnapshotState.RUNNING
