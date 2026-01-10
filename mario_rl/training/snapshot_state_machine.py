"""
Snapshot/Restore State Machine for Mario RL Training.

This module implements a DFA-based state machine to manage emulator snapshotting
and restoring based on death hotspots. The goal is to improve learning efficiency
by allowing the agent to practice difficult sections identified by frequent deaths.

DFA Design
==========

The state machine uses a table-driven approach:
1. Events are computed from context (timeout, death, approaching hotspot, etc.)
2. Transitions are defined as a table: (State, Event) → (State, Action)
3. Side effects (counter updates) are handled separately on state entry/exit

State Machine Diagram
=====================

                                        ┌─────────────────┐
                                        │                 │
                                        │    RUNNING      │◄──────────────────────┐
                                        │                 │                       │
                                        └────────┬────────┘                       │
                                                 │                                │
                 ┌───────────────────────────────┼────────────────────────┐       │
                 │                               │                        │       │
                 │          ┌────────────────────┼────────────────────┐   │       │
                 │          │                    │                    │   │       │
                 ▼          ▼                    ▼                    ▼   ▼       │
    ┌──────────────┐ ┌───────────────┐ ┌───────────────┐    ┌──────────────┐      │
    │              │ │               │ │               │    │              │      │
    │   TIMEOUT    │ │ APPROACHING_  │ │ CHECKPOINT_   │    │    DEAD      │      │
    │  (timer out) │ │ HOTSPOT       │ │ DUE           │    │              │      │
    │              │ │               │ │               │    └──────┬───────┘      │
    └──────┬───────┘ └───────┬───────┘ └───────┬───────┘           │              │
           │                 │                 │                   │              │
           │                 │                 │         ┌─────────┴─────────┐    │
           │                 ▼                 ▼         ▼                   ▼    │
           │      ┌───────────────────┐ ┌─────────────────┐    ┌───────────────────┐
           │      │                   │ │                 │    │                   │
           │      │  SAVE_SNAPSHOT    │►│ SNAPSHOT_SAVED  │    │ EVALUATE_RESTORE  │
           │      │                   │ │                 │    │                   │
           │      └───────────────────┘ └────────┬────────┘    └────────┬──────────┘
           │                                     │                      │
           │                                     │          ┌───────────┴───────────┐
           │                                     │          │                       │
           │                                     │          ▼                       ▼
           │                                     │ ┌─────────────────┐     ┌─────────────────┐
           │                                     │ │                 │     │                 │
           │                                     │ │   RESTORING     │     │   GIVE_UP       │
           │                                     │ │                 │     │                 │
           │                                     │ └────────┬────────┘     └────────┬────────┘
           │                                     │          │                       │
           │                                     │          │ (success)             │
           │  (end episode)                      └──────────┴───────────────────────┘
           │                                                │
           └────────────────────────────────────────────────┘
                                                            │
                                                            ▼
                                                     (back to RUNNING
                                                      or episode ends)

Transition Table
================

The complete transition table is defined in TRANSITIONS. Each entry maps
(current_state, event) to (next_state, action).
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
    TIMEOUT = auto()
    EVALUATE_RESTORE = auto()
    RESTORING = auto()
    GIVE_UP = auto()


class SnapshotAction(Enum):
    """Actions that can be triggered by state transitions."""

    NONE = auto()
    SAVE_SNAPSHOT = auto()
    RESTORE_SNAPSHOT = auto()
    END_EPISODE = auto()


class Event(Enum):
    """Events that trigger state transitions.

    Events are computed from SnapshotContext and internal state.
    Priority (highest first): TIMEOUT > DEATH > AT_SAVE_POSITION > etc.
    """

    # Episode-ending events (highest priority)
    TIMEOUT = auto()  # Timer ran out
    DEATH = auto()  # Skill-based death

    # Hotspot-related events
    APPROACHING_HOTSPOT = auto()  # Entering approach zone
    AT_SAVE_POSITION = auto()  # At optimal save position
    PASSED_HOTSPOT = auto()  # Passed hotspot without saving

    # Checkpoint events
    CHECKPOINT_DUE = auto()  # Time-based checkpoint interval reached

    # Restore evaluation events
    MAX_RESTORES_EXCEEDED = auto()  # Too many restore attempts
    NO_SNAPSHOT = auto()  # No snapshot available
    SNAPSHOT_AVAILABLE = auto()  # Snapshot available for restore

    # Progress tracking
    PROGRESS_MADE = auto()  # Made progress since last restore
    NO_PROGRESS = auto()  # No progress since last restore

    # Default/continuation
    CONTINUE = auto()  # No event, stay in current state or auto-transition


@dataclass(frozen=True)
class SnapshotContext:
    """Context data for state transitions.

    Attributes:
        x_pos: Mario's current x position in pixels.
        level_id: Current level identifier (e.g., "1-1").
        game_time: Current game time in ticks.
        is_dead: Whether Mario just died (skill-based death, excludes timeouts).
        is_timeout: Whether the episode ended due to timer running out.
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
    is_timeout: bool
    flag_get: bool
    best_x: int
    hotspot_positions: tuple[int, ...]
    snapshot_available: bool
    suggested_restore_x: int | None = None


# Type alias for transition table
_Transition = tuple[SnapshotState, SnapshotAction]

# Transition table: (state, event) → (new_state, action)
# Missing entries default to (current_state, NONE)
TRANSITIONS: dict[tuple[SnapshotState, Event], _Transition] = {
    # === RUNNING state ===
    (SnapshotState.RUNNING, Event.TIMEOUT): (SnapshotState.TIMEOUT, SnapshotAction.NONE),
    (SnapshotState.RUNNING, Event.DEATH): (SnapshotState.DEAD, SnapshotAction.NONE),
    (SnapshotState.RUNNING, Event.APPROACHING_HOTSPOT): (
        SnapshotState.APPROACHING_HOTSPOT,
        SnapshotAction.NONE,
    ),
    (SnapshotState.RUNNING, Event.CHECKPOINT_DUE): (
        SnapshotState.CHECKPOINT_DUE,
        SnapshotAction.NONE,
    ),
    # === TIMEOUT state ===
    (SnapshotState.TIMEOUT, Event.CONTINUE): (SnapshotState.RUNNING, SnapshotAction.END_EPISODE),
    # === DEAD state ===
    (SnapshotState.DEAD, Event.CONTINUE): (SnapshotState.EVALUATE_RESTORE, SnapshotAction.NONE),
    # === APPROACHING_HOTSPOT state ===
    (SnapshotState.APPROACHING_HOTSPOT, Event.TIMEOUT): (
        SnapshotState.TIMEOUT,
        SnapshotAction.NONE,
    ),
    (SnapshotState.APPROACHING_HOTSPOT, Event.DEATH): (SnapshotState.DEAD, SnapshotAction.NONE),
    (SnapshotState.APPROACHING_HOTSPOT, Event.AT_SAVE_POSITION): (
        SnapshotState.SAVE_SNAPSHOT,
        SnapshotAction.SAVE_SNAPSHOT,
    ),
    (SnapshotState.APPROACHING_HOTSPOT, Event.PASSED_HOTSPOT): (
        SnapshotState.RUNNING,
        SnapshotAction.NONE,
    ),
    # === CHECKPOINT_DUE state ===
    (SnapshotState.CHECKPOINT_DUE, Event.TIMEOUT): (SnapshotState.TIMEOUT, SnapshotAction.NONE),
    (SnapshotState.CHECKPOINT_DUE, Event.DEATH): (SnapshotState.DEAD, SnapshotAction.NONE),
    (SnapshotState.CHECKPOINT_DUE, Event.CONTINUE): (
        SnapshotState.SAVE_SNAPSHOT,
        SnapshotAction.SAVE_SNAPSHOT,
    ),
    # === SAVE_SNAPSHOT state ===
    (SnapshotState.SAVE_SNAPSHOT, Event.CONTINUE): (
        SnapshotState.SNAPSHOT_SAVED,
        SnapshotAction.NONE,
    ),
    # === SNAPSHOT_SAVED state ===
    (SnapshotState.SNAPSHOT_SAVED, Event.CONTINUE): (SnapshotState.RUNNING, SnapshotAction.NONE),
    # === EVALUATE_RESTORE state ===
    (SnapshotState.EVALUATE_RESTORE, Event.MAX_RESTORES_EXCEEDED): (
        SnapshotState.GIVE_UP,
        SnapshotAction.NONE,
    ),
    (SnapshotState.EVALUATE_RESTORE, Event.NO_SNAPSHOT): (
        SnapshotState.GIVE_UP,
        SnapshotAction.NONE,
    ),
    (SnapshotState.EVALUATE_RESTORE, Event.SNAPSHOT_AVAILABLE): (
        SnapshotState.RESTORING,
        SnapshotAction.RESTORE_SNAPSHOT,
    ),
    # === RESTORING state ===
    (SnapshotState.RESTORING, Event.PROGRESS_MADE): (SnapshotState.RUNNING, SnapshotAction.NONE),
    (SnapshotState.RESTORING, Event.NO_PROGRESS): (SnapshotState.RUNNING, SnapshotAction.NONE),
    # === GIVE_UP state ===
    (SnapshotState.GIVE_UP, Event.CONTINUE): (SnapshotState.RUNNING, SnapshotAction.END_EPISODE),
}


@dataclass
class SnapshotStateMachine:
    """DFA-based state machine for managing emulator snapshots.

    This state machine decides when to save snapshots and when/where to restore
    from them based on death hotspots and other heuristics.

    The implementation uses a table-driven approach:
    1. _compute_event() determines the current event from context
    2. TRANSITIONS table defines state transitions
    3. _on_exit()/_on_enter() handle side effects

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
            Tuple of (state, action) where:
            - For END_EPISODE actions: returns the terminal state that triggered the end
            - Otherwise: returns the new state after transition
        """
        # Compute event from context
        event = self._compute_event(ctx)

        # Look up transition
        key = (self.state, event)
        if key not in TRANSITIONS:
            # No transition defined - stay in current state
            return (self.state, SnapshotAction.NONE)

        new_state, action = TRANSITIONS[key]

        # Execute side effects
        self._on_exit(self.state, event, ctx)
        old_state = self.state
        self.state = new_state
        self._on_enter(old_state, new_state, event, ctx)

        # For END_EPISODE, return the terminal state that triggered it
        # (TIMEOUT, GIVE_UP) so callers know why the episode ended
        if action == SnapshotAction.END_EPISODE:
            return (old_state, action)

        return (new_state, action)

    def _compute_event(self, ctx: SnapshotContext) -> Event:
        """Compute the current event from context.

        Events are checked in priority order (highest first).
        """
        match self.state:
            case SnapshotState.RUNNING:
                return self._compute_running_event(ctx)

            case SnapshotState.APPROACHING_HOTSPOT:
                return self._compute_approaching_event(ctx)

            case SnapshotState.CHECKPOINT_DUE:
                return self._compute_checkpoint_event(ctx)

            case SnapshotState.EVALUATE_RESTORE:
                return self._compute_restore_event(ctx)

            case SnapshotState.RESTORING:
                return self._compute_restoring_event(ctx)

            case _:
                # States that auto-transition (DEAD, TIMEOUT, SAVE_SNAPSHOT, etc.)
                return Event.CONTINUE

    def _compute_running_event(self, ctx: SnapshotContext) -> Event:
        """Compute event for RUNNING state."""
        # Priority: timeout > death > approaching hotspot > checkpoint
        if ctx.is_timeout:
            return Event.TIMEOUT
        if ctx.is_dead:
            return Event.DEATH

        # Check if approaching a hotspot
        for hotspot_x in ctx.hotspot_positions:
            approach_start = hotspot_x - self.hotspot_approach_distance
            if approach_start <= ctx.x_pos < hotspot_x:
                self._current_hotspot_target = hotspot_x
                return Event.APPROACHING_HOTSPOT

        # Check for time-based checkpoint
        if ctx.game_time - self._last_checkpoint_time >= self.checkpoint_interval:
            return Event.CHECKPOINT_DUE

        return Event.CONTINUE

    def _compute_approaching_event(self, ctx: SnapshotContext) -> Event:
        """Compute event for APPROACHING_HOTSPOT state."""
        # Priority: timeout > death > at save position > passed hotspot
        if ctx.is_timeout:
            return Event.TIMEOUT
        if ctx.is_dead:
            return Event.DEATH

        if self._current_hotspot_target is not None:
            snapshot_x = self._current_hotspot_target - self.hotspot_save_offset
            if abs(ctx.x_pos - snapshot_x) <= self.hotspot_save_tolerance:
                return Event.AT_SAVE_POSITION
            if ctx.x_pos > self._current_hotspot_target:
                return Event.PASSED_HOTSPOT

        return Event.CONTINUE

    def _compute_checkpoint_event(self, ctx: SnapshotContext) -> Event:
        """Compute event for CHECKPOINT_DUE state."""
        if ctx.is_timeout:
            return Event.TIMEOUT
        if ctx.is_dead:
            return Event.DEATH
        return Event.CONTINUE

    def _compute_restore_event(self, ctx: SnapshotContext) -> Event:
        """Compute event for EVALUATE_RESTORE state."""
        if self._restores_without_progress >= self.max_restores_without_progress:
            return Event.MAX_RESTORES_EXCEEDED
        if not ctx.snapshot_available:
            return Event.NO_SNAPSHOT
        return Event.SNAPSHOT_AVAILABLE

    def _compute_restoring_event(self, ctx: SnapshotContext) -> Event:
        """Compute event for RESTORING state."""
        if ctx.x_pos > self._last_restore_x + self.progress_threshold:
            return Event.PROGRESS_MADE
        return Event.NO_PROGRESS

    def _on_exit(
        self, state: SnapshotState, event: Event, ctx: SnapshotContext
    ) -> None:
        """Handle side effects when exiting a state."""
        match state:
            case SnapshotState.APPROACHING_HOTSPOT:
                # Clear hotspot target when leaving
                if event in (Event.TIMEOUT, Event.DEATH, Event.PASSED_HOTSPOT):
                    self._current_hotspot_target = None

            case SnapshotState.SAVE_SNAPSHOT:
                # Clear hotspot target after saving
                self._current_hotspot_target = None

            case SnapshotState.CHECKPOINT_DUE:
                # Update last checkpoint time
                if event == Event.CONTINUE:
                    self._last_checkpoint_time = ctx.game_time

            case SnapshotState.TIMEOUT:
                # Reset restore counter on timeout
                self._restores_without_progress = 0

            case SnapshotState.GIVE_UP:
                # Reset restore counter on give up
                self._restores_without_progress = 0

            case SnapshotState.RESTORING:
                # Track progress
                if event == Event.PROGRESS_MADE:
                    self._restores_without_progress = 0
                else:
                    self._restores_without_progress += 1
                self._last_restore_x = ctx.x_pos

    def _on_enter(
        self,
        old_state: SnapshotState,
        new_state: SnapshotState,
        event: Event,
        ctx: SnapshotContext,
    ) -> None:
        """Handle side effects when entering a state.

        Currently unused but available for future extensions.
        """
        pass

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
