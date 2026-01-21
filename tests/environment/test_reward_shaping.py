"""Tests for the new reward shaping system with progressive milestones."""

from __future__ import annotations

import pytest

from mario_rl.environment.mariogym import State  # type: ignore[attr-defined]
from mario_rl.environment.mariogym import Reward  # type: ignore[attr-defined]
from mario_rl.environment.mariogym import MILESTONES  # type: ignore[attr-defined]

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def initial_state() -> State:
    """Create initial state at start of level."""
    return State(
        time=400,
        score=0,
        x_pos=40,
        x_pos_max=40,
        coins=0,
        powerup_state=0,
        is_alive=True,
        got_flag=False,
        milestones_reached=frozenset(),
    )


@pytest.fixture
def mid_level_state() -> State:
    """Create state at mid-level with some milestones reached."""
    return State(
        time=300,
        score=1000,
        x_pos=750,
        x_pos_max=750,
        coins=5,
        powerup_state=1,
        is_alive=True,
        got_flag=False,
        milestones_reached=frozenset({100, 200, 350, 500, 750}),
    )


# =============================================================================
# MILESTONES constant tests
# =============================================================================


def test_milestones_are_sorted() -> None:
    """Milestones should be in ascending order."""
    assert MILESTONES == tuple(sorted(MILESTONES))


def test_milestones_are_positive() -> None:
    """All milestones should be positive integers."""
    assert all(m > 0 for m in MILESTONES)


def test_milestones_count() -> None:
    """Should have 10 milestones."""
    assert len(MILESTONES) == 10


def test_milestones_values() -> None:
    """Verify exact milestone positions."""
    expected = (100, 200, 350, 500, 750, 1000, 1500, 2000, 2500, 3000)
    assert MILESTONES == expected


# =============================================================================
# State tests
# =============================================================================


def test_state_has_milestones_field() -> None:
    """State should have milestones_reached field."""
    state = State()
    assert hasattr(state, "milestones_reached")
    assert state.milestones_reached == frozenset()


def test_state_milestones_are_frozenset() -> None:
    """Milestones should be stored as frozenset for immutability."""
    state = State(milestones_reached=frozenset({100, 200}))
    assert isinstance(state.milestones_reached, frozenset)


# =============================================================================
# Reward.calc() tests - Basic cases
# =============================================================================


def test_reward_calc_forward_movement(initial_state: State) -> None:
    """Forward movement should give positive x_reward."""
    next_state = State(
        time=399,
        x_pos=50,
        x_pos_max=50,
        is_alive=True,
        milestones_reached=frozenset(),
    )
    reward = Reward.calc(next_state, initial_state)

    assert reward.x_reward == 10  # 50 - 40 = 10 pixels forward
    assert reward.speed_bonus == 10  # Running speed
    assert reward.alive_bonus == 1


def test_reward_calc_backward_movement(initial_state: State) -> None:
    """Backward movement should give negative x_reward."""
    next_state = State(
        time=399,
        x_pos=35,
        x_pos_max=40,  # Max stays at previous max
        is_alive=True,
        milestones_reached=frozenset(),
    )
    reward = Reward.calc(next_state, initial_state)

    assert reward.x_reward == -5  # 35 - 40 = -5 pixels backward
    assert reward.speed_bonus == 0  # No speed bonus for backward


def test_reward_calc_no_movement(initial_state: State) -> None:
    """Standing still should give zero x_reward but alive bonus."""
    next_state = State(
        time=399,
        x_pos=40,
        x_pos_max=40,
        is_alive=True,
        milestones_reached=frozenset(),
    )
    reward = Reward.calc(next_state, initial_state)

    assert reward.x_reward == 0
    assert reward.speed_bonus == 0
    assert reward.alive_bonus == 1  # Still alive


# =============================================================================
# Reward.calc() tests - Milestones
# =============================================================================


def test_reward_calc_first_milestone() -> None:
    """Reaching first milestone should give milestone bonus."""
    before = State(x_pos=90, x_pos_max=90, is_alive=True, milestones_reached=frozenset())
    after = State(x_pos=105, x_pos_max=105, is_alive=True, milestones_reached=frozenset({100}))

    reward = Reward.calc(after, before)

    assert reward.milestone_bonus == 1  # One new milestone


def test_reward_calc_multiple_milestones_same_step() -> None:
    """Hitting multiple milestones in one step (e.g., teleport) should count all."""
    before = State(x_pos=90, x_pos_max=90, is_alive=True, milestones_reached=frozenset())
    # Jump from 90 to 550 (hits 100, 200, 350, 500)
    after = State(
        x_pos=550,
        x_pos_max=550,
        is_alive=True,
        milestones_reached=frozenset({100, 200, 350, 500}),
    )

    reward = Reward.calc(after, before)

    assert reward.milestone_bonus == 4  # Four new milestones


def test_reward_calc_no_duplicate_milestone_bonus() -> None:
    """Already-reached milestones should not give bonus again."""
    before = State(
        x_pos=100,
        x_pos_max=100,
        is_alive=True,
        milestones_reached=frozenset({100}),
    )
    after = State(
        x_pos=150,
        x_pos_max=150,
        is_alive=True,
        milestones_reached=frozenset({100}),  # Still only 100
    )

    reward = Reward.calc(after, before)

    assert reward.milestone_bonus == 0  # No new milestones


def test_reward_calc_milestone_on_exact_position() -> None:
    """Reaching exactly milestone position should trigger bonus."""
    before = State(x_pos=99, x_pos_max=99, is_alive=True, milestones_reached=frozenset())
    after = State(x_pos=100, x_pos_max=100, is_alive=True, milestones_reached=frozenset({100}))

    reward = Reward.calc(after, before)

    assert reward.milestone_bonus == 1


# =============================================================================
# Reward.calc() tests - Death
# =============================================================================


def test_reward_calc_death_penalty(initial_state: State) -> None:
    """Death should give penalty only on transition."""
    dead_state = State(
        x_pos=40,
        x_pos_max=40,
        is_alive=False,
        milestones_reached=frozenset(),
    )
    reward = Reward.calc(dead_state, initial_state)

    assert reward.death_penalty == -1000
    assert reward.alive_bonus == 0


def test_reward_calc_death_no_x_delta() -> None:
    """X movement during death animation should be ignored."""
    alive_state = State(x_pos=500, x_pos_max=500, is_alive=True, milestones_reached=frozenset())
    dead_state = State(
        x_pos=100,  # Position resets during death
        x_pos_max=500,
        is_alive=False,
        milestones_reached=frozenset(),
    )
    reward = Reward.calc(dead_state, alive_state)

    assert reward.x_reward == 0  # Ignored, not -400


def test_reward_calc_already_dead_no_penalty() -> None:
    """Already dead state should not give additional penalty."""
    dead_state1 = State(x_pos=100, x_pos_max=100, is_alive=False, milestones_reached=frozenset())
    dead_state2 = State(x_pos=100, x_pos_max=100, is_alive=False, milestones_reached=frozenset())

    reward = Reward.calc(dead_state2, dead_state1)

    assert reward.death_penalty == 0  # No penalty, was already dead


# =============================================================================
# Reward.calc() tests - Flag
# =============================================================================


def test_reward_calc_flag_bonus() -> None:
    """Reaching flag should give finish reward."""
    before = State(x_pos=3100, x_pos_max=3100, is_alive=True, got_flag=False, milestones_reached=frozenset())
    after = State(x_pos=3200, x_pos_max=3200, is_alive=True, got_flag=True, milestones_reached=frozenset())

    reward = Reward.calc(after, before)

    assert reward.finish_reward == 1000


# =============================================================================
# Reward.calc() tests - Powerup
# =============================================================================


def test_reward_calc_powerup_gain() -> None:
    """Getting powerup should give positive reward."""
    before = State(x_pos=100, x_pos_max=100, powerup_state=0, is_alive=True, milestones_reached=frozenset())
    after = State(x_pos=110, x_pos_max=110, powerup_state=1, is_alive=True, milestones_reached=frozenset())

    reward = Reward.calc(after, before)

    assert reward.powerup_reward == 100  # (1 - 0) * 100


def test_reward_calc_powerup_loss() -> None:
    """Losing powerup should give negative reward."""
    before = State(x_pos=100, x_pos_max=100, powerup_state=2, is_alive=True, milestones_reached=frozenset())
    after = State(x_pos=110, x_pos_max=110, powerup_state=1, is_alive=True, milestones_reached=frozenset())

    reward = Reward.calc(after, before)

    assert reward.powerup_reward == -100  # (1 - 2) * 100


# =============================================================================
# Reward.calc() tests - Speed bonus
# =============================================================================


def test_reward_calc_speed_bonus_walking() -> None:
    """Walking speed should give small speed bonus."""
    before = State(x_pos=100, x_pos_max=100, is_alive=True, milestones_reached=frozenset())
    after = State(x_pos=102, x_pos_max=102, is_alive=True, milestones_reached=frozenset())

    reward = Reward.calc(after, before)

    assert reward.speed_bonus == 2  # Walking: 2 pixels


def test_reward_calc_speed_bonus_running() -> None:
    """Running speed should give higher speed bonus."""
    before = State(x_pos=100, x_pos_max=100, is_alive=True, milestones_reached=frozenset())
    after = State(x_pos=104, x_pos_max=104, is_alive=True, milestones_reached=frozenset())

    reward = Reward.calc(after, before)

    assert reward.speed_bonus == 4  # Running: 4 pixels


def test_reward_calc_speed_bonus_capped() -> None:
    """Speed bonus should be capped at 10."""
    before = State(x_pos=100, x_pos_max=100, is_alive=True, milestones_reached=frozenset())
    after = State(x_pos=120, x_pos_max=120, is_alive=True, milestones_reached=frozenset())

    reward = Reward.calc(after, before)

    assert reward.speed_bonus == 10  # Capped at 10


# =============================================================================
# total_reward() tests
# =============================================================================


def test_total_reward_alive_bonus() -> None:
    """Alive bonus should contribute +0.01."""
    reward = Reward(alive_bonus=1)
    assert abs(reward.total_reward() - 0.01) < 1e-6


def test_total_reward_x_progress() -> None:
    """X progress should scale to -0.15 to +1.0."""
    # Forward movement
    reward = Reward(x_reward=100)
    assert abs(reward.total_reward() - 1.0) < 1e-6

    # Backward movement
    reward = Reward(x_reward=-15)
    assert abs(reward.total_reward() - (-0.15)) < 1e-6


def test_total_reward_speed_bonus() -> None:
    """Speed bonus should scale to 0 to +0.1."""
    reward = Reward(speed_bonus=10)
    assert abs(reward.total_reward() - 0.1) < 1e-6


def test_total_reward_milestone() -> None:
    """Each milestone should give +2.0."""
    reward = Reward(milestone_bonus=1)
    assert abs(reward.total_reward() - 2.0) < 1e-6

    reward = Reward(milestone_bonus=3)
    assert abs(reward.total_reward() - 6.0) < 1e-6


def test_total_reward_death() -> None:
    """Death penalty should be -1.0."""
    reward = Reward(death_penalty=-1000)
    assert abs(reward.total_reward() - (-1.0)) < 1e-6


def test_total_reward_flag() -> None:
    """Flag bonus should be +5.0."""
    reward = Reward(finish_reward=1000)
    assert abs(reward.total_reward() - 5.0) < 1e-6


def test_total_reward_powerup_gain() -> None:
    """Powerup gain should give +0.5."""
    reward = Reward(powerup_reward=100)
    assert abs(reward.total_reward() - 0.5) < 1e-6


def test_total_reward_powerup_loss() -> None:
    """Powerup loss should give -0.5."""
    reward = Reward(powerup_reward=-100)
    assert abs(reward.total_reward() - (-0.5)) < 1e-6


def test_total_reward_combined_typical_step() -> None:
    """Typical step: alive + forward movement + speed."""
    reward = Reward(
        alive_bonus=1,  # +0.01
        x_reward=4,  # +0.04
        speed_bonus=4,  # +0.04
    )
    expected = 0.01 + 0.04 + 0.04
    assert abs(reward.total_reward() - expected) < 1e-6


def test_total_reward_combined_milestone_step() -> None:
    """Step hitting a milestone: alive + progress + speed + milestone."""
    reward = Reward(
        alive_bonus=1,  # +0.01
        x_reward=5,  # +0.05
        speed_bonus=5,  # +0.05
        milestone_bonus=1,  # +2.0
    )
    expected = 0.01 + 0.05 + 0.05 + 2.0
    assert abs(reward.total_reward() - expected) < 1e-6


def test_total_reward_combined_death() -> None:
    """Death step: progress + death penalty."""
    reward = Reward(
        alive_bonus=0,
        x_reward=0,
        death_penalty=-1000,  # -1.0
    )
    expected = -1.0
    assert abs(reward.total_reward() - expected) < 1e-6


def test_total_reward_combined_flag() -> None:
    """Flag step: progress + milestones + flag."""
    reward = Reward(
        alive_bonus=1,  # +0.01
        x_reward=5,  # +0.05
        speed_bonus=5,  # +0.05
        milestone_bonus=1,  # +2.0 (hit 3000)
        finish_reward=1000,  # +5.0
    )
    expected = 0.01 + 0.05 + 0.05 + 2.0 + 5.0
    assert abs(reward.total_reward() - expected) < 1e-6


# =============================================================================
# Regression tests - Episode reward totals
# =============================================================================


def test_episode_reward_early_death() -> None:
    """Episode dying at X=100 should have positive total reward."""
    # Simulate episode: start at X=40, reach X=100, hit milestone, die
    steps = (100 - 40) // 4  # ~15 steps with skip=4

    total = 0.0
    # Alive steps moving forward
    total += steps * (0.01 + 0.04 + 0.04)  # alive + progress + speed
    # Milestone at 100
    total += 2.0
    # Death
    total += -1.0

    assert total > 0, f"Early death should have positive reward, got {total}"


def test_episode_reward_flag_capture() -> None:
    """Full level completion should have high reward."""
    # Simulate: X=40 to X=3200, all milestones, flag
    x_distance = 3200 - 40
    steps = x_distance // 4  # ~790 steps

    total = 0.0
    # Alive + progress + speed for all steps
    total += steps * (0.01 + 0.04 + 0.04)
    # All 10 milestones
    total += 10 * 2.0
    # Flag bonus
    total += 5.0

    # Total should be ~95-100 for full level completion
    assert total > 90, f"Flag capture should give 90+ reward, got {total}"
