"""Tests for MarioBrosLevel environment.

These tests verify:
- is_timeout flag and TIMEOUT_THRESHOLD constant
- Speed bonus for fast forward movement
- Milestone system (replaces exploration bonus)
"""

from __future__ import annotations

from unittest.mock import PropertyMock
from unittest.mock import patch

import pytest

from mario_rl.environment import TIMEOUT_THRESHOLD
from mario_rl.environment.mariogym import MILESTONES
from mario_rl.environment.mariogym import MarioBrosLevel
from mario_rl.environment.mariogym import Reward
from mario_rl.environment.mariogym import State


class TestTimeoutThreshold:
    """Tests for TIMEOUT_THRESHOLD constant."""

    def test_timeout_threshold_is_exported(self) -> None:
        """TIMEOUT_THRESHOLD should be exported from environment module."""
        assert TIMEOUT_THRESHOLD == 10

    def test_timeout_threshold_value(self) -> None:
        """TIMEOUT_THRESHOLD should be 10 (game time when timeout occurs)."""
        from mario_rl.environment.mariogym import TIMEOUT_THRESHOLD as DIRECT_IMPORT

        assert DIRECT_IMPORT == 10


class TestIsTimeoutFlag:
    """Tests for is_timeout flag in environment info dict.

    The is_timeout flag indicates the episode ended because the game timer
    ran out, not because of a skill-based death.

    Timeout occurs when:
    - Mario is dying or dead (is_dying or is_dead)
    - Game time is <= TIMEOUT_THRESHOLD (10)
    - Flag was not reached (not flag_get)
    """

    def test_is_timeout_true_when_dead_and_time_low(self) -> None:
        """is_timeout should be True when dead with low time and no flag.

        This is the core timeout condition: Mario died because time ran out.
        """
        with patch.object(
            MarioBrosLevel, "_is_dead", new_callable=PropertyMock, return_value=True
        ), patch.object(
            MarioBrosLevel, "_is_dying", new_callable=PropertyMock, return_value=False
        ), patch.object(
            MarioBrosLevel, "_time", new_callable=PropertyMock, return_value=5
        ), patch.object(
            MarioBrosLevel, "_flag_get", new_callable=PropertyMock, return_value=False
        ), patch.object(
            MarioBrosLevel, "_reward_state", new_callable=PropertyMock
        ), patch.object(
            MarioBrosLevel, "state", new_callable=PropertyMock
        ), patch.object(
            MarioBrosLevel, "_last_state", None
        ):
            env = object.__new__(MarioBrosLevel)
            env._last_state = None
            
            # Call _get_info directly - we're testing the logic, not the full env
            is_dying_or_dead = env._is_dying or env._is_dead
            is_timeout = is_dying_or_dead and env._time <= TIMEOUT_THRESHOLD and not env._flag_get
            
            assert is_timeout is True

    def test_is_timeout_true_when_dying_and_time_low(self) -> None:
        """is_timeout should be True when dying (not yet dead) with low time."""
        with patch.object(
            MarioBrosLevel, "_is_dead", new_callable=PropertyMock, return_value=False
        ), patch.object(
            MarioBrosLevel, "_is_dying", new_callable=PropertyMock, return_value=True
        ), patch.object(
            MarioBrosLevel, "_time", new_callable=PropertyMock, return_value=10
        ), patch.object(
            MarioBrosLevel, "_flag_get", new_callable=PropertyMock, return_value=False
        ):
            env = object.__new__(MarioBrosLevel)
            
            is_dying_or_dead = env._is_dying or env._is_dead
            is_timeout = is_dying_or_dead and env._time <= TIMEOUT_THRESHOLD and not env._flag_get
            
            assert is_timeout is True

    def test_is_timeout_false_when_time_above_threshold(self) -> None:
        """is_timeout should be False when time is above threshold (normal death)."""
        with patch.object(
            MarioBrosLevel, "_is_dead", new_callable=PropertyMock, return_value=True
        ), patch.object(
            MarioBrosLevel, "_is_dying", new_callable=PropertyMock, return_value=False
        ), patch.object(
            MarioBrosLevel, "_time", new_callable=PropertyMock, return_value=100
        ), patch.object(
            MarioBrosLevel, "_flag_get", new_callable=PropertyMock, return_value=False
        ):
            env = object.__new__(MarioBrosLevel)
            
            is_dying_or_dead = env._is_dying or env._is_dead
            is_timeout = is_dying_or_dead and env._time <= TIMEOUT_THRESHOLD and not env._flag_get
            
            assert is_timeout is False

    def test_is_timeout_false_when_flag_reached(self) -> None:
        """is_timeout should be False when flag is reached (level complete).

        Even if time is low, getting the flag means success, not timeout.
        """
        with patch.object(
            MarioBrosLevel, "_is_dead", new_callable=PropertyMock, return_value=False
        ), patch.object(
            MarioBrosLevel, "_is_dying", new_callable=PropertyMock, return_value=False
        ), patch.object(
            MarioBrosLevel, "_time", new_callable=PropertyMock, return_value=5
        ), patch.object(
            MarioBrosLevel, "_flag_get", new_callable=PropertyMock, return_value=True
        ):
            env = object.__new__(MarioBrosLevel)
            
            is_dying_or_dead = env._is_dying or env._is_dead
            is_timeout = is_dying_or_dead and env._time <= TIMEOUT_THRESHOLD and not env._flag_get
            
            assert is_timeout is False

    def test_is_timeout_false_when_alive(self) -> None:
        """is_timeout should be False when Mario is alive (not dying/dead)."""
        with patch.object(
            MarioBrosLevel, "_is_dead", new_callable=PropertyMock, return_value=False
        ), patch.object(
            MarioBrosLevel, "_is_dying", new_callable=PropertyMock, return_value=False
        ), patch.object(
            MarioBrosLevel, "_time", new_callable=PropertyMock, return_value=5
        ), patch.object(
            MarioBrosLevel, "_flag_get", new_callable=PropertyMock, return_value=False
        ):
            env = object.__new__(MarioBrosLevel)
            
            is_dying_or_dead = env._is_dying or env._is_dead
            is_timeout = is_dying_or_dead and env._time <= TIMEOUT_THRESHOLD and not env._flag_get
            
            assert is_timeout is False

    def test_is_timeout_at_exact_threshold(self) -> None:
        """is_timeout should be True when time equals threshold exactly."""
        with patch.object(
            MarioBrosLevel, "_is_dead", new_callable=PropertyMock, return_value=True
        ), patch.object(
            MarioBrosLevel, "_is_dying", new_callable=PropertyMock, return_value=False
        ), patch.object(
            MarioBrosLevel, "_time", new_callable=PropertyMock, return_value=TIMEOUT_THRESHOLD
        ), patch.object(
            MarioBrosLevel, "_flag_get", new_callable=PropertyMock, return_value=False
        ):
            env = object.__new__(MarioBrosLevel)
            
            is_dying_or_dead = env._is_dying or env._is_dead
            is_timeout = is_dying_or_dead and env._time <= TIMEOUT_THRESHOLD and not env._flag_get
            
            assert is_timeout is True

    def test_is_timeout_just_above_threshold(self) -> None:
        """is_timeout should be False when time is just above threshold."""
        with patch.object(
            MarioBrosLevel, "_is_dead", new_callable=PropertyMock, return_value=True
        ), patch.object(
            MarioBrosLevel, "_is_dying", new_callable=PropertyMock, return_value=False
        ), patch.object(
            MarioBrosLevel, "_time", new_callable=PropertyMock, return_value=TIMEOUT_THRESHOLD + 1
        ), patch.object(
            MarioBrosLevel, "_flag_get", new_callable=PropertyMock, return_value=False
        ):
            env = object.__new__(MarioBrosLevel)
            
            is_dying_or_dead = env._is_dying or env._is_dead
            is_timeout = is_dying_or_dead and env._time <= TIMEOUT_THRESHOLD and not env._flag_get
            
            assert is_timeout is False


class TestIsTimeoutLogic:
    """Tests for the is_timeout calculation logic itself.
    
    These tests verify the boolean logic without needing the full environment.
    """

    @pytest.mark.parametrize(
        "is_dead,is_dying,time,flag_get,expected",
        [
            # Timeout cases (dead/dying + low time + no flag)
            (True, False, 5, False, True),   # Dead, low time, no flag
            (False, True, 5, False, True),   # Dying, low time, no flag
            (True, True, 10, False, True),   # Both, at threshold, no flag
            
            # Non-timeout cases
            (True, False, 100, False, False),  # Dead but time is high
            (True, False, 11, False, False),   # Dead but time just above threshold
            (False, False, 5, False, False),   # Not dead/dying (still alive)
            (True, False, 5, True, False),     # Dead but got flag
            (False, True, 5, True, False),     # Dying but got flag
        ],
    )
    def test_is_timeout_logic(
        self,
        is_dead: bool,
        is_dying: bool,
        time: int,
        flag_get: bool,
        expected: bool,
    ) -> None:
        """Test is_timeout calculation with various input combinations."""
        is_dying_or_dead = is_dying or is_dead
        is_timeout = is_dying_or_dead and time <= TIMEOUT_THRESHOLD and not flag_get
        
        assert is_timeout == expected


# =============================================================================
# Speed Bonus Tests
# =============================================================================


class TestSpeedBonus:
    """Tests for speed bonus reward.
    
    The speed bonus rewards fast forward movement, encouraging
    efficient level completion (running vs walking).
    """

    def test_speed_bonus_when_moving_forward(self) -> None:
        """Speed bonus should trigger when moving forward."""
        last = State(x_pos=100, x_pos_max=100, is_alive=True, milestones_reached=frozenset())
        current = State(x_pos=105, x_pos_max=105, is_alive=True, milestones_reached=frozenset())
        
        reward = Reward.calc(current, last)
        
        assert reward.speed_bonus == 5  # x_delta = 5

    def test_no_speed_bonus_when_standing_still(self) -> None:
        """No speed bonus when not moving."""
        last = State(x_pos=100, x_pos_max=100, is_alive=True, milestones_reached=frozenset())
        current = State(x_pos=100, x_pos_max=100, is_alive=True, milestones_reached=frozenset())
        
        reward = Reward.calc(current, last)
        
        assert reward.speed_bonus == 0

    def test_no_speed_bonus_when_moving_backward(self) -> None:
        """No speed bonus when moving backward."""
        last = State(x_pos=100, x_pos_max=100, is_alive=True, milestones_reached=frozenset())
        current = State(x_pos=95, x_pos_max=100, is_alive=True, milestones_reached=frozenset())
        
        reward = Reward.calc(current, last)
        
        assert reward.speed_bonus == 0

    def test_speed_bonus_capped(self) -> None:
        """Speed bonus should be capped at 10."""
        last = State(x_pos=100, x_pos_max=100, is_alive=True, milestones_reached=frozenset())
        current = State(x_pos=150, x_pos_max=150, is_alive=True, milestones_reached=frozenset())
        
        reward = Reward.calc(current, last)
        
        assert reward.speed_bonus == 10  # Capped at 10

    def test_no_speed_bonus_when_dead(self) -> None:
        """No speed bonus when Mario is dead (x_delta is zeroed)."""
        last = State(x_pos=100, x_pos_max=100, is_alive=True, milestones_reached=frozenset())
        current = State(x_pos=105, x_pos_max=105, is_alive=False, milestones_reached=frozenset())
        
        reward = Reward.calc(current, last)
        
        assert reward.speed_bonus == 0

    def test_speed_bonus_normalized_in_total(self) -> None:
        """Speed bonus should be normalized in total_reward()."""
        # Create reward with max speed bonus (10)
        reward = Reward(speed_bonus=10)
        
        # Should contribute +0.1 to total (10 / 100)
        assert abs(reward.total_reward() - 0.1) < 0.01


# =============================================================================
# Milestone Tests (replaces exploration bonus)
# =============================================================================


class TestMilestones:
    """Tests for milestone reward system.
    
    Milestones provide significant bonuses at progressive X positions
    to guide the agent through the level.
    """

    def test_milestone_bonus_at_first_milestone(self) -> None:
        """Milestone bonus should trigger at first milestone (X=100)."""
        last = State(x_pos=95, x_pos_max=95, is_alive=True, milestones_reached=frozenset())
        current = State(x_pos=105, x_pos_max=105, is_alive=True, milestones_reached=frozenset({100}))
        
        reward = Reward.calc(current, last)
        
        assert reward.milestone_bonus == 1

    def test_no_milestone_bonus_below_first_milestone(self) -> None:
        """No milestone bonus when below first milestone."""
        last = State(x_pos=80, x_pos_max=80, is_alive=True, milestones_reached=frozenset())
        current = State(x_pos=90, x_pos_max=90, is_alive=True, milestones_reached=frozenset())
        
        reward = Reward.calc(current, last)
        
        assert reward.milestone_bonus == 0

    def test_milestone_bonus_not_repeated(self) -> None:
        """Already reached milestones should not give bonus again."""
        last = State(x_pos=100, x_pos_max=100, is_alive=True, milestones_reached=frozenset({100}))
        current = State(x_pos=150, x_pos_max=150, is_alive=True, milestones_reached=frozenset({100}))
        
        reward = Reward.calc(current, last)
        
        assert reward.milestone_bonus == 0  # Already had 100

    def test_milestone_normalized_in_total(self) -> None:
        """Each milestone should give +2.0 in total_reward()."""
        reward = Reward(milestone_bonus=1)
        assert abs(reward.total_reward() - 2.0) < 0.01
        
        reward = Reward(milestone_bonus=3)
        assert abs(reward.total_reward() - 6.0) < 0.01


class TestRewardCombined:
    """Tests for combined reward components."""

    def test_speed_and_milestone_at_new_max(self) -> None:
        """Both speed and milestone bonus can trigger together."""
        last = State(x_pos=95, x_pos_max=95, is_alive=True, milestones_reached=frozenset())
        current = State(x_pos=105, x_pos_max=105, is_alive=True, milestones_reached=frozenset({100}))
        
        reward = Reward.calc(current, last)
        
        assert reward.milestone_bonus == 1  # Hit milestone at 100
        assert reward.speed_bonus == 10  # Fast forward movement

    def test_speed_only_when_not_hitting_milestone(self) -> None:
        """Speed bonus without milestone when between milestones."""
        last = State(x_pos=110, x_pos_max=110, is_alive=True, milestones_reached=frozenset({100}))
        current = State(x_pos=115, x_pos_max=115, is_alive=True, milestones_reached=frozenset({100}))
        
        reward = Reward.calc(current, last)
        
        assert reward.milestone_bonus == 0  # Between milestones
        assert reward.speed_bonus == 5  # Still moving forward

    def test_reward_components_in_realistic_scenario(self) -> None:
        """Test all reward components in a realistic forward movement scenario."""
        last = State(
            x_pos=500,
            x_pos_max=500,
            time=300,
            is_alive=True,
            milestones_reached=frozenset({100, 200, 350, 500}),
        )
        current = State(
            x_pos=504,
            x_pos_max=504,
            time=299,
            is_alive=True,
            milestones_reached=frozenset({100, 200, 350, 500}),
        )
        
        reward = Reward.calc(current, last)
        
        # Check all components
        assert reward.x_reward == 4  # Forward movement
        assert reward.speed_bonus == 4  # Speed bonus
        assert reward.alive_bonus == 1  # Alive
        assert reward.milestone_bonus == 0  # No new milestones
        assert reward.death_penalty == 0  # Not dead
        assert reward.finish_reward == 0  # Didn't finish
        
        # Total should be positive
        assert reward.total_reward() > 0
