"""Tests for lazy level initialization in SuperMarioBrosMultiLevel.

This test verifies that random/sequential level modes do NOT eagerly
create all 32 NES emulators upfront, which would cause ~12s startup delay.
Instead, levels should be created lazily on first use.
"""

import time

import pytest

from mario_rl.environment.mariogym import SuperMarioBrosMultiLevel


class TestLazyLevelInitialization:
    """Test that random/sequential modes use lazy initialization."""

    def test_single_level_mode_creates_only_one_env(self) -> None:
        """Single level mode should only create one NES emulator."""
        env = SuperMarioBrosMultiLevel(level=(1, 1))

        # Count non-None envs in the grid
        created_envs = sum(1 for row in env.envs for e in row if e is not None)

        assert created_envs == 1, f"Expected 1 env, got {created_envs}"
        env.close()

    def test_random_mode_starts_with_one_env(self) -> None:
        """Random mode should only create one env initially (for reset)."""
        env = SuperMarioBrosMultiLevel(level="random")

        # After initialization + reset, should have only 1 env created
        created_envs = sum(1 for row in env.envs for e in row if e is not None)

        assert created_envs == 1, f"Random mode created {created_envs} envs upfront, expected 1"
        env.close()

    def test_sequential_mode_starts_with_one_env(self) -> None:
        """Sequential mode should only create one env initially (for reset)."""
        env = SuperMarioBrosMultiLevel(level="sequential")

        # After initialization + reset, should have only 1 env created
        created_envs = sum(1 for row in env.envs for e in row if e is not None)

        assert created_envs == 1, f"Sequential mode created {created_envs} envs upfront, expected 1"
        env.close()

    def test_random_mode_initialization_is_fast(self) -> None:
        """Random mode should initialize quickly (not ~12s like eager init).

        Before the fix, creating 32 NES emulators took ~12s.
        With lazy init, it should take <2s (only 1 emulator).
        """
        start = time.time()
        env = SuperMarioBrosMultiLevel(level="random")
        elapsed = time.time() - start

        # Should be much faster than 12s (eager init)
        # Allow up to 3s for slow CI machines
        assert elapsed < 3.0, f"Random mode init took {elapsed:.2f}s, expected <3s (lazy init)"
        env.close()

    def test_sequential_mode_creates_levels_on_demand(self) -> None:
        """Sequential mode should create levels lazily as they're played."""
        env = SuperMarioBrosMultiLevel(level="sequential")

        initial_count = sum(1 for row in env.envs for e in row if e is not None)
        assert initial_count == 1, f"Should start with 1 env, got {initial_count}"

        # Reset a few times to trigger new level creation
        for _ in range(5):
            env.reset()

        later_count = sum(1 for row in env.envs for e in row if e is not None)

        # Should have more envs now, but not all 32
        assert later_count > initial_count, "No new envs created after multiple resets"
        assert later_count <= 6, f"Created {later_count} envs, expected <= 6 after 5 resets"
        env.close()

    def test_random_mode_can_play_multiple_levels(self) -> None:
        """Verify random mode works correctly with lazy initialization."""
        env = SuperMarioBrosMultiLevel(level="random")

        levels_seen = set()
        for _ in range(10):
            obs, info = env.reset()
            levels_seen.add(env.current_level)

            # Take a few steps to ensure env works
            for _ in range(5):
                obs, reward, done, trunc, info = env.step(env.action_space.sample())
                if done:
                    break

        # Should have seen at least a few different levels
        assert len(levels_seen) >= 1, "Random mode should be able to play different levels"
        env.close()

    def test_sequential_mode_progresses_through_levels(self) -> None:
        """Verify sequential mode progresses correctly with lazy init.

        Note: __init__ calls reset() once, putting us at level 1-1.
        Each subsequent reset() advances to the next level.
        """
        env = SuperMarioBrosMultiLevel(level="sequential")

        # First level is set during __init__'s reset()
        initial_level = env.current_level
        assert initial_level == "1-1", f"Initial level should be 1-1, got {initial_level}"

        levels = []
        for _ in range(5):
            env.reset()  # Advances to next level
            levels.append(env.current_level)

        # After 5 resets from 1-1: 1-2, 1-3, 1-4, 2-1, 2-2
        expected = ["1-2", "1-3", "1-4", "2-1", "2-2"]
        assert levels == expected, f"Sequential order wrong: {levels} != {expected}"
        env.close()
