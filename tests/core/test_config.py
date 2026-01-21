"""Tests for training configuration.

These tests verify the epsilon floor calculation and configuration defaults.

The formula is: epsilon_end = epsilon_base ** (1 + (worker_id + 1) / num_workers)

With epsilon_base=0.15 and 4 workers:
- Worker 0: 0.15^1.25 ≈ 8.7%
- Worker 1: 0.15^1.50 ≈ 5.8%
- Worker 2: 0.15^1.75 ≈ 3.9%
- Worker 3: 0.15^2.00 ≈ 2.3%
"""

from __future__ import annotations

import pytest

from mario_rl.core.config import TrainingConfig


def test_default_epsilon_base() -> None:
    """Default epsilon_base should be 0.15 for low floor."""
    config = TrainingConfig()
    assert config.epsilon_base == 0.15


def test_epsilon_floor_with_4_workers() -> None:
    """Epsilon floor should range from ~2% to ~9% with 4 workers."""
    config = TrainingConfig(num_workers=4)
    worker_configs = config.create_worker_configs()

    # Get epsilon_end for each worker
    epsilons = [wc.exploration.epsilon_end for wc in worker_configs]

    # Worker 0 should have highest epsilon (most exploration)
    # Worker 3 should have lowest epsilon (most exploitation)
    assert epsilons[0] > epsilons[1] > epsilons[2] > epsilons[3]

    # All should be under 10%
    assert all(eps < 0.10 for eps in epsilons)

    # Most exploitative should be around 2%
    assert epsilons[3] < 0.03

    # Most explorative should be around 9%
    assert 0.05 < epsilons[0] < 0.12


def test_epsilon_floor_calculation_formula() -> None:
    """Verify exact epsilon calculation formula."""
    config = TrainingConfig(num_workers=4)
    worker_configs = config.create_worker_configs()

    base = config.epsilon_base
    n = config.num_workers

    for i, wc in enumerate(worker_configs):
        expected = base ** (1 + (i + 1) / n)
        actual = wc.exploration.epsilon_end
        assert abs(actual - expected) < 0.0001, f"Worker {i}: expected {expected}, got {actual}"


@pytest.mark.parametrize(
    "eps_base,num_workers,expected_range",
    [
        (0.15, 4, (0.02, 0.10)),  # Default: 2-9%
        (0.15, 8, (0.01, 0.10)),  # More workers: tighter low end
        (0.10, 4, (0.01, 0.06)),  # Lower base: lower floors
        (0.20, 4, (0.04, 0.13)),  # Higher base: higher floors
    ],
)
def test_epsilon_range_with_different_configs(
    eps_base: float,
    num_workers: int,
    expected_range: tuple[float, float],
) -> None:
    """Epsilon range should vary with base and num_workers."""
    config = TrainingConfig(num_workers=num_workers, epsilon_base=eps_base)
    worker_configs = config.create_worker_configs()

    epsilons = [wc.exploration.epsilon_end for wc in worker_configs]

    min_eps, max_eps = expected_range
    assert epsilons[-1] >= min_eps * 0.5  # Allow some margin
    assert epsilons[-1] <= max_eps
    assert epsilons[0] <= max_eps * 1.5  # Allow some margin


def test_old_epsilon_base_was_too_high() -> None:
    """Document that old eps_base=0.4 gave too high floors.

    This test documents the problem we fixed: with eps_base=0.4,
    even the most exploitative worker had ~16% random actions.
    """
    # Old config with eps_base=0.4
    old_base = 0.4
    num_workers = 4

    # Calculate old epsilon floors
    old_epsilons = [old_base ** (1 + (i + 1) / num_workers) for i in range(num_workers)]

    # The old most exploitative worker had ~16% epsilon - too high!
    assert old_epsilons[3] > 0.15  # Was 0.16

    # The old most explorative had ~32% - way too high!
    assert old_epsilons[0] > 0.30  # Was 0.315

    # New config should be much better
    new_config = TrainingConfig(num_workers=4)
    new_epsilons = [wc.exploration.epsilon_end for wc in new_config.create_worker_configs()]

    # New floors are much lower
    assert new_epsilons[3] < 0.03  # ~2.3%
    assert new_epsilons[0] < 0.10  # ~8.7%
