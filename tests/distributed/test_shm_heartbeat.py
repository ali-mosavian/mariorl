"""Tests for SharedHeartbeat behavior.

These tests verify the SharedHeartbeat interface:
- Workers can send heartbeats
- Monitor can check heartbeats
- Stale workers are detected
"""

import time
from pathlib import Path

import pytest


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def shm_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for shared memory files."""
    shm = tmp_path / "shm"
    shm.mkdir()
    return shm


@pytest.fixture
def heartbeat(shm_dir: Path):
    """Create a SharedHeartbeat for testing."""
    from mario_rl.distributed.shm_heartbeat import SharedHeartbeat

    hb = SharedHeartbeat(num_workers=3, shm_dir=shm_dir)
    yield hb
    hb.close()


# =============================================================================
# Creation Tests
# =============================================================================


def test_heartbeat_creates_successfully(heartbeat) -> None:
    """Heartbeat should create without error."""
    assert heartbeat is not None


def test_heartbeat_creates_shm_file(heartbeat, shm_dir: Path) -> None:
    """Shared memory file should be created."""
    files = list(shm_dir.glob("*.shm"))
    assert len(files) == 1


# =============================================================================
# Update Tests
# =============================================================================


def test_update_worker_heartbeat(heartbeat) -> None:
    """Worker can update its heartbeat."""
    # Should not raise
    heartbeat.update(worker_id=0)


def test_update_multiple_workers(heartbeat) -> None:
    """Multiple workers can update heartbeats."""
    heartbeat.update(worker_id=0)
    heartbeat.update(worker_id=1)
    heartbeat.update(worker_id=2)


def test_get_heartbeat_time(heartbeat) -> None:
    """Should retrieve heartbeat time for worker."""
    before = time.time()
    heartbeat.update(worker_id=0)
    after = time.time()

    hb_time = heartbeat.get(worker_id=0)

    assert before <= hb_time <= after


# =============================================================================
# Stale Detection Tests
# =============================================================================


def test_is_stale_false_for_recent(heartbeat) -> None:
    """Recent heartbeat should not be stale."""
    heartbeat.update(worker_id=0)

    assert not heartbeat.is_stale(worker_id=0, timeout=5.0)


def test_is_stale_true_for_old(heartbeat) -> None:
    """Old heartbeat should be stale."""
    # Manually set old timestamp
    heartbeat._set_time(worker_id=0, timestamp=time.time() - 10.0)

    assert heartbeat.is_stale(worker_id=0, timeout=5.0)


def test_is_stale_true_for_never_updated(heartbeat) -> None:
    """Worker that never sent heartbeat should be stale."""
    assert heartbeat.is_stale(worker_id=0, timeout=5.0)


def test_stale_workers(heartbeat) -> None:
    """stale_workers should return list of stale workers."""
    # Worker 0: recent
    heartbeat.update(worker_id=0)

    # Worker 1: old
    heartbeat._set_time(worker_id=1, timestamp=time.time() - 10.0)

    # Worker 2: never updated (starts as 0)

    stale = heartbeat.stale_workers(timeout=5.0)

    assert 0 not in stale
    assert 1 in stale
    assert 2 in stale


# =============================================================================
# All Heartbeats Tests
# =============================================================================


def test_all_heartbeats(heartbeat) -> None:
    """all_heartbeats should return all heartbeats."""
    heartbeat.update(worker_id=0)
    heartbeat.update(worker_id=1)

    all_hb = heartbeat.all_heartbeats()

    assert len(all_hb) == 3
    assert all_hb[0] > 0  # Updated
    assert all_hb[1] > 0  # Updated
    assert all_hb[2] == 0.0  # Never updated


# =============================================================================
# Attach Tests
# =============================================================================


def test_attach_to_existing(shm_dir: Path) -> None:
    """Should attach to existing heartbeat."""
    from mario_rl.distributed.shm_heartbeat import SharedHeartbeat

    creator = SharedHeartbeat(num_workers=3, shm_dir=shm_dir, create=True)
    creator.update(worker_id=0)

    attacher = SharedHeartbeat(num_workers=3, shm_dir=shm_dir, create=False)

    # Should see the update
    assert attacher.get(worker_id=0) > 0

    creator.close()
    attacher.close()


# =============================================================================
# Cleanup Tests
# =============================================================================


def test_close_cleans_up(shm_dir: Path) -> None:
    """close should release resources."""
    from mario_rl.distributed.shm_heartbeat import SharedHeartbeat

    hb = SharedHeartbeat(num_workers=3, shm_dir=shm_dir)
    hb.close()

    # Double close should not raise
    hb.close()


def test_unlink_removes_file(shm_dir: Path) -> None:
    """unlink should remove shared memory file."""
    from mario_rl.distributed.shm_heartbeat import SharedHeartbeat

    hb = SharedHeartbeat(num_workers=3, shm_dir=shm_dir)
    hb.unlink()

    files = list(shm_dir.glob("*.shm"))
    assert len(files) == 0
