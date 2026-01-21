"""Shared Memory Heartbeat for worker monitoring.

Simple mmap-based heartbeat system for detecting stale workers.
Each worker periodically updates its timestamp; the monitor
checks for stale workers that haven't updated recently.
"""

import mmap
import time
import struct
from typing import Any
from pathlib import Path
from dataclasses import field
from dataclasses import dataclass


@dataclass
class SharedHeartbeat:
    """Shared memory heartbeat for worker monitoring.

    Uses mmap'd file with one 8-byte double per worker for timestamp.
    Simple and deadlock-free.
    """

    num_workers: int
    shm_dir: Path | None = None
    create: bool = True

    # Internal state
    _mmap: mmap.mmap | None = field(init=False, default=None, repr=False)
    _fd: Any = field(init=False, default=None, repr=False)
    _shm_path: Path = field(init=False, repr=False)
    _buffer_size: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        """Initialize shared memory."""
        # 8 bytes per worker (double timestamp)
        self._buffer_size = self.num_workers * 8

        # Determine shm path
        if self.shm_dir is None:
            if Path("/dev/shm").exists():
                self.shm_dir = Path("/dev/shm")
            else:
                self.shm_dir = Path("/tmp")
        else:
            self.shm_dir = Path(self.shm_dir)

        self._shm_path = self.shm_dir / "heartbeats.shm"

        if self.create:
            self._create()
        else:
            self._attach()

    def _create(self) -> None:
        """Create new shared memory file."""
        if self.shm_dir is None:
            raise ValueError("shm_dir must be set when creating shared memory")
        self.shm_dir.mkdir(parents=True, exist_ok=True)

        # Create file with zeros
        with open(self._shm_path, "wb") as f:
            f.write(b"\x00" * self._buffer_size)

        self._open_mmap()

    def _attach(self) -> None:
        """Attach to existing shared memory file."""
        if not self._shm_path.exists():
            raise FileNotFoundError(f"Heartbeat file not found: {self._shm_path}")

        self._open_mmap()

    def _open_mmap(self) -> None:
        """Open mmap."""
        self._fd = open(self._shm_path, "r+b")
        self._mmap = mmap.mmap(self._fd.fileno(), self._buffer_size)

    def update(self, worker_id: int) -> None:
        """Update heartbeat timestamp for worker.

        Args:
            worker_id: Worker to update
        """
        if self._mmap is None:
            return

        if worker_id >= self.num_workers:
            return

        now = time.time()
        offset = worker_id * 8
        self._mmap[offset : offset + 8] = struct.pack("d", now)
        self._mmap.flush()

    def _set_time(self, worker_id: int, timestamp: float) -> None:
        """Set specific timestamp (for testing).

        Args:
            worker_id: Worker to update
            timestamp: Timestamp to set
        """
        if self._mmap is None:
            return

        if worker_id >= self.num_workers:
            return

        offset = worker_id * 8
        self._mmap[offset : offset + 8] = struct.pack("d", timestamp)
        self._mmap.flush()

    def get(self, worker_id: int) -> float:
        """Get heartbeat timestamp for worker.

        Args:
            worker_id: Worker to check

        Returns:
            Timestamp of last heartbeat (0 if never updated)
        """
        if self._mmap is None:
            return 0.0

        if worker_id >= self.num_workers:
            return 0.0

        offset = worker_id * 8
        return struct.unpack("d", self._mmap[offset : offset + 8])[0]

    def all_heartbeats(self) -> list[float]:
        """Get all worker heartbeats.

        Returns:
            List of timestamps for each worker
        """
        return [self.get(i) for i in range(self.num_workers)]

    def is_stale(self, worker_id: int, timeout: float) -> bool:
        """Check if worker heartbeat is stale.

        Args:
            worker_id: Worker to check
            timeout: Seconds since last heartbeat to consider stale

        Returns:
            True if worker is stale
        """
        last_hb = self.get(worker_id)
        if last_hb == 0.0:
            return True  # Never updated

        return time.time() - last_hb > timeout

    def stale_workers(self, timeout: float) -> list[int]:
        """Get list of stale workers.

        Args:
            timeout: Seconds since last heartbeat to consider stale

        Returns:
            List of stale worker IDs
        """
        return [i for i in range(self.num_workers) if self.is_stale(i, timeout)]

    def close(self) -> None:
        """Close mmap and file descriptor."""
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None

        if self._fd is not None:
            self._fd.close()
            self._fd = None

    def unlink(self) -> None:
        """Close and remove shared memory file."""
        self.close()
        if self._shm_path.exists():
            self._shm_path.unlink()

    def __del__(self) -> None:
        """Cleanup on garbage collection."""
        self.close()
