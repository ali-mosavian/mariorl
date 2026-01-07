"""
Shared Memory Gradient Exchange for Distributed DDQN.

This module provides a deadlock-free alternative to multiprocessing Queues
for exchanging gradients between workers and the learner.

Architecture:
    - Each worker has a dedicated SharedGradientBuffer
    - Workers write gradients directly to shared memory (no pipes/sockets)
    - Atomic flags signal when data is ready
    - Learner polls flags non-blockingly
    - Impossible to deadlock - no blocking I/O

Why this is better than Queues:
    - Queue.get() can block forever if sender dies mid-transfer
    - Shared memory writes are atomic (mmap)
    - Flags are checked without blocking
    - Worker crashes don't corrupt the communication channel
"""

from __future__ import annotations

import io
import os
import mmap
import struct
import ctypes
import multiprocessing as mp
from pathlib import Path
from typing import Any
from dataclasses import dataclass

import torch


# Header format for seqlock synchronization:
# seq (4 bytes) - sequence number (odd = writing, even = stable)
# ready (1 byte) - 0 = empty, 1 = data available
# size (4 bytes) - data size
# version (4 bytes) - weight version
# worker_id (4 bytes) - worker id
# padding (7 bytes) - align to 24 bytes
HEADER_SIZE = 24
HEADER_FORMAT = "=IBIIIxxxxxxx"  # seq, ready, size, version, worker_id, padding

# Default buffer size: 16MB per worker (enough for ~7MB gradients + overhead)
DEFAULT_BUFFER_SIZE = 16 * 1024 * 1024


@dataclass
class GradientPacket:
    """Gradient packet read from shared memory."""

    grads: dict[str, torch.Tensor]
    worker_id: int
    weight_version: int
    metrics: dict[str, Any]
    timesteps: int
    episodes: int


class SharedGradientBuffer:
    """
    Shared memory buffer for gradient exchange between one worker and the learner.
    
    Memory layout:
        [0:1]   - ready flag (1 = data available, 0 = empty/being written)
        [1:5]   - data size (uint32)
        [5:9]   - weight version (uint32)
        [9:13]  - worker id (uint32)
        [13:16] - reserved/padding
        [16:]   - serialized gradient data (torch.save bytes)
    
    Protocol:
        Worker:
            1. Set ready = 0 (claiming buffer)
            2. Write data to buffer[HEADER_SIZE:]
            3. Write size to buffer[1:5]
            4. Set ready = 1 (releasing buffer)
        
        Learner:
            1. Check if ready == 1
            2. If ready, read size and data
            3. Set ready = 0 (acknowledge receipt)
    """

    def __init__(
        self,
        worker_id: int,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        create: bool = True,
        shm_dir: Path | None = None,
    ):
        """
        Initialize shared memory buffer.
        
        Args:
            worker_id: Unique worker identifier
            buffer_size: Total buffer size in bytes (including header)
            create: If True, create new buffer. If False, attach to existing.
            shm_dir: Directory for shared memory files (default: /dev/shm or temp)
        """
        self.worker_id = worker_id
        self.buffer_size = buffer_size
        self.max_data_size = buffer_size - HEADER_SIZE
        
        # Shared memory file path
        if shm_dir is None:
            # Prefer /dev/shm (RAM-backed) if available
            if Path("/dev/shm").exists():
                shm_dir = Path("/dev/shm")
            else:
                shm_dir = Path("/tmp")
        
        self.shm_path = shm_dir / f"mario_grads_{os.getpid()}_{worker_id}.shm"
        
        if create:
            self._create_buffer()
        else:
            self._attach_buffer()
    
    def _create_buffer(self) -> None:
        """Create a new shared memory buffer."""
        # Ensure parent directory exists
        self.shm_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create file with correct size
        with open(self.shm_path, "wb") as f:
            f.write(b"\x00" * self.buffer_size)
        
        # Memory-map it
        self._fd = open(self.shm_path, "r+b")
        self._mmap = mmap.mmap(self._fd.fileno(), self.buffer_size)
        
        # Initialize sequence number (even = stable)
        self._seq = 0
        
        # Initialize header (seq=0, ready=0, size=0, version=0, worker_id)
        self._write_header(seq=0, ready=0, size=0, version=0)
    
    def _attach_buffer(self) -> None:
        """Attach to existing shared memory buffer."""
        if not self.shm_path.exists():
            raise FileNotFoundError(f"Shared memory file not found: {self.shm_path}")
        
        self._fd = open(self.shm_path, "r+b")
        self._mmap = mmap.mmap(self._fd.fileno(), self.buffer_size)
        
        # Read current sequence number from buffer
        self._seq = self._read_seq()
    
    def _write_header(self, seq: int, ready: int, size: int, version: int) -> None:
        """Write header to shared memory."""
        header = struct.pack(HEADER_FORMAT, seq, ready, size, version, self.worker_id)
        self._mmap[0:HEADER_SIZE] = header
    
    def _read_header(self) -> tuple[int, int, int, int, int]:
        """Read header from shared memory. Returns (seq, ready, size, version, worker_id)."""
        header = self._mmap[0:HEADER_SIZE]
        return struct.unpack(HEADER_FORMAT, header)
    
    def _read_seq(self) -> int:
        """Read just the sequence number (first 4 bytes)."""
        return struct.unpack("=I", self._mmap[0:4])[0]
    
    def is_ready(self) -> bool:
        """Check if buffer has data ready (non-blocking)."""
        seq, ready, size, version, worker_id = self._read_header()
        # Data is ready if: ready==1 AND seq is even (not being written)
        return ready == 1 and (seq % 2) == 0
    
    def write(self, packet: dict[str, Any]) -> bool:
        """
        Write gradient packet to shared memory using seqlock protocol.
        
        Args:
            packet: Dictionary containing grads, metrics, etc.
        
        Returns:
            True if write succeeded, False if buffer too small or busy.
        """
        # Serialize packet to bytes
        buffer = io.BytesIO()
        torch.save(packet, buffer)
        data = buffer.getvalue()
        
        if len(data) > self.max_data_size:
            return False  # Data too large
        
        # Seqlock write protocol:
        # 1. Increment seq to odd (signals "writing in progress")
        self._seq += 1
        version = packet.get("weight_version", 0)
        self._write_header(seq=self._seq, ready=0, size=len(data), version=version)
        self._mmap.flush()
        
        # 2. Write data
        self._mmap[HEADER_SIZE:HEADER_SIZE + len(data)] = data
        self._mmap.flush()
        
        # 3. Increment seq to even (signals "write complete")
        self._seq += 1
        self._write_header(seq=self._seq, ready=1, size=len(data), version=version)
        self._mmap.flush()
        
        return True
    
    def read(self) -> dict[str, Any] | None:
        """
        Read gradient packet from shared memory using seqlock protocol.
        
        Returns:
            Packet dict if data available, None otherwise.
        """
        # Seqlock read protocol:
        # 1. Read seq and header
        seq1, ready, size, version, worker_id = self._read_header()
        
        # Check if data is available and not being written (seq must be even)
        if ready != 1 or (seq1 % 2) != 0:
            return None
        
        # Validate size
        if size == 0 or size > self.max_data_size:
            return None
        
        # 2. Read data
        data = bytes(self._mmap[HEADER_SIZE:HEADER_SIZE + size])
        
        # 3. Re-read seq to detect concurrent write
        seq2 = self._read_seq()
        
        if seq1 != seq2:
            # Write occurred during our read - data is invalid
            return None
        
        # 4. Clear ready flag to signal we've consumed the data
        # Use a header write with same seq to avoid triggering false "writing" state
        self._mmap[4] = 0  # Set ready byte to 0 (offset 4 after seq)
        self._mmap.flush()
        
        # 5. Deserialize
        try:
            buffer = io.BytesIO(data)
            packet = torch.load(buffer, map_location="cpu", weights_only=False)
            return packet
        except Exception:
            # Corrupted data (shouldn't happen with seqlock, but be safe)
            return None
    
    def clear(self) -> None:
        """Clear the buffer (mark as empty)."""
        self._mmap[0] = 0
    
    def close(self) -> None:
        """Close and optionally cleanup shared memory."""
        if hasattr(self, "_mmap") and self._mmap is not None:
            self._mmap.close()
        if hasattr(self, "_fd") and self._fd is not None:
            self._fd.close()
    
    def unlink(self) -> None:
        """Remove the shared memory file."""
        self.close()
        if self.shm_path.exists():
            self.shm_path.unlink()
    
    def __del__(self):
        self.close()


class SharedGradientPool:
    """
    Pool of shared gradient buffers for all workers.
    
    Usage (Learner side):
        pool = SharedGradientPool(num_workers=16, create=True)
        
        while training:
            packets = pool.collect_ready()
            for packet in packets:
                apply_gradients(packet)
    
    Usage (Worker side):
        buffer = pool.get_worker_buffer(worker_id)
        buffer.write(gradient_packet)
    """

    def __init__(
        self,
        num_workers: int,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        create: bool = True,
        shm_dir: Path | None = None,
    ):
        """
        Initialize pool of shared gradient buffers.
        
        Args:
            num_workers: Number of workers (buffers to create)
            buffer_size: Size of each buffer in bytes
            create: If True, create new buffers. If False, attach to existing.
            shm_dir: Directory for shared memory files
        """
        self.num_workers = num_workers
        self.buffer_size = buffer_size
        self.shm_dir = shm_dir
        
        # Create/attach buffers for each worker
        self.buffers: list[SharedGradientBuffer] = []
        for worker_id in range(num_workers):
            buf = SharedGradientBuffer(
                worker_id=worker_id,
                buffer_size=buffer_size,
                create=create,
                shm_dir=shm_dir,
            )
            self.buffers.append(buf)
    
    def get_buffer(self, worker_id: int) -> SharedGradientBuffer:
        """Get the buffer for a specific worker."""
        return self.buffers[worker_id]
    
    def collect_ready(self) -> list[dict[str, Any]]:
        """
        Collect all ready gradient packets (non-blocking).
        
        Returns:
            List of gradient packets that were ready.
        """
        packets = []
        for buf in self.buffers:
            packet = buf.read()
            if packet is not None:
                packets.append(packet)
        return packets
    
    def any_ready(self) -> bool:
        """Check if any buffer has data ready."""
        return any(buf.is_ready() for buf in self.buffers)
    
    def count_ready(self) -> int:
        """Count how many buffers have data ready."""
        return sum(1 for buf in self.buffers if buf.is_ready())
    
    def get_shm_paths(self) -> list[Path]:
        """Get paths to all shared memory files (for passing to workers)."""
        return [buf.shm_path for buf in self.buffers]
    
    def close(self) -> None:
        """Close all buffers."""
        for buf in self.buffers:
            buf.close()
    
    def unlink(self) -> None:
        """Remove all shared memory files."""
        for buf in self.buffers:
            buf.unlink()
    
    def __del__(self):
        self.close()


# Convenience function for workers to attach to their buffer
def attach_worker_buffer(
    worker_id: int,
    shm_path: Path,
    buffer_size: int = DEFAULT_BUFFER_SIZE,
) -> SharedGradientBuffer:
    """
    Attach to an existing shared memory buffer (for workers).
    
    Args:
        worker_id: Worker's ID
        shm_path: Path to the shared memory file
        buffer_size: Size of the buffer
    
    Returns:
        SharedGradientBuffer attached to the existing shared memory.
    """
    buf = SharedGradientBuffer.__new__(SharedGradientBuffer)
    buf.worker_id = worker_id
    buf.buffer_size = buffer_size
    buf.max_data_size = buffer_size - HEADER_SIZE
    buf.shm_path = shm_path
    buf._attach_buffer()
    return buf


class SharedHeartbeats:
    """
    Shared memory for worker heartbeats.
    
    Uses mmap'd file with one 8-byte double per worker for timestamp.
    No queues = no deadlocks.
    """
    
    def __init__(
        self,
        num_workers: int,
        shm_dir: Path | None = None,
        create: bool = True,
    ):
        self.num_workers = num_workers
        # 8 bytes per worker (double timestamp)
        self.buffer_size = num_workers * 8
        
        if shm_dir is None:
            if Path("/dev/shm").exists():
                shm_dir = Path("/dev/shm")
            else:
                shm_dir = Path("/tmp")
        
        self.shm_path = shm_dir / f"mario_heartbeats_{os.getpid()}.shm"
        
        if create:
            self._create()
        else:
            self._attach()
    
    def _create(self) -> None:
        """Create new shared memory."""
        self.shm_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.shm_path, "wb") as f:
            # Initialize all timestamps to 0
            f.write(b"\x00" * self.buffer_size)
        
        self._fd = open(self.shm_path, "r+b")
        self._mmap = mmap.mmap(self._fd.fileno(), self.buffer_size)
    
    def _attach(self) -> None:
        """Attach to existing shared memory."""
        if not self.shm_path.exists():
            raise FileNotFoundError(f"Heartbeat file not found: {self.shm_path}")
        
        self._fd = open(self.shm_path, "r+b")
        self._mmap = mmap.mmap(self._fd.fileno(), self.buffer_size)
    
    def update(self, worker_id: int, timestamp: float | None = None) -> None:
        """Update heartbeat timestamp for a worker."""
        if timestamp is None:
            import time
            timestamp = time.time()
        
        offset = worker_id * 8
        self._mmap[offset:offset + 8] = struct.pack("=d", timestamp)
        self._mmap.flush()
    
    def get(self, worker_id: int) -> float:
        """Get heartbeat timestamp for a worker."""
        offset = worker_id * 8
        return struct.unpack("=d", self._mmap[offset:offset + 8])[0]
    
    def get_all(self) -> list[float]:
        """Get all heartbeat timestamps."""
        return [self.get(i) for i in range(self.num_workers)]
    
    def close(self) -> None:
        """Close the mmap and file descriptor."""
        if hasattr(self, "_mmap"):
            self._mmap.close()
        if hasattr(self, "_fd"):
            self._fd.close()
    
    def unlink(self) -> None:
        """Remove shared memory file."""
        self.close()
        if self.shm_path.exists():
            self.shm_path.unlink()
    
    def __del__(self):
        self.close()

