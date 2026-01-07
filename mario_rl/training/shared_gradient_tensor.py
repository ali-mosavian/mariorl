"""
SharedGradientTensor - Zero-copy gradient transfer via shared memory ring buffer.

Uses numpy array backed by mmap, then torch.from_numpy() for tensor access.
Ring buffer design allows writer to keep writing while reader catches up.
Seqlock synchronization for crash safety per slot.
"""
from __future__ import annotations

import mmap
import struct
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True, slots=True)
class GradientPacket:
    """Immutable gradient packet from shared memory.
    
    Contains gradients and metadata from a worker's gradient computation.
    Frozen for thread safety and to prevent accidental mutation.
    """
    grads: dict[str, torch.Tensor]
    version: int
    worker_id: int
    timesteps: int
    episodes: int
    # Worker-computed metrics (optional, for learner display)
    loss: float = 0.0
    q_mean: float = 0.0
    td_error: float = 0.0


# Per-slot header layout (64 bytes):
# - seq: 4 bytes (uint32) - sequence number for seqlock
# - ready: 1 byte (uint8) - ready flag
# - version: 4 bytes (uint32) - weight version
# - worker_id: 4 bytes (uint32) - worker ID
# - timesteps: 4 bytes (uint32) - timesteps in this gradient batch
# - episodes: 4 bytes (uint32) - total episodes from worker
# - loss: 4 bytes (float32) - worker-computed loss
# - q_mean: 4 bytes (float32) - worker-computed Q mean
# - td_error: 4 bytes (float32) - worker-computed TD error
# - padding: 31 bytes
SLOT_HEADER_SIZE = 64
SEQ_OFFSET = 0
READY_OFFSET = 4
VERSION_OFFSET = 5
WORKER_ID_OFFSET = 9
TIMESTEPS_OFFSET = 13
EPISODES_OFFSET = 17
LOSS_OFFSET = 21
Q_MEAN_OFFSET = 25
TD_ERROR_OFFSET = 29

# Global header (64 bytes at start of file):
# - write_idx: 4 bytes (uint32) - next slot to write
# - read_idx: 4 bytes (uint32) - next slot to read  
# - num_slots: 4 bytes (uint32) - number of slots
# - padding: 52 bytes
GLOBAL_HEADER_SIZE = 64


class SharedGradientTensor:
    """
    Zero-copy gradient transfer using ring buffer in shared memory.

    Ring buffer design:
    - N slots, each with header + gradient data
    - Writer writes to slot[write_idx % N], increments write_idx
    - Reader reads from oldest available slot
    - Seqlock per slot for crash safety

    Usage:
        # Worker (writer)
        buffer = SharedGradientTensor(model, shm_path, create=True, num_slots=4)
        buffer.write(grads)  # Writes to next slot

        # Learner (reader)  
        buffer = SharedGradientTensor(model, shm_path, create=False, num_slots=4)
        grads = buffer.read()  # Reads oldest unread slot
    """

    def __init__(
        self,
        model: nn.Module,
        shm_path: Path,
        create: bool = True,
        num_slots: int = 4,
    ):
        """
        Initialize shared gradient ring buffer.

        Args:
            model: PyTorch model to extract parameter layout from
            shm_path: Path to shared memory file
            create: If True, create new buffer. If False, attach to existing.
            num_slots: Number of slots in ring buffer (default 4)
        """
        self.shm_path = Path(shm_path)
        self.num_slots = num_slots

        # Build parameter layout from model and detect dtype
        self.param_layout: list[tuple[str, int, torch.Size]] = []
        total_numel = 0
        self.dtype: torch.dtype = torch.float32  # default
        self.np_dtype = np.float32  # default

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.param_layout.append((name, param.numel(), param.shape))
                total_numel += param.numel()
                # Use dtype from first parameter
                if len(self.param_layout) == 1:
                    self.dtype = param.dtype
                    self.np_dtype = self._torch_to_numpy_dtype(param.dtype)

        self.total_numel = total_numel
        self.element_size = self._dtype_size(self.dtype)
        self.slot_data_bytes = total_numel * self.element_size
        self.slot_total_bytes = SLOT_HEADER_SIZE + self.slot_data_bytes
        self.total_bytes = GLOBAL_HEADER_SIZE + num_slots * self.slot_total_bytes

        if create:
            self._create()
        else:
            self._attach()

    @staticmethod
    def _torch_to_numpy_dtype(dtype: torch.dtype) -> np.dtype:
        """Convert torch dtype to numpy dtype."""
        mapping = {
            torch.float16: np.float16,
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.bfloat16: np.float32,  # numpy doesn't have bfloat16, use float32
            torch.int8: np.int8,
            torch.int16: np.int16,
            torch.int32: np.int32,
            torch.int64: np.int64,
        }
        return mapping.get(dtype, np.float32)

    @staticmethod
    def _dtype_size(dtype: torch.dtype) -> int:
        """Get element size in bytes for dtype."""
        sizes = {
            torch.float16: 2,
            torch.float32: 4,
            torch.float64: 8,
            torch.bfloat16: 2,
            torch.int8: 1,
            torch.int16: 2,
            torch.int32: 4,
            torch.int64: 8,
        }
        return sizes.get(dtype, 4)

    def _create(self) -> None:
        """Create new shared memory file and mmap it."""
        self.shm_path.parent.mkdir(parents=True, exist_ok=True)

        # Create file with zeros
        with open(self.shm_path, "wb") as f:
            f.write(b"\x00" * self.total_bytes)

        self._open_mmap()

        # Initialize global header
        self._write_global_header(write_idx=0, read_idx=0)

    def _attach(self) -> None:
        """Attach to existing shared memory file."""
        if not self.shm_path.exists():
            raise FileNotFoundError(f"Shared memory file not found: {self.shm_path}")

        self._open_mmap()

    def _open_mmap(self) -> None:
        """Open mmap and create numpy views for each slot."""
        self._fd = open(self.shm_path, "r+b")
        self._mmap = mmap.mmap(self._fd.fileno(), self.total_bytes)

        # Create numpy array views for each slot's data section
        self._slot_arrays: list[np.ndarray] = []
        for i in range(self.num_slots):
            slot_start = GLOBAL_HEADER_SIZE + i * self.slot_total_bytes
            data_start = slot_start + SLOT_HEADER_SIZE

            arr = np.ndarray(
                shape=(self.total_numel,),
                dtype=self.np_dtype,
                buffer=self._mmap,
                offset=data_start,
            )
            self._slot_arrays.append(arr)

    def _slot_offset(self, slot_idx: int) -> int:
        """Get byte offset for slot header."""
        return GLOBAL_HEADER_SIZE + slot_idx * self.slot_total_bytes

    def _read_global_header(self) -> tuple[int, int]:
        """Read global header. Returns (write_idx, read_idx)."""
        data = self._mmap[0:8]
        return struct.unpack("II", data)

    def _write_global_header(self, write_idx: int, read_idx: int) -> None:
        """Write global header."""
        self._mmap[0:8] = struct.pack("II", write_idx, read_idx)

    def _read_slot_header(self, slot_idx: int) -> tuple[int, int, int, int, int, int, float, float, float]:
        """Read slot header. Returns (seq, ready, version, worker_id, timesteps, episodes, loss, q_mean, td_error)."""
        offset = self._slot_offset(slot_idx)
        seq = struct.unpack("I", self._mmap[offset : offset + 4])[0]
        ready = self._mmap[offset + READY_OFFSET]
        version = struct.unpack("I", self._mmap[offset + VERSION_OFFSET : offset + VERSION_OFFSET + 4])[0]
        worker_id = struct.unpack("I", self._mmap[offset + WORKER_ID_OFFSET : offset + WORKER_ID_OFFSET + 4])[0]
        timesteps = struct.unpack("I", self._mmap[offset + TIMESTEPS_OFFSET : offset + TIMESTEPS_OFFSET + 4])[0]
        episodes = struct.unpack("I", self._mmap[offset + EPISODES_OFFSET : offset + EPISODES_OFFSET + 4])[0]
        loss = struct.unpack("f", self._mmap[offset + LOSS_OFFSET : offset + LOSS_OFFSET + 4])[0]
        q_mean = struct.unpack("f", self._mmap[offset + Q_MEAN_OFFSET : offset + Q_MEAN_OFFSET + 4])[0]
        td_error = struct.unpack("f", self._mmap[offset + TD_ERROR_OFFSET : offset + TD_ERROR_OFFSET + 4])[0]
        return seq, ready, version, worker_id, timesteps, episodes, loss, q_mean, td_error

    def _write_slot_header(
        self,
        slot_idx: int,
        seq: int,
        ready: int,
        version: int,
        worker_id: int = 0,
        timesteps: int = 0,
        episodes: int = 0,
        loss: float = 0.0,
        q_mean: float = 0.0,
        td_error: float = 0.0,
    ) -> None:
        """Write slot header."""
        offset = self._slot_offset(slot_idx)
        self._mmap[offset : offset + 4] = struct.pack("I", seq)
        self._mmap[offset + READY_OFFSET] = ready
        self._mmap[offset + VERSION_OFFSET : offset + VERSION_OFFSET + 4] = struct.pack("I", version)
        self._mmap[offset + WORKER_ID_OFFSET : offset + WORKER_ID_OFFSET + 4] = struct.pack("I", worker_id)
        self._mmap[offset + TIMESTEPS_OFFSET : offset + TIMESTEPS_OFFSET + 4] = struct.pack("I", timesteps)
        self._mmap[offset + EPISODES_OFFSET : offset + EPISODES_OFFSET + 4] = struct.pack("I", episodes)
        self._mmap[offset + LOSS_OFFSET : offset + LOSS_OFFSET + 4] = struct.pack("f", loss)
        self._mmap[offset + Q_MEAN_OFFSET : offset + Q_MEAN_OFFSET + 4] = struct.pack("f", q_mean)
        self._mmap[offset + TD_ERROR_OFFSET : offset + TD_ERROR_OFFSET + 4] = struct.pack("f", td_error)
        self._mmap[offset + TIMESTEPS_OFFSET : offset + TIMESTEPS_OFFSET + 4] = struct.pack("I", timesteps)
        self._mmap[offset + EPISODES_OFFSET : offset + EPISODES_OFFSET + 4] = struct.pack("I", episodes)

    def _read_seq(self, slot_idx: int = 0) -> int:
        """Read sequence number from slot header."""
        offset = self._slot_offset(slot_idx)
        return struct.unpack("I", self._mmap[offset : offset + 4])[0]

    def _write_seq(self, seq: int, slot_idx: int = 0) -> None:
        """Write sequence number to slot header."""
        offset = self._slot_offset(slot_idx)
        self._mmap[offset : offset + 4] = struct.pack("I", seq)

    def _read_ready(self, slot_idx: int = 0) -> int:
        """Read ready flag from slot header."""
        offset = self._slot_offset(slot_idx)
        return self._mmap[offset + READY_OFFSET]

    def _write_ready(self, ready: int, slot_idx: int = 0) -> None:
        """Write ready flag to slot header."""
        offset = self._slot_offset(slot_idx)
        self._mmap[offset + READY_OFFSET] = ready

    def is_ready(self, slot_idx: int = 0) -> bool:
        """Check if slot has data ready to read."""
        seq, ready, *_ = self._read_slot_header(slot_idx)
        return ready == 1 and (seq % 2) == 0

    def count_ready(self) -> int:
        """Count number of slots with data ready."""
        return sum(1 for i in range(self.num_slots) if self.is_ready(i))

    def write(
        self,
        grads: dict[str, torch.Tensor],
        version: int = 0,
        worker_id: int = 0,
        timesteps: int = 0,
        episodes: int = 0,
        loss: float = 0.0,
        q_mean: float = 0.0,
        td_error: float = 0.0,
    ) -> bool:
        """
        Write gradients to next slot in ring buffer.

        Args:
            grads: Dict mapping parameter names to gradient tensors
            version: Weight version number
            worker_id: Worker ID that computed these gradients
            timesteps: Number of timesteps in this gradient batch
            episodes: Total episodes from worker
            loss: Worker-computed loss value
            q_mean: Worker-computed mean Q value
            td_error: Worker-computed TD error

        Returns:
            True if write succeeded
        """
        # Get next write slot
        write_idx, _ = self._read_global_header()
        slot_idx = write_idx % self.num_slots

        # Read current seq for this slot
        current_seq, *_ = self._read_slot_header(slot_idx)

        # 1. Increment seq to odd (writing in progress)
        new_seq = current_seq + 1
        self._write_slot_header(
            slot_idx,
            seq=new_seq,
            ready=0,
            version=version,
            worker_id=worker_id,
            timesteps=timesteps,
            episodes=episodes,
            loss=loss,
            q_mean=q_mean,
            td_error=td_error,
        )

        # 2. Copy gradients to slot's numpy array
        slot_array = self._slot_arrays[slot_idx]
        offset = 0
        for name, numel, shape in self.param_layout:
            if name in grads:
                grad = grads[name]
                if grad.is_cuda:
                    grad = grad.cpu()
                np.copyto(
                    slot_array[offset : offset + numel],
                    grad.detach().view(-1).numpy(),
                )
            offset += numel

        # 3. Increment seq to even (write complete), set ready
        new_seq += 1
        self._write_slot_header(
            slot_idx,
            seq=new_seq,
            ready=1,
            version=version,
            worker_id=worker_id,
            timesteps=timesteps,
            episodes=episodes,
            loss=loss,
            q_mean=q_mean,
            td_error=td_error,
        )

        # 4. Increment write index
        self._write_global_header(write_idx=write_idx + 1, read_idx=self._read_global_header()[1])

        self._mmap.flush()
        return True

    def read(self) -> GradientPacket | None:
        """
        Read gradients from oldest unread slot.

        Returns:
            GradientPacket if data available, None otherwise
        """
        write_idx, read_idx = self._read_global_header()

        # Check if there's data to read
        if read_idx >= write_idx:
            return None  # No unread data

        slot_idx = read_idx % self.num_slots

        # 1. Check ready and seq
        seq1, ready, version, worker_id, timesteps, episodes, loss, q_mean, td_error = self._read_slot_header(slot_idx)

        if ready != 1 or (seq1 % 2) == 1:
            return None  # Not ready or write in progress

        # 2. Read gradients
        slot_array = self._slot_arrays[slot_idx]
        grads: dict[str, torch.Tensor] = {}
        offset = 0

        for name, numel, shape in self.param_layout:
            grad_flat = torch.from_numpy(slot_array[offset : offset + numel].copy())
            grads[name] = grad_flat.view(shape)
            offset += numel

        # 3. Verify seq didn't change (seqlock check)
        seq2 = self._read_seq(slot_idx)

        if seq1 != seq2:
            return None  # Concurrent write detected

        # 4. Clear ready and increment read index
        self._write_ready(0, slot_idx)
        self._write_global_header(write_idx=write_idx, read_idx=read_idx + 1)
        self._mmap.flush()

        return GradientPacket(
            grads=grads,
            version=version,
            worker_id=worker_id,
            timesteps=timesteps,
            episodes=episodes,
            loss=loss,
            q_mean=q_mean,
            td_error=td_error,
        )

    def read_latest(self) -> GradientPacket | None:
        """
        Read gradients from latest slot, skipping any older unread data.

        Useful when reader is falling behind and wants to catch up.

        Returns:
            GradientPacket if data available, None otherwise
        """
        write_idx, read_idx = self._read_global_header()

        if read_idx >= write_idx:
            return None  # No unread data

        # Jump to latest slot
        latest_idx = write_idx - 1
        slot_idx = latest_idx % self.num_slots

        # 1. Check ready and seq
        seq1, ready, version, worker_id, timesteps, episodes, loss, q_mean, td_error = self._read_slot_header(slot_idx)

        if ready != 1 or (seq1 % 2) == 1:
            # Latest not ready, try previous
            if latest_idx > read_idx:
                latest_idx -= 1
                slot_idx = latest_idx % self.num_slots
                seq1, ready, version, worker_id, timesteps, episodes, loss, q_mean, td_error = self._read_slot_header(slot_idx)
                if ready != 1 or (seq1 % 2) == 1:
                    return None
            else:
                return None

        # 2. Read gradients
        slot_array = self._slot_arrays[slot_idx]
        grads: dict[str, torch.Tensor] = {}
        offset = 0

        for name, numel, shape in self.param_layout:
            grad_flat = torch.from_numpy(slot_array[offset : offset + numel].copy())
            grads[name] = grad_flat.view(shape)
            offset += numel

        # 3. Verify seq didn't change
        seq2 = self._read_seq(slot_idx)

        if seq1 != seq2:
            return None

        # 4. Update read index to skip all older slots
        self._write_ready(0, slot_idx)
        self._write_global_header(write_idx=write_idx, read_idx=latest_idx + 1)
        self._mmap.flush()

        return GradientPacket(
            grads=grads,
            version=version,
            worker_id=worker_id,
            timesteps=timesteps,
            episodes=episodes,
            loss=loss,
            q_mean=q_mean,
            td_error=td_error,
        )

    def close(self) -> None:
        """Close mmap and file descriptor."""
        if hasattr(self, "_slot_arrays"):
            del self._slot_arrays

        if hasattr(self, "_mmap") and self._mmap is not None:
            self._mmap.close()
            self._mmap = None

        if hasattr(self, "_fd") and self._fd is not None:
            self._fd.close()
            self._fd = None

    def unlink(self) -> None:
        """Close and remove shared memory file."""
        self.close()
        if self.shm_path.exists():
            self.shm_path.unlink()

    def __del__(self):
        """Cleanup on garbage collection."""
        self.close()


class SharedGradientTensorPool:
    """
    Pool of SharedGradientTensor buffers for multiple workers.

    Each worker gets its own ring buffer for gradient transfer.
    The learner polls all buffers to collect gradients.
    """

    def __init__(
        self,
        num_workers: int,
        model: nn.Module,
        shm_dir: Path,
        num_slots: int = 8,  # Slots per worker (8 = good buffer for async writes)
        create: bool = True,
    ):
        """
        Initialize pool of SharedGradientTensor buffers.

        Args:
            num_workers: Number of workers (one buffer per worker)
            model: PyTorch model for parameter layout
            shm_dir: Directory for shared memory files
            num_slots: Number of slots per worker's ring buffer
            create: If True, create new buffers. If False, attach to existing.
        """
        self.num_workers = num_workers
        self.shm_dir = Path(shm_dir)
        self.num_slots = num_slots

        if create:
            self.shm_dir.mkdir(parents=True, exist_ok=True)

        self.buffers: list[SharedGradientTensor] = []
        for i in range(num_workers):
            shm_path = self.shm_dir / f"grad_tensor_{i}.shm"
            buffer = SharedGradientTensor(
                model=model,
                shm_path=shm_path,
                create=create,
                num_slots=num_slots,
            )
            self.buffers.append(buffer)

    def get_shm_paths(self) -> list[Path]:
        """Get paths to all shared memory files."""
        return [buf.shm_path for buf in self.buffers]

    def read_worker_buffer(self, worker_id: int) -> GradientPacket | None:
        """Read gradients from a specific worker's buffer.
        
        Returns:
            GradientPacket if data available, None otherwise
        """
        if worker_id >= len(self.buffers):
            return None
        return self.buffers[worker_id].read()

    def read_all_available(self) -> list[GradientPacket]:
        """Read all available gradients from all workers.
        
        Returns:
            List of GradientPacket instances
        """
        results: list[GradientPacket] = []
        for buffer in self.buffers:
            # Read all available from this worker's ring buffer
            while True:
                packet = buffer.read()
                if packet is None:
                    break
                results.append(packet)
        return results

    def count_total_ready(self) -> int:
        """Count total number of ready gradients across all workers."""
        return sum(buf.count_ready() for buf in self.buffers)

    def close(self) -> None:
        """Close all buffers."""
        for buffer in self.buffers:
            buffer.close()

    def unlink(self) -> None:
        """Close and remove all shared memory files."""
        for buffer in self.buffers:
            buffer.unlink()

        # Remove directory if empty
        if self.shm_dir.exists():
            try:
                self.shm_dir.rmdir()
            except OSError:
                pass  # Directory not empty

    def __del__(self):
        """Cleanup on garbage collection."""
        self.close()


def attach_tensor_buffer(
    worker_id: int,
    model: nn.Module,
    shm_path: Path,
    num_slots: int = 8,
) -> SharedGradientTensor:
    """
    Attach to an existing SharedGradientTensor buffer.

    Used by workers to attach to buffers created by main process.

    Args:
        worker_id: Worker ID (for identification)
        model: PyTorch model for parameter layout
        shm_path: Path to shared memory file
        num_slots: Number of slots (must match creator)

    Returns:
        SharedGradientTensor attached to existing buffer
    """
    return SharedGradientTensor(
        model=model,
        shm_path=shm_path,
        create=False,
        num_slots=num_slots,
    )
