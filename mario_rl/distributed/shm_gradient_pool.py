"""Shared Memory Gradient Pool for distributed training.

Wraps SharedGradientTensorPool with a cleaner interface for the
new modular training infrastructure.
"""

from typing import Any
from pathlib import Path
from dataclasses import dataclass
from dataclasses import field

import torch
from torch import nn
from torch import Tensor

from mario_rl.training.shared_gradient_tensor import SharedGradientTensor
from mario_rl.training.shared_gradient_tensor import GradientPacket


@dataclass
class SharedGradientPool:
    """Pool of per-worker gradient buffers using shared memory.

    Each worker has its own ring buffer for zero-copy gradient transfer.
    The coordinator polls all buffers to collect gradients.
    """

    num_workers: int
    model: nn.Module
    shm_dir: Path
    num_slots: int = 8
    create: bool = True

    # Internal state
    buffers: list[SharedGradientTensor] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize per-worker buffers."""
        self.shm_dir = Path(self.shm_dir)

        if self.create:
            self.shm_dir.mkdir(parents=True, exist_ok=True)

        self.buffers = []
        for i in range(self.num_workers):
            shm_path = self.shm_dir / f"grads_worker_{i}.shm"
            buffer = SharedGradientTensor(
                model=self.model,
                shm_path=shm_path,
                create=self.create,
                num_slots=self.num_slots,
            )
            self.buffers.append(buffer)

    def write(
        self,
        worker_id: int,
        grads: dict[str, Tensor],
        worker_meta: dict[str, Any] | None = None,
        version: int = 0,
        timesteps: int = 0,
        episodes: int = 0,
    ) -> bool:
        """Write gradients to worker's buffer.

        Args:
            worker_id: Worker that computed the gradients
            grads: Dict mapping param names to gradient tensors
            worker_meta: Optional metadata dict
            version: Weight version used for these gradients
            timesteps: Number of timesteps in this batch
            episodes: Total episodes from worker

        Returns:
            True if write succeeded
        """
        if worker_id >= len(self.buffers):
            return False

        meta = worker_meta or {}

        return self.buffers[worker_id].write(
            grads=grads,
            version=version,
            worker_id=worker_id,
            timesteps=timesteps,
            episodes=episodes,
            loss=meta.get("loss", 0.0),
            q_mean=meta.get("q_mean", 0.0),
            td_error=meta.get("td_error", 0.0),
            avg_reward=meta.get("avg_reward", 0.0),
            avg_speed=meta.get("avg_speed", 0.0),
            entropy=meta.get("entropy", 0.0),
            deaths=meta.get("deaths", 0),
            flags=meta.get("flags", 0),
            best_x=meta.get("best_x", 0),
        )

    def read(self, worker_id: int) -> GradientPacket | None:
        """Read gradients from worker's buffer.

        Args:
            worker_id: Worker to read from

        Returns:
            GradientPacket if data available, None otherwise
        """
        if worker_id >= len(self.buffers):
            return None

        return self.buffers[worker_id].read()

    def read_all(self) -> list[GradientPacket]:
        """Read all available gradients from all workers.

        Returns:
            List of GradientPackets
        """
        results: list[GradientPacket] = []
        for buffer in self.buffers:
            while True:
                packet = buffer.read()
                if packet is None:
                    break
                results.append(packet)
        return results

    def buffer_path(self, worker_id: int) -> Path:
        """Get path to worker's shared memory file."""
        return self.buffers[worker_id].shm_path

    def count_ready(self) -> int:
        """Count total ready gradients across all workers."""
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
                pass

    def __del__(self) -> None:
        """Cleanup on garbage collection."""
        self.close()
