"""Distributed training infrastructure."""

from mario_rl.distributed.coordinator import Coordinator
from mario_rl.distributed.worker import Worker

__all__ = [
    "Coordinator",
    "Worker",
]
