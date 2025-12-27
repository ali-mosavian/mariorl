"""
Weight synchronization for distributed training.

Handles loading weights from disk and tracking sync state.
"""

import time
from typing import Any
from pathlib import Path
from dataclasses import field
from dataclasses import dataclass

import torch


@dataclass
class WeightSync:
    """
    Handles weight synchronization from learner.

    Loads weights from disk periodically and tracks sync statistics.
    """

    path: Path
    device: str
    interval: float = 5.0

    # State (not initialized from args)
    last_sync: float = field(init=False, default=0.0)
    version: int = field(init=False, default=0)
    count: int = field(init=False, default=0)

    def should_sync(self) -> bool:
        """Check if enough time has passed to sync weights."""
        return time.time() - self.last_sync >= self.interval

    def maybe_sync(self, net: Any) -> bool:
        """Sync weights if enough time has passed."""
        if not self.should_sync():
            return False
        return self.load(net)

    def load(self, net: Any) -> bool:
        """
        Load weights from disk into network.

        Args:
            net: Network with load_state_dict and sync_target methods

        Returns:
            True if weights were loaded successfully
        """
        if not self.path.exists():
            return False

        for attempt in range(3):
            try:
                checkpoint = torch.load(
                    self.path,
                    map_location=self.device,
                    weights_only=True,
                )
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    net.load_state_dict(checkpoint["state_dict"])
                    self.version = checkpoint.get("version", 0)
                else:
                    net.load_state_dict(checkpoint)
                net.sync_target()
                self.last_sync = time.time()
                self.count += 1
                return True
            except Exception:
                if attempt < 2:
                    time.sleep(0.1)
                continue
        return False
