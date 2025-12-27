"""
Device detection utilities.

Single source of truth for detecting the best available compute device.
"""

import torch


def detect_device() -> str:
    """
    Detect the best available compute device.

    Returns:
        Device string: "cuda", "mps", or "cpu"
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
