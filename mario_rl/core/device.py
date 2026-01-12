"""
Device detection utilities.

Single source of truth for detecting the best available compute device.
Supports multi-GPU distribution for distributed training.
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


def get_gpu_count() -> int:
    """
    Get the number of available CUDA GPUs.

    Returns:
        Number of CUDA devices (0 if CUDA not available)
    """
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def assign_device(process_id: int, num_processes: int) -> str:
    """
    Assign a device to a process for even GPU distribution.

    Distributes processes across available GPUs in round-robin fashion.
    Falls back to MPS or CPU if no CUDA GPUs available.

    Args:
        process_id: The ID of the process (0-indexed)
        num_processes: Total number of processes to distribute

    Returns:
        Device string (e.g., "cuda:0", "cuda:1", "mps", or "cpu")

    Example:
        With 2 GPUs and 4 workers + 1 coordinator (5 processes):
        - process 0 (coordinator): cuda:0
        - process 1 (worker 0): cuda:1
        - process 2 (worker 1): cuda:0
        - process 3 (worker 2): cuda:1
        - process 4 (worker 3): cuda:0
    """
    gpu_count = get_gpu_count()

    if gpu_count == 0:
        # No CUDA, fall back to MPS or CPU
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if gpu_count == 1:
        # Single GPU, everything goes there
        return "cuda:0"

    # Multiple GPUs: distribute evenly using round-robin
    gpu_index = process_id % gpu_count
    return f"cuda:{gpu_index}"


def get_device_assignment_summary(num_workers: int) -> str:
    """
    Get a human-readable summary of device assignments.

    Args:
        num_workers: Number of worker processes

    Returns:
        Summary string showing device distribution
    """
    gpu_count = get_gpu_count()
    num_processes = num_workers + 1  # workers + coordinator

    if gpu_count == 0:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        return f"All {num_processes} processes on {device}"

    if gpu_count == 1:
        return f"All {num_processes} processes on cuda:0"

    # Count processes per GPU
    gpu_assignments: dict[int, list[str]] = {i: [] for i in range(gpu_count)}

    # Coordinator is process 0
    gpu_assignments[0 % gpu_count].append("coordinator")

    # Workers are processes 1 to num_workers
    for i in range(num_workers):
        gpu_idx = (i + 1) % gpu_count
        gpu_assignments[gpu_idx].append(f"worker_{i}")

    lines = [f"{gpu_count} GPUs detected:"]
    for gpu_idx in range(gpu_count):
        procs = gpu_assignments[gpu_idx]
        lines.append(f"  cuda:{gpu_idx}: {', '.join(procs)} ({len(procs)} processes)")

    return "\n".join(lines)
