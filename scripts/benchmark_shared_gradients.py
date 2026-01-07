#!/usr/bin/env python3
"""
Benchmark SharedGradientBuffer (torch.save/load) vs SharedGradientTensor (zero-copy).
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn

from mario_rl.training.shared_gradients import SharedGradientBuffer
from mario_rl.training.shared_gradient_tensor import SharedGradientTensor


class DQNNetwork(nn.Module):
    """Approximate DQN network size for realistic benchmark."""

    def __init__(self):
        super().__init__()
        # Similar to DDQNNet architecture
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Calculate conv output size for 64x64 input
        # After conv1: (64-8)/4+1 = 15
        # After conv2: (15-4)/2+1 = 6
        # After conv3: (6-3)/1+1 = 4
        # Output: 64 * 4 * 4 = 1024
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 12),  # action space
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.fc(x)


def count_parameters(model: nn.Module) -> int:
    """Count total parameters in model."""
    return sum(p.numel() for p in model.parameters())


def benchmark_shared_gradient_buffer(
    model: nn.Module, num_iterations: int = 100
) -> dict[str, float]:
    """Benchmark SharedGradientBuffer (torch.save/load serialization)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        shm_path = Path(tmpdir) / "grads.shm"

        # Create buffer
        buffer = SharedGradientBuffer(
            worker_id=0,
            buffer_size=32 * 1024 * 1024,  # 32MB
            create=True,
            shm_dir=Path(tmpdir),
        )

        # Generate gradients
        x = torch.randn(1, 4, 64, 64)
        loss = model(x).sum()
        model.zero_grad()
        loss.backward()

        grads = {
            name: param.grad.clone()
            for name, param in model.named_parameters()
            if param.grad is not None
        }

        # Warm up
        for _ in range(5):
            packet = {"grads": grads, "version": 0}
            buffer.write(packet)
            buffer.read()

        # Benchmark write
        write_times = []
        for _ in range(num_iterations):
            packet = {"grads": grads, "version": 0}
            start = time.perf_counter()
            buffer.write(packet)
            write_times.append(time.perf_counter() - start)

        # Benchmark read
        read_times = []
        for _ in range(num_iterations):
            buffer.write({"grads": grads, "version": 0})
            start = time.perf_counter()
            result = buffer.read()
            read_times.append(time.perf_counter() - start)

        buffer.unlink()

        return {
            "write_mean_ms": sum(write_times) / len(write_times) * 1000,
            "write_std_ms": (sum((t - sum(write_times) / len(write_times)) ** 2 for t in write_times) / len(write_times)) ** 0.5 * 1000,
            "read_mean_ms": sum(read_times) / len(read_times) * 1000,
            "read_std_ms": (sum((t - sum(read_times) / len(read_times)) ** 2 for t in read_times) / len(read_times)) ** 0.5 * 1000,
        }


def benchmark_shared_gradient_tensor(
    model: nn.Module, num_iterations: int = 100, num_slots: int = 4
) -> dict[str, float]:
    """Benchmark SharedGradientTensor (zero-copy ring buffer)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        shm_path = Path(tmpdir) / "grads_tensor.shm"

        # Create buffer with ring buffer
        buffer = SharedGradientTensor(
            model=model,
            shm_path=shm_path,
            create=True,
            num_slots=num_slots,
        )

        # Generate gradients
        x = torch.randn(1, 4, 64, 64)
        loss = model(x).sum()
        model.zero_grad()
        loss.backward()

        grads = {
            name: param.grad.clone()
            for name, param in model.named_parameters()
            if param.grad is not None
        }

        # Warm up
        for _ in range(5):
            buffer.write(grads)
            buffer.read()

        # Benchmark write
        write_times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            buffer.write(grads)
            write_times.append(time.perf_counter() - start)

        # Benchmark read
        read_times = []
        for _ in range(num_iterations):
            buffer.write(grads)
            start = time.perf_counter()
            result = buffer.read()
            read_times.append(time.perf_counter() - start)

        buffer.unlink()

        return {
            "write_mean_ms": sum(write_times) / len(write_times) * 1000,
            "write_std_ms": (sum((t - sum(write_times) / len(write_times)) ** 2 for t in write_times) / len(write_times)) ** 0.5 * 1000,
            "read_mean_ms": sum(read_times) / len(read_times) * 1000,
            "read_std_ms": (sum((t - sum(read_times) / len(read_times)) ** 2 for t in read_times) / len(read_times)) ** 0.5 * 1000,
        }


class LargerDQNNetwork(nn.Module):
    """Larger DQN network for benchmark scaling tests."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 64, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 12),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.fc(x)


def main():
    print("=" * 70)
    print("SharedGradientBuffer vs SharedGradientTensor Benchmark")
    print("=" * 70)

    models = [
        ("DQNNetwork (small)", DQNNetwork()),
        ("LargerDQNNetwork", LargerDQNNetwork()),
    ]

    num_iterations = 100

    for model_name, model in models:
        num_params = count_parameters(model)
        grad_size_mb = num_params * 4 / 1024 / 1024  # float32

        print(f"\n{'=' * 70}")
        print(f"Model: {model_name}")
        print(f"  Parameters: {num_params:,}")
        print(f"  Gradient size: {grad_size_mb:.2f} MB")
        print(f"  Running {num_iterations} iterations each...")
        print()

        # Benchmark SharedGradientBuffer
        print("  Benchmarking SharedGradientBuffer (torch.save/load)...")
        buffer_results = benchmark_shared_gradient_buffer(model, num_iterations)

        # Benchmark SharedGradientTensor
        print("  Benchmarking SharedGradientTensor (zero-copy)...")
        tensor_results = benchmark_shared_gradient_tensor(model, num_iterations)

        # Results
        print("\n  ┌─────────────────────────┬────────────────────┬────────────────────┐")
        print("  │                         │ SharedGradientBuf  │ SharedGradientTens │")
        print("  │                         │ (torch.save/load)  │ (zero-copy)        │")
        print("  ├─────────────────────────┼────────────────────┼────────────────────┤")
        print(f"  │ Write (mean)            │ {buffer_results['write_mean_ms']:>14.3f} ms │ {tensor_results['write_mean_ms']:>14.3f} ms │")
        print(f"  │ Write (std)             │ {buffer_results['write_std_ms']:>14.3f} ms │ {tensor_results['write_std_ms']:>14.3f} ms │")
        print(f"  │ Read (mean)             │ {buffer_results['read_mean_ms']:>14.3f} ms │ {tensor_results['read_mean_ms']:>14.3f} ms │")
        print(f"  │ Read (std)              │ {buffer_results['read_std_ms']:>14.3f} ms │ {tensor_results['read_std_ms']:>14.3f} ms │")
        print("  └─────────────────────────┴────────────────────┴────────────────────┘")

        # Speedup
        write_speedup = buffer_results["write_mean_ms"] / tensor_results["write_mean_ms"]
        read_speedup = buffer_results["read_mean_ms"] / tensor_results["read_mean_ms"]
        total_buffer = buffer_results["write_mean_ms"] + buffer_results["read_mean_ms"]
        total_tensor = tensor_results["write_mean_ms"] + tensor_results["read_mean_ms"]
        total_speedup = total_buffer / total_tensor

        print(f"\n  📊 Speedup (SharedGradientTensor vs SharedGradientBuffer):")
        print(f"     Write: {write_speedup:.1f}x faster")
        print(f"     Read:  {read_speedup:.1f}x faster")
        print(f"     Total: {total_speedup:.1f}x faster")

        # Throughput
        print(f"\n  ⚡ Throughput (roundtrips/sec):")
        print(f"     SharedGradientBuffer: {1000 / total_buffer:.0f} ops/sec")
        print(f"     SharedGradientTensor: {1000 / total_tensor:.0f} ops/sec")


if __name__ == "__main__":
    main()

