#!/usr/bin/env python3
"""
Benchmark worker inner loop operations to identify heartbeat delays.

Times:
1. collect_steps(N) - environment stepping and experience collection
2. compute_and_send_gradients() - gradient computation and transfer
"""
from __future__ import annotations

import time
from pathlib import Path

import torch

from mario_rl.core.config import WorkerConfig, BufferConfig, ExplorationConfig, SnapshotConfig
from mario_rl.environment.factory import create_env
from mario_rl.training.ddqn_worker import DDQNWorker
from mario_rl.training.shared_gradients import SharedGradientBuffer


def benchmark_collect_steps(num_iterations: int = 10, steps_per_collection: int = 64) -> dict[str, float]:
    """Benchmark collect_steps operation."""
    print(f"\n{'='*70}")
    print(f"Benchmarking collect_steps({steps_per_collection})")
    print(f"{'='*70}")
    
    # Create minimal worker config
    buffer_config = BufferConfig(
        capacity=10000,
        batch_size=128,
        n_step=3,
        gamma=0.99,
        alpha=0.0,  # No PER for simplicity
    )
    
    exploration_config = ExplorationConfig(
        epsilon_start=1.0,
        epsilon_end=0.01,
        decay_steps=1000000,
    )
    
    snapshot_config = SnapshotConfig(enabled=False)
    
    worker_config = WorkerConfig(
        worker_id=0,
        level="random",
        steps_per_collection=steps_per_collection,
        train_steps=0,  # Don't train, just collect
        buffer=buffer_config,
        exploration=exploration_config,
        snapshot=snapshot_config,
    )
    
    # Create temporary shared memory buffer (not used for collect_steps)
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        shm_dir = Path(tmpdir)
        gradient_buffer = SharedGradientBuffer(
            worker_id=0,
            buffer_size=16 * 1024 * 1024,
            create=True,
            shm_dir=shm_dir,
        )
        
        # Create worker
        weights_path = Path("/tmp/test_weights.pth")
        worker = DDQNWorker(
            config=worker_config,
            weights_path=weights_path,
            gradient_buffer=gradient_buffer,
            ui_queue=None,
        )
        
        # Warm up
        print("Warming up...")
        worker.collect_steps(steps_per_collection)
        
        # Benchmark
        print(f"Running {num_iterations} iterations...")
        times = []
        for i in range(num_iterations):
            start = time.perf_counter()
            worker.collect_steps(steps_per_collection)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(f"  Iteration {i+1}: {elapsed*1000:.1f} ms")
        
        mean_time = sum(times) / len(times)
        std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
        
        print(f"\nResults:")
        print(f"  Mean: {mean_time*1000:.1f} ms")
        print(f"  Std:  {std_time*1000:.1f} ms")
        print(f"  Min:  {min(times)*1000:.1f} ms")
        print(f"  Max:  {max(times)*1000:.1f} ms")
        print(f"  Per step: {mean_time/steps_per_collection*1000:.2f} ms")
        
        gradient_buffer.unlink()
        
        return {
            "mean_ms": mean_time * 1000,
            "std_ms": std_time * 1000,
            "min_ms": min(times) * 1000,
            "max_ms": max(times) * 1000,
            "per_step_ms": mean_time / steps_per_collection * 1000,
        }


def benchmark_compute_and_send_gradients(
    num_iterations: int = 20,
    batch_size: int = 128,
    train_steps: int = 8,
) -> dict[str, float]:
    """Benchmark compute_and_send_gradients operation."""
    print(f"\n{'='*70}")
    print(f"Benchmarking compute_and_send_gradients()")
    print(f"  batch_size={batch_size}, train_steps={train_steps}")
    print(f"{'='*70}")
    
    # Create worker config
    buffer_config = BufferConfig(
        capacity=10000,
        batch_size=batch_size,
        n_step=3,
        gamma=0.99,
        alpha=0.0,  # No PER for simplicity
    )
    
    exploration_config = ExplorationConfig(
        epsilon_start=1.0,
        epsilon_end=0.01,
        decay_steps=1000000,
    )
    
    snapshot_config = SnapshotConfig(enabled=False)
    
    worker_config = WorkerConfig(
        worker_id=0,
        level="random",
        steps_per_collection=64,
        train_steps=train_steps,
        buffer=buffer_config,
        exploration=exploration_config,
        snapshot=snapshot_config,
    )
    
    # Create temporary shared memory buffer
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        shm_dir = Path(tmpdir)
        gradient_buffer = SharedGradientBuffer(
            worker_id=0,
            buffer_size=16 * 1024 * 1024,
            create=True,
            shm_dir=shm_dir,
        )
        
        # Create worker
        weights_path = Path("/tmp/test_weights.pth")
        worker = DDQNWorker(
            config=worker_config,
            weights_path=weights_path,
            gradient_buffer=gradient_buffer,
            ui_queue=None,
        )
        
        # Fill buffer with enough data
        print("Filling buffer...")
        while len(worker.buffer) < batch_size:
            worker.collect_steps(64)
        
        print(f"Buffer size: {len(worker.buffer)}")
        
        # Warm up
        print("Warming up...")
        for _ in range(3):
            if len(worker.buffer) >= batch_size:
                worker.compute_and_send_gradients()
        
        # Benchmark individual gradient computations
        print(f"Running {num_iterations} iterations...")
        times = []
        for i in range(num_iterations):
            # Ensure buffer has enough data
            while len(worker.buffer) < batch_size:
                worker.collect_steps(32)
            
            start = time.perf_counter()
            worker.compute_and_send_gradients()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(f"  Iteration {i+1}: {elapsed*1000:.1f} ms")
        
        mean_time = sum(times) / len(times)
        std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
        
        print(f"\nResults:")
        print(f"  Mean: {mean_time*1000:.1f} ms")
        print(f"  Std:  {std_time*1000:.1f} ms")
        print(f"  Min:  {min(times)*1000:.1f} ms")
        print(f"  Max:  {max(times)*1000:.1f} ms")
        
        # Estimate total time for train_steps iterations
        total_time_ms = mean_time * train_steps * 1000
        print(f"\nEstimated time for {train_steps} train_steps: {total_time_ms:.1f} ms")
        
        gradient_buffer.unlink()
        
        return {
            "mean_ms": mean_time * 1000,
            "std_ms": std_time * 1000,
            "min_ms": min(times) * 1000,
            "max_ms": max(times) * 1000,
            "total_for_train_steps_ms": total_time_ms,
        }


def main():
    print("="*70)
    print("Worker Inner Loop Operations Benchmark")
    print("="*70)
    
    # Benchmark collect_steps
    collect_results = benchmark_collect_steps(num_iterations=10, steps_per_collection=64)
    
    # Benchmark compute_and_send_gradients
    grad_results = benchmark_compute_and_send_gradients(
        num_iterations=20,
        batch_size=128,
        train_steps=8,
    )
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n1. collect_steps(64):")
    print(f"   Mean: {collect_results['mean_ms']:.1f} ms")
    print(f"   Per step: {collect_results['per_step_ms']:.2f} ms")
    print(f"   Could delay heartbeat: {'YES' if collect_results['mean_ms'] > 10000 else 'NO'} (>10s)")
    
    print(f"\n2. compute_and_send_gradients() (single):")
    print(f"   Mean: {grad_results['mean_ms']:.1f} ms")
    print(f"   For 8 train_steps: {grad_results['total_for_train_steps_ms']:.1f} ms")
    print(f"   Could delay heartbeat: {'YES' if grad_results['total_for_train_steps_ms'] > 10000 else 'NO'} (>10s)")
    
    print(f"\n3. Combined worst case:")
    total_worst = collect_results['max_ms'] + grad_results['total_for_train_steps_ms']
    print(f"   collect_steps(max) + train_steps(8): {total_worst:.1f} ms")
    print(f"   Could delay heartbeat: {'YES' if total_worst > 10000 else 'NO'} (>10s)")


if __name__ == "__main__":
    main()

