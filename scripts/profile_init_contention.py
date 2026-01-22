#!/usr/bin/env python3
"""
Profile to confirm orthogonal initialization causes CPU contention.
"""

import os
import time
import multiprocessing as mp

mp.set_start_method("spawn", force=True)


def worker_orthogonal(worker_id: int, result_queue: mp.Queue):
    """Worker that uses orthogonal init."""
    import torch
    from torch import nn
    import numpy as np
    
    def layer_init_orthogonal(layer, std=np.sqrt(2)):
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, std)
        return layer
    
    start = time.time()
    
    # Create layers similar to DoubleDQN (multiple large Linear layers)
    layers = []
    for _ in range(8):  # DoubleDQN has ~8 large linear layers (online + target)
        layer = nn.Linear(2048, 512)
        layer_init_orthogonal(layer)
        layers.append(layer)
    
    elapsed = time.time() - start
    print(f"Worker {worker_id} (orthogonal): {elapsed:.2f}s")
    result_queue.put(("orthogonal", worker_id, elapsed))


def worker_kaiming(worker_id: int, result_queue: mp.Queue):
    """Worker that uses kaiming init (fast)."""
    import torch
    from torch import nn
    
    def layer_init_kaiming(layer):
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
        return layer
    
    start = time.time()
    
    # Same layers but with kaiming init
    layers = []
    for _ in range(8):
        layer = nn.Linear(2048, 512)
        layer_init_kaiming(layer)
        layers.append(layer)
    
    elapsed = time.time() - start
    print(f"Worker {worker_id} (kaiming): {elapsed:.2f}s")
    result_queue.put(("kaiming", worker_id, elapsed))


def run_test(name: str, worker_fn, num_workers: int):
    """Run parallel workers and measure time."""
    print(f"\n{'='*60}")
    print(f"Testing {name} with {num_workers} parallel workers")
    print(f"{'='*60}")
    
    result_queue = mp.Queue()
    processes = []
    
    start = time.time()
    for i in range(num_workers):
        p = mp.Process(target=worker_fn, args=(i, result_queue))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    wall_time = time.time() - start
    
    # Collect results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    
    worker_times = [r[2] for r in results]
    print(f"\nWall-clock time: {wall_time:.2f}s")
    print(f"Avg worker time: {sum(worker_times)/len(worker_times):.2f}s")
    print(f"Max worker time: {max(worker_times):.2f}s")
    
    return wall_time


def main():
    num_workers = 4
    
    print("Testing CPU contention in weight initialization")
    print("=" * 60)
    print(f"Each worker creates 8 Linear(2048, 512) layers")
    print(f"Running {num_workers} workers in parallel")
    
    # First, baseline with single worker
    print("\n" + "="*60)
    print("BASELINE: Single worker")
    print("="*60)
    
    result_queue = mp.Queue()
    
    start = time.time()
    worker_orthogonal(0, result_queue)
    single_orthogonal = time.time() - start
    
    start = time.time() 
    worker_kaiming(0, result_queue)
    single_kaiming = time.time() - start
    
    print(f"\nSingle worker orthogonal: {single_orthogonal:.2f}s")
    print(f"Single worker kaiming: {single_kaiming:.2f}s")
    
    # Now parallel
    parallel_orthogonal = run_test("ORTHOGONAL init", worker_orthogonal, num_workers)
    parallel_kaiming = run_test("KAIMING init", worker_kaiming, num_workers)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n{'Method':<20} {'Single':>10} {'Parallel':>10} {'Slowdown':>10}")
    print("-" * 50)
    print(f"{'Orthogonal':<20} {single_orthogonal:>9.2f}s {parallel_orthogonal:>9.2f}s {parallel_orthogonal/single_orthogonal:>9.1f}x")
    print(f"{'Kaiming':<20} {single_kaiming:>9.2f}s {parallel_kaiming:>9.2f}s {parallel_kaiming/single_kaiming:>9.1f}x")
    print()
    print(f"Switching to Kaiming would save: {parallel_orthogonal - parallel_kaiming:.1f}s")


if __name__ == "__main__":
    main()
