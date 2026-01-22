#!/usr/bin/env python3
"""
Profile worker startup time in the actual distributed training scenario.

This script replicates the exact startup sequence from train_distributed.py
but with detailed timing instrumentation to identify bottlenecks when
multiple workers start simultaneously.
"""

from __future__ import annotations

import os
import sys
import time
import signal
from pathlib import Path
from dataclasses import dataclass
import multiprocessing as mp

mp.set_start_method("spawn", force=True)

# Global start time for all workers
GLOBAL_START = time.time()


def format_time(t: float) -> str:
    """Format time relative to global start."""
    return f"{t - GLOBAL_START:6.2f}s"


def install_exit_handler():
    """Install signal handler that exits cleanly in child processes."""
    def handler(sig, frame):
        os._exit(0)
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


@dataclass
class TimingResult:
    """Timing results from a worker."""
    worker_id: int
    timings: dict[str, float]
    total_time: float


def run_worker_with_timing(
    worker_id: int,
    num_workers: int,
    weights_path: Path,
    result_queue: mp.Queue,
) -> None:
    """Run worker startup with detailed timing."""
    install_exit_handler()
    
    timings = {}
    worker_start = time.time()
    
    def record(name: str):
        timings[name] = time.time() - worker_start
    
    print(f"[{format_time(time.time())}] Worker {worker_id}: Starting...")
    
    # ==========================================================================
    # Phase 1: Heavy imports (this is what each spawned process must do)
    # ==========================================================================
    t0 = time.time()
    import torch
    record("import_torch")
    print(f"[{format_time(time.time())}] Worker {worker_id}: torch imported ({time.time()-t0:.2f}s)")
    
    t1 = time.time()
    from mario_rl.metrics import DDQNMetrics
    from mario_rl.metrics import MetricLogger
    record("import_metrics")
    
    t2 = time.time()
    from mario_rl.distributed.shm_heartbeat import SharedHeartbeat
    from mario_rl.distributed.training_worker import TrainingWorker
    from mario_rl.training.shared_gradient_tensor import attach_tensor_buffer
    record("import_distributed")
    
    t3 = time.time()
    from mario_rl.environment.snapshot_wrapper import create_snapshot_mario_env
    record("import_environment")
    
    print(f"[{format_time(time.time())}] Worker {worker_id}: All imports done ({time.time()-t1:.2f}s)")
    
    # ==========================================================================
    # Phase 2: Device setup
    # ==========================================================================
    t4 = time.time()
    from mario_rl.core.device import assign_device
    device_str = assign_device(process_id=worker_id + 1, num_processes=num_workers + 1)
    device = torch.device(device_str)
    record("device_setup")
    print(f"[{format_time(time.time())}] Worker {worker_id}: Device={device} ({time.time()-t4:.2f}s)")
    
    # ==========================================================================
    # Phase 3: Environment creation
    # ==========================================================================
    t5 = time.time()
    env = create_snapshot_mario_env(
        level="random",  # Test with random levels (was 12s before lazy init fix!)
        render_frames=False,
        hotspot_path=None,
        checkpoint_interval=500,
        max_restores_without_progress=3,
        enabled=True,
        sum_rewards=True,
        action_history_len=4,
        input_type="frames",
    )
    record("create_environment")
    print(f"[{format_time(time.time())}] Worker {worker_id}: Environment created ({time.time()-t5:.2f}s)")
    
    # ==========================================================================
    # Phase 4: Model and learner creation (skip init - will load weights)
    # ==========================================================================
    t6 = time.time()
    from mario_rl.agent.ddqn_net import DoubleDQN, set_skip_weight_init
    from mario_rl.learners.ddqn import DDQNLearner
    
    # Skip expensive orthogonal init - we'll load pre-initialized weights
    set_skip_weight_init(True)
    model = DoubleDQN(
        input_shape=(4, 64, 64),
        num_actions=7,
        feature_dim=512,
        hidden_dim=256,
        dropout=0.1,
        action_history_len=4,
        danger_prediction_bins=16,
    )
    set_skip_weight_init(False)
    record("create_model_cpu")
    print(f"[{format_time(time.time())}] Worker {worker_id}: Model created on CPU (skip init) ({time.time()-t6:.2f}s)")
    
    t7 = time.time()
    model = model.to(device)
    record("model_to_device")
    print(f"[{format_time(time.time())}] Worker {worker_id}: Model moved to {device} ({time.time()-t7:.2f}s)")
    
    # Load pre-initialized weights from main process
    t_load = time.time()
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    record("load_weights")
    print(f"[{format_time(time.time())}] Worker {worker_id}: Loaded pre-init weights ({time.time()-t_load:.2f}s)")
    
    t8 = time.time()
    learner = DDQNLearner(model=model, gamma=0.99, n_step=10, entropy_coef=0.01)
    record("create_learner")
    
    # ==========================================================================
    # Phase 5: Training worker creation
    # ==========================================================================
    t9 = time.time()
    worker = TrainingWorker(
        env=env,
        learner=learner,
        buffer_capacity=25_000,
        batch_size=128,
        n_step=10,
        gamma=0.99,
        alpha=0.6,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_steps=500_000,
        logger=None,
        flush_every=320,
        action_history_len=4,
        danger_prediction_bins=16,
    )
    record("create_training_worker")
    print(f"[{format_time(time.time())}] Worker {worker_id}: TrainingWorker created ({time.time()-t9:.2f}s)")
    
    # ==========================================================================
    # Phase 6: Weight sync (simulates actual startup)
    # ==========================================================================
    t10 = time.time()
    worker.sync_weights(weights_path)
    record("sync_weights")
    print(f"[{format_time(time.time())}] Worker {worker_id}: Weights synced ({time.time()-t10:.2f}s)")
    
    # Cleanup
    env.close()
    
    total_time = time.time() - worker_start
    record("total")
    
    print(f"[{format_time(time.time())}] Worker {worker_id}: READY (total: {total_time:.2f}s)")
    
    result_queue.put(TimingResult(
        worker_id=worker_id,
        timings=timings,
        total_time=total_time,
    ))


def main():
    import click
    
    @click.command()
    @click.option("--workers", "-w", default=4, help="Number of workers to start")
    @click.option("--level", "-l", default="1,1", help="Level to use (1,1 or random)")
    def run(workers: int, level: str):
        """Profile parallel worker startup."""
        global GLOBAL_START
        GLOBAL_START = time.time()
        
        print("=" * 70)
        print(f"PROFILING PARALLEL WORKER STARTUP ({workers} workers)")
        print("=" * 70)
        print()
        
        # Setup (same as train_distributed.py)
        from mario_rl.agent.ddqn_net import DoubleDQN
        
        # Create temp directory and weights file
        import tempfile
        import shutil
        
        tmp_dir = Path(tempfile.mkdtemp(prefix="mario_profile_"))
        weights_path = tmp_dir / "weights.pt"
        
        # Create reference model and save weights
        print(f"[{format_time(time.time())}] Main: Creating reference model...")
        ref_model = DoubleDQN(
            input_shape=(4, 64, 64),
            num_actions=7,
            feature_dim=512,
            hidden_dim=256,
            dropout=0.1,
            action_history_len=4,
            danger_prediction_bins=16,
        )
        
        import torch
        torch.save(ref_model.state_dict(), weights_path)
        del ref_model
        print(f"[{format_time(time.time())}] Main: Weights saved to {weights_path}")
        
        # Result queue
        result_queue = mp.Queue()
        
        # Start all workers simultaneously
        print()
        print(f"[{format_time(time.time())}] Main: Starting {workers} workers...")
        print("-" * 70)
        
        processes = []
        start_time = time.time()
        
        for i in range(workers):
            p = mp.Process(
                target=run_worker_with_timing,
                args=(i, workers, weights_path, result_queue),
            )
            p.start()
            processes.append(p)
        
        # Wait for all workers
        for p in processes:
            p.join()
        
        wall_clock_time = time.time() - start_time
        
        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        
        results.sort(key=lambda r: r.worker_id)
        
        # Print summary
        print()
        print("-" * 70)
        print("TIMING SUMMARY")
        print("-" * 70)
        print()
        
        # Aggregate timings
        all_timings = {}
        for r in results:
            for name, t in r.timings.items():
                if name not in all_timings:
                    all_timings[name] = []
                all_timings[name].append(t)
        
        print(f"{'Phase':<25} {'Min':>8} {'Max':>8} {'Avg':>8}")
        print("-" * 55)
        
        phases = [
            ("import_torch", "Import torch"),
            ("import_metrics", "Import metrics"),
            ("import_distributed", "Import distributed"),
            ("import_environment", "Import environment"),
            ("device_setup", "Device setup"),
            ("create_environment", "Create environment"),
            ("create_model_cpu", "Create model (skip init)"),
            ("model_to_device", "Model to device"),
            ("load_weights", "Load pre-init weights"),
            ("create_learner", "Create learner"),
            ("create_training_worker", "Create TrainingWorker"),
            ("sync_weights", "Sync weights"),
            ("total", "TOTAL"),
        ]
        
        for key, label in phases:
            if key in all_timings:
                times = all_timings[key]
                print(f"{label:<25} {min(times):>7.2f}s {max(times):>7.2f}s {sum(times)/len(times):>7.2f}s")
        
        print()
        print(f"Wall-clock time for all {workers} workers: {wall_clock_time:.2f}s")
        print(f"Sum of individual worker times: {sum(r.total_time for r in results):.2f}s")
        print(f"Parallelism efficiency: {sum(r.total_time for r in results) / wall_clock_time / workers * 100:.1f}%")
        
        # Cleanup
        shutil.rmtree(tmp_dir)
        
        print()
        print("=" * 70)
    
    run()


if __name__ == "__main__":
    main()
