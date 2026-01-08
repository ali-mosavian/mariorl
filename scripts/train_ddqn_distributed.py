#!/usr/bin/env python3
from __future__ import annotations

"""
Distributed DDQN training with async gradient updates.

Workers compute gradients locally and send them to the learner.
Similar to Gorila DQN / APPO architecture.

Architecture
============

┌─────────────────────────────────────────────────────────────────────────┐
│                           MAIN PROCESS                                   │
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                      Gradient Queue                              │   │
│   │              (workers → learner, ~2MB per packet)                │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
          │                    │                    │
          │                    │                    │
          ▼                    ▼                    ▼
   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
   │  WORKER 0   │      │  WORKER 1   │      │  WORKER N   │
   │             │      │             │      │             │
   │ ε = 0.4^1   │      │ ε = 0.4^2   │      │ ε = 0.4^N   │
   │             │      │             │      │             │
   │ 1. Collect  │      │ 1. Collect  │      │ 1. Collect  │
   │    steps    │      │    steps    │      │    steps    │
   │             │      │             │      │             │
   │ 2. Sample   │      │ 2. Sample   │      │ 2. Sample   │
   │    batch    │      │    batch    │      │    batch    │
   │             │      │             │      │             │
   │ 3. Compute  │      │ 3. Compute  │      │ 3. Compute  │
   │    DQN loss │      │    DQN loss │      │    DQN loss │
   │             │      │             │      │             │
   │ 4. Backward │      │ 4. Backward │      │ 4. Backward │
   │    pass     │      │    pass     │      │    pass     │
   │             │      │             │      │             │
   │ 5. Send     │      │ 5. Send     │      │ 5. Send     │
   │    gradients│      │    gradients│      │    gradients│
   └──────┬──────┘      └──────┬──────┘      └──────┬──────┘
          │                    │                    │
          │     GRADIENT QUEUE (small, ~2MB each)   │
          └────────────────────┼────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      LEARNER        │
                    │                     │
                    │  1. Receive grads   │
                    │  2. Accumulate      │
                    │  3. Average         │
                    │  4. Clip            │
                    │  5. Optimizer step  │
                    │  6. Soft update     │
                    │  7. Save weights    │
                    └─────────────────────┘

Comparison with Standard DDQN
=============================

    Standard DDQN:
    Single process: collect → store → sample → forward → backward → step

    Distributed DDQN:
    Worker: collect → sample → forward → backward → send grads (async)
    Learner: receive grads → accumulate → step → soft update

    Benefits:
    - ~Nx faster with N workers
    - Distributed computation (workers do forward+backward)
    - ~4x less data through IPC (grads ~2MB vs experiences ~8MB)
    - GPU utilization for training
    - Diverse exploration (different epsilons per worker)

Usage:
    uv run mario-train-ddqn-dist --workers 4 --level random
    uv run mario-train-ddqn-dist --workers 8 --level 1,1 --accumulate-grads 4
    uv run mario-train-ddqn-dist --workers 4 --no-ui  # Disable ncurses UI
"""

import os
import sys
import time
import atexit
import shutil
import signal
import threading
import multiprocessing as mp
from typing import cast
from pathlib import Path
from typing import Literal
from datetime import datetime

# Use 'spawn' start method for cleaner CUDA handling (avoids fork issues)
# Must be done before importing torch or creating any processes
mp.set_start_method("spawn", force=True)

from multiprocessing import Queue
from multiprocessing import Process

# Ensure unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import click

sys.path.insert(0, str(Path(__file__).parent.parent))

from mario_rl.core.config import BufferConfig
from mario_rl.core.config import WorkerConfig
from mario_rl.core.config import SnapshotConfig
from mario_rl.core.config import ExplorationConfig
from mario_rl.training.training_ui import TrainingUI
from mario_rl.training.ddqn_worker import run_ddqn_worker
from mario_rl.training.ddqn_learner import run_ddqn_learner
from mario_rl.core.config import LevelType as ConfigLevelType
from mario_rl.training.shared_gradients import SharedHeartbeats
from mario_rl.training.shared_gradient_tensor import SharedGradientTensorPool
from mario_rl.agent.ddqn_net import DoubleDQN

LevelType = Literal["sequential", "random"] | tuple[int, int]


def parse_level(level_str: str) -> LevelType:
    """Parse level string into LevelType."""
    if level_str == "random":
        return "random"
    elif level_str == "sequential":
        return "sequential"
    else:
        parts = level_str.split(",")
        if len(parts) == 2:
            return (int(parts[0]), int(parts[1]))
        raise ValueError(f"Invalid level format: {level_str}")


def run_training_ui(num_workers: int, ui_queue: Queue) -> None:
    """Run the training UI in a separate process."""
    ui = TrainingUI(num_workers=num_workers, ui_queue=ui_queue, use_ppo=False)
    ui.run()


def run_worker_silent(
    config: WorkerConfig,
    weights_path: Path,
    shm_path: Path,
    ui_queue: "Queue[object] | None" = None,
    heartbeat_path: Path | None = None,
    crash_log_dir: Path | None = None,
) -> None:
    """Run worker with stdout/stderr suppressed (for UI mode)."""
    import os
    import sys
    import warnings

    # Suppress all warnings
    warnings.filterwarnings("ignore")

    # Redirect stdout/stderr to /dev/null
    devnull = open(os.devnull, "w")
    sys.stdout = devnull
    sys.stderr = devnull

    # Pass shm_path directly - worker will attach internally
    run_ddqn_worker(config, weights_path, shm_path, ui_queue, heartbeat_path, crash_log_dir)


def run_learner_silent(
    weights_path: Path,
    save_dir: Path,
    shm_dir: Path,
    num_workers: int,
    restore_snapshot: bool = False,
    snapshot_path: Path | None = None,
    **kwargs,
) -> None:
    """Run learner with stdout/stderr suppressed (for UI mode)."""
    import os
    import sys
    import warnings
    import traceback

    # Suppress all warnings
    warnings.filterwarnings("ignore")

    # Redirect stdout/stderr to /dev/null
    devnull = open(os.devnull, "w")
    sys.stdout = devnull
    sys.stderr = devnull

    # Now run the actual learner with crash logging
    try:
        run_ddqn_learner(
            weights_path,
            save_dir,
            shm_dir,
            num_workers,
            restore_snapshot=restore_snapshot,
            snapshot_path=snapshot_path,
            **kwargs,
        )
    except Exception as e:
        # Log crash to file (stdout/stderr are /dev/null)
        crash_log_path = save_dir / "crash_logs" / "learner_crash.log"
        crash_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(crash_log_path, "w") as f:
            f.write(f"LEARNER CRASHED\n")
            f.write(f"Error: {e}\n")
            f.write(f"Type: {type(e).__name__}\n")
            f.write(f"\nFull traceback:\n")
            f.write(traceback.format_exc())
        
        # Re-raise to exit process
        raise


def monitor_workers(
    worker_list: list[tuple[Process, WorkerConfig]],
    heartbeats: SharedHeartbeats,
    weights_path: Path,
    shm_paths: list[Path],
    ui_queue: Queue | None,
    worker_target,
    stop_event: threading.Event,
    monitor_log_file: Path | None = None,
    crash_log_dir: Path | None = None,
) -> None:
    """Monitor worker health and restart crashed/zombie workers."""
    from datetime import datetime
    
    def log_monitor(msg: str):
        """Log to file and/or UI queue (avoid stderr when UI is active)."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] [MONITOR] {msg}"
        
        # Write to log file
        if monitor_log_file is not None:
            try:
                with open(monitor_log_file, "a") as f:
                    f.write(log_line + "\n")
                    f.flush()
            except Exception:
                pass
        
        # Send to UI queue if available, otherwise print to stderr
        if ui_queue is not None:
            try:
                from mario_rl.training.training_ui import UIMessage, MessageType
                ui_queue.put_nowait(
                    UIMessage(
                        msg_type=MessageType.SYSTEM_LOG,
                        source_id=-1,
                        data={"text": msg},
                    )
                )
            except Exception:
                pass
        else:
            # No UI - print to stderr
            print(log_line, file=sys.stderr, flush=True)
    
    while not stop_event.is_set():
        # Check each worker process
        current_time = time.time()
        
        # Read all heartbeat timestamps from shared memory (no blocking!)
        worker_heartbeats = heartbeats.get_all()
        
        for i, (proc, config) in enumerate(worker_list):
            worker_id = config.worker_id
            last_heartbeat = worker_heartbeats[worker_id]
            
            # Forward heartbeat to UI if recent (updated in last 30s)
            if last_heartbeat > 0 and current_time - last_heartbeat < 30 and ui_queue is not None:
                try:
                    from mario_rl.training.training_ui import UIMessage, MessageType
                    ui_queue.put_nowait(
                        UIMessage(
                            msg_type=MessageType.WORKER_HEARTBEAT,
                            source_id=worker_id,
                            data={"status": "alive", "worker_id": worker_id},
                        )
                    )
                except Exception:
                    pass
            
            # Check if process is dead (zombie or terminated)
            if not proc.is_alive():
                exit_code = proc.exitcode
                exit_reason = {
                    0: "normal exit",
                    -1: "killed (SIGHUP)",
                    -2: "interrupted (SIGINT)",
                    -9: "killed (SIGKILL)",
                    -11: "segmentation fault (SIGSEGV)",
                    -15: "terminated (SIGTERM)",
                }.get(exit_code, f"unknown (code {exit_code})")
                
                log_monitor(f"Worker {worker_id} is DEAD")
                log_monitor(f"  Exit code: {exit_code} ({exit_reason})")
                log_monitor(f"  Restarting worker {worker_id}...")
                
                # Start a new worker process
                new_proc = Process(
                    target=worker_target,
                    args=(config, weights_path, shm_paths[worker_id]),
                    kwargs={"ui_queue": ui_queue, "heartbeat_path": heartbeats.shm_path, "crash_log_dir": crash_log_dir},
                    daemon=True,
                )
                new_proc.start()
                worker_list[i] = (new_proc, config)
                log_monitor(f"  Worker {worker_id} restarted with PID {new_proc.pid}")
                continue
            
            # Check for stalled workers (no heartbeat in 120 seconds)
            if last_heartbeat > 0:
                time_since_heartbeat = current_time - last_heartbeat
                if time_since_heartbeat > 120:
                    log_monitor(f"Worker {worker_id} is STALLED")
                    log_monitor(f"  Last heartbeat: {time_since_heartbeat:.0f}s ago")
                    log_monitor(f"  PID: {proc.pid}")
                    
                    # Send SIGUSR1 to trigger stack trace dump
                    log_monitor(f"  Requesting stack trace dump...")
                    try:
                        import signal
                        import os
                        os.kill(proc.pid, signal.SIGUSR1)
                        time.sleep(0.5)  # Give worker time to write trace
                        log_monitor(f"  Stack trace requested (check worker_{worker_id}_stack.log)")
                    except Exception as e:
                        log_monitor(f"  Failed to request stack trace: {e}")
                    
                    log_monitor(f"  Terminating worker {worker_id}...")
                    
                    # Terminate and restart
                    proc.terminate()
                    proc.join(timeout=5)
                    if proc.is_alive():
                        log_monitor(f"  Worker {worker_id} didn't terminate, forcing kill...")
                        proc.kill()
                    
                    # Start a new worker process
                    new_proc = Process(
                        target=worker_target,
                        args=(config, weights_path, shm_paths[worker_id]),
                        kwargs={"ui_queue": ui_queue, "heartbeat_path": heartbeats.shm_path, "crash_log_dir": crash_log_dir},
                        daemon=True,
                    )
                    new_proc.start()
                    worker_list[i] = (new_proc, config)
                    log_monitor(f"  Worker {worker_id} restarted with PID {new_proc.pid}")
        
        # Sleep before next check
        time.sleep(10)


@click.command()
@click.option("--workers", "-w", default=4, help="Number of worker processes")
@click.option("--level", "-l", default="random", help="Level: 'random', 'sequential', or 'W,S'")
@click.option("--save-dir", default="checkpoints", help="Directory for saving checkpoints")
# Learning hyperparameters
@click.option("--lr", default=2.5e-4, help="Initial learning rate")
@click.option("--lr-end", default=1e-5, help="Final learning rate")
@click.option("--gamma", default=0.99, help="Discount factor")
@click.option("--n-step", default=3, help="N-step returns")
@click.option("--tau", default=0.001, help="Soft update coefficient (lower = more stable)")
# Worker settings
@click.option("--local-buffer-size", default=10_000, help="Local replay buffer per worker")
@click.option("--batch-size", default=32, help="Training batch size per worker")
@click.option("--collect-steps", default=64, help="Steps to collect per cycle")
@click.option("--train-steps", default=4, help="Gradient computations per cycle")
# Learner settings
@click.option("--accumulate-grads", default=1, help="Gradients to accumulate before update")
# Epsilon settings (per worker)
@click.option("--eps-base", default=0.4, help="Base for per-worker epsilon (ε = base^(1+i/N))")
@click.option("--eps-decay-steps", default=1_000_000, help="Steps for epsilon decay (1M = ~1hr with 16 workers)")
# Stability settings
@click.option("--q-clip", default=100.0, help="Clip Q-values to [-x, x] to prevent explosion (0 to disable)")
@click.option("--loss-threshold", default=1000.0, help="Skip gradient if loss exceeds this threshold")
# Other
@click.option("--total-steps", default=2_000_000, help="Total training steps (for LR schedule)")
@click.option("--max-grad-norm", default=10.0, help="Maximum gradient norm")
@click.option("--weight-decay", default=1e-4, help="L2 regularization")
@click.option("--no-ui", is_flag=True, help="Disable ncurses UI")
@click.option("--resume", is_flag=True, help="Resume from latest checkpoint")
@click.option("--restore-snapshot", type=str, default=None, help="Restore from specific snapshot file")
@click.option("--no-game-snapshots", is_flag=True, help="Disable game state snapshots (save/restore on death)")
@click.option("--no-per", is_flag=True, help="Disable Prioritized Experience Replay (use uniform sampling)")
@click.option("--reward-norm", type=click.Choice(["none", "scale", "running"]), default="none",
              help="Reward normalization: 'none' (recommended - env normalizes), 'scale', 'running'")
@click.option("--reward-scale", default=1.0, help="Scale factor when reward-norm=scale")
@click.option("--reward-clip", default=5.0, help="Clip rewards to [-x, x] to prevent instability (0 to disable)")
@click.option("--entropy-coef", default=0.01, help="Entropy regularization coefficient (encourages exploration)")
# Network architecture
@click.option("--dreamer", is_flag=True, help="Use Dreamer-style network (encoder + latent Q-network)")
@click.option("--latent-dim", default=128, help="Latent dimension for Dreamer network")
def main(
    workers: int,
    level: str,
    save_dir: str,
    lr: float,
    lr_end: float,
    gamma: float,
    n_step: int,
    tau: float,
    local_buffer_size: int,
    batch_size: int,
    collect_steps: int,
    train_steps: int,
    accumulate_grads: int,
    eps_base: float,
    eps_decay_steps: int,
    q_clip: float,
    loss_threshold: float,
    total_steps: int,
    max_grad_norm: float,
    weight_decay: float,
    no_ui: bool,
    resume: bool,
    restore_snapshot: str | None,
    no_game_snapshots: bool,
    no_per: bool,
    reward_norm: str,
    reward_scale: float,
    reward_clip: float,
    entropy_coef: float,
    dreamer: bool,
    latent_dim: int,
) -> None:
    """Train Mario using Distributed DDQN with async gradient updates."""
    # Parse level
    level_type = parse_level(level)

    # Create save directory
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = Path(save_dir) / f"ddqn_dist_{timestamp}"

    # Parse snapshot path if provided
    snapshot_path: Path | None = None
    if restore_snapshot:
        snapshot_path = Path(restore_snapshot)
        # If it's a directory, look for snapshot.pt inside
        if snapshot_path.is_dir():
            snapshot_path = snapshot_path / "snapshot.pt"
        # Use the parent directory as run_dir
        run_dir = snapshot_path.parent
        print(f"Restoring from snapshot: {snapshot_path}")

    # Check for resume (only if not restoring from specific snapshot)
    elif resume:
        # Find all checkpoint directories (sort by name which contains timestamp)
        checkpoint_dirs = sorted(Path(save_dir).glob("ddqn_dist_*"), key=lambda p: p.name, reverse=True)
        
        # Prefer directories that have both weights.pt AND snapshot.pt
        best_checkpoint = None
        for ckpt_dir in checkpoint_dirs:
            if (ckpt_dir / "weights.pt").exists():
                if (ckpt_dir / "snapshot.pt").exists():
                    best_checkpoint = ckpt_dir
                    break  # Found one with snapshot, use it
                elif best_checkpoint is None:
                    best_checkpoint = ckpt_dir  # Fallback to weights-only if no better option
        
        if best_checkpoint:
            run_dir = best_checkpoint
            print(f"Found checkpoint: {run_dir}")
            print(f"  Available checkpoints: {[d.name for d in checkpoint_dirs[:5]]}")  # Show top 5
            # Check if snapshot exists
            if (run_dir / "snapshot.pt").exists():
                snapshot_path = run_dir / "snapshot.pt"
                snapshot_size = snapshot_path.stat().st_size / 1024 / 1024
                print(f"  → Snapshot found: {snapshot_path.name} ({snapshot_size:.1f}MB)")
            else:
                print(f"  → WARNING: No snapshot.pt found, will only load weights")
                print(f"    (Training state like update_count will reset to 0)")

    # Load initial_steps from snapshot if resuming
    initial_steps = 0
    if snapshot_path is not None and snapshot_path.exists():
        try:
            import torch
            snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=False)
            initial_steps = snapshot.get("total_timesteps_collected", 0)
            print(f"Resuming from timestep: {initial_steps:,}")
        except Exception as e:
            print(f"Warning: Could not load initial_steps from snapshot: {e}")

    run_dir.mkdir(parents=True, exist_ok=True)
    weights_path = run_dir / "weights.pt"
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Create crash logs directory
    crash_log_dir = run_dir / "crash_logs"
    crash_log_dir.mkdir(exist_ok=True)
    monitor_log_file = crash_log_dir / "monitor.log"

    # Print config (will be hidden by UI if enabled)
    if no_ui:
        print("=" * 70)
        print("Distributed DDQN Training (Async Gradients)")
        print("=" * 70)
        print(f"  Workers: {workers}")
        print(f"  Level: {level}")
        print(f"  Save dir: {run_dir}")
        print(f"  LR: {lr} → {lr_end} (cosine)")
        print(f"  Gamma: {gamma}, N-step: {n_step}")
        print(f"  Tau: {tau}, Entropy coef: {entropy_coef}")
        print(f"  Q-clip: {q_clip}, Loss threshold: {loss_threshold}, Reward clip: {reward_clip}")
        print(f"  Local buffer: {local_buffer_size:,}, Batch: {batch_size}")
        print(f"  Collect steps: {collect_steps}, Train steps: {train_steps}")
        print(f"  Accumulate grads: {accumulate_grads}")
        print(f"  Per-worker epsilon: {eps_base}^(1+i/N)")
        print(f"  Total steps: {total_steps:,}")
        print("=" * 70)

    # Create shared memory gradient pool in /dev/shm (RAM-backed, fast)
    # Use unique subdir based on PID to avoid conflicts
    shm_dir = Path("/dev/shm") / f"mario_ddqn_{os.getpid()}"
    shm_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a temporary model for gradient pool layout (same architecture as workers)
    state_dim = (4, 64, 64)
    action_dim = 12  # COMPLEX_MOVEMENT
    temp_model = DoubleDQN(input_shape=state_dim, num_actions=action_dim)
    
    gradient_pool = SharedGradientTensorPool(
        num_workers=workers,
        model=temp_model.online,
        shm_dir=shm_dir,
        num_slots=8,  # 8 slots per worker for good async buffering
        create=True,
    )
    shm_paths = gradient_pool.get_shm_paths()
    
    # Register cleanup handler to remove shm directory on exit
    def cleanup_shm():
        try:
            gradient_pool.unlink()
        except Exception:
            pass
        try:
            if shm_dir.exists():
                shutil.rmtree(shm_dir)
        except Exception:
            pass
    atexit.register(cleanup_shm)
    
    # Create queues for UI (unbounded to prevent blocking)
    ui_queue: Queue | None = Queue() if not no_ui else None
    
    # Create shared memory heartbeats (no queues = no deadlocks)
    heartbeats = SharedHeartbeats(num_workers=workers, shm_dir=shm_dir, create=True)
    atexit.register(heartbeats.unlink)

    # Start UI process
    ui_process = None
    if not no_ui:
        ui_process = Process(
            target=run_training_ui,
            args=(workers, ui_queue),
            daemon=True,
        )
        ui_process.start()

    # Choose worker/learner targets based on UI mode
    worker_target = run_worker_silent if not no_ui else run_ddqn_worker
    learner_target = run_learner_silent if not no_ui else run_ddqn_learner

    # Create buffer config (shared by all workers)
    # alpha=0 disables prioritization (uniform sampling)
    buffer_config = BufferConfig(
        capacity=local_buffer_size,
        batch_size=batch_size,
        n_step=n_step,
        gamma=gamma,
        alpha=0.0 if no_per else 0.6,
    )

    # Create snapshot config
    snapshot_config = SnapshotConfig(
        enabled=not no_game_snapshots,
    )

    # Start worker processes with different epsilons
    worker_processes = []
    for i in range(workers):
        # Each worker has different exploration rate
        # Worker 0: most exploration, Worker N-1: least exploration
        worker_eps_end = eps_base ** (1 + (i + 1) / workers)

        # Create exploration config for this worker
        exploration_config = ExplorationConfig(
            epsilon_start=1.0,
            epsilon_end=worker_eps_end,
            decay_steps=eps_decay_steps,
        )

        # Create worker config
        worker_config = WorkerConfig(
            worker_id=i,
            level=cast(ConfigLevelType, level_type),
            use_dreamer=dreamer,
            latent_dim=latent_dim,
            steps_per_collection=collect_steps,
            train_steps=train_steps,
            max_grad_norm=max_grad_norm,
            reward_norm=reward_norm,
            reward_scale=reward_scale,
            reward_clip=reward_clip,
            entropy_coef=entropy_coef,
            q_clip=q_clip,
            loss_threshold=loss_threshold,
            initial_steps=initial_steps,  # Resume from correct timestep for epsilon decay
            buffer=buffer_config,
            exploration=exploration_config,
            snapshot=snapshot_config,
        )

        p = Process(
            target=worker_target,
            args=(worker_config, weights_path, shm_paths[i]),
            kwargs={"ui_queue": ui_queue, "heartbeat_path": heartbeats.shm_path, "crash_log_dir": crash_log_dir},
            daemon=True,
        )
        p.start()
        worker_processes.append((p, worker_config))
        if no_ui:
            print(f"Started worker {i} (ε_end = {worker_eps_end:.4f})")

    # Start learner process (pass dir and count, not pool - pool can't be pickled with spawn)
    learner_process = Process(
        target=learner_target,
        args=(weights_path, run_dir, shm_dir, workers),
        kwargs={
            "learning_rate": lr,
            "lr_end": lr_end,
            "total_timesteps": total_steps,
            "tau": tau,
            "accumulate_grads": accumulate_grads,
            "max_grad_norm": max_grad_norm,
            "weight_decay": weight_decay,
            "q_clip": q_clip,
            "use_dreamer": dreamer,
            "latent_dim": latent_dim,
            "ui_queue": ui_queue,
            "restore_snapshot": snapshot_path is not None,
            "snapshot_path": snapshot_path,
        },
        daemon=True,
    )
    learner_process.start()
    if no_ui:
        print("Started learner (shared memory mode)")
    
    # Start worker monitoring thread
    stop_monitor = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_workers,
        args=(
            worker_processes,
            heartbeats,
            weights_path,
            shm_paths,
            ui_queue,
            worker_target,
            stop_monitor,
            monitor_log_file,
            crash_log_dir,
        ),
        daemon=True,
    )
    monitor_thread.start()
    if no_ui:
        print("Started worker monitor")

    # Store main process PID for signal handler check
    main_pid = os.getpid()

    # Shutdown function
    def shutdown():
        print("\nShutting down...")
        stop_monitor.set()  # Stop monitoring thread
        for p, _ in worker_processes:
            if p.is_alive():
                p.terminate()
        if learner_process.is_alive():
            learner_process.terminate()
        if ui_process and ui_process.is_alive():
            ui_process.terminate()
        # Clean up shared memory
        gradient_pool.unlink()

    # Set up signal handler (only runs shutdown in main process)
    def signal_handler(sig, frame):
        # Only run shutdown if we're in the main process
        # Child processes inherit this handler but shouldn't execute shutdown
        if os.getpid() == main_pid:
            shutdown()
            sys.exit(0)
        else:
            # Child process received signal - just exit gracefully
            sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Wait for processes - UI controls lifetime when enabled
    try:
        if ui_process:
            # When UI exits (user pressed 'q'), shut everything down
            ui_process.join()
            shutdown()
        else:
            # No UI - wait for learner
            learner_process.join()
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    main()
