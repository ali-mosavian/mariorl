#!/usr/bin/env python3
"""
Distributed training launcher.

Spawns multiple worker processes and one learner process.
Workers collect experiences, learner trains the network.

Usage:
    python -m distributed.train -n 4
    python -m distributed.train -n 4 --render --level 1,2
"""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import signal
from typing import Any
from pathlib import Path
from typing import Optional
from datetime import datetime
from multiprocessing import Queue
from multiprocessing import Process
from multiprocessing import set_start_method

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import click

from mario_rl.training.worker import run_worker
from mario_rl.training.learner import run_learner
from mario_rl.training.shared_buffer import SharedReplayBuffer
from mario_rl.training.world_model_learner import run_world_model_learner


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    print("\nShutting down...")
    sys.exit(0)


def logged_worker(log_file: Path, *args, **kwargs):
    """Wrapper for run_worker that redirects stdout/stderr to a log file."""
    import traceback

    # Open log file with line buffering
    try:
        f = open(log_file, "w", buffering=1)
    except Exception as e:
        print(f"Failed to open log file {log_file}: {e}", file=sys.stderr)
        raise

    # Redirect stdout/stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = f
    sys.stderr = f

    try:
        print(f"=== Worker started at {datetime.now()} ===")
        print(f"Args: {args}")
        print(f"Kwargs: {kwargs}")
        print("=" * 60)
        sys.stdout.flush()

        run_worker(*args, **kwargs)

        print(f"\n=== Worker exited normally at {datetime.now()} ===")
    except BaseException as e:
        # Catch ALL exceptions including KeyboardInterrupt, SystemExit
        print(f"\n{'='*60}")
        print(f"FATAL: Worker crashed with {type(e).__name__}: {e}")
        print(f"Timestamp: {datetime.now()}")
        print(f"{'='*60}")
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        # Ensure we close the file and restore streams
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        f.close()


def logged_learner(log_file: Path, *args, **kwargs):
    """Wrapper for run_learner that redirects stdout/stderr to a log file."""
    import traceback

    try:
        f = open(log_file, "w", buffering=1)
    except Exception as e:
        print(f"Failed to open log file {log_file}: {e}", file=sys.stderr)
        raise

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = f
    sys.stderr = f

    try:
        print(f"=== Learner started at {datetime.now()} ===")
        print(f"Args: {args}")
        print(f"Kwargs: {kwargs}")
        print("=" * 60)
        sys.stdout.flush()

        run_learner(*args, **kwargs)

        print(f"\n=== Learner exited normally at {datetime.now()} ===")
    except BaseException as e:
        print(f"\n{'='*60}")
        print(f"FATAL: Learner crashed with {type(e).__name__}: {e}")
        print(f"Timestamp: {datetime.now()}")
        print(f"{'='*60}")
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        f.close()


def logged_world_model_learner(log_file: Path, *args, **kwargs):
    """Wrapper for run_world_model_learner that redirects stdout/stderr to a log file."""
    import traceback

    try:
        f = open(log_file, "w", buffering=1)
    except Exception as e:
        print(f"Failed to open log file {log_file}: {e}", file=sys.stderr)
        raise

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = f
    sys.stderr = f

    try:
        print(f"=== World Model Learner started at {datetime.now()} ===")
        print(f"Args: {args}")
        print(f"Kwargs: {kwargs}")
        print("=" * 60)
        sys.stdout.flush()

        run_world_model_learner(*args, **kwargs)

        print(f"\n=== World Model Learner exited normally at {datetime.now()} ===")
    except BaseException as e:
        print(f"\n{'='*60}")
        print(f"FATAL: World Model Learner crashed with {type(e).__name__}: {e}")
        print(f"Timestamp: {datetime.now()}")
        print(f"{'='*60}")
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        f.close()


@click.command()
@click.option("-n", "--num-workers", type=int, default=4, help="Number of worker processes")
@click.option(
    "-l",
    "--level",
    type=str,
    default="1,1",
    help="Level to play: '1,1' for specific level, 'random' for random levels, 'sequential' for all levels in order",
)
@click.option("-b", "--buffer-size", type=int, default=250000, help="Replay buffer size")
@click.option("--batch-size", type=int, default=64, help="Training batch size")
@click.option("--render", is_flag=True, help="Render first worker (for visualization)")
@click.option("--episodes", type=int, default=-1, help="Episodes per worker (-1 for infinite)")
@click.option("--learner-steps", type=int, default=-1, help="Max learner steps (-1 for infinite)")
@click.option("--save-dir", type=Path, default=None, help="Directory to save checkpoints")
@click.option("--ui/--no-ui", default=True, help="Use curses UI (default: yes)")
# World model options
@click.option(
    "--world-model/--no-world-model",
    default=False,
    help="Use world model learner (default: no)",
)
@click.option("--latent-dim", type=int, default=128, help="World model latent dimension")
@click.option(
    "--wm-steps",
    type=int,
    default=500,
    help="World model training steps per cycle",
)
@click.option(
    "--q-steps",
    type=int,
    default=500,
    help="Q-network training steps per cycle",
)
@click.option("--wm-lr", type=float, default=1e-4, help="World model learning rate")
@click.option("--q-lr", type=float, default=1e-4, help="Q-network learning rate")
def main(
    num_workers: int,
    level: str,
    buffer_size: int,
    batch_size: int,
    render: bool,
    episodes: int,
    learner_steps: int,
    save_dir: Path,
    ui: bool,
    # World model options
    world_model: bool,
    latent_dim: int,
    wm_steps: int,
    q_steps: int,
    wm_lr: float,
    q_lr: float,
):
    """Launch distributed Mario training with multiple workers."""

    # Parse level: can be "random", "sequential", or "world,stage"
    level_mode: str | tuple[int, int]
    if level in ("random", "sequential"):
        level_mode = level
    else:
        parts = tuple(map(int, level.split(",")))
        assert len(parts) == 2, "Level must be 'world,stage' with 2 integers"
        level_mode = (parts[0], parts[1])

    # Setup save directory
    if save_dir is None:
        save_dir = Path("checkpoints") / f"distributed_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}"
    save_dir.mkdir(parents=True, exist_ok=True)
    weights_path = save_dir / "weights.pt"

    # Create logs directory
    logs_dir = save_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Print startup info (before UI takes over)
    print("=" * 60)
    if world_model:
        print("Distributed Mario Training (WORLD MODEL)")
    else:
        print("Distributed Mario Training")
    print("=" * 60)
    print(f"  Workers:      {num_workers}")
    print(f"  Level:        {level_mode}")
    print(f"  Buffer size:  {buffer_size:,}")
    print(f"  Batch size:   {batch_size}")
    print(f"  Save dir:     {save_dir}")
    print(f"  Weights:      {weights_path}")
    print(f"  UI:           {'curses' if ui else 'text'}")
    if world_model:
        print(f"  Latent dim:   {latent_dim}")
        print(f"  WM steps:     {wm_steps}")
        print(f"  Q steps:      {q_steps}")
        print(f"  WM LR:        {wm_lr}")
        print(f"  Q LR:         {q_lr}")
    print("=" * 60)

    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create shared queue BEFORE spawning processes (required for spawn method)
    from multiprocessing import Queue as MPQueue

    experience_queue: MPQueue = MPQueue(maxsize=10000)  # type: ignore[type-arg]

    # Create shared buffer with the queue
    shared_buffer = SharedReplayBuffer(max_len=buffer_size, queue=experience_queue)

    # Create UI queue if using UI
    ui_queue: Optional[Queue[Any]] = Queue(maxsize=1000) if ui else None

    processes = []

    try:
        # Start learner process (world model or standard)
        if world_model:
            learner_log = logs_dir / "learner_world_model.log"
            learner_proc = Process(
                target=logged_world_model_learner,
                args=(learner_log, shared_buffer, weights_path, save_dir),
                kwargs={
                    "batch_size": batch_size,
                    "max_steps": learner_steps,
                    "ui_queue": ui_queue,
                    "latent_dim": latent_dim,
                    "wm_steps": wm_steps,
                    "q_steps": q_steps,
                    "wm_lr": wm_lr,
                    "q_lr": q_lr,
                },
                name="WorldModelLearner",
            )
            learner_proc.start()
            processes.append(learner_proc)
            print(f"Started World Model Learner (PID: {learner_proc.pid}, log: {learner_log})")
        else:
            learner_log = logs_dir / "learner.log"
            learner_proc = Process(
                target=logged_learner,
                args=(learner_log, shared_buffer, weights_path, save_dir),
                kwargs={
                    "batch_size": batch_size,
                    "max_steps": learner_steps,
                    "ui_queue": ui_queue,
                },
                name="Learner",
            )
            learner_proc.start()
            processes.append(learner_proc)
            print(f"Started Learner (PID: {learner_proc.pid}, log: {learner_log})")

        # Start worker processes
        for i in range(num_workers):
            # Only render first worker if requested
            worker_render = render and (i == 0)
            worker_log = logs_dir / f"worker_{i}.log"

            worker_proc = Process(
                target=logged_worker,
                args=(worker_log, i, shared_buffer, weights_path),
                kwargs={
                    "level": level_mode,
                    "render_frames": worker_render,
                    "num_episodes": episodes,
                    "ui_queue": ui_queue,
                    "use_world_model": world_model,
                    "latent_dim": latent_dim,
                },
                name=f"Worker-{i}",
            )
            worker_proc.start()
            processes.append(worker_proc)
            print(f"Started Worker {i} (PID: {worker_proc.pid}, log: {worker_log})")

        print("\nAll processes started. Press Ctrl+C to stop.\n")

        # Run UI in main process (if enabled)
        if ui and ui_queue is not None:
            from mario_rl.training.training_ui import run_ui

            run_ui(num_workers=num_workers, ui_queue=ui_queue)
        else:
            # Wait for all processes without UI
            for proc in processes:
                proc.join()

    except KeyboardInterrupt:
        print("\nReceived interrupt, terminating processes...")

    finally:
        # Terminate all processes
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()

        print("All processes terminated.")


if __name__ == "__main__":
    # Use 'spawn' for cross-platform compatibility
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass  # Already set

    main()
