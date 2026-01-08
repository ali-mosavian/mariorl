#!/usr/bin/env python3
"""
Distributed training with async gradient updates.

Supports multiple model types (DDQN, Dreamer) via --model flag.
Uses A3C-style gradient sharing where workers compute gradients locally
and send them to a central coordinator.

Usage:
    uv run python scripts/train_distributed.py --model ddqn --workers 4
    uv run python scripts/train_distributed.py --model dreamer --workers 8 --no-ui
"""
from __future__ import annotations

import os
import sys
import time
import atexit
import shutil
import signal
import threading
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

# Use 'spawn' start method for cleaner CUDA handling
mp.set_start_method("spawn", force=True)

from multiprocessing import Queue
from multiprocessing import Process

import click
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class TrainingConfig:
    """Training configuration."""

    model: str = "ddqn"
    num_workers: int = 4
    collect_steps: int = 64
    train_steps: int = 4
    learning_rate: float = 1e-4
    lr_min: float = 1e-5
    lr_decay_steps: int = 1_000_000
    gamma: float = 0.99
    tau: float = 0.005
    max_grad_norm: float = 10.0
    weight_decay: float = 1e-4
    buffer_capacity: int = 10_000
    batch_size: int = 32
    n_step: int = 3
    alpha: float = 0.6
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 1_000_000
    q_scale: float = 100.0
    latent_dim: int = 128
    target_update_interval: int = 100
    checkpoint_interval: int = 10_000


# =============================================================================
# Worker Process
# =============================================================================


def run_worker(
    worker_id: int,
    config: TrainingConfig,
    weights_path: Path,
    shm_path: Path,
    shm_dir: Path,
    ui_queue: Queue | None = None,
) -> None:
    """Run a training worker process."""
    # Each process sets up its own signal handler
    def worker_signal_handler(sig, frame):
        os._exit(0)  # Use os._exit to avoid cleanup issues

    signal.signal(signal.SIGINT, worker_signal_handler)
    signal.signal(signal.SIGTERM, worker_signal_handler)

    import torch

    from mario_rl.environment.factory import create_mario_env
    from mario_rl.distributed.training_worker import TrainingWorker
    from mario_rl.distributed.shm_heartbeat import SharedHeartbeat
    from mario_rl.training.shared_gradient_tensor import attach_tensor_buffer

    # Log to UI
    def log(msg: str) -> None:
        if ui_queue:
            try:
                ui_queue.put_nowait({"type": "log", "data": {"message": f"[W{worker_id}] {msg}"}})
            except Exception:
                pass
        else:
            print(f"[W{worker_id}] {msg}", flush=True)

    try:
        log("Starting...")

        # Worker-specific epsilon (diverse exploration)
        eps_base = 0.4
        epsilon_end = eps_base ** (1 + (worker_id + 1) / config.num_workers)

        # Create environment
        env = create_mario_env(level=(1, 1), render_frames=False)

        # Create model and learner
        if config.model == "ddqn":
            from mario_rl.models.ddqn import DoubleDQN, DDQNConfig
            from mario_rl.learners.ddqn import DDQNLearner

            model_config = DDQNConfig(
                input_shape=(4, 64, 64),
                num_actions=12,
                q_scale=config.q_scale,
            )
            model = DoubleDQN(
                input_shape=model_config.input_shape,
                num_actions=model_config.num_actions,
                feature_dim=model_config.feature_dim,
                hidden_dim=model_config.hidden_dim,
                dropout=model_config.dropout,
                q_scale=model_config.q_scale,
            )
            learner = DDQNLearner(model=model, gamma=config.gamma)
        else:
            from mario_rl.models.dreamer import DreamerModel, DreamerModelConfig
            from mario_rl.learners.dreamer import DreamerLearner

            model_config = DreamerModelConfig(
                input_shape=(4, 64, 64),
                num_actions=12,
                latent_dim=config.latent_dim,
            )
            model = DreamerModel(
                input_shape=model_config.input_shape,
                num_actions=model_config.num_actions,
                latent_dim=model_config.latent_dim,
                hidden_dim=model_config.hidden_dim,
                rssm_hidden_dim=model_config.rssm_hidden_dim,
                actor_hidden_dim=model_config.actor_hidden_dim,
                critic_hidden_dim=model_config.critic_hidden_dim,
                imagination_horizon=model_config.imagination_horizon,
            )
            learner = DreamerLearner(model=model, gamma=config.gamma)

        # Create training worker
        worker = TrainingWorker(
            env=env,
            learner=learner,
            buffer_capacity=config.buffer_capacity,
            batch_size=config.batch_size,
            n_step=config.n_step,
            gamma=config.gamma,
            alpha=config.alpha,
            epsilon_start=config.epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_steps=config.epsilon_decay_steps,
        )

        # Attach to shared memory (created by main process)
        gradient_buffer = attach_tensor_buffer(
            worker_id=worker_id,
            model=worker.model,
            shm_path=shm_path,
            num_slots=8,
        )

        heartbeat = SharedHeartbeat(
            num_workers=config.num_workers,
            shm_dir=shm_dir,
            create=False,
        )

        log(f"Started (ε_end={epsilon_end:.4f})")

        # Main loop
        version = 0
        total_episodes = 0
        best_x = 0
        grads_sent = 0

        while True:
            # Update heartbeat
            heartbeat.update(worker_id)

            # Sync weights
            worker.sync_weights(weights_path)

            # Run cycle
            result = worker.run_cycle(
                collect_steps=config.collect_steps,
                train_steps=config.train_steps,
            )

            grads = result["gradients"]
            info = result["collection_info"]

            # Track stats
            total_episodes += info.get("episodes_completed", 0)
            if info.get("final_x_pos", 0) > best_x:
                best_x = info["final_x_pos"]

            # Write gradients to shared memory
            if grads:
                loss = 0.0
                if result.get("train_metrics"):
                    loss = result["train_metrics"][0].get("loss", 0.0)

                gradient_buffer.write(
                    grads=grads,
                    version=version,
                    worker_id=worker_id,
                    timesteps=config.collect_steps,
                    episodes=total_episodes,
                    loss=loss,
                    best_x=best_x,
                )
                grads_sent += 1

            # Periodic log
            if grads_sent % 10 == 0:
                eps = worker.epsilon_at(worker.total_steps)
                log(f"steps={worker.total_steps}, eps={total_episodes}, ε={eps:.4f}, best_x={best_x}, grads={grads_sent}")

            # Send UI update
            if ui_queue:
                try:
                    ui_queue.put_nowait({
                        "type": "worker_status",
                        "data": {
                            "worker_id": worker_id,
                            "episodes": total_episodes,
                            "steps": worker.total_steps,
                            "best_x": best_x,
                            "epsilon": worker.epsilon_at(worker.total_steps),
                        },
                    })
                except Exception:
                    pass

    except Exception as e:
        log(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# =============================================================================
# Coordinator Process
# =============================================================================


def run_coordinator(
    config: TrainingConfig,
    weights_path: Path,
    checkpoint_dir: Path,
    shm_dir: Path,
    ui_queue: Queue | None = None,
) -> None:
    """Run the training coordinator process."""
    # Each process sets up its own signal handler
    def coord_signal_handler(sig, frame):
        os._exit(0)  # Use os._exit to avoid cleanup issues

    signal.signal(signal.SIGINT, coord_signal_handler)
    signal.signal(signal.SIGTERM, coord_signal_handler)

    import torch

    from mario_rl.distributed.training_coordinator import TrainingCoordinator

    def log(msg: str) -> None:
        if ui_queue:
            try:
                ui_queue.put_nowait({"type": "log", "data": {"message": f"[COORD] {msg}"}})
            except Exception:
                pass
        else:
            print(f"[COORD] {msg}", flush=True)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log(f"Starting on {device}...")

        # Create model and learner
        if config.model == "ddqn":
            from mario_rl.models.ddqn import DoubleDQN, DDQNConfig
            from mario_rl.learners.ddqn import DDQNLearner

            model_config = DDQNConfig(
                input_shape=(4, 64, 64),
                num_actions=12,
                q_scale=config.q_scale,
            )
            model = DoubleDQN(
                input_shape=model_config.input_shape,
                num_actions=model_config.num_actions,
                feature_dim=model_config.feature_dim,
                hidden_dim=model_config.hidden_dim,
                dropout=model_config.dropout,
                q_scale=model_config.q_scale,
            ).to(device)
            learner = DDQNLearner(model=model, gamma=config.gamma)
        else:
            from mario_rl.models.dreamer import DreamerModel, DreamerModelConfig
            from mario_rl.learners.dreamer import DreamerLearner

            model_config = DreamerModelConfig(
                input_shape=(4, 64, 64),
                num_actions=12,
                latent_dim=config.latent_dim,
            )
            model = DreamerModel(
                input_shape=model_config.input_shape,
                num_actions=model_config.num_actions,
                latent_dim=model_config.latent_dim,
                hidden_dim=model_config.hidden_dim,
                rssm_hidden_dim=model_config.rssm_hidden_dim,
                actor_hidden_dim=model_config.actor_hidden_dim,
                critic_hidden_dim=model_config.critic_hidden_dim,
                imagination_horizon=model_config.imagination_horizon,
            ).to(device)
            learner = DreamerLearner(model=model, gamma=config.gamma)

        # Create coordinator (attach to existing shm)
        coordinator = TrainingCoordinator(
            learner=learner,
            num_workers=config.num_workers,
            shm_dir=shm_dir,
            checkpoint_dir=checkpoint_dir,
            learning_rate=config.learning_rate,
            lr_min=config.lr_min,
            lr_decay_steps=config.lr_decay_steps,
            weight_decay=config.weight_decay,
            max_grad_norm=config.max_grad_norm,
            target_update_interval=config.target_update_interval,
            checkpoint_interval=config.checkpoint_interval,
            create_shm=False,
        )

        log("Started")

        # Main loop
        last_log = time.time()
        grads_since_log = 0

        while True:
            result = coordinator.training_step()
            grads_since_log += result["gradients_processed"]

            # Log periodically
            if time.time() - last_log > 5.0:
                elapsed = time.time() - last_log
                grads_per_sec = grads_since_log / elapsed
                log(f"update={result['update_count']}, steps={result['total_steps']}, grads/s={grads_per_sec:.1f}")

                if ui_queue:
                    try:
                        ui_queue.put_nowait({
                            "type": "learner_status",
                            "data": {
                                "update_count": result["update_count"],
                                "total_steps": result["total_steps"],
                                "gradients_per_sec": grads_per_sec,
                            },
                        })
                    except Exception:
                        pass

                last_log = time.time()
                grads_since_log = 0

            # Small sleep if no work
            if result["gradients_processed"] == 0:
                time.sleep(0.01)

    except Exception as e:
        log(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# =============================================================================
# Main CLI
# =============================================================================


@click.command()
@click.option("--model", type=click.Choice(["ddqn", "dreamer"]), default="ddqn", help="Model type")
@click.option("--workers", default=4, help="Number of workers")
@click.option("--save-dir", default="checkpoints", help="Directory for checkpoints")
@click.option("--lr", default=1e-4, help="Learning rate")
@click.option("--gamma", default=0.99, help="Discount factor")
@click.option("--batch-size", default=32, help="Batch size")
@click.option("--buffer-size", default=10000, help="Buffer capacity per worker")
@click.option("--no-ui", is_flag=True, help="Disable ncurses UI")
def main(
    model: str,
    workers: int,
    save_dir: str,
    lr: float,
    gamma: float,
    batch_size: int,
    buffer_size: int,
    no_ui: bool,
) -> None:
    """Train Mario using distributed gradient updates."""
    print("=" * 70)
    print("Distributed Training (Modular)")
    print("=" * 70)
    print(f"  Model: {model}")
    print(f"  Workers: {workers}")
    print(f"  LR: {lr}, Gamma: {gamma}")
    print(f"  Batch: {batch_size}, Buffer: {buffer_size}")
    print("=" * 70)

    config = TrainingConfig(
        model=model,
        num_workers=workers,
        learning_rate=lr,
        gamma=gamma,
        batch_size=batch_size,
        buffer_capacity=buffer_size,
    )

    # Create directories
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = Path(save_dir) / f"{model}_dist_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_path = run_dir / "weights.pt"

    # Shared memory directory
    shm_dir = Path("/dev/shm") / f"mario_{model}_{os.getpid()}"
    shm_dir.mkdir(parents=True, exist_ok=True)

    # Create reference model for shm initialization
    if model == "ddqn":
        from mario_rl.models.ddqn import DoubleDQN, DDQNConfig

        ref_config = DDQNConfig(input_shape=(4, 64, 64), num_actions=12, q_scale=config.q_scale)
        ref_model = DoubleDQN(
            input_shape=ref_config.input_shape,
            num_actions=ref_config.num_actions,
            feature_dim=ref_config.feature_dim,
            hidden_dim=ref_config.hidden_dim,
            dropout=ref_config.dropout,
            q_scale=ref_config.q_scale,
        )
    else:
        from mario_rl.models.dreamer import DreamerModel, DreamerModelConfig

        ref_config = DreamerModelConfig(input_shape=(4, 64, 64), num_actions=12, latent_dim=config.latent_dim)
        ref_model = DreamerModel(
            input_shape=ref_config.input_shape,
            num_actions=ref_config.num_actions,
            latent_dim=ref_config.latent_dim,
            hidden_dim=ref_config.hidden_dim,
            rssm_hidden_dim=ref_config.rssm_hidden_dim,
            actor_hidden_dim=ref_config.actor_hidden_dim,
            critic_hidden_dim=ref_config.critic_hidden_dim,
            imagination_horizon=ref_config.imagination_horizon,
        )

    # Create gradient pool in main process
    from mario_rl.distributed.shm_gradient_pool import SharedGradientPool

    gradient_pool = SharedGradientPool(
        num_workers=workers,
        model=ref_model,
        shm_dir=shm_dir,
        create=True,
    )
    shm_paths = [gradient_pool.buffer_path(i) for i in range(workers)]

    # Save initial weights
    torch.save(ref_model.state_dict(), weights_path)
    del ref_model

    # Register cleanup
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

    # Create heartbeats
    from mario_rl.distributed.shm_heartbeat import SharedHeartbeat

    heartbeats = SharedHeartbeat(num_workers=workers, shm_dir=shm_dir, create=True)
    atexit.register(heartbeats.unlink)

    # UI queue
    ui_queue: Queue | None = Queue() if not no_ui else None

    # Start UI process (if enabled)
    ui_process = None
    if not no_ui:
        from mario_rl.training.training_ui import TrainingUI

        def run_ui():
            ui = TrainingUI(num_workers=workers, message_queue=ui_queue)
            ui.run()

        ui_process = Process(target=run_ui, daemon=True)
        ui_process.start()

    # Start workers (pass shm_dir so they can attach to heartbeats)
    worker_processes = []
    for i in range(workers):
        p = Process(
            target=run_worker,
            args=(i, config, weights_path, shm_paths[i], shm_dir),
            kwargs={"ui_queue": ui_queue},
            daemon=True,
        )
        p.start()
        worker_processes.append((p, i))
        if no_ui:
            print(f"Started worker {i}")

    # Start coordinator
    coord_process = Process(
        target=run_coordinator,
        args=(config, weights_path, run_dir, shm_dir),
        kwargs={"ui_queue": ui_queue},
        daemon=True,
    )
    coord_process.start()
    if no_ui:
        print("Started coordinator")

    # Start worker monitor thread
    stop_monitor = threading.Event()

    def monitor_workers():
        # Initial delay to let workers start up
        time.sleep(60.0)
        while not stop_monitor.is_set():
            try:
                stale = heartbeats.stale_workers(timeout=60.0)
                for wid in stale:
                    if stop_monitor.is_set():
                        break
                    print(f"Warning: Worker {wid} appears stale")
            except Exception:
                pass
            time.sleep(10.0)

    monitor_thread = threading.Thread(target=monitor_workers, daemon=True)
    monitor_thread.start()

    # Store main PID for signal handler
    main_pid = os.getpid()

    def shutdown():
        print("\nShutting down...")
        stop_monitor.set()
        for p, _ in worker_processes:
            if p.is_alive():
                p.terminate()
        if coord_process.is_alive():
            coord_process.terminate()
        if ui_process and ui_process.is_alive():
            ui_process.terminate()
        gradient_pool.unlink()

    def signal_handler(sig, frame):
        if os.getpid() == main_pid:
            shutdown()
            os._exit(0)
        else:
            os._exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Wait for processes
    try:
        if ui_process:
            ui_process.join()
            shutdown()
        else:
            coord_process.join()
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    main()
