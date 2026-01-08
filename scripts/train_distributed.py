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
from typing import Any

mp.set_start_method("spawn", force=True)

from multiprocessing import Queue, Process

import click
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from mario_rl.training.training_ui import UIMessage, MessageType


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class Config:
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
# Helpers
# =============================================================================


def get_device() -> torch.device:
    """Get best available accelerator device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_model_and_learner(
    config: Config,
    device: torch.device | None = None,
) -> tuple[nn.Module, Any]:
    """Create model and learner from config."""
    if config.model == "ddqn":
        from mario_rl.models.ddqn import DoubleDQN, DDQNConfig
        from mario_rl.learners.ddqn import DDQNLearner

        cfg = DDQNConfig(input_shape=(4, 64, 64), num_actions=12, q_scale=config.q_scale)
        model = DoubleDQN(
            input_shape=cfg.input_shape,
            num_actions=cfg.num_actions,
            feature_dim=cfg.feature_dim,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
            q_scale=cfg.q_scale,
        )
        if device:
            model = model.to(device)
        learner = DDQNLearner(model=model, gamma=config.gamma)
    else:
        from mario_rl.models.dreamer import DreamerModel, DreamerConfig
        from mario_rl.learners.dreamer import DreamerLearner

        cfg = DreamerConfig(
            input_shape=(4, 64, 64),
            num_actions=12,
            latent_dim=config.latent_dim,
        )
        model = DreamerModel(
            input_shape=cfg.input_shape,
            num_actions=cfg.num_actions,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim,
        )
        if device:
            model = model.to(device)
        learner = DreamerLearner(model=model, gamma=config.gamma)

    return model, learner


def make_logger(prefix: str, ui_queue: Queue | None, source_id: int = -1):
    """Create a logger function that writes to UI queue or stdout."""

    def log(msg: str) -> None:
        if ui_queue:
            try:
                msg_type = MessageType.WORKER_LOG if source_id >= 0 else MessageType.LEARNER_LOG
                ui_queue.put_nowait(UIMessage(
                    msg_type=msg_type,
                    source_id=source_id,
                    data={"text": f"{prefix} {msg}"},
                ))
            except Exception:
                pass
        print(f"{prefix} {msg}", flush=True)  # Always print to stdout too

    return log


def install_exit_handler():
    """Install signal handler that exits cleanly in child processes."""

    def handler(sig, frame):
        os._exit(0)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


# =============================================================================
# Worker Process
# =============================================================================


def run_worker(
    worker_id: int,
    config: Config,
    weights_path: Path,
    shm_path: Path,
    shm_dir: Path,
    ui_queue: Queue | None = None,
) -> None:
    """Run a training worker process."""
    install_exit_handler()
    log = make_logger(f"[W{worker_id}]", ui_queue, source_id=worker_id)

    try:
        from mario_rl.environment.factory import create_mario_env
        from mario_rl.distributed.training_worker import TrainingWorker
        from mario_rl.distributed.shm_heartbeat import SharedHeartbeat
        from mario_rl.training.shared_gradient_tensor import attach_tensor_buffer

        device = get_device()
        log(f"Starting on {device}...")

        # Worker-specific epsilon for diverse exploration
        eps_base = 0.4
        epsilon_end = eps_base ** (1 + (worker_id + 1) / config.num_workers)

        # Create components
        env = create_mario_env(level=(1, 1), render_frames=False)
        _, learner = create_model_and_learner(config, device)

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

        # Training loop state
        version = 0
        total_episodes = 0
        best_x = 0
        grads_sent = 0

        while True:
            heartbeat.update(worker_id)
            worker.sync_weights(weights_path)

            result = worker.run_cycle(
                collect_steps=config.collect_steps,
                train_steps=config.train_steps,
            )

            # Update stats
            info = result["collection_info"]
            total_episodes += info.get("episodes_completed", 0)
            best_x = max(best_x, info.get("final_x_pos", 0))

            # Write gradients to shared memory
            if grads := result["gradients"]:
                loss = result.get("train_metrics", [{}])[0].get("loss", 0.0)
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

            # Periodic logging
            if grads_sent % 10 == 0:
                eps = worker.epsilon_at(worker.total_steps)
                log(f"steps={worker.total_steps}, eps={total_episodes}, ε={eps:.4f}, best_x={best_x}, grads={grads_sent}")

            # UI update
            if ui_queue:
                try:
                    ui_queue.put_nowait(UIMessage(
                        msg_type=MessageType.WORKER_STATUS,
                        source_id=worker_id,
                        data={
                            "episode": total_episodes,
                            "step": worker.total_steps,
                            "best_x": best_x,
                            "best_x_ever": best_x,
                            "epsilon": worker.epsilon_at(worker.total_steps),
                            "x_pos": info.get("final_x_pos", 0),
                            "reward": info.get("episode_reward", 0),
                        },
                    ))
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
    config: Config,
    weights_path: Path,
    checkpoint_dir: Path,
    shm_dir: Path,
    ui_queue: Queue | None = None,
) -> None:
    """Run the training coordinator process."""
    install_exit_handler()
    log = make_logger("[COORD]", ui_queue)

    try:
        from mario_rl.distributed.training_coordinator import TrainingCoordinator

        device = get_device()
        log(f"Starting on {device}...")

        _, learner = create_model_and_learner(config, device)

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

        last_log = time.time()
        grads_since_log = 0

        while True:
            result = coordinator.training_step()
            grads_since_log += result["gradients_processed"]

            # Periodic logging
            if time.time() - last_log > 5.0:
                elapsed = time.time() - last_log
                grads_per_sec = grads_since_log / elapsed
                log(f"update={result['update_count']}, steps={result['total_steps']}, grads/s={grads_per_sec:.1f}")

                if ui_queue:
                    try:
                        ui_queue.put_nowait(UIMessage(
                            msg_type=MessageType.LEARNER_STATUS,
                            source_id=-1,
                            data={
                                "step": result["update_count"],
                                "timesteps": result["total_steps"],
                                "grads_per_sec": grads_per_sec,
                                "gradients_received": result.get("gradients_processed", 0),
                            },
                        ))
                    except Exception:
                        pass

                last_log = time.time()
                grads_since_log = 0

            # Avoid busy loop when no gradients
            if result["gradients_processed"] == 0:
                time.sleep(0.01)

    except Exception as e:
        log(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# =============================================================================
# Process Management
# =============================================================================


@dataclass
class ProcessManager:
    """Manages worker and coordinator processes."""

    workers: list[tuple[Process, int]]
    coordinator: Process
    ui: Process | None
    gradient_pool: Any
    heartbeats: Any
    stop_event: threading.Event

    def terminate_all(self) -> None:
        """Terminate all managed processes."""
        self.stop_event.set()

        for p, _ in self.workers:
            if p.is_alive():
                p.terminate()

        if self.coordinator.is_alive():
            self.coordinator.terminate()

        if self.ui and self.ui.is_alive():
            self.ui.terminate()

    def cleanup(self) -> None:
        """Clean up shared memory resources."""
        try:
            self.gradient_pool.unlink()
        except Exception:
            pass
        try:
            self.heartbeats.unlink()
        except Exception:
            pass


def start_monitor_thread(
    heartbeats: Any,
    stop_event: threading.Event,
    timeout: float = 60.0,
) -> threading.Thread:
    """Start a thread that monitors worker heartbeats."""

    def monitor():
        time.sleep(timeout)  # Initial delay
        while not stop_event.is_set():
            try:
                for wid in heartbeats.stale_workers(timeout=timeout):
                    if stop_event.is_set():
                        break
                    print(f"Warning: Worker {wid} appears stale")
            except Exception:
                pass
            time.sleep(10.0)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    return thread


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

    config = Config(
        model=model,
        num_workers=workers,
        learning_rate=lr,
        gamma=gamma,
        batch_size=batch_size,
        buffer_capacity=buffer_size,
    )

    # Setup directories
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = Path(save_dir) / f"{model}_dist_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_path = run_dir / "weights.pt"

    shm_dir = Path("/dev/shm") / f"mario_{model}_{os.getpid()}"
    shm_dir.mkdir(parents=True, exist_ok=True)

    # Create reference model for shm sizing (CPU only, deleted after)
    ref_model, _ = create_model_and_learner(config, device=None)
    torch.save(ref_model.state_dict(), weights_path)

    # Initialize shared memory
    from mario_rl.distributed.shm_gradient_pool import SharedGradientPool
    from mario_rl.distributed.shm_heartbeat import SharedHeartbeat

    gradient_pool = SharedGradientPool(
        num_workers=workers,
        model=ref_model,
        shm_dir=shm_dir,
        create=True,
    )
    shm_paths = [gradient_pool.buffer_path(i) for i in range(workers)]
    del ref_model

    heartbeats = SharedHeartbeat(num_workers=workers, shm_dir=shm_dir, create=True)

    # Register cleanup
    def cleanup_shm():
        try:
            gradient_pool.unlink()
        except Exception:
            pass
        try:
            heartbeats.unlink()
        except Exception:
            pass
        try:
            if shm_dir.exists():
                shutil.rmtree(shm_dir)
        except Exception:
            pass

    atexit.register(cleanup_shm)

    # UI setup
    ui_queue: Queue | None = Queue() if not no_ui else None
    ui_process = None

    if not no_ui:
        from mario_rl.training.training_ui import run_ui

        ui_process = Process(
            target=run_ui,
            args=(workers, ui_queue),
            daemon=True,
        )
        ui_process.start()

    # Start workers
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

    # Process manager
    stop_event = threading.Event()
    manager = ProcessManager(
        workers=worker_processes,
        coordinator=coord_process,
        ui=ui_process,
        gradient_pool=gradient_pool,
        heartbeats=heartbeats,
        stop_event=stop_event,
    )

    # Monitor thread
    start_monitor_thread(heartbeats, stop_event)

    # Signal handling
    main_pid = os.getpid()

    def shutdown():
        print("\nShutting down...")
        manager.terminate_all()
        manager.cleanup()

    def signal_handler(sig, frame):
        if os.getpid() == main_pid:
            shutdown()
            os._exit(0)
        else:
            os._exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Main loop
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
