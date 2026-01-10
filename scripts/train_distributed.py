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

from mario_rl.distributed.events import (
    EventPublisher,
    EventSubscriber,
    format_event,
    event_to_ui_message,
    make_endpoint,
)
from mario_rl.metrics import MetricAggregator, DeathHotspotAggregate


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class Config:
    """Training configuration."""

    model: str = "ddqn"
    num_workers: int = 4
    level: str = "1,1"
    collect_steps: int = 64
    train_steps: int = 4
    accumulate_grads: int = 1  # Gradients to accumulate before update
    learning_rate: float = 1e-4
    lr_min: float = 1e-5
    lr_decay_steps: int = 1_000_000
    gamma: float = 0.99
    tau: float = 0.004  # 4x higher to compensate for averaging 4 gradients into 1 update
    max_grad_norm: float = 10.0
    weight_decay: float = 1e-4
    buffer_capacity: int = 10_000
    batch_size: int = 32
    n_step: int = 3
    alpha: float = 0.6
    eps_base: float = 0.15  # Base for per-worker epsilon (floor ~2-9%)
    epsilon_decay_steps: int = 1_000_000
    q_scale: float = 100.0
    latent_dim: int = 128
    entropy_coef: float = 0.01  # Entropy regularization for exploration
    target_update_interval: int = 1  # Update target every step like working script
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
        learner = DDQNLearner(
            model=model,
            gamma=config.gamma,
            n_step=config.n_step,
            entropy_coef=config.entropy_coef,
        )
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


def make_event_publisher(endpoint: str, source_id: int = -1) -> EventPublisher:
    """Create an EventPublisher for a child process."""
    return EventPublisher(endpoint=endpoint, source_id=source_id)


def install_exit_handler():
    """Install signal handler that exits cleanly in child processes."""

    def handler(sig, frame):
        os._exit(0)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def parse_level(level_str: str) -> tuple[int, int] | str:
    """Parse level string to tuple or special mode.
    
    Args:
        level_str: 'random', 'sequential', or 'W,S' (e.g. '1,1')
    
    Returns:
        (world, stage) tuple or 'random'/'sequential' string
    """
    if level_str in ("random", "sequential"):
        return level_str
    try:
        parts = level_str.split(",")
        return (int(parts[0]), int(parts[1]))
    except (ValueError, IndexError):
        return (1, 1)  # Default to 1-1


# =============================================================================
# Worker Process
# =============================================================================


def run_worker(
    worker_id: int,
    config: Config,
    weights_path: Path,
    shm_path: Path,
    shm_dir: Path,
    zmq_endpoint: str,
    run_dir: Path,
) -> None:
    """Run a training worker process."""
    install_exit_handler()
    events = make_event_publisher(zmq_endpoint, source_id=worker_id)

    try:
        from mario_rl.environment.snapshot_wrapper import create_snapshot_mario_env
        from mario_rl.distributed.training_worker import TrainingWorker
        from mario_rl.distributed.shm_heartbeat import SharedHeartbeat
        from mario_rl.training.shared_gradient_tensor import attach_tensor_buffer
        from mario_rl.metrics import MetricLogger, DDQNMetrics, DreamerMetrics

        device = get_device()
        events.log(f"Starting on {device}...")

        # Worker-specific epsilon for diverse exploration
        epsilon_end = config.eps_base ** (1 + (worker_id + 1) / config.num_workers)

        # Create components with snapshot support
        level = parse_level(config.level)
        hotspot_path = run_dir / "death_hotspots.json"
        env = create_snapshot_mario_env(
            level=level,
            render_frames=False,
            hotspot_path=hotspot_path,
            checkpoint_interval=500,
            max_restores_without_progress=3,
            enabled=True,
        )
        _, learner = create_model_and_learner(config, device)

        # Create metrics logger for this worker
        schema = DDQNMetrics if config.model == "ddqn" else DreamerMetrics
        csv_path = run_dir / f"worker_{worker_id}.csv"
        logger = MetricLogger(
            source_id=f"worker.{worker_id}",
            schema=schema,
            csv_path=csv_path,
            publisher=events,
        )

        worker = TrainingWorker(
            env=env,
            learner=learner,
            buffer_capacity=config.buffer_capacity,
            batch_size=config.batch_size,
            n_step=config.n_step,
            gamma=config.gamma,
            alpha=config.alpha,
            epsilon_start=1.0,
            epsilon_end=epsilon_end,
            epsilon_decay_steps=config.epsilon_decay_steps,
            logger=logger,
            flush_every=config.collect_steps * 10,  # Flush every 10 cycles
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

        events.log(f"Started (ε_end={epsilon_end:.4f})")

        # Training loop state
        version = 0
        total_episodes = 0
        best_x = 0
        grads_sent = 0
        
        # Track snapshot stats (handled by wrapper, but we log them)
        last_snapshot_saves = 0
        last_snapshot_restores = 0

        while True:
            heartbeat.update(worker_id)
            worker.sync_weights(weights_path)

            result = worker.run_cycle(
                collect_steps=config.collect_steps,
                train_steps=config.train_steps,
            )

            # Update stats from raw env info dicts
            info = result["collection_info"]
            total_episodes += info.get("episodes_completed", 0)
            
            # Extract game-specific metrics from step_infos (raw env info dicts)
            step_infos = info.get("step_infos", [])
            episode_end_infos = info.get("episode_end_infos", [])
            
            # Get current game state from last step's info
            if step_infos:
                last_info = step_infos[-1]
                x_pos = last_info.get("x_pos", 0)
                game_time = last_info.get("time", 0)
                world = last_info.get("world", 1)
                stage = last_info.get("stage", 1)
            else:
                x_pos = 0
                game_time = 0
                world = 1
                stage = 1
            
            best_x = max(best_x, x_pos)
            
            # Count deaths, timeouts, and flags
            # Include both: deaths where we restored AND deaths where episode ended
            deaths_this_cycle = 0
            timeouts_this_cycle = 0
            death_positions: list[int] = []
            flags_this_cycle = 0
            
            # Count deaths that were restored (from step_infos)
            for step_info in step_infos:
                if step_info.get("snapshot_restored", False):
                    death_x = step_info.get("death_position", step_info.get("x_pos", 0))
                    if death_x > 0:
                        deaths_this_cycle += 1
                        death_positions.append(death_x)
            
            # Count deaths/timeouts from actual episode endings
            for ep_info in episode_end_infos:
                if ep_info.get("flag_get", False):
                    # Completed level
                    flags_this_cycle += 1
                elif ep_info.get("is_timeout", False):
                    # Timeout - ran out of time (not a skill failure)
                    timeouts_this_cycle += 1
                else:
                    # Actual death (skill failure)
                    deaths_this_cycle += 1
                    death_x = ep_info.get("x_pos", 0)
                    if death_x > 0:
                        death_positions.append(death_x)
            
            # Log deaths to CSV and publish for aggregation
            level_id = f"{world}-{stage}"
            if deaths_this_cycle > 0:
                logger.count("deaths", n=deaths_this_cycle)
                # Log death positions to CSV (format: "level:pos1,pos2,pos3")
                positions_str = ",".join(str(p) for p in death_positions)
                logger.text("death_positions", f"{level_id}:{positions_str}")
                # Publish for main process aggregation
                events.publish("death_positions", {
                    "level_id": level_id,
                    "positions": death_positions,
                })
            else:
                # Clear death positions if no deaths this cycle
                logger.text("death_positions", "")
            if timeouts_this_cycle > 0:
                logger.count("timeouts", n=timeouts_this_cycle)
            if flags_this_cycle > 0:
                logger.count("flags", n=flags_this_cycle)

            # Update game-specific metrics in logger
            logger.gauge("x_pos", x_pos)
            logger.gauge("best_x", best_x)
            logger.gauge("best_x_ever", best_x)
            logger.gauge("game_time", game_time)
            logger.gauge("world", world)
            logger.gauge("stage", stage)
            
            # Calculate speed from episode end infos (x_pos / time_spent)
            for ep_info in episode_end_infos:
                ep_time = ep_info.get("time", 0)
                ep_x = ep_info.get("x_pos", 0)
                if ep_time > 0:
                    speed = ep_x / ep_time
                    logger.observe("speed", speed)

            # Log snapshot metrics (from the wrapper)
            if hasattr(worker.env, "snapshot_stats"):
                stats = worker.env.snapshot_stats
                current_saves = stats.get("snapshot_saves", 0)
                current_restores = stats.get("snapshot_restores", 0)
                
                # Log incremental counts
                new_saves = current_saves - last_snapshot_saves
                new_restores = current_restores - last_snapshot_restores
                
                if new_saves > 0:
                    logger.count("snapshot_saves", n=new_saves)
                if new_restores > 0:
                    logger.count("snapshot_restores", n=new_restores)
                
                last_snapshot_saves = current_saves
                last_snapshot_restores = current_restores

            # Write gradients to shared memory
            if grads := result["gradients"]:
                train_metrics = result.get("train_metrics", [{}])
                if train_metrics:
                    m = train_metrics[0]
                    loss = m.get("loss", 0.0)
                    # Log training metrics
                    if loss:
                        logger.observe("loss", float(loss))
                    if "q_mean" in m:
                        logger.observe("q_mean", float(m["q_mean"]))
                    if "td_error" in m:
                        logger.observe("td_error", float(m["td_error"]))
                    if "q_max" in m:
                        logger.gauge("q_max", float(m["q_max"]))
                else:
                    loss = 0.0
                
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
                logger.count("grads_sent")

            # Force periodic flush to send metrics to UI
            if grads_sent % 5 == 0:
                logger.flush()

            # Periodic logging
            if grads_sent % 10 == 0:
                eps = worker.epsilon_at(worker.total_steps)
                events.log(f"steps={worker.total_steps}, eps={total_episodes}, ε={eps:.4f}, best_x={best_x}, grads={grads_sent}")

    except Exception as e:
        events.log(f"Error: {e}")
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
    zmq_endpoint: str,
    run_dir: Path,
) -> None:
    """Run the training coordinator process."""
    install_exit_handler()
    events = make_event_publisher(zmq_endpoint, source_id=-1)

    try:
        from mario_rl.distributed.training_coordinator import TrainingCoordinator
        from mario_rl.metrics import MetricLogger, CoordinatorMetrics

        device = get_device()
        events.log(f"Starting on {device}...")

        _, learner = create_model_and_learner(config, device)

        # No logger here - main process writes aggregated metrics to coordinator.csv
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
            tau=config.tau,
            accumulate_count=config.accumulate_grads,
            target_update_interval=config.target_update_interval,
            checkpoint_interval=config.checkpoint_interval,
            create_shm=False,
            logger=None,  # Main process handles CSV writing with aggregated metrics
        )

        events.log("Started")

        last_log = time.time()
        grads_since_log = 0

        while True:
            result = coordinator.training_step()
            grads_since_log += result["gradients_processed"]

            # Periodic logging
            if time.time() - last_log > 5.0:
                elapsed = time.time() - last_log
                grads_per_sec = grads_since_log / elapsed
                events.log(f"update={result['update_count']}, steps={result['total_steps']}, grads/s={grads_per_sec:.1f}")

                # Status update (sent via ZMQ)
                events.status(
                    step=result["update_count"],
                    timesteps=result["total_steps"],
                    grads_per_sec=grads_per_sec,
                    gradients_received=result.get("gradients_processed", 0),
                    grad_norm=result.get("grad_norm", 0.0),
                    weight_version=result.get("weight_version", 0),
                    lr=result.get("lr", 0.0),
                )

                last_log = time.time()
                grads_since_log = 0

            # Avoid busy loop when no gradients
            if result["gradients_processed"] == 0:
                time.sleep(0.01)

    except Exception as e:
        events.log(f"Error: {e}")
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
@click.option("--workers", "-w", default=4, help="Number of workers")
@click.option("--level", "-l", default="random", help="Level: 'random', 'sequential', or 'W,S' (e.g. '1,1')")
@click.option("--save-dir", default="checkpoints", help="Directory for checkpoints")
# Learning hyperparameters
@click.option("--lr", default=2.5e-4, help="Initial learning rate")
@click.option("--lr-end", default=1e-5, help="Final learning rate")
@click.option("--gamma", default=0.99, help="Discount factor")
@click.option("--n-step", default=3, help="N-step returns")
@click.option("--tau", default=0.001, help="Soft update coefficient")
# Worker settings
@click.option("--buffer-size", default=10_000, help="Buffer capacity per worker")
@click.option("--batch-size", default=32, help="Batch size per worker")
@click.option("--collect-steps", default=64, help="Steps to collect per cycle")
@click.option("--train-steps", default=4, help="Gradient computations per cycle")
# Learner settings
@click.option("--accumulate-grads", default=1, help="Gradients to accumulate before update")
# Epsilon settings
@click.option("--eps-base", default=0.4, help="Base for per-worker epsilon (ε = base^(1+i/N))")
@click.option("--eps-decay-steps", default=1_000_000, help="Steps for epsilon decay")
# Stability settings
@click.option("--q-scale", default=100.0, help="Q-value scale for softsign")
@click.option("--max-grad-norm", default=10.0, help="Maximum gradient norm")
@click.option("--weight-decay", default=1e-4, help="L2 regularization")
# Dreamer specific
@click.option("--latent-dim", default=128, help="Latent dimension for Dreamer")
# Other
@click.option("--total-steps", default=2_000_000, help="Total training steps (for LR schedule)")
@click.option("--no-ui", is_flag=True, help="Disable ncurses UI")
def main(
    model: str,
    workers: int,
    level: str,
    save_dir: str,
    lr: float,
    lr_end: float,
    gamma: float,
    n_step: int,
    tau: float,
    buffer_size: int,
    batch_size: int,
    collect_steps: int,
    train_steps: int,
    accumulate_grads: int,
    eps_base: float,
    eps_decay_steps: int,
    q_scale: float,
    max_grad_norm: float,
    weight_decay: float,
    latent_dim: int,
    total_steps: int,
    no_ui: bool,
) -> None:
    """Train Mario using distributed gradient updates."""
    print("=" * 70)
    print("Distributed Training")
    print("=" * 70)
    print(f"  Model: {model}")
    print(f"  Workers: {workers}, Level: {level}")
    print(f"  LR: {lr} → {lr_end} (cosine), Gamma: {gamma}")
    print(f"  Tau: {tau}, N-step: {n_step}")
    print(f"  Batch: {batch_size}, Buffer: {buffer_size}")
    print(f"  Collect steps: {collect_steps}, Train steps: {train_steps}")
    print(f"  Accumulate grads: {accumulate_grads}")
    print(f"  Epsilon: {eps_base}^(1+i/N), decay: {eps_decay_steps:,}")
    print(f"  Q-scale: {q_scale}, Max grad norm: {max_grad_norm}")
    print("=" * 70)

    config = Config(
        model=model,
        num_workers=workers,
        level=level,
        collect_steps=collect_steps,
        train_steps=train_steps,
        accumulate_grads=accumulate_grads,
        learning_rate=lr,
        lr_min=lr_end,
        lr_decay_steps=total_steps,
        gamma=gamma,
        tau=tau,
        max_grad_norm=max_grad_norm,
        weight_decay=weight_decay,
        buffer_capacity=buffer_size,
        batch_size=batch_size,
        n_step=n_step,
        eps_base=eps_base,
        epsilon_decay_steps=eps_decay_steps,
        q_scale=q_scale,
        latent_dim=latent_dim,
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

    # ZMQ event system
    zmq_endpoint = make_endpoint()
    event_sub = EventSubscriber(zmq_endpoint)
    
    # Metrics aggregator for combining worker/coordinator stats
    aggregator = MetricAggregator(num_workers=workers)
    
    # Main process CSV logger for aggregated metrics (coordinator only logs basic stats)
    main_csv_path = run_dir / "coordinator.csv"
    main_csv_file = None
    main_csv_writer = None
    main_csv_header_written = False
    
    def write_main_metrics(data: dict) -> None:
        """Write aggregated metrics to coordinator CSV."""
        nonlocal main_csv_file, main_csv_writer, main_csv_header_written
        import csv
        
        if main_csv_file is None:
            main_csv_file = open(main_csv_path, "w", newline="")
            fieldnames = [
                "timestamp", "update_count", "total_steps", "total_episodes",
                "grads_per_sec", "total_sps", "learning_rate", "weight_version",
                "avg_reward", "avg_speed", "avg_loss", "q_mean", "td_error", "grad_norm",
            ]
            main_csv_writer = csv.DictWriter(main_csv_file, fieldnames=fieldnames)
        
        if not main_csv_header_written:
            main_csv_writer.writeheader()
            main_csv_header_written = True
        
        row = {
            "timestamp": time.time(),
            "update_count": data.get("step", data.get("update_count", 0)),
            "total_steps": data.get("timesteps", data.get("total_steps", 0)),
            "total_episodes": data.get("total_episodes", 0),
            "grads_per_sec": data.get("grads_per_sec", 0),
            "total_sps": data.get("total_sps", 0),
            "learning_rate": data.get("lr", data.get("learning_rate", 0)),
            "weight_version": data.get("weight_version", 0),
            "avg_reward": data.get("avg_reward", 0),
            "avg_speed": data.get("avg_speed", 0),
            "avg_loss": data.get("avg_loss", 0),
            "q_mean": data.get("q_mean", 0),
            "td_error": data.get("td_error", 0),
            "grad_norm": data.get("grad_norm", 0),
        }
        main_csv_writer.writerow(row)
        main_csv_file.flush()
    
    # Death hotspot aggregator for snapshot/restore decisions
    hotspot_path = run_dir / "death_hotspots.json"
    hotspot_aggregator = DeathHotspotAggregate.load_or_create(hotspot_path)
    last_hotspot_save = time.time()
    
    # Cleanup ZMQ on exit
    def cleanup_zmq():
        event_sub.close()
        # Save hotspots on exit
        hotspot_aggregator.save_if_dirty()
    atexit.register(cleanup_zmq)

    # UI setup (only queue for forwarding to UI process)
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

    # Start workers (pass zmq_endpoint, not ui_queue)
    worker_processes = []
    for i in range(workers):
        p = Process(
            target=run_worker,
            args=(i, config, weights_path, shm_paths[i], shm_dir, zmq_endpoint, run_dir),
            daemon=True,
        )
        p.start()
        worker_processes.append((p, i))
        print(f"Started worker {i}")

    # Start coordinator (pass zmq_endpoint, not ui_queue)
    coord_process = Process(
        target=run_coordinator,
        args=(config, weights_path, run_dir, shm_dir, zmq_endpoint, run_dir),
        daemon=True,
    )
    coord_process.start()
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
        # Close main process CSV
        if main_csv_file is not None:
            main_csv_file.close()
        # Save death hotspots on exit
        hotspot_aggregator.save_if_dirty()

    def signal_handler(sig, frame):
        if os.getpid() == main_pid:
            shutdown()
            os._exit(0)
        else:
            os._exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Main event loop - poll ZMQ and route events
    try:
        while True:
            # Check if coordinator is still alive
            if not coord_process.is_alive():
                print("Coordinator exited")
                break
            
            # Check if UI exited (user pressed 'q')
            if ui_process is not None and not ui_process.is_alive():
                print("UI exited, stopping training...")
                break
            
            # Poll events from ZMQ (non-blocking)
            for event in event_sub.poll(timeout_ms=50):
                msg_type = event.get("msg_type", "")
                
                # Update aggregator with metrics events
                if msg_type == "metrics":
                    source = event.get("data", {}).get("source", "")
                    snapshot = event.get("data", {}).get("snapshot", {})
                    aggregator.update(source, snapshot)
                
                # Aggregate death positions for snapshot/restore hints
                if msg_type == "death_positions":
                    level_id = event.get("data", {}).get("level_id", "")
                    positions = event.get("data", {}).get("positions", [])
                    if level_id and positions:
                        hotspot_aggregator.record_deaths_batch(level_id, positions)
                
                # Enhance learner_status with aggregated worker metrics
                if msg_type == "learner_status":
                    agg = aggregator.aggregate()
                    event["data"]["avg_reward"] = agg.get("mean_reward", 0.0)
                    event["data"]["avg_speed"] = agg.get("mean_speed", 0.0)
                    event["data"]["avg_loss"] = agg.get("mean_loss", 0.0)
                    event["data"]["loss"] = agg.get("mean_loss", 0.0)
                    event["data"]["q_mean"] = agg.get("mean_q_mean", 0.0)
                    event["data"]["td_error"] = agg.get("mean_td_error", 0.0)
                    event["data"]["total_episodes"] = agg.get("total_episodes", 0)
                    event["data"]["total_sps"] = agg.get("total_steps_per_sec", 0.0)
                    
                    # Write enhanced metrics to CSV
                    write_main_metrics(event["data"])
                
                if ui_queue:
                    # Forward to UI process (skip None messages)
                    try:
                        ui_msg = event_to_ui_message(event)
                        if ui_msg is not None:
                            ui_queue.put_nowait(ui_msg)
                    except Exception:
                        pass
                else:
                    # Print to stdout
                    text = format_event(event)
                    if text:
                        print(text, flush=True)
            
            # Periodically save death hotspots (every 30 seconds)
            now = time.time()
            if now - last_hotspot_save > 30:
                if hotspot_aggregator.save_if_dirty():
                    pass  # Saved successfully
                last_hotspot_save = now
            
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown()


if __name__ == "__main__":
    main()
