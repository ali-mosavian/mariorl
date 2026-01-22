"""
ZeroMQ-based event system for distributed training.

Uses PUSH/PULL pattern for many-to-one messaging (workers -> main).
Child processes push events, main process pulls and routes them.

Uses msgpack for fast binary serialization.
"""

from __future__ import annotations

import os
import logging
from typing import Any
from dataclasses import field
from dataclasses import dataclass

import zmq
import msgpack

from mario_rl.training.training_ui import UIMessage
from mario_rl.training.training_ui import MessageType

# =============================================================================
# Event Publisher (used by workers/coordinator)
# =============================================================================


@dataclass
class EventPublisher:
    """ZMQ PUSH socket wrapper for sending events."""

    endpoint: str
    source_id: int = -1
    _sock: zmq.Socket = field(init=False, repr=False)

    def __post_init__(self) -> None:
        ctx = zmq.Context.instance()
        self._sock = ctx.socket(zmq.PUSH)
        self._sock.setsockopt(zmq.LINGER, 0)  # Don't block on close
        self._sock.connect(self.endpoint)

    def publish(self, msg_type: str, data: dict[str, Any]) -> None:
        """Publish an event (non-blocking, fire-and-forget)."""
        try:
            payload = msgpack.packb(
                {
                    "msg_type": msg_type,
                    "source_id": self.source_id,
                    "data": data,
                },
                use_bin_type=True,
            )
            self._sock.send(payload, zmq.NOBLOCK)
        except (zmq.ZMQError, msgpack.PackException):
            pass  # Drop message if can't send

    def log(self, text: str) -> None:
        """Publish a log message."""
        msg_type = "worker_log" if self.source_id >= 0 else "learner_log"
        self.publish(msg_type, {"text": text})

    def status(self, **kwargs: Any) -> None:
        """Publish a status update."""
        msg_type = "worker_status" if self.source_id >= 0 else "learner_status"
        self.publish(msg_type, kwargs)

    def close(self) -> None:
        """Close the socket."""
        self._sock.close()


# =============================================================================
# Event Subscriber (used by main process)
# =============================================================================


@dataclass
class EventSubscriber:
    """ZMQ PULL socket wrapper for receiving events."""

    endpoint: str
    _sock: zmq.Socket = field(init=False, repr=False)

    def __post_init__(self) -> None:
        ctx = zmq.Context.instance()
        self._sock = ctx.socket(zmq.PULL)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind(self.endpoint)

    def poll(self, timeout_ms: int = 10) -> list[dict[str, Any]]:
        """Poll for events (non-blocking). Returns list of event dicts."""
        events = []
        while self._sock.poll(timeout_ms, zmq.POLLIN):
            try:
                payload = self._sock.recv(zmq.NOBLOCK)
                events.append(msgpack.unpackb(payload, raw=False))
            except (zmq.ZMQError, msgpack.UnpackException):
                break
        return events

    def close(self) -> None:
        """Close the socket."""
        self._sock.close()


# =============================================================================
# Event Routing Helpers
# =============================================================================


def format_event(event: dict[str, Any]) -> str | None:
    """Format an event dict for stdout. Returns None for status updates."""
    msg_type = event.get("msg_type", "")
    source_id = event.get("source_id", -1)
    data = event.get("data", {})

    match msg_type:
        case "worker_log":
            return f"[W{source_id}] {data.get('text', '')}"
        case "learner_log":
            return f"[COORD] {data.get('text', '')}"
        case "system_log":
            return f"[SYS] {data.get('text', '')}"
        case _:
            return None  # Status updates not printed to stdout


def event_to_ui_message(event: dict[str, Any]) -> UIMessage | None:
    """Convert event dict to UIMessage for UI queue.

    Returns None for events that shouldn't be displayed in the UI.
    """
    msg_type_str = event.get("msg_type", "system_log")
    source_id = event.get("source_id", -1)
    data = event.get("data", {})

    # Skip events that are for internal aggregation only (not for UI display)
    if msg_type_str in ("death_positions", "stuck_positions", "timeout_positions"):
        return None

    # Handle new metrics events
    if msg_type_str == "metrics":
        # Route to appropriate status type based on source
        source = data.get("source", "")
        snapshot = data.get("snapshot", {})
        if source.startswith("worker."):
            # Extract worker ID from source string
            try:
                worker_id = int(source.split(".")[1])
            except (IndexError, ValueError):
                worker_id = source_id
            return UIMessage(
                msg_type=MessageType.WORKER_STATUS,
                source_id=worker_id,
                data=snapshot,
            )
        else:
            # Coordinator metrics
            return UIMessage(
                msg_type=MessageType.LEARNER_STATUS,
                source_id=-1,
                data=snapshot,
            )

    msg_type_map = {
        "worker_log": MessageType.WORKER_LOG,
        "learner_log": MessageType.LEARNER_LOG,
        "system_log": MessageType.SYSTEM_LOG,
        "worker_status": MessageType.WORKER_STATUS,
        "learner_status": MessageType.LEARNER_STATUS,
        "worker_heartbeat": MessageType.WORKER_HEARTBEAT,
    }
    msg_type = msg_type_map.get(msg_type_str)
    if msg_type is None:
        # Unknown event type - skip it instead of creating empty system log
        return None

    return UIMessage(msg_type=msg_type, source_id=source_id, data=data)


def make_endpoint(pid: int | None = None) -> str:
    """Create an IPC endpoint path for the event system."""
    if pid is None:
        pid = os.getpid()
    return f"ipc:///tmp/mario_events_{pid}.sock"


# =============================================================================
# Logging Adapter
# =============================================================================


class ZMQLogHandler(logging.Handler):
    """Logging handler that publishes log records via ZMQ.

    Usage:
        publisher = EventPublisher(endpoint, source_id=0)
        handler = ZMQLogHandler(publisher)
        logger = logging.getLogger("worker")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        logger.info("Starting worker...")  # Sends via ZMQ
    """

    def __init__(self, publisher: EventPublisher, level: int = logging.NOTSET) -> None:
        super().__init__(level)
        self.publisher = publisher

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record via ZMQ."""
        try:
            msg = self.format(record)
            self.publisher.log(msg)
        except Exception:
            self.handleError(record)


def get_logger(
    name: str,
    publisher: EventPublisher,
    level: int = logging.INFO,
    fmt: str = "%(message)s",
) -> logging.Logger:
    """Create a logger that publishes via ZMQ.

    Args:
        name: Logger name (e.g., "worker.0", "coordinator")
        publisher: EventPublisher to send log messages through
        level: Logging level (default: INFO)
        fmt: Log format string (default: just the message)

    Returns:
        Configured logger instance

    Usage:
        publisher = EventPublisher(endpoint, source_id=0)
        log = get_logger("worker.0", publisher)
        log.info("Starting...")
        log.warning("Something happened")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Add ZMQ handler
    handler = ZMQLogHandler(publisher, level)
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)

    # Don't propagate to root logger
    logger.propagate = False

    return logger
