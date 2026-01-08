"""
ZeroMQ-based event system for distributed training.

Provides non-blocking pub/sub messaging between child processes and the main process.
Child processes publish events, main process subscribes and routes them.

Uses msgpack for fast binary serialization.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import msgpack
import zmq

from mario_rl.training.training_ui import MessageType, UIMessage


# =============================================================================
# Event Publisher (used by workers/coordinator)
# =============================================================================


@dataclass
class EventPublisher:
    """ZMQ PUB socket wrapper for sending events."""
    
    endpoint: str
    source_id: int = -1
    _sock: zmq.Socket | None = None
    
    def __post_init__(self) -> None:
        ctx = zmq.Context.instance()
        self._sock = ctx.socket(zmq.PUB)
        self._sock.connect(self.endpoint)
        # Small delay to let socket connect before first message
        time.sleep(0.05)
    
    def publish(self, msg_type: str, data: dict[str, Any]) -> None:
        """Publish an event (non-blocking, fire-and-forget)."""
        if self._sock is None:
            return
        try:
            payload = msgpack.packb({
                "msg_type": msg_type,
                "source_id": self.source_id,
                "data": data,
            }, use_bin_type=True)
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
        if self._sock is not None:
            self._sock.close()
            self._sock = None


# =============================================================================
# Event Subscriber (used by main process)
# =============================================================================


@dataclass
class EventSubscriber:
    """ZMQ SUB socket wrapper for receiving events."""
    
    endpoint: str
    _sock: zmq.Socket | None = None
    
    def __post_init__(self) -> None:
        ctx = zmq.Context.instance()
        self._sock = ctx.socket(zmq.SUB)
        self._sock.bind(self.endpoint)
        self._sock.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all
    
    def poll(self, timeout_ms: int = 10) -> list[dict[str, Any]]:
        """Poll for events (non-blocking). Returns list of event dicts."""
        if self._sock is None:
            return []
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
        if self._sock is not None:
            self._sock.close()
            self._sock = None


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


def event_to_ui_message(event: dict[str, Any]) -> UIMessage:
    """Convert event dict to UIMessage for UI queue."""
    msg_type_str = event.get("msg_type", "system_log")
    source_id = event.get("source_id", -1)
    data = event.get("data", {})
    
    msg_type_map = {
        "worker_log": MessageType.WORKER_LOG,
        "learner_log": MessageType.LEARNER_LOG,
        "system_log": MessageType.SYSTEM_LOG,
        "worker_status": MessageType.WORKER_STATUS,
        "learner_status": MessageType.LEARNER_STATUS,
        "worker_heartbeat": MessageType.WORKER_HEARTBEAT,
    }
    msg_type = msg_type_map.get(msg_type_str, MessageType.SYSTEM_LOG)
    
    return UIMessage(msg_type=msg_type, source_id=source_id, data=data)


def make_endpoint(pid: int | None = None) -> str:
    """Create an IPC endpoint path for the event system."""
    if pid is None:
        pid = os.getpid()
    return f"ipc:///tmp/mario_events_{pid}.sock"
