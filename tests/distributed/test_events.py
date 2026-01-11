"""Tests for ZeroMQ event system."""

import logging

from typing import Iterator

import pytest

from mario_rl.distributed.events import (
    EventPublisher,
    EventSubscriber,
    ZMQLogHandler,
    format_event,
    get_logger,
    make_endpoint,
)


@pytest.fixture
def endpoint() -> str:
    """Create a unique TCP endpoint for each test."""
    import random
    port = random.randint(30000, 60000)
    return f"tcp://127.0.0.1:{port}"


@pytest.fixture
def pub_sub(endpoint: str) -> Iterator[tuple[EventPublisher, EventSubscriber]]:
    """Create a publisher/subscriber pair (PUSH/PULL)."""
    sub = EventSubscriber(endpoint)  # PULL binds
    pub = EventPublisher(endpoint, source_id=0)  # PUSH connects
    yield pub, sub
    pub.close()
    sub.close()


# =============================================================================
# EventPublisher/EventSubscriber Tests
# =============================================================================


def test_publisher_sends_and_subscriber_receives(pub_sub: tuple[EventPublisher, EventSubscriber]) -> None:
    """Publisher sends message, subscriber receives it."""
    pub, sub = pub_sub
    
    pub.publish("test_event", {"key": "value"})
    
    import time
    time.sleep(0.1)  # Allow message to propagate
    
    events = sub.poll(timeout_ms=100)
    assert len(events) == 1
    assert events[0]["msg_type"] == "test_event"
    assert events[0]["source_id"] == 0
    assert events[0]["data"] == {"key": "value"}


def test_log_method_sends_worker_log(pub_sub: tuple[EventPublisher, EventSubscriber]) -> None:
    """log() sends worker_log when source_id >= 0."""
    pub, sub = pub_sub
    
    pub.log("Hello world")
    
    import time
    time.sleep(0.1)
    
    events = sub.poll(timeout_ms=100)
    assert len(events) == 1
    assert events[0]["msg_type"] == "worker_log"
    assert events[0]["data"]["text"] == "Hello world"


def test_log_method_sends_learner_log_when_source_id_negative(endpoint: str) -> None:
    """log() sends learner_log when source_id < 0."""
    import random
    import time
    port = random.randint(30000, 60000)
    ep = f"tcp://127.0.0.1:{port}"
    
    sub = EventSubscriber(ep)  # PULL binds
    pub = EventPublisher(ep, source_id=-1)  # Coordinator
    
    pub.log("Coordinator message")
    time.sleep(0.05)
    
    events = sub.poll(timeout_ms=100)
    assert len(events) == 1
    assert events[0]["msg_type"] == "learner_log"
    
    pub.close()
    sub.close()


def test_status_method_sends_worker_status(pub_sub: tuple[EventPublisher, EventSubscriber]) -> None:
    """status() sends worker_status with kwargs."""
    pub, sub = pub_sub
    
    pub.status(steps=100, epsilon=0.5, reward=10.0)
    
    import time
    time.sleep(0.1)
    
    events = sub.poll(timeout_ms=100)
    assert len(events) == 1
    assert events[0]["msg_type"] == "worker_status"
    assert events[0]["data"]["steps"] == 100
    assert events[0]["data"]["epsilon"] == 0.5
    assert events[0]["data"]["reward"] == 10.0


# =============================================================================
# format_event Tests
# =============================================================================


def test_format_event_worker_log():
    """format_event formats worker log messages."""
    event = {"msg_type": "worker_log", "source_id": 2, "data": {"text": "Starting"}}
    assert format_event(event) == "[W2] Starting"


def test_format_event_learner_log():
    """format_event formats learner log messages."""
    event = {"msg_type": "learner_log", "source_id": -1, "data": {"text": "Update 10"}}
    assert format_event(event) == "[COORD] Update 10"


def test_format_event_system_log():
    """format_event formats system log messages."""
    event = {"msg_type": "system_log", "source_id": -1, "data": {"text": "Shutdown"}}
    assert format_event(event) == "[SYS] Shutdown"


def test_format_event_status_returns_none():
    """format_event returns None for status updates (not printed)."""
    event = {"msg_type": "worker_status", "source_id": 0, "data": {"steps": 100}}
    assert format_event(event) is None


# =============================================================================
# ZMQLogHandler Tests
# =============================================================================


def test_zmq_log_handler_emits_via_publisher(pub_sub: tuple[EventPublisher, EventSubscriber]) -> None:
    """ZMQLogHandler routes log records through EventPublisher."""
    pub, sub = pub_sub
    
    handler = ZMQLogHandler(pub)
    logger = logging.getLogger("test.handler")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    logger.info("Test message")
    
    import time
    time.sleep(0.1)
    
    events = sub.poll(timeout_ms=100)
    assert len(events) == 1
    assert events[0]["msg_type"] == "worker_log"
    assert events[0]["data"]["text"] == "Test message"


def test_zmq_log_handler_respects_level(pub_sub: tuple[EventPublisher, EventSubscriber]) -> None:
    """ZMQLogHandler respects logging level."""
    pub, sub = pub_sub
    
    handler = ZMQLogHandler(pub, level=logging.WARNING)
    logger = logging.getLogger("test.level")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    
    logger.debug("Debug message")  # Should be filtered
    logger.info("Info message")    # Should be filtered
    logger.warning("Warning message")  # Should pass
    
    import time
    time.sleep(0.1)
    
    events = sub.poll(timeout_ms=100)
    assert len(events) == 1
    assert "Warning message" in events[0]["data"]["text"]


def test_zmq_log_handler_formats_message(pub_sub: tuple[EventPublisher, EventSubscriber]) -> None:
    """ZMQLogHandler uses formatter."""
    pub, sub = pub_sub
    
    handler = ZMQLogHandler(pub)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    
    logger = logging.getLogger("test.format")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    logger.warning("Formatted")
    
    import time
    time.sleep(0.1)
    
    events = sub.poll(timeout_ms=100)
    assert len(events) == 1
    assert events[0]["data"]["text"] == "[WARNING] Formatted"


# =============================================================================
# get_logger Tests
# =============================================================================


def test_get_logger_creates_configured_logger(pub_sub: tuple[EventPublisher, EventSubscriber]) -> None:
    """get_logger returns a logger that publishes via ZMQ."""
    pub, sub = pub_sub
    
    log = get_logger("test.factory", pub)
    log.info("Factory test")
    
    import time
    time.sleep(0.1)
    
    events = sub.poll(timeout_ms=100)
    assert len(events) == 1
    assert events[0]["data"]["text"] == "Factory test"


def test_get_logger_sets_level(pub_sub: tuple[EventPublisher, EventSubscriber]) -> None:
    """get_logger respects level parameter."""
    pub, sub = pub_sub
    
    log = get_logger("test.level2", pub, level=logging.ERROR)
    log.warning("Should not appear")
    log.error("Should appear")
    
    import time
    time.sleep(0.1)
    
    events = sub.poll(timeout_ms=100)
    assert len(events) == 1
    assert "Should appear" in events[0]["data"]["text"]


def test_get_logger_uses_custom_format(pub_sub: tuple[EventPublisher, EventSubscriber]) -> None:
    """get_logger applies custom format string."""
    pub, sub = pub_sub
    
    log = get_logger("test.fmt", pub, fmt="[CUSTOM] %(message)s")
    log.info("Hello")
    
    import time
    time.sleep(0.1)
    
    events = sub.poll(timeout_ms=100)
    assert len(events) == 1
    assert events[0]["data"]["text"] == "[CUSTOM] Hello"


def test_get_logger_does_not_propagate(pub_sub: tuple[EventPublisher, EventSubscriber]) -> None:
    """get_logger disables propagation to root logger."""
    pub, _ = pub_sub
    
    log = get_logger("test.propagate", pub)
    assert log.propagate is False
