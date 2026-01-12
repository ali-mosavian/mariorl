"""ncurses-based Training UI for distributed training.

Provides real-time monitoring of:
- Per-worker stats (steps, episodes, epsilon, loss)
- Learner stats (updates, total steps, LR)
- Global stats (total episodes, flags, deaths)
- Keyboard controls (q to quit, r to clear/redraw)
"""

import time
import curses
from typing import Any
from typing import Callable
from collections import deque
from dataclasses import field
from dataclasses import dataclass
import multiprocessing as mp

from mario_rl.ui.metrics import MetricsAggregator


@dataclass
class TrainingUI:
    """ncurses-based training monitoring UI.

    Displays real-time training statistics in a terminal interface.
    Polls a message queue for updates from workers and learner.
    """

    num_workers: int
    message_queue: mp.Queue | None = None
    refresh_rate_ms: int = 50

    # Callbacks
    on_quit: Callable[[], None] | None = None

    # State
    running: bool = field(init=False, default=True)
    metrics: MetricsAggregator = field(init=False, repr=False)
    logs: deque[str] = field(init=False, repr=False)
    max_logs: int = 50

    def __post_init__(self) -> None:
        """Initialize state."""
        self.running = True
        self.metrics = MetricsAggregator(num_workers=self.num_workers)
        self.logs = deque(maxlen=self.max_logs)

    def run(self) -> None:
        """Run the UI in a curses wrapper."""
        curses.wrapper(self._main)

    def _main(self, stdscr) -> None:
        """Main curses loop."""
        try:
            # Setup
            curses.curs_set(0)  # Hide cursor
            stdscr.nodelay(True)  # Non-blocking input
            stdscr.timeout(self.refresh_rate_ms)

            # Initialize colors
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_GREEN, -1)   # Success
            curses.init_pair(2, curses.COLOR_YELLOW, -1)  # Warning
            curses.init_pair(3, curses.COLOR_RED, -1)     # Error
            curses.init_pair(4, curses.COLOR_CYAN, -1)    # Info
            curses.init_pair(5, curses.COLOR_MAGENTA, -1) # Headers

            while self.running:
                # Process messages
                self._process_messages()

                # Handle input
                try:
                    key = stdscr.getch()
                    if key == ord("q") or key == ord("Q"):
                        self.running = False
                        if self.on_quit:
                            self.on_quit()
                        break
                    elif key == ord("r") or key == ord("R"):
                        stdscr.clearok(True)
                except curses.error:
                    pass

                # Draw
                self._draw(stdscr)

        except Exception as e:
            self.log(f"UI Error: {e}")

    def _process_messages(self) -> None:
        """Process messages from queue."""
        if self.message_queue is None:
            return

        # Process up to 100 messages per refresh
        for _ in range(100):
            try:
                msg = self.message_queue.get_nowait()
                self._handle_message(msg)
            except Exception:
                break

    def _handle_message(self, msg: dict[str, Any]) -> None:
        """Handle a single message."""
        msg_type = msg.get("type")
        data = msg.get("data", {})

        if msg_type == "worker_status":
            worker_id = data.get("worker_id", 0)
            self.metrics.update_worker(
                worker_id=worker_id,
                episodes=data.get("episodes"),
                steps=data.get("steps"),
                reward=data.get("reward"),
                x_pos=data.get("x_pos"),
                best_x=data.get("best_x"),
                epsilon=data.get("epsilon"),
                loss=data.get("loss"),
                deaths=data.get("deaths"),
                flags=data.get("flags"),
                last_heartbeat=data.get("last_heartbeat"),
            )
        elif msg_type == "learner_status":
            self.metrics.update_learner(
                update_count=data.get("update_count"),
                total_steps=data.get("total_steps"),
                learning_rate=data.get("learning_rate"),
                loss=data.get("loss"),
                gradients_per_sec=data.get("gradients_per_sec"),
            )
        elif msg_type == "log":
            self.log(data.get("message", ""))

    def _draw(self, stdscr) -> None:
        """Draw the UI."""
        try:
            stdscr.clear()
            height, width = stdscr.getmaxyx()

            # Title
            title = "=== Distributed Training Monitor ==="
            stdscr.addstr(0, (width - len(title)) // 2, title, curses.A_BOLD)

            # Learner section
            row = 2
            stdscr.addstr(row, 0, "LEARNER", curses.A_BOLD | curses.color_pair(4))
            row += 1
            stdscr.addstr(row, 0, self.metrics.format_learner_status())
            row += 2

            # Global stats
            stdscr.addstr(row, 0, "GLOBAL STATS", curses.A_BOLD | curses.color_pair(4))
            row += 1
            global_line = (
                f"Episodes: {self.metrics.total_episodes:6d} | "
                f"Flags: {self.metrics.total_flags:4d} | "
                f"Deaths: {self.metrics.total_deaths:6d} | "
                f"Best X: {self.metrics.best_x_ever:6d}"
            )
            stdscr.addstr(row, 0, global_line)
            row += 2

            # Workers section
            stdscr.addstr(row, 0, "WORKERS", curses.A_BOLD | curses.color_pair(5))
            row += 1

            for worker_id in range(self.num_workers):
                if row >= height - 5:
                    break
                status_line = self.metrics.format_worker_status(worker_id)
                stdscr.addstr(row, 0, status_line[:width - 1])
                row += 1

            # Logs section
            row += 1
            if row < height - 2:
                stdscr.addstr(row, 0, "LOGS", curses.A_BOLD | curses.color_pair(2))
                row += 1

                for log_line in list(self.logs)[-5:]:
                    if row >= height - 1:
                        break
                    stdscr.addstr(row, 0, log_line[:width - 1])
                    row += 1

            # Footer
            footer = "Press 'q' to quit | 'r' to clear/redraw"
            if height > 1:
                stdscr.addstr(height - 1, (width - len(footer)) // 2, footer)

            stdscr.refresh()

        except curses.error:
            pass  # Terminal too small

    def log(self, message: str) -> None:
        """Add a log message."""
        timestamp = time.strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")

    def update_worker(self, worker_id: int, **kwargs) -> None:
        """Update worker metrics directly (for --no-ui mode)."""
        self.metrics.update_worker(worker_id, **kwargs)

    def update_learner(self, **kwargs) -> None:
        """Update learner metrics directly (for --no-ui mode)."""
        self.metrics.update_learner(**kwargs)

    def stop(self) -> None:
        """Stop the UI."""
        self.running = False
