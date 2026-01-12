"""
Curses-based UI for distributed training.
Separates learner and worker outputs into distinct panels.

Keyboard controls:
- q/Q: Quit
- r/R: Force clear and redraw screen
"""

import time
import curses
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
import multiprocessing as mp
from dataclasses import field
from dataclasses import dataclass


class MessageType(Enum):
    LEARNER_STATUS = "learner_status"
    LEARNER_LOG = "learner_log"
    WORKER_STATUS = "worker_status"
    WORKER_LOG = "worker_log"
    WORKER_HEARTBEAT = "worker_heartbeat"
    SYSTEM_LOG = "system_log"
    WORLD_MODEL_STATUS = "world_model_status"
    PPO_STATUS = "ppo_status"


@dataclass(frozen=True)
class UIMessage:
    """Immutable message for UI queue."""

    msg_type: MessageType
    source_id: int  # -1 for learner, 0+ for workers
    data: dict


@dataclass
class TrainingUI:
    """Curses-based UI for distributed training."""

    num_workers: int
    ui_queue: mp.Queue
    use_world_model: bool = False  # Toggle for world model UI
    use_ppo: bool = False  # Toggle for PPO UI

    # State tracking (initialized in __post_init__)
    running: bool = field(init=False, default=True)
    learner_status: Dict[str, Any] = field(init=False, default_factory=dict)
    world_model_status: Dict[str, Any] = field(init=False, default_factory=dict)
    ppo_status: Dict[str, Any] = field(init=False, default_factory=dict)
    worker_statuses: Dict[int, Dict[str, Any]] = field(init=False, repr=False)
    logs: List[str] = field(init=False, default_factory=list)
    max_logs: int = field(init=False, default=100)

    # Loss history for charts
    loss_history: List[float] = field(init=False, default_factory=list)
    wm_loss_history: List[float] = field(init=False, default_factory=list)
    q_loss_history: List[float] = field(init=False, default_factory=list)
    ppo_policy_loss_history: List[float] = field(init=False, default_factory=list)
    ppo_value_loss_history: List[float] = field(init=False, default_factory=list)
    max_history: int = field(init=False, default=100)

    # Global convergence tracking (aggregated from all workers)
    reward_history: List[float] = field(init=False, default_factory=list)
    x_pos_history: List[int] = field(init=False, default_factory=list)
    flag_history: List[int] = field(init=False, default_factory=list)
    total_episodes: int = field(init=False, default=0)
    total_flags: int = field(init=False, default=0)
    first_flag_episode: int = field(init=False, default=0)

    # Graph history (from learner aggregated stats)
    graph_reward_history: List[float] = field(init=False, default_factory=list)
    graph_speed_history: List[float] = field(init=False, default_factory=list)
    graph_entropy_history: List[float] = field(init=False, default_factory=list)
    graph_steps_history: List[int] = field(init=False, default_factory=list)
    max_graph_history: int = field(init=False, default=500)

    def __post_init__(self):
        """Initialize state tracking after dataclass fields are set."""
        self.running = True
        self.learner_status = {}
        self.world_model_status = {}
        self.ppo_status = {}
        self.worker_statuses = {i: {} for i in range(self.num_workers)}
        self.logs = []
        self.max_logs = 100
        self.loss_history = []
        self.wm_loss_history = []
        self.q_loss_history = []
        self.ppo_policy_loss_history = []
        self.ppo_value_loss_history = []
        self.max_history = 100
        # Global convergence tracking
        self.reward_history = []
        self.x_pos_history = []
        self.flag_history = []
        self.total_episodes = 0
        self.total_flags = 0
        self.first_flag_episode = 0
        # Graph history
        self.graph_reward_history = []
        self.graph_speed_history = []
        self.graph_entropy_history = []
        self.graph_steps_history = []
        self.max_graph_history = 500

    def run(self):
        """Main entry point - runs the curses UI."""
        curses.wrapper(self._main)

    def _main(self, stdscr):
        """Main curses loop."""
        try:
            # Setup
            curses.curs_set(0)  # Hide cursor
            stdscr.nodelay(True)  # Non-blocking input
            stdscr.timeout(50)  # 50ms refresh

            # Initialize colors
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_GREEN, -1)  # Success/good
            curses.init_pair(2, curses.COLOR_YELLOW, -1)  # Warning/info
            curses.init_pair(3, curses.COLOR_RED, -1)  # Error/death
            curses.init_pair(4, curses.COLOR_CYAN, -1)  # Learner
            curses.init_pair(5, curses.COLOR_MAGENTA, -1)  # Worker headers

            while self.running:
                try:
                    # Process incoming messages
                    self._process_messages()

                    # Handle input
                    try:
                        key = stdscr.getch()
                        if key == ord("q") or key == ord("Q"):
                            self.running = False
                            break
                        elif key == ord("r") or key == ord("R"):
                            stdscr.clearok(True)
                    except curses.error:
                        pass

                    # Draw UI
                    self._draw(stdscr)
                except Exception as e:
                    # Log error but continue
                    import traceback

                    self._add_log(f"UI Error: {e}")
                    self._add_log(traceback.format_exc()[:100])
        except Exception as e:
            # Fatal error in UI setup
            import sys

            print(f"\nFatal UI error: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()

    def _process_messages(self):
        """Process all pending UI messages."""
        while True:
            try:
                msg: UIMessage = self.ui_queue.get_nowait()

                if msg.msg_type == MessageType.LEARNER_STATUS:
                    self.learner_status = msg.data
                    # Track loss history for standard learner (support both field names)
                    loss = msg.data.get("loss", 0)
                    if loss:
                        self.loss_history.append(loss)
                        if len(self.loss_history) > self.max_history:
                            self.loss_history.pop(0)
                    # Track graph history for distributed DDQN
                    # Support both old (timesteps) and new (total_steps) field names
                    timesteps = msg.data.get("timesteps", msg.data.get("total_steps", 0))
                    avg_reward = msg.data.get("avg_reward", 0)
                    if timesteps > 0:
                        # Only add new data points (avoid duplicates)
                        if not self.graph_steps_history or timesteps > self.graph_steps_history[-1]:
                            self.graph_reward_history.append(avg_reward)
                            self.graph_speed_history.append(msg.data.get("avg_speed", 0))
                            self.graph_entropy_history.append(msg.data.get("avg_entropy", 0))
                            self.graph_steps_history.append(timesteps)
                            # Trim to max history
                            if len(self.graph_reward_history) > self.max_graph_history:
                                self.graph_reward_history.pop(0)
                                self.graph_speed_history.pop(0)
                                self.graph_entropy_history.pop(0)
                                self.graph_steps_history.pop(0)
                elif msg.msg_type == MessageType.LEARNER_LOG:
                    self._add_log(f"[LEARNER] {msg.data.get('text', '')}")
                elif msg.msg_type == MessageType.WORKER_STATUS:
                    old_data = self.worker_statuses.get(msg.source_id, {})
                    new_data = msg.data
                    # Preserve last_heartbeat when updating status
                    if "last_heartbeat" in old_data and "last_heartbeat" not in new_data:
                        new_data["last_heartbeat"] = old_data["last_heartbeat"]
                    self.worker_statuses[msg.source_id] = new_data

                    # Track global convergence metrics when episode ends
                    # Support both old (episode) and new (episodes) field names
                    old_episode = old_data.get("episode", old_data.get("episodes", 0))
                    new_episode = new_data.get("episode", new_data.get("episodes", 0))
                    if new_episode > old_episode and new_episode > 0:
                        # New episode completed - track metrics
                        self.total_episodes += 1
                        reward = new_data.get("rolling_avg_reward", 0)
                        x_pos = new_data.get("x_pos", 0)
                        flags = new_data.get("flags", 0)

                        self.reward_history.append(reward)
                        if len(self.reward_history) > self.max_history:
                            self.reward_history.pop(0)

                        self.x_pos_history.append(x_pos)
                        if len(self.x_pos_history) > self.max_history:
                            self.x_pos_history.pop(0)

                        # Track flag progress
                        if flags > self.total_flags:
                            if self.total_flags == 0 and flags > 0:
                                self.first_flag_episode = self.total_episodes
                            self.total_flags = flags
                        self.flag_history.append(self.total_flags)
                        if len(self.flag_history) > self.max_history:
                            self.flag_history.pop(0)
                elif msg.msg_type == MessageType.WORKER_LOG:
                    self._add_log(f"[W{msg.source_id}] {msg.data.get('text', '')}")
                elif msg.msg_type == MessageType.WORKER_HEARTBEAT:
                    # Update heartbeat timestamp for this worker
                    worker_id = msg.source_id
                    if worker_id not in self.worker_statuses:
                        self.worker_statuses[worker_id] = {}
                    self.worker_statuses[worker_id]["last_heartbeat"] = msg.data.get("timestamp", time.time())
                    # Also update some stats if provided
                    if "episodes" in msg.data:
                        self.worker_statuses[worker_id]["heartbeat_episodes"] = msg.data["episodes"]
                    if "steps" in msg.data:
                        self.worker_statuses[worker_id]["heartbeat_steps"] = msg.data["steps"]
                elif msg.msg_type == MessageType.SYSTEM_LOG:
                    self._add_log(f"[SYSTEM] {msg.data.get('text', '')}")
                elif msg.msg_type == MessageType.WORLD_MODEL_STATUS:
                    self.world_model_status = msg.data
                    self.use_world_model = True  # Auto-detect world model mode
                    # Track loss history for world model
                    wm_metrics = msg.data.get("wm_metrics")
                    if wm_metrics:
                        recon_mse = (
                            wm_metrics.get("recon_mse", 0) if isinstance(wm_metrics, dict) else wm_metrics.recon_mse
                        )
                        self.wm_loss_history.append(float(recon_mse))
                        if len(self.wm_loss_history) > self.max_history:
                            self.wm_loss_history.pop(0)
                    if "q_loss" in msg.data:
                        self.q_loss_history.append(msg.data["q_loss"])
                        if len(self.q_loss_history) > self.max_history:
                            self.q_loss_history.pop(0)
                elif msg.msg_type == MessageType.PPO_STATUS:
                    self.ppo_status = msg.data
                    self.use_ppo = True  # Auto-detect PPO mode
                    # Track loss history for PPO
                    policy_loss = msg.data.get("policy_loss", 0.0)
                    value_loss = msg.data.get("value_loss", 0.0)
                    if policy_loss != 0.0:
                        self.ppo_policy_loss_history.append(policy_loss)
                        if len(self.ppo_policy_loss_history) > self.max_history:
                            self.ppo_policy_loss_history.pop(0)
                    if value_loss != 0.0:
                        self.ppo_value_loss_history.append(value_loss)
                        if len(self.ppo_value_loss_history) > self.max_history:
                            self.ppo_value_loss_history.pop(0)

            except Exception:
                break  # Queue empty

    def _add_log(self, text: str):
        """Add a log entry."""
        timestamp = time.strftime("%H:%M:%S")
        self.logs.append(f"{timestamp} {text}")
        if len(self.logs) > self.max_logs:
            self.logs.pop(0)

    def _draw_sparkline(self, stdscr, y: int, x: int, width: int, data: List[float], label: str = ""):
        """Draw a simple ASCII sparkline chart."""
        if not data or width < 10:
            return

        # Downsample data to fit width if needed
        if len(data) > width:
            # Take evenly spaced samples
            step = len(data) / width
            sampled = [data[int(i * step)] for i in range(width)]
        else:
            sampled = data

        # Normalize data to fit in 8 height levels
        if len(sampled) > 0:
            min_val = min(sampled)
            max_val = max(sampled)
            range_val = max_val - min_val

            if range_val > 0:
                # Map to spark characters (8 levels)
                spark_chars = " ▁▂▃▄▅▆▇█"
                spark_str = ""
                for val in sampled:
                    normalized = (val - min_val) / range_val
                    idx = min(int(normalized * 8), 8)
                    spark_str += spark_chars[idx]

                # Draw the sparkline
                try:
                    if label:
                        stdscr.addstr(y, x, f"{label}: ")
                        x += len(label) + 2
                    stdscr.addstr(y, x, spark_str, curses.A_DIM)
                    # Add min/max values
                    stdscr.addstr(f" {min_val:.3f}→{max_val:.3f}")
                except curses.error:
                    pass  # Ignore if we run out of space

    def _draw_graph_with_axes(
        self,
        stdscr,
        y: int,
        x: int,
        width: int,
        height: int,
        data: List[float],
        x_data: List[int] | None,
        title: str,
        x_label: str = "steps",
        color: int = 1,
    ) -> None:
        """
        Draw an ASCII graph with proper X and Y axes.

        Args:
            stdscr: Curses screen
            y: Top row of graph area
            x: Left column of graph area
            width: Total width including axes
            height: Total height including axes
            data: Y values to plot
            x_data: X values (e.g., step numbers), or None for auto
            title: Graph title
            x_label: X-axis label
            color: Color pair for plot line
        """
        if not data or width < 25 or height < 6:
            return

        # Reserve space for axes labels
        y_axis_width = 7  # "12345 │"
        x_axis_height = 2  # X-axis line + labels
        title_height = 1

        plot_width = width - y_axis_width - 2
        plot_height = height - x_axis_height - title_height

        if plot_width < 10 or plot_height < 3:
            return

        # Downsample ALL data to fit in plot_width (show full history)
        if len(data) > plot_width:
            step = len(data) / plot_width
            display_data = [data[int(i * step)] for i in range(plot_width)]
            if x_data:
                display_x: List[int] | None = [x_data[int(i * step)] for i in range(plot_width)]
            else:
                display_x = None
        else:
            display_data = data
            display_x = x_data

        # Calculate data range - always include 0 in Y-axis for context
        data_min = min(data)  # Use full data range for Y-axis
        data_max = max(data)

        # Y-axis always includes 0 for proper scale context
        if data_min >= 0:
            min_val = 0.0
            max_val = data_max * 1.1  # 10% padding above
        elif data_max <= 0:
            min_val = data_min * 1.1  # 10% padding below
            max_val = 0.0
        else:
            # Data spans across 0
            min_val = data_min * 1.1
            max_val = data_max * 1.1

        range_val = max_val - min_val if max_val != min_val else 1.0

        # Draw title
        try:
            title_x = x + y_axis_width + (plot_width - len(title)) // 2
            stdscr.addstr(y, title_x, title, curses.A_BOLD | curses.color_pair(color))
        except curses.error:
            pass

        # Draw Y-axis with labels and ticks
        for row in range(plot_height):
            try:
                # Calculate Y value for this row
                val = max_val - (row / max(plot_height - 1, 1)) * range_val

                # Format label based on magnitude
                if abs(val) >= 1000:
                    label = f"{val/1000:5.1f}k"
                elif abs(val) >= 1:
                    label = f"{val:6.1f}"
                else:
                    label = f"{val:6.2f}"

                stdscr.addstr(y + title_height + row, x, label)
                stdscr.addstr(y + title_height + row, x + y_axis_width - 1, "│")
            except curses.error:
                pass

        # Draw X-axis
        try:
            axis_y = y + title_height + plot_height
            stdscr.addstr(axis_y, x + y_axis_width - 1, "└" + "─" * plot_width)
        except curses.error:
            pass

        # Draw X-axis labels (always from 0 to current max)
        try:
            label_y = axis_y + 1

            # X-axis always starts at 0
            start_val = 0
            if x_data:
                end_val = x_data[-1]
            else:
                end_val = len(data)
            mid_val = end_val // 2

            # Format function based on magnitude
            def fmt_steps(v: int) -> str:
                if v >= 1_000_000:
                    return f"{v/1_000_000:.1f}M"
                elif v >= 1000:
                    return f"{v/1000:.0f}k"
                else:
                    return f"{v:.0f}"

            start_str = fmt_steps(start_val)
            mid_str = fmt_steps(mid_val)
            end_str = fmt_steps(end_val)

            # Draw start, middle, and end labels
            stdscr.addstr(label_y, x + y_axis_width, start_str)
            mid_x = x + y_axis_width + plot_width // 2 - len(mid_str) // 2
            stdscr.addstr(label_y, mid_x, mid_str, curses.A_DIM)
            stdscr.addstr(label_y, x + y_axis_width + plot_width - len(end_str), end_str)
        except curses.error:
            pass

        # Plot the data using dots - position at actual step value (relative to 0)
        max_step = x_data[-1] if x_data else len(data)
        for i, val in enumerate(display_data):
            try:
                # Normalize value to row position (Y)
                normalized_y = (val - min_val) / range_val if range_val > 0 else 0.5
                normalized_y = max(0, min(1, normalized_y))  # Clamp to [0, 1]
                row = int((1 - normalized_y) * (plot_height - 1))

                # Position X at actual step value (relative to 0 to max_step)
                if display_x and max_step > 0:
                    step_val = display_x[i]
                    col = int((step_val / max_step) * (plot_width - 1))
                else:
                    col = int(i * (plot_width - 1) / max(len(display_data) - 1, 1))

                # Draw the point
                plot_x = x + y_axis_width + col
                plot_y = y + title_height + row

                stdscr.addstr(plot_y, plot_x, "•", curses.color_pair(color))
            except curses.error:
                pass

    def _draw_graphs_section(self, stdscr, y: int, width: int, height: int) -> None:
        """Draw the graphs section with reward, speed, and entropy plots."""
        # Header with data count
        data_count = len(self.graph_reward_history)
        stdscr.addstr(y, 2, f"┌─ GRAPHS ({data_count} pts) ", curses.A_BOLD | curses.color_pair(4))
        stdscr.addstr(y, 22, "─" * (width - 24))

        if not self.graph_reward_history:
            stdscr.addstr(y + 1, 4, "Waiting for data...", curses.A_DIM)
            return

        # Calculate graph dimensions - 3 graphs side by side
        graph_height = height - 2  # Leave room for header
        third_width = (width - 8) // 3

        # Draw reward graph on the left
        self._draw_graph_with_axes(
            stdscr,
            y=y + 1,
            x=2,
            width=third_width,
            height=graph_height,
            data=self.graph_reward_history,
            x_data=self.graph_steps_history,
            title="Avg Reward",
            x_label="steps",
            color=1,  # Green
        )

        # Draw speed graph in the middle
        self._draw_graph_with_axes(
            stdscr,
            y=y + 1,
            x=2 + third_width + 2,
            width=third_width,
            height=graph_height,
            data=self.graph_speed_history,
            x_data=self.graph_steps_history,
            title="Avg Speed",
            x_label="steps",
            color=4,  # Blue/Cyan
        )

        # Draw entropy graph on the right
        self._draw_graph_with_axes(
            stdscr,
            y=y + 1,
            x=2 + 2 * (third_width + 2),
            width=third_width,
            height=graph_height,
            data=self.graph_entropy_history,
            x_data=self.graph_steps_history,
            title="Q-Entropy",
            x_label="steps",
            color=5,  # Magenta
        )

    def _draw(self, stdscr):
        """Draw the entire UI."""
        stdscr.erase()
        height, width = stdscr.getmaxyx()

        # Check minimum terminal size
        min_height = 20
        min_width = 80
        if height < min_height or width < min_width:
            msg = f"Terminal too small! Need at least {min_width}x{min_height}, got {width}x{height}"
            stdscr.addstr(0, 0, msg[: width - 1])
            stdscr.addstr(1, 0, "Please resize your terminal window.", curses.A_BOLD)
            stdscr.refresh()
            return

        # Calculate layout
        header_height = 3
        world_model_height = 7 if self.use_world_model else 0  # Increased for charts
        ppo_height = 7 if self.use_ppo else 0  # PPO section
        learner_height = 7 if not self.use_world_model and not self.use_ppo else 0  # Standard DQN learner
        convergence_height = 4  # Global convergence charts

        # Workers grid: calculate columns and rows dynamically
        min_worker_width = 48
        num_cols = max(1, width // min_worker_width)
        num_rows = (self.num_workers + num_cols - 1) // num_cols
        workers_total_height = num_rows * 4 + 1  # (separator + 3 content) per row + bottom border

        # Graphs section - show if we have distributed DDQN data and enough height
        has_graph_data = len(self.graph_reward_history) > 2
        # Calculate available space for graphs
        used_height = header_height + world_model_height + ppo_height + learner_height + workers_total_height + convergence_height + 3
        available_for_graphs_and_logs = height - used_height
        # Only show graphs if we have enough space (need at least 5 lines for logs)
        graphs_height = 12 if has_graph_data and available_for_graphs_and_logs > 17 else 0

        log_height = max(3, available_for_graphs_and_logs - graphs_height)

        current_y = 0

        # Header
        self._draw_header(stdscr, current_y, width)
        current_y += header_height

        # Learner section (mutually exclusive: World Model, PPO, or standard DQN)
        if self.use_world_model:
            self._draw_world_model(stdscr, current_y, width, world_model_height)
            current_y += world_model_height
        elif self.use_ppo:
            self._draw_ppo(stdscr, current_y, width, ppo_height)
            current_y += ppo_height
        else:
            # Standard DQN Learner section
            self._draw_learner(stdscr, current_y, width, learner_height)
            current_y += learner_height

        # Global convergence section (right under learner)
        self._draw_convergence(stdscr, current_y, width, convergence_height)
        current_y += convergence_height

        # Graphs section (if we have data and enough height)
        if graphs_height > 0:
            self._draw_graphs_section(stdscr, current_y, width, graphs_height)
            current_y += graphs_height

        # Worker sections - compact multi-column grid
        workers_height = self._draw_workers_grid(stdscr, current_y, width)
        current_y += workers_height

        # Separator
        stdscr.addstr(current_y, 0, "─" * (width - 1), curses.A_DIM)
        current_y += 1

        # Log section
        self._draw_logs(stdscr, current_y, width, log_height)

        # Footer
        footer_y = height - 1
        footer = " Press 'q' to quit | 'r' to clear/redraw "
        stdscr.addstr(footer_y, 0, "─" * (width - 1), curses.A_DIM)
        stdscr.addstr(footer_y, (width - len(footer)) // 2, footer, curses.A_DIM)

        stdscr.refresh()

    def _draw_header(self, stdscr, y: int, width: int):
        """Draw the header section."""
        title = "═══ DISTRIBUTED MARIO TRAINING ═══"
        stdscr.addstr(y, (width - len(title)) // 2, title, curses.A_BOLD | curses.color_pair(1))

        info = f"Workers: {self.num_workers}"
        stdscr.addstr(y + 1, 2, info)

        stdscr.addstr(y + 2, 0, "═" * (width - 1))

    def _draw_learner(self, stdscr, y: int, width: int, height: int):
        """Draw the learner section."""
        # Header
        stdscr.addstr(y, 2, "┌─ LEARNER ", curses.A_BOLD | curses.color_pair(4))
        stdscr.addstr(y, 13, "─" * (width - 15))

        ls = self.learner_status
        if ls:
            # Support both old and new field names
            step = ls.get("step", ls.get("update_count", 0))
            loss = ls.get("loss", 0)
            avg_loss = ls.get("avg_loss", 0)
            buf_size = ls.get("buf_size", 0)
            status = ls.get("status", "running")
            q_mean = ls.get("q_mean", 0)
            q_max = ls.get("q_max", 0)
            td_error = ls.get("td_error", 0)
            grad_norm = ls.get("grad_norm", 0)
            reward_mean = ls.get("reward_mean", 0)
            steps_per_sec = ls.get("steps_per_sec", 0)
            queue_msgs_per_sec = ls.get("queue_msgs_per_sec", 0)
            queue_kb_per_sec = ls.get("queue_kb_per_sec", 0)
            # New diagnostics from DDQN learner / new metrics system
            lr = ls.get("lr", ls.get("learning_rate", 0))
            grads_per_sec = ls.get("grads_per_sec", 0)
            gradients_received = ls.get("gradients_received", 0)
            timesteps = ls.get("timesteps", ls.get("total_steps", 0))
            total_episodes = ls.get("total_episodes", ls.get("episodes", 0))
            weight_version = ls.get("weight_version", 0)

            # Progress bar for steps
            max_steps = ls.get("max_steps", 1)
            if max_steps > 0:
                progress = min(step / max_steps, 1.0)
                bar_width = 30
                filled = int(progress * bar_width)
                bar = "█" * filled + "░" * (bar_width - filled)
                pct = f"{progress * 100:.1f}%"
                stdscr.addstr(y + 1, 4, f"Progress: [{bar}] {pct}  ", curses.color_pair(4))
                stdscr.addstr(f"({step:,} / {max_steps:,})  {steps_per_sec:.1f} sps")
            else:
                # Show step count with grads/sec for distributed training
                stdscr.addstr(y + 1, 4, f"Step: {step:,}", curses.color_pair(4))
                if grads_per_sec > 0:
                    stdscr.addstr(f"  {grads_per_sec:.1f} grads/s")
                elif steps_per_sec > 0:
                    stdscr.addstr(f"  {steps_per_sec:.1f} sps")
                # Show timesteps and episodes if available (from distributed DDQN)
                if timesteps > 0:
                    stdscr.addstr(f"  Timesteps: {timesteps:,}")
                if total_episodes > 0:
                    stdscr.addstr(f"  Eps: {total_episodes:,}")

            # Loss stats - adapt based on available data
            stdscr.addstr(y + 2, 4, "Loss: ")
            loss_color = curses.color_pair(1) if avg_loss > 0 and loss < avg_loss else curses.A_NORMAL
            stdscr.addstr(f"{loss:.4f}", loss_color)
            if avg_loss > 0:
                stdscr.addstr(f"  Avg: {avg_loss:.4f}")
            if buf_size > 0:
                stdscr.addstr(f"  Buffer: {buf_size:,}")
            if queue_msgs_per_sec > 0:
                stdscr.addstr(f"  Queue: {queue_msgs_per_sec:.0f} msg/s")
            # Show gradients received for distributed training
            if gradients_received > 0:
                stdscr.addstr(f"  ↓{gradients_received:,} grads")

            # Model-specific stats
            model_type = ls.get("model_type", "ddqn")
            if model_type == "dreamer":
                wm_loss = ls.get("wm_loss", 0)
                actor_loss = ls.get("actor_loss", 0)
                critic_loss = ls.get("critic_loss", 0)
                entropy = ls.get("entropy", 0)
                stdscr.addstr(y + 3, 4, f"WM: {wm_loss:.4f}  Actor: {actor_loss:.4f}  Critic: {critic_loss:.4f}  H: {entropy:.3f}")
            else:
                stdscr.addstr(y + 3, 4, f"Q: μ={q_mean:.2f}")
                if q_max > 0:
                    stdscr.addstr(f" max={q_max:.1f}")
                stdscr.addstr(f"  TD={td_error:.4f}")
            if grad_norm > 0:
                stdscr.addstr(f"  ∇={grad_norm:.1f}")
            if reward_mean != 0:
                stdscr.addstr(f"  r̄={reward_mean:.1f}")

            # Learning rate, weight version, and device (diagnostic info)
            stdscr.addstr(y + 4, 4, "")
            if lr > 0:
                stdscr.addstr("LR=")
                # Color code LR - green if high, yellow if medium, red if very low
                if lr >= 1e-4:
                    lr_color = curses.color_pair(1)  # Green
                elif lr >= 1e-5:
                    lr_color = curses.color_pair(2)  # Yellow
                else:
                    lr_color = curses.color_pair(3)  # Red
                stdscr.addstr(f"{lr:.2e}", lr_color)
                stdscr.addstr("  ")
            if weight_version > 0:
                stdscr.addstr(f"Weights: v{weight_version}")
            # Device info
            device = ls.get("device", "")
            if device:
                stdscr.addstr(f"  📍{device}", curses.color_pair(4))
            # Status indicator
            status_color = curses.color_pair(1) if status == "training" else curses.color_pair(2)
            if status != "running":
                stdscr.addstr(f"  Status: {status}", status_color)

            # Loss chart
            if self.loss_history:
                self._draw_sparkline(stdscr, y + 5, 4, width - 8, self.loss_history, "Loss")
        else:
            stdscr.addstr(y + 1, 4, "Initializing...", curses.A_DIM)

    def _draw_ppo(self, stdscr, y: int, width: int, height: int):
        """Draw the PPO learner section."""
        # Header
        stdscr.addstr(y, 2, "┌─ PPO ", curses.A_BOLD | curses.color_pair(4))
        stdscr.addstr(y, 8, "─" * (width - 10))

        ppo = self.ppo_status
        if ppo:
            step = ppo.get("step", 0)
            timesteps = ppo.get("timesteps", 0)
            total_episodes = ppo.get("total_episodes", 0)
            policy_loss = ppo.get("policy_loss", 0.0)
            value_loss = ppo.get("value_loss", 0.0)
            entropy = ppo.get("entropy", 0.0)
            clip_fraction = ppo.get("clip_fraction", 0.0)
            steps_per_sec = ppo.get("steps_per_sec", 0.0)
            elapsed = ppo.get("elapsed_time", 0.0)

            # Format elapsed time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            # Main stats line with total steps and episodes
            stdscr.addstr(y + 1, 4, f"Step: {step:,}  ", curses.color_pair(4))
            stdscr.addstr(f"Steps: {timesteps:,}  Eps: {total_episodes:,}  {steps_per_sec:.1f} gps  Time: {time_str}")

            # Loss stats
            stdscr.addstr(y + 2, 4, "Policy Loss: ")
            policy_color = curses.color_pair(1) if abs(policy_loss) < 0.1 else curses.color_pair(3)
            stdscr.addstr(f"{policy_loss:.4f}", policy_color)

            stdscr.addstr("  Value Loss: ")
            value_color = curses.color_pair(1) if value_loss < 1.0 else curses.color_pair(3)
            stdscr.addstr(f"{value_loss:.4f}", value_color)

            # Entropy and clip fraction + gradient counts
            grads_recv = ppo.get("gradients_received", 0)
            weight_ver = ppo.get("weight_version", 0)
            stdscr.addstr(y + 3, 4, f"Entropy: {entropy:.4f}  Clip: {clip_fraction:.3f}  ↓{grads_recv} v{weight_ver}")

            # Status indicator
            stdscr.addstr(y + 4, 4, "Status: ", curses.A_DIM)
            stdscr.addstr("training", curses.color_pair(1))

            # Loss charts (side by side)
            chart_width = (width - 20) // 2
            if self.ppo_policy_loss_history:
                stdscr.addstr(y + 5, 4, "π:")
                self._draw_sparkline(stdscr, y + 5, 7, chart_width, self.ppo_policy_loss_history)
            if self.ppo_value_loss_history:
                stdscr.addstr(y + 5, 10 + chart_width, "V:")
                self._draw_sparkline(stdscr, y + 5, 13 + chart_width, chart_width, self.ppo_value_loss_history)
        else:
            stdscr.addstr(y + 1, 4, "Initializing...", curses.A_DIM)

    def _draw_world_model(self, stdscr, y: int, width: int, height: int):
        """Draw the world model learner section with all metrics."""
        # Header
        stdscr.addstr(y, 2, "┌─ WORLD MODEL ", curses.A_BOLD | curses.color_pair(4))
        stdscr.addstr(y, 17, "─" * (width - 19))

        wm = self.world_model_status
        if wm:
            step = wm.get("step", 0)
            wm_step = wm.get("wm_step", 0)
            q_step = wm.get("q_step", 0)
            buf_size = wm.get("buf_size", 0)
            status = wm.get("status", "running")
            phase = wm.get("phase", "world_model")
            phase_step = wm.get("phase_step", 0)
            phase_total = wm.get("phase_total", 1)
            steps_per_sec = wm.get("steps_per_sec", 0)

            # World model metrics
            wm_metrics = wm.get("wm_metrics")

            # Q-network metrics
            q_mean = wm.get("q_mean", 0)
            q_max = wm.get("q_max", 0)
            td_error = wm.get("td_error", 0)
            q_loss = wm.get("q_loss", 0)

            # Phase progress bar
            progress = phase_step / max(phase_total, 1)
            bar_width = 20
            filled = int(progress * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)

            phase_label = "WM" if phase == "world_model" else "Q"
            phase_color = curses.color_pair(4) if phase == "world_model" else curses.color_pair(5)

            stdscr.addstr(y + 1, 4, f"Step: {step:,}  ", curses.color_pair(4))
            stdscr.addstr("Phase: ")
            stdscr.addstr(f"{phase_label}", phase_color | curses.A_BOLD)
            stdscr.addstr(f" [{bar}] {phase_step}/{phase_total}  ")
            stdscr.addstr(f"WM:{wm_step:,} Q:{q_step:,}  {steps_per_sec:.1f} sps  buf={buf_size:,}")

            # World model metrics (if available)
            if wm_metrics:
                recon_mse = wm_metrics.get("recon_mse", 0) if isinstance(wm_metrics, dict) else wm_metrics.recon_mse
                pred_mse = wm_metrics.get("pred_mse", 0) if isinstance(wm_metrics, dict) else wm_metrics.pred_mse
                ssim_val = wm_metrics.get("ssim", 0) if isinstance(wm_metrics, dict) else wm_metrics.ssim
                dynamics_loss = (
                    wm_metrics.get("dynamics_loss", 0) if isinstance(wm_metrics, dict) else wm_metrics.dynamics_loss
                )
                reward_loss = (
                    wm_metrics.get("reward_loss", 0) if isinstance(wm_metrics, dict) else wm_metrics.reward_loss
                )
                kl_loss = wm_metrics.get("kl_loss", 0) if isinstance(wm_metrics, dict) else wm_metrics.kl_loss

                # Color SSIM based on quality
                ssim_color = (
                    curses.color_pair(1)
                    if ssim_val > 0.8
                    else (curses.color_pair(2) if ssim_val > 0.5 else curses.color_pair(3))
                )

                stdscr.addstr(y + 2, 4, f"Recon MSE: {recon_mse:.4f}  Pred MSE: {pred_mse:.4f}  SSIM: ")
                stdscr.addstr(f"{ssim_val:.3f}", ssim_color)
                stdscr.addstr(f"  Dynamics: {dynamics_loss:.4f}  Reward: {reward_loss:.4f}  KL: {kl_loss:.4f}")
            else:
                stdscr.addstr(y + 2, 4, "World model metrics: waiting for training...", curses.A_DIM)

            # Q-network metrics
            stdscr.addstr(
                y + 3, 4, f"Q-value: μ={q_mean:.1f} max={q_max:.1f}  TD-err: {td_error:.2f}  Q-loss: {q_loss:.4f}"
            )

            # Status indicator
            status_color = curses.color_pair(1) if status == "training" else curses.color_pair(2)
            stdscr.addstr(y + 4, 4, f"Status: {status}", status_color)

            # Loss charts
            chart_width = (width - 12) // 2
            if self.wm_loss_history:
                self._draw_sparkline(stdscr, y + 5, 4, chart_width, self.wm_loss_history, "WM Loss")
            if self.q_loss_history:
                self._draw_sparkline(stdscr, y + 6, 4, chart_width, self.q_loss_history, "Q Loss")
        else:
            stdscr.addstr(y + 1, 4, "Initializing world model...", curses.A_DIM)

    def _draw_worker(self, stdscr, y: int, width: int, height: int, worker_id: int):
        """Draw a worker section."""
        ws = self.worker_statuses.get(worker_id, {})

        # Header with level info and device
        if ws:
            # Try new world/stage format first, fall back to current_level string
            world = ws.get("world", 0)
            stage = ws.get("stage", 0)
            if world > 0 and stage > 0:
                current_level = f"{int(world)}-{int(stage)}"
            else:
                current_level = ws.get("current_level", "?")
            device = ws.get("device", "")
        else:
            current_level = "?"
            device = ""
        device_str = f" @{device}" if device else ""
        header = f"├─ WORKER {worker_id} [{current_level}]{device_str} "
        stdscr.addstr(y, 2, header, curses.A_BOLD | curses.color_pair(5))
        stdscr.addstr(y, 2 + len(header), "─" * (width - len(header) - 4))

        if ws:
            # Support both old and new field names
            # Convert to int for fields formatted with :d specifier
            episode = int(ws.get("episode", ws.get("episodes", 0)))
            # episode_reward = last episode's reward (GAUGE)
            # reward = rolling average (ROLLING)
            episode_reward = ws.get("episode_reward", ws.get("reward", 0))
            rolling_avg_reward = ws.get("reward", 0)  # Rolling average from schema
            x_pos = int(ws.get("x_pos", 0))
            best_x = int(ws.get("best_x", 0))
            deaths = int(ws.get("deaths", 0))
            flags = int(ws.get("flags", 0))
            epsilon = ws.get("epsilon", 1.0)
            exp = int(ws.get("experiences", ws.get("buffer_size", 0)))
            q_mean = ws.get("q_mean", 0)
            q_max = ws.get("q_max", 0)
            steps_per_sec = ws.get("steps_per_sec", 0)
            step = int(ws.get("step", ws.get("steps", 0)))
            ws.get("curr_step", 0)
            last_weight_sync = ws.get("last_weight_sync", 0)
            weight_sync_count = int(ws.get("weight_sync_count", 0))
            snapshot_saves = int(ws.get("snapshot_saves", 0))
            snapshot_restores = int(ws.get("snapshot_restores", 0))
            restores_without_progress = int(ws.get("restores_without_progress", 0))
            max_restores = int(ws.get("max_restores", 3))
            # Also support old rolling_avg_reward field
            rolling_avg_reward = ws.get("rolling_avg_reward", rolling_avg_reward)
            ws.get("first_flag_time", 0)
            best_x_ever = int(ws.get("best_x_ever", 0))

            # Calculate time since last weight sync
            if last_weight_sync > 0:
                seconds_ago = int(time.time() - last_weight_sync)
                if seconds_ago < 60:
                    sync_str = f"{seconds_ago}s"
                elif seconds_ago < 3600:
                    sync_str = f"{seconds_ago // 60}m"
                else:
                    sync_str = f"{seconds_ago // 3600}h"
            else:
                sync_str = "never"

            # Main stats line with model-specific metrics and weight sync
            game_time = int(ws.get("game_time", 0))
            model_type = ws.get("model_type", "ddqn")
            
            # Build model-specific metric string
            if model_type == "dreamer":
                wm_loss = ws.get("wm_loss", 0)
                actor_loss = ws.get("actor_loss", 0)
                metric_str = f"WM: {wm_loss:.3f}  Act: {actor_loss:.3f}"
            else:
                metric_str = f"Q: {q_mean:.1f}/{q_max:.1f}"
            
            stats = f"Ep: {episode:4d}  Step: {step:4d}  X: {x_pos:4d}  ⏱ {game_time:3d}  Best: {best_x:4d}  {metric_str}  {steps_per_sec:.0f} sps  Wgt: {sync_str}"
            stdscr.addstr(y + 1, 4, stats)

            # Secondary stats
            death_color = curses.color_pair(3) if deaths > 0 else curses.A_NORMAL
            flag_color = curses.color_pair(1) if flags > 0 else curses.A_NORMAL

            stdscr.addstr(y + 2, 4, f"r={episode_reward:7.1f}  💀=")
            stdscr.addstr(f"{deaths:2d}", death_color)
            # Show saves and restores: saves/restores(stuck/max) - color red if near limit
            restore_color = curses.color_pair(3) if restores_without_progress >= max_restores - 1 else curses.A_NORMAL
            stdscr.addstr(f"  💾 {snapshot_saves:2d}")
            stdscr.addstr(f"  ⏮ {snapshot_restores:2d}")
            stdscr.addstr(f"({restores_without_progress}/{max_restores})", restore_color)
            stdscr.addstr("  🏁=")
            stdscr.addstr(f"{flags:2d}", flag_color)
            grads_sent = ws.get("gradients_sent", ws.get("grads_sent", 0))
            stdscr.addstr(f"  ε={epsilon:.3f}  Exp={exp:,}  ↑{grads_sent} ↓{weight_sync_count}")

            # Convergence metrics line (rolling avg + speed + flag time)
            avg_color = curses.color_pair(1) if rolling_avg_reward > 0 else curses.color_pair(3)
            # Support both old (avg_speed) and new (speed) field names
            avg_speed = ws.get("avg_speed", ws.get("speed", 0))
            avg_x_at_death = ws.get("avg_x_at_death", 0)
            avg_time_to_flag = ws.get("avg_time_to_flag", 0)
            total_deaths = ws.get("total_deaths", ws.get("deaths", 0))

            # Buffer diagnostics
            buffer_fill_pct = ws.get("buffer_fill_pct", 0)
            can_train = ws.get("can_train", False)

            stdscr.addstr(y + 3, 4, "r̄=")
            stdscr.addstr(f"{rolling_avg_reward:6.1f}", avg_color)
            stdscr.addstr(f"  BestX={best_x_ever:4d}")
            stdscr.addstr(f"  Spd={avg_speed:4.1f}x/t")
            stdscr.addstr(f"  💀X̄={avg_x_at_death:4.0f}")

            if avg_time_to_flag > 0:
                stdscr.addstr("  🏁T̄=", curses.A_BOLD)
                stdscr.addstr(f"{avg_time_to_flag:3.0f}", curses.color_pair(1) | curses.A_BOLD)
            else:
                stdscr.addstr("  🏁T̄=", curses.A_DIM)
                stdscr.addstr("---", curses.A_DIM)

            # Buffer fill indicator with color coding
            stdscr.addstr("  Buf=")
            if buffer_fill_pct >= 50:
                buf_color = curses.color_pair(1)  # Green
            elif buffer_fill_pct >= 10:
                buf_color = curses.color_pair(2)  # Yellow
            else:
                buf_color = curses.color_pair(3)  # Red
            stdscr.addstr(f"{buffer_fill_pct:4.1f}%", buf_color)

            # Elite buffer indicator
            elite_size = ws.get("elite_size", 0)
            elite_capacity = ws.get("elite_capacity", 1000)
            if elite_size > 0:
                elite_pct = elite_size / elite_capacity * 100
                stdscr.addstr(f" E={elite_size}", curses.color_pair(4))  # Cyan

            # Training indicator
            if can_train:
                stdscr.addstr(" ✓", curses.color_pair(1))
            else:
                stdscr.addstr(" ✗", curses.color_pair(3) | curses.A_DIM)

            # Show seconds since last action (for stuck detection)
            last_action_time = ws.get("last_action_time", 0)
            if last_action_time > 0:
                idle_secs = int(time.time() - last_action_time)
                if idle_secs > 5:
                    # Worker appears stuck
                    stdscr.addstr(f"  ⚠ idle {idle_secs}s", curses.color_pair(3) | curses.A_BOLD)
        else:
            stdscr.addstr(y + 1, 4, "Starting...", curses.A_DIM)

    def _draw_workers_grid(self, stdscr, y: int, width: int) -> int:
        """Draw workers in a compact multi-column grid layout. Returns height used."""
        screen_height = stdscr.getmaxyx()[0]

        # Calculate columns based on width (minimum 48 chars per worker including borders)
        min_worker_width = 48
        num_cols = max(1, width // min_worker_width)
        col_width = width // num_cols

        # Calculate rows needed
        num_rows = (self.num_workers + num_cols - 1) // num_cols
        content_lines = 3  # 3 lines of worker content per row

        rows_drawn = 0
        current_y = y

        for row in range(num_rows):
            # Check if we have space for this row (need content_lines + 1 for separator)
            if current_y + content_lines >= screen_height - 3:
                break

            # Draw horizontal separator line
            try:
                line_parts = []
                for col in range(num_cols):
                    if row == 0:
                        # Top border
                        left = "┌" if col == 0 else "┬"
                    else:
                        # Row separator
                        left = "├" if col == 0 else "┼"
                    line_parts.append(left + "─" * (col_width - 1))
                # Add right edge
                right = "┐" if row == 0 else "┤"
                stdscr.addstr(current_y, 0, "".join(line_parts)[:width-1] + right, curses.A_DIM)
            except curses.error:
                pass
            current_y += 1

            # Draw worker content for this row
            for line in range(content_lines):
                for col in range(num_cols):
                    worker_id = row * num_cols + col
                    col_x = col * col_width

                    # Draw left border
                    try:
                        stdscr.addstr(current_y, col_x, "│", curses.A_DIM)
                    except curses.error:
                        pass

                    # Draw worker content on first pass only
                    if line == 0 and worker_id < self.num_workers:
                        self._draw_worker_compact(stdscr, current_y, col_x + 1, col_width - 1, worker_id)

                # Draw right border
                try:
                    stdscr.addstr(current_y, num_cols * col_width, "│", curses.A_DIM)
                except curses.error:
                    pass

                current_y += 1

            rows_drawn += 1

        # Draw bottom border
        if rows_drawn > 0:
            try:
                line_parts = []
                for col in range(num_cols):
                    left = "└" if col == 0 else "┴"
                    line_parts.append(left + "─" * (col_width - 1))
                stdscr.addstr(current_y, 0, "".join(line_parts)[:width-1] + "┘", curses.A_DIM)
            except curses.error:
                pass
            current_y += 1

        return current_y - y

    def _draw_worker_compact(self, stdscr, y: int, x: int, width: int, worker_id: int):
        """Draw a compact 3-line worker display within a column."""
        ws = self.worker_statuses.get(worker_id, {})

        # Header with worker ID, level, and device
        if ws:
            # Try new world/stage format first, fall back to current_level string
            world = ws.get("world", 0)
            stage = ws.get("stage", 0)
            if world > 0 and stage > 0:
                current_level = f"{int(world)}-{int(stage)}"
            else:
                current_level = ws.get("current_level", "?")
            # Get device (e.g., "cuda:0" -> "0", "cpu" -> "cpu")
            device = ws.get("device", "")
            if device.startswith("cuda:"):
                device_short = f"G{device[5:]}"  # "cuda:0" -> "G0"
            elif device:
                device_short = device[:3]  # "cpu" -> "cpu", "mps" -> "mps"
            else:
                device_short = ""
        else:
            current_level = "?"
            device_short = ""
        header = f"W{worker_id}[{current_level}]" + (f"({device_short})" if device_short else "")
        
        # Calculate heartbeat status
        last_heartbeat = ws.get("last_heartbeat", 0) if ws else 0
        if last_heartbeat > 0:
            seconds_since_heartbeat = int(time.time() - last_heartbeat)
            if seconds_since_heartbeat < 60:
                heartbeat_status = "💚"  # Recent heartbeat (green)
                heartbeat_attr = curses.color_pair(1)  # Green
            elif seconds_since_heartbeat < 120:
                heartbeat_status = "💛"  # Stale heartbeat (yellow)
                heartbeat_attr = curses.color_pair(2)  # Yellow
            else:
                heartbeat_status = "💔"  # No heartbeat (red)
                heartbeat_attr = curses.color_pair(3)  # Red
        else:
            heartbeat_status = "⚪"  # No heartbeat data yet
            heartbeat_attr = curses.A_DIM
        
        try:
            stdscr.addstr(y, x, header, curses.A_BOLD | curses.color_pair(5))
            stdscr.addstr(y, x + len(header), heartbeat_status, heartbeat_attr)
        except curses.error:
            return

        if not ws:
            try:
                stdscr.addstr(y, x + len(header) + 2, "Starting...", curses.A_DIM)
            except curses.error:
                pass
            return

        # Extract all metrics (support both old and new field names)
        # Convert to int for fields formatted with :d specifier
        episode = int(ws.get("episode", ws.get("episodes", 0)))
        step = int(ws.get("step", ws.get("steps", 0)))
        x_pos = int(ws.get("x_pos", 0))
        game_time = int(ws.get("game_time", 0))
        best_x_ever = int(ws.get("best_x_ever", 0))
        epsilon = ws.get("epsilon", 1.0)
        steps_per_sec = ws.get("steps_per_sec", 0)
        # episode_reward = last episode's reward (GAUGE), reward = rolling average
        episode_reward = ws.get("episode_reward", ws.get("reward", 0))
        rolling_avg_reward = ws.get("reward", 0)
        rolling_avg_reward = ws.get("rolling_avg_reward", rolling_avg_reward)  # Support old field
        deaths = int(ws.get("deaths", 0))
        flags = int(ws.get("flags", 0))
        buffer_fill_pct = ws.get("buffer_fill_pct", 0)
        buffer_size = int(ws.get("buffer_size", 0))
        # Compute buffer fill % from buffer_size if available
        if buffer_fill_pct == 0 and buffer_size > 0:
            buffer_fill_pct = min(100.0, buffer_size / 100.0)  # Rough estimate
        can_train = ws.get("can_train", buffer_size >= 32)
        # Support both old (avg_speed) and new (speed) field names
        avg_speed = ws.get("avg_speed", ws.get("speed", 0))
        total_deaths = int(ws.get("total_deaths", ws.get("deaths", 0)))
        snapshot_saves = int(ws.get("snapshot_saves", 0))
        snapshot_restores = int(ws.get("snapshot_restores", 0))
        last_weight_sync = ws.get("last_weight_sync", 0)

        # Calculate sync time string
        if last_weight_sync > 0:
            seconds_ago = int(time.time() - last_weight_sync)
            sync_str = f"{seconds_ago}s" if seconds_ago < 60 else f"{seconds_ago // 60}m"
        else:
            sync_str = "-"

        try:
            # Line 1: header + heartbeat + Ep, Step, X, Time, sps
            line1 = f" Ep:{episode:3d} St:{step:3d} X:{x_pos:4d} ⏱ {game_time:3d} {steps_per_sec:2.0f}sps"
            header_length = len(header) + 1  # +1 for heartbeat emoji
            stdscr.addstr(y, x + header_length, line1[: width - header_length - 1])

            # Line 2: reward, deaths, flags, restores, epsilon
            stdscr.addstr(y + 1, x, f"r={episode_reward:6.1f} 💀 ")

            death_color = curses.color_pair(3) if deaths > 0 else curses.A_DIM
            stdscr.addstr(f"{deaths:2d}", death_color)

            flag_color = curses.color_pair(1) if flags > 0 else curses.A_DIM
            stdscr.addstr(" 🏁 ")
            stdscr.addstr(f"{flags}", flag_color)

            stdscr.addstr(f" 💾 {snapshot_saves} ⏮ {snapshot_restores}")
            stdscr.addstr(f" ε={epsilon:.2f}")

            # Line 3: avg reward, BestX, Spd, Buf%, sync
            stdscr.addstr(y + 2, x, "r̄=")
            avg_color = curses.color_pair(1) if rolling_avg_reward > 0 else curses.color_pair(3)
            stdscr.addstr(f"{rolling_avg_reward:6.1f}", avg_color)

            stdscr.addstr(f" Bst:{best_x_ever:4d}")
            stdscr.addstr(f" Spd:{avg_speed:4.1f}")

            # Buffer fill with color
            if buffer_fill_pct >= 50:
                buf_color = curses.color_pair(1)
            elif buffer_fill_pct >= 10:
                buf_color = curses.color_pair(2)
            else:
                buf_color = curses.color_pair(3)
            stdscr.addstr(" ")
            stdscr.addstr(f"{buffer_fill_pct:3.0f}%", buf_color)

            # Elite buffer indicator
            elite_size = ws.get("elite_size", 0)
            if elite_size > 0:
                stdscr.addstr(f"E{elite_size}", curses.color_pair(4))

            # Training indicator
            train_char = "✓" if can_train else "✗"
            train_color = curses.color_pair(1) if can_train else curses.color_pair(3)
            stdscr.addstr(train_char, train_color)

            # Sync indicator
            stdscr.addstr(f" ↓{sync_str}")

        except curses.error:
            pass  # Ignore if we can't fit

    def _draw_convergence(self, stdscr, y: int, width: int, height: int):
        """Draw global convergence metrics with sparklines."""
        # Title
        stdscr.addstr(y, 2, "┌─ CONVERGENCE ", curses.A_BOLD)
        stdscr.addstr(y, 16, "─" * (width - 18))

        # Summary stats
        avg_reward = sum(self.reward_history) / len(self.reward_history) if self.reward_history else 0.0
        best_x = max(self.x_pos_history) if self.x_pos_history else 0

        # Color for average reward
        if avg_reward > 0:
            reward_color = curses.color_pair(1)  # Green
        elif avg_reward < -50:
            reward_color = curses.color_pair(3)  # Red
        else:
            reward_color = curses.A_NORMAL

        # First line: Summary stats
        stdscr.addstr(y + 1, 4, f"Episodes: {self.total_episodes:5d}  ")
        stdscr.addstr("Avg r̄₁₀₀: ")
        stdscr.addstr(f"{avg_reward:7.1f}", reward_color)
        stdscr.addstr(f"  Best X: {best_x:5d}  ")
        stdscr.addstr("🏁: ")
        flag_color = curses.color_pair(1) if self.total_flags > 0 else curses.A_DIM
        stdscr.addstr(f"{self.total_flags:3d}", flag_color)

        if self.first_flag_episode > 0:
            stdscr.addstr(f"  (1st @ ep {self.first_flag_episode})", curses.color_pair(1))

        # Second line: Reward history sparkline
        chart_width = (width - 40) // 2
        if chart_width > 10 and self.reward_history:
            stdscr.addstr(y + 2, 4, "Reward: ")
            self._draw_sparkline(stdscr, y + 2, 12, chart_width, self.reward_history)

            # X position history sparkline
            stdscr.addstr(y + 2, 15 + chart_width, "X-pos: ")
            if self.x_pos_history:
                self._draw_sparkline(
                    stdscr, y + 2, 22 + chart_width, chart_width, [float(x) for x in self.x_pos_history]
                )
        elif not self.reward_history:
            stdscr.addstr(y + 2, 4, "Waiting for episode completions...", curses.A_DIM)

        # Third line: Min/max for the sparklines
        if self.reward_history and len(self.reward_history) >= 2:
            r_min, r_max = min(self.reward_history), max(self.reward_history)
            stdscr.addstr(y + 3, 4, f"r∈[{r_min:.1f},{r_max:.1f}]", curses.A_DIM)

            if self.x_pos_history and len(self.x_pos_history) >= 2:
                x_min, x_max = min(self.x_pos_history), max(self.x_pos_history)
                stdscr.addstr(y + 3, 15 + chart_width, f"x∈[{x_min},{x_max}]", curses.A_DIM)

    def _draw_logs(self, stdscr, y: int, width: int, height: int):
        """Draw the log section."""
        stdscr.addstr(y, 2, "┌─ LOGS ", curses.A_BOLD)
        stdscr.addstr(y, 10, "─" * (width - 12))

        # Show most recent logs that fit
        visible_logs = self.logs[-(height - 1) :] if height > 1 else []
        for i, log in enumerate(visible_logs):
            if y + 1 + i < stdscr.getmaxyx()[0] - 1:
                # Truncate if too long
                display_log = log[: width - 6] if len(log) > width - 6 else log

                # Color based on content
                if "[LEARNER]" in log:
                    color = curses.color_pair(4)
                elif "FLAG" in log or "🏁" in log:
                    color = curses.color_pair(1)
                elif "Synced" in log or "🔄" in log:
                    color = curses.color_pair(2)
                else:
                    color = curses.A_NORMAL

                try:
                    stdscr.addstr(y + 1 + i, 4, display_log, color)
                except curses.error:
                    pass  # Ignore if we can't fit


def create_ui_queue() -> mp.Queue:
    """Create a queue for UI messages."""
    return mp.Queue(maxsize=1000)


def send_learner_status(
    queue: mp.Queue,
    step: int,
    loss: float,
    avg_loss: float,
    buf_size: int,
    pulled: int,
    max_steps: int,
    status: str = "training",
    q_mean: float = 0.0,
    q_max: float = 0.0,
    td_error: float = 0.0,
    grad_norm: float = 0.0,
    reward_mean: float = 0.0,
    steps_per_sec: float = 0.0,
    queue_msgs_per_sec: float = 0.0,
    queue_kb_per_sec: float = 0.0,
    # Dreamer-specific metrics (optional)
    wm_loss: float | None = None,
    actor_loss: float | None = None,
    critic_loss: float | None = None,
    entropy: float | None = None,
    model_type: str = "ddqn",
):
    """Send learner status update to UI."""
    try:
        data = {
            "step": step,
            "loss": loss,
            "avg_loss": avg_loss,
            "buf_size": buf_size,
            "pulled": pulled,
            "max_steps": max_steps,
            "status": status,
            "q_mean": q_mean,
            "q_max": q_max,
            "td_error": td_error,
            "grad_norm": grad_norm,
            "reward_mean": reward_mean,
            "steps_per_sec": steps_per_sec,
            "queue_msgs_per_sec": queue_msgs_per_sec,
            "queue_kb_per_sec": queue_kb_per_sec,
            "model_type": model_type,
        }
        
        # Add Dreamer metrics if provided
        if wm_loss is not None:
            data["wm_loss"] = wm_loss
        if actor_loss is not None:
            data["actor_loss"] = actor_loss
        if critic_loss is not None:
            data["critic_loss"] = critic_loss
        if entropy is not None:
            data["entropy"] = entropy
            
        queue.put_nowait(
            UIMessage(
                msg_type=MessageType.LEARNER_STATUS,
                source_id=-1,
                data=data,
            )
        )
    except Exception:
        pass  # Queue full, skip update


def send_learner_log(queue: mp.Queue, text: str):
    """Send learner log message to UI."""
    try:
        queue.put_nowait(UIMessage(msg_type=MessageType.LEARNER_LOG, source_id=-1, data={"text": text}))
    except Exception:
        pass


def send_worker_status(
    queue: mp.Queue,
    worker_id: int,
    episode: int,
    reward: float,
    x_pos: int,
    best_x: int,
    deaths: int,
    flags: int,
    epsilon: float,
    experiences: int,
    q_mean: float = 0.0,
    q_max: float = 0.0,
    steps_per_sec: float = 0.0,
    step: int = 0,
    curr_step: int = 0,
    last_weight_sync: float = 0.0,
    weight_sync_count: int = 0,
    snapshot_restores: int = 0,
    current_level: str = "?",
    game_time: int = 0,
    # Convergence metrics
    rolling_avg_reward: float = 0.0,
    first_flag_time: float = 0.0,
    best_x_ever: int = 0,
    # Dreamer-specific metrics (optional)
    wm_loss: float | None = None,
    actor_loss: float | None = None,
    critic_loss: float | None = None,
    entropy: float | None = None,
    model_type: str = "ddqn",
):
    """Send worker status update to UI."""
    try:
        data = {
            "episode": episode,
            "reward": reward,
            "x_pos": x_pos,
            "best_x": best_x,
            "deaths": deaths,
            "flags": flags,
            "epsilon": epsilon,
            "experiences": experiences,
            "q_mean": q_mean,
            "q_max": q_max,
            "steps_per_sec": steps_per_sec,
            "step": step,
            "curr_step": curr_step,
            "last_weight_sync": last_weight_sync,
            "weight_sync_count": weight_sync_count,
            "snapshot_restores": snapshot_restores,
            "current_level": current_level,
            "game_time": game_time,
            # Convergence metrics
            "rolling_avg_reward": rolling_avg_reward,
            "first_flag_time": first_flag_time,
            "best_x_ever": best_x_ever,
            # Model type for UI to know which metrics to show
            "model_type": model_type,
        }
        
        # Add Dreamer metrics if provided
        if wm_loss is not None:
            data["wm_loss"] = wm_loss
        if actor_loss is not None:
            data["actor_loss"] = actor_loss
        if critic_loss is not None:
            data["critic_loss"] = critic_loss
        if entropy is not None:
            data["entropy"] = entropy
            
        queue.put_nowait(
            UIMessage(
                msg_type=MessageType.WORKER_STATUS,
                source_id=worker_id,
                data=data,
            )
        )
    except Exception:
        pass


def send_worker_log(queue: mp.Queue, worker_id: int, text: str):
    """Send worker log message to UI."""
    try:
        queue.put_nowait(
            UIMessage(
                msg_type=MessageType.WORKER_LOG,
                source_id=worker_id,
                data={"text": text},
            )
        )
    except Exception:
        pass


def send_system_log(queue: mp.Queue, text: str):
    """Send system log message to UI."""
    try:
        queue.put_nowait(UIMessage(msg_type=MessageType.SYSTEM_LOG, source_id=-1, data={"text": text}))
    except Exception:
        pass


def send_world_model_status(
    queue: mp.Queue,
    step: int,
    wm_step: int,
    q_step: int,
    buf_size: int,
    pulled: int,
    max_steps: int,
    status: str = "training",
    phase: str = "world_model",
    phase_step: int = 0,
    phase_total: int = 1,
    # World model metrics
    wm_metrics: Any = None,
    # Q-network metrics
    q_mean: float = 0.0,
    q_max: float = 0.0,
    td_error: float = 0.0,
    q_loss: float = 0.0,
    grad_norm: float = 0.0,
    reward_mean: float = 0.0,
    steps_per_sec: float = 0.0,
    queue_msgs_per_sec: float = 0.0,
    queue_kb_per_sec: float = 0.0,
):
    """Send world model learner status update to UI."""
    try:
        # Convert WorldModelMetrics to dict if needed
        wm_metrics_dict = None
        if wm_metrics is not None:
            if hasattr(wm_metrics, "_asdict"):
                wm_metrics_dict = wm_metrics._asdict()
            elif isinstance(wm_metrics, dict):
                wm_metrics_dict = wm_metrics
            else:
                wm_metrics_dict = {
                    "total_loss": getattr(wm_metrics, "total_loss", 0),
                    "recon_mse": getattr(wm_metrics, "recon_mse", 0),
                    "pred_mse": getattr(wm_metrics, "pred_mse", 0),
                    "ssim": getattr(wm_metrics, "ssim", 0),
                    "dynamics_loss": getattr(wm_metrics, "dynamics_loss", 0),
                    "reward_loss": getattr(wm_metrics, "reward_loss", 0),
                    "kl_loss": getattr(wm_metrics, "kl_loss", 0),
                }

        queue.put_nowait(
            UIMessage(
                msg_type=MessageType.WORLD_MODEL_STATUS,
                source_id=-1,
                data={
                    "step": step,
                    "wm_step": wm_step,
                    "q_step": q_step,
                    "buf_size": buf_size,
                    "pulled": pulled,
                    "max_steps": max_steps,
                    "status": status,
                    "phase": phase,
                    "phase_step": phase_step,
                    "phase_total": phase_total,
                    "wm_metrics": wm_metrics_dict,
                    "q_mean": q_mean,
                    "q_max": q_max,
                    "td_error": td_error,
                    "q_loss": q_loss,
                    "grad_norm": grad_norm,
                    "reward_mean": reward_mean,
                    "steps_per_sec": steps_per_sec,
                    "queue_msgs_per_sec": queue_msgs_per_sec,
                    "queue_kb_per_sec": queue_kb_per_sec,
                },
            )
        )
    except Exception:
        pass  # Queue full, skip update


def run_ui(num_workers: int, ui_queue: mp.Queue):
    """Entry point for UI process."""
    ui = TrainingUI(num_workers=num_workers, ui_queue=ui_queue)
    ui.run()
