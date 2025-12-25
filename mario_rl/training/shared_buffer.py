"""
Shared replay buffer for distributed training.

Workers push experiences to a queue, learner maintains local buffer for sampling.
"""

import sys
import multiprocessing as mp

from typing import Any
from typing import List
from typing import Tuple
from typing import Optional
from pathlib import Path
from collections import deque
from dataclasses import field
from dataclasses import dataclass

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mario_rl.agent.replay import Memory
from mario_rl.agent.replay import Experience
from mario_rl.agent.replay import ExperienceBatch
from mario_rl.agent.replay import prepare_batch
from mario_rl.agent.replay import pack_experience
from mario_rl.agent.replay import unpack_experience


@dataclass
class SequenceBatch:
    """Batch of sequential experiences for dynamics training."""

    states: np.ndarray  # (batch, seq_len, F, H, W, C)
    actions: np.ndarray  # (batch, seq_len)
    rewards: np.ndarray  # (batch, seq_len)
    dones: np.ndarray  # (batch, seq_len)


class SharedReplayBuffer:
    """
    Multiprocessing-safe replay buffer.

    Workers push experiences via push() to a shared queue.
    Learner calls pull_all() to get new experiences, then samples locally.
    
    The queue must be created BEFORE spawning processes and passed in.
    """

    def __init__(self, max_len: int, queue: Optional[mp.Queue] = None):
        """
        Args:
            max_len: Maximum size of local replay buffer
            queue: Shared multiprocessing Queue (create with mp.Queue() before spawning)
                   If None, creates a new queue (only works with fork, not spawn)
        """
        self.max_len = max_len
        
        # Queue for cross-process communication
        # Must be created in parent before spawn and passed to children
        self._queue = queue if queue is not None else mp.Queue(maxsize=10000)
        
        # Local memory (each process has its own copy)
        self._memory: deque = deque(maxlen=max_len)
        self._priorities: Optional[np.ndarray] = None
        
        # Local throughput stats (not shared between processes)
        self._total_messages = 0
        self._total_bytes = 0
        self._last_throughput_time = 0.0
        self._last_messages = 0
        self._last_bytes = 0
        self._msgs_per_sec = 0.0
        self._kb_per_sec = 0.0

    def __len__(self) -> int:
        return len(self._memory)

    def __getitem__(self, i: int) -> Experience:
        return unpack_experience(self._memory[i])

    def _clear_caches(self):
        self._priorities = None

    @property
    def priorities(self) -> np.ndarray:
        if self._priorities is None:
            self._priorities = np.array([m["p"] for m in self._memory], "f4")
        return self._priorities

    def _update_throughput(self):
        """Update throughput statistics (local to this process)."""
        import time
        
        now = time.time()
        elapsed = now - self._last_throughput_time
        
        # Update every second
        if elapsed >= 1.0:
            msg_delta = self._total_messages - self._last_messages
            byte_delta = self._total_bytes - self._last_bytes
            
            self._msgs_per_sec = msg_delta / elapsed
            self._kb_per_sec = (byte_delta / 1024.0) / elapsed
            
            self._last_throughput_time = now
            self._last_messages = self._total_messages
            self._last_bytes = self._total_bytes

    def get_throughput(self) -> tuple[float, float]:
        """Get current throughput (msgs/s, KB/s)."""
        return self._msgs_per_sec, self._kb_per_sec

    # === Worker methods ===

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        actions: List[int],
        priority: float = 0.1,
    ) -> bool:
        """
        Push an experience to the shared queue (called by workers).
        Returns True if successful, False if queue is full.
        """
        try:
            packed = pack_experience(
                s=state,
                a=action,
                r=reward,
                s_=next_state,
                d=done,
                a_=actions,
                p=priority,
            )
            
            self._queue.put_nowait(packed)
            
            # Track throughput (local stats, not shared)
            self._total_messages += 1
            byte_count = state.nbytes + next_state.nbytes + 100
            self._total_bytes += byte_count
            
            return True
        except Exception:
            # Queue full, drop experience
            return False

    # === Learner methods ===

    def pull_all(self) -> int:
        """
        Pull all available experiences from queue into local buffer.
        Called by learner before sampling. Returns number of experiences pulled.
        """
        count = 0
        while not self._queue.empty():
            try:
                packed = self._queue.get_nowait()
                self._memory.append(packed)
                self._clear_caches()
                count += 1
            except Exception:
                break
        
        # Update throughput stats after pulling
        self._update_throughput()
        
        return count

    def sample(self, batch_size: int) -> Tuple[ExperienceBatch, np.ndarray]:
        """
        Sample a batch from local buffer using priority sampling.
        Called by learner after pull_all().
        """
        if len(self._memory) < batch_size:
            raise ValueError(
                f"Not enough experiences: {len(self._memory)} < {batch_size}"
            )

        p = np.abs(self.priorities) + 1e-36
        p /= p.sum()

        indices = np.random.choice(
            np.arange(0, len(p)),
            size=batch_size,
            p=p,
            replace=False,
        )

        return prepare_batch((self[i] for i in indices)), indices

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences."""
        self._clear_caches()
        for i, p in zip(indices, priorities):
            self._memory[i]["p"] = float(p)

    def sample_sequences(
        self, batch_size: int, seq_len: int = 4
    ) -> Tuple[SequenceBatch, np.ndarray]:
        """
        Sample consecutive frame sequences for dynamics/temporal training.

        Each sequence contains seq_len consecutive experiences from the same episode.

        Args:
            batch_size: Number of sequences to sample
            seq_len: Length of each sequence

        Returns:
            SequenceBatch with consecutive experiences, and starting indices
        """
        if len(self._memory) < batch_size * seq_len:
            raise ValueError(
                f"Not enough experiences: {len(self._memory)} < {batch_size * seq_len}"
            )

        # Find valid starting positions (sequences that don't cross episode boundaries)
        valid_starts = []
        for i in range(len(self._memory) - seq_len + 1):
            # Check that no experience in the sequence is terminal (except possibly the last)
            valid = True
            for j in range(seq_len - 1):
                exp = unpack_experience(self._memory[i + j])
                if exp.done:
                    valid = False
                    break
            if valid:
                valid_starts.append(i)

        if len(valid_starts) < batch_size:
            # Fall back to random sampling if not enough valid sequences
            valid_starts = list(range(len(self._memory) - seq_len + 1))

        # Sample starting indices
        if len(valid_starts) < batch_size:
            raise ValueError(
                f"Not enough valid sequences: {len(valid_starts)} < {batch_size}"
            )

        indices = np.random.choice(valid_starts, size=batch_size, replace=False)

        # Collect sequences
        states_list = []
        actions_list = []
        rewards_list = []
        dones_list = []

        for start_idx in indices:
            seq_states = []
            seq_actions = []
            seq_rewards = []
            seq_dones = []

            for offset in range(seq_len):
                exp = unpack_experience(self._memory[start_idx + offset])
                seq_states.append(np.array(exp.state))
                seq_actions.append(exp.action)
                seq_rewards.append(exp.reward)
                seq_dones.append(exp.done)

            states_list.append(np.stack(seq_states))
            actions_list.append(np.array(seq_actions))
            rewards_list.append(np.array(seq_rewards))
            dones_list.append(np.array(seq_dones))

        batch = SequenceBatch(
            states=np.stack(states_list),
            actions=np.stack(actions_list),
            rewards=np.stack(rewards_list),
            dones=np.stack(dones_list),
        )

        return batch, indices

    def get_queue_size(self) -> int:
        """Approximate size of the queue (may not be exact due to race conditions)."""
        try:
            return self._queue.qsize()
        except NotImplementedError:
            # qsize() not available on macOS
            return -1
