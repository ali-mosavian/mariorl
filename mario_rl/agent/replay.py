# +
import os
import pickle

from typing import List
from typing import Deque
from typing import Tuple
from typing import Union
from typing import Literal
from typing import TypeVar
from typing import Iterable
from typing import Optional
from typing import Sequence
from typing import TypedDict
from typing import NamedTuple
from itertools import starmap
from collections import deque
from dataclasses import dataclass

import numpy as np
import lz4.block
import lz4.frame

from frame_stack import LazyFrames


T = TypeVar("T")


class Experience(NamedTuple):
    state: Union[LazyFrames, np.ndarray]
    action: int
    reward: float
    state_next: Union[LazyFrames, np.ndarray]
    done: bool
    actions: List[int]


class ExperienceBatch(NamedTuple):
    state: np.array
    action: np.array
    reward: np.array
    state_next: np.array
    done: np.array
    actions: np.array


class PackedFrames(TypedDict):
    shape: Tuple[int, ...]
    dtype: str
    data: bytes


class Memory(TypedDict):
    data: bytes
    p: float


def _pack(x):
    return lz4.frame.compress(pickle.dumps(x))


def _unpack(x):
    return pickle.loads(lz4.frame.decompress(x))


def pack_experience(
    s: Union[np.ndarray, LazyFrames],
    a: int,
    r: float,
    s_: Union[np.ndarray, LazyFrames],
    d: bool,
    a_: List[int],
    p: float,
) -> Memory:
    def frames_to_dict(f: Union[np.ndarray, LazyFrames]) -> PackedFrames:
        if isinstance(f, LazyFrames):
            f = np.stack(np.array(f))

        return {"shape": f.shape, "dtype": str(f.dtype), "data": bytes(f)}

    return {
        "data": _pack(
            {
                "s": frames_to_dict(s),
                "s_": frames_to_dict(s_),
                "a": int(a),
                "r": float(r),
                "d": bool(d),
                "a_": a_,
            }
        ),
        "p": p,
    }


def unpack_frames(s: PackedFrames) -> np.ndarray:
    def unpack(data: bytes, dtype: str, shape: Tuple[int, ...]) -> np.ndarray:
        return np.frombuffer(data, dtype=dtype).reshape(shape)

    return unpack(**s)


def unpack_experience(m: Memory) -> Experience:
    e = _unpack(m["data"])
    return Experience(
        state=unpack_frames(e["s"]),
        action=e["a"],
        reward=e["r"],
        state_next=unpack_frames(e["s_"]),
        done=e["d"],
        actions=e["a_"],
    )


def batch_sequence(sequence: Sequence[T], batch_size: int) -> Iterable[T]:
    it, N = iter(sequence), len(sequence)
    for i in range(0, N, batch_size):
        yield (next(it) for _ in range(min(N - i, batch_size)))


def prepare_batch(batch: Iterable[Experience]) -> ExperienceBatch:
    state, action, reward, next_state, done, actions = starmap(
        np.ndarray.astype,
        zip(map(np.stack, zip(*batch)), ["u1", "i8", "f4", "u1", "?", "i8"]),
    )

    return ExperienceBatch(
        state,
        action.squeeze(),
        reward.squeeze(),
        next_state,
        done.squeeze(),
        actions,
    )


@dataclass
class PriorityReplayBuffer:
    _memory: Deque[Memory]
    _priorities: Optional[np.ndarray] = None

    def __init__(self, max_len: int, memory: List[Memory] = tuple()):
        self._memory = deque(memory, maxlen=max_len)

    def __len__(self):
        return len(self._memory)

    def __getitem__(self, i: int) -> Experience:
        return unpack_experience(self._memory[i])

    def __iter__(self):
        return (unpack_experience(m) for m in self._memory)

    def to_dict(self) -> dict:
        return {"max_len": self._memory.maxlen, "memory": [m for m in self._memory]}

    def _clear_caches(self):
        self._priorities = None

    @property
    def priorities(self):
        if self._priorities is None:
            self._priorities = np.array([m["p"] for m in self._memory], "f4")
        return self._priorities

    def add(
        self, s: LazyFrames, a: int, r: float, s_: LazyFrames, d: bool, a_: List[int]
    ):
        self._clear_caches()
        self._memory.append(pack_experience(s=s, a=a, r=r, s_=s_, d=d, a_=a_, p=0.1))

    def batch(self, batch_size: int) -> Iterable[Tuple[ExperienceBatch, np.ndarray]]:
        for indices in map(list, batch_sequence(range(len(self)), batch_size)):
            yield (
                prepare_batch((self[i] for i in indices)),
                np.array(indices, dtype="u4"),
            )

    def recall(self, batch_size: int) -> Tuple[ExperienceBatch, np.ndarray]:
        p = np.abs(self.priorities) + 1e-36
        p /= p.sum()

        indices = np.random.choice(
            np.arange(0, len(p)), size=batch_size, p=p, replace=False
        )

        return prepare_batch((self[i] for i in indices)), indices

    def update_priorities(self, indices: Sequence[int], priorities: np.array):
        self._clear_caches()
        for i, p in zip(indices, priorities):
            self._memory[i]["p"] = p

    def save(self, path: os.PathLike):
        with lz4.frame.open(path, "wb") as fp:
            pickle.dump(self.to_dict(), fp)

    @staticmethod
    def load(path: os.PathLike) -> "PriorityReplayBuffer":
        with lz4.frame.open(path, "rb") as fp:
            return PriorityReplayBuffer(**pickle.load(fp))
