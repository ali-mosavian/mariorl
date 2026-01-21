"""Wrapper that stacks frames."""

from typing import overload
from collections import deque
from dataclasses import dataclass

import lz4.block
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class LazyFrames[T: np.generic]:
    """Ensures common frames are only stored once to optimize memory use.

    To further reduce the memory use, it is optionally to turn on lz4 to compress the observations.

    Note:
        This object should only be converted to numpy array just before forward pass.
    """

    frames: list[NDArray[T] | bytes]
    _shape: tuple[int, ...]
    _dtype: np.dtype[T]
    _compressed: bool

    @property
    def is_compressed(self) -> bool:
        """Whether frames are LZ4 compressed."""
        return self._compressed

    @property
    def dtype(self) -> np.dtype[T]:
        """Data type of frames."""
        return self._dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of stacked frames (num_frames, *frame_shape)."""
        return (len(self.frames), *self._shape)

    def __len__(self) -> int:
        """Return number of stacked frames."""
        return len(self.frames)

    def __getitem__(self, idx: int | slice) -> NDArray[T]:
        """Get frame(s) by index or slice."""
        if isinstance(idx, slice):
            return np.stack([self._decode(frame) for frame in self.frames[idx]], axis=0)
        return self._decode(self.frames[idx])

    @overload
    def __array__(self, dtype: None = None) -> NDArray[T]: ...

    @overload
    def __array__[T2: np.generic](self, dtype: np.dtype[T2]) -> NDArray[T2]: ...

    def __array__(self, dtype: np.dtype | None = None) -> NDArray:
        """Convert to numpy array, optionally with dtype conversion."""
        arr = self[:]
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    def __eq__(self, other: NDArray[T]) -> NDArray[np.bool_]:  # type: ignore[override]
        """Element-wise equality with another array-like object."""
        return self.__array__() == other  # type: ignore[return-value]

    def _decode(self, frame: bytes | NDArray[T]) -> NDArray[T]:
        """Decompress frame if compressed, otherwise return as-is."""
        if self._compressed:
            return np.frombuffer(lz4.block.decompress(frame), dtype=self._dtype).reshape(self._shape)
        return frame  # type: ignore[return-value]

    @classmethod
    def from_frames(
        cls,
        frames: list[NDArray[T]],
        compressed: bool = False,
        dtype: np.dtype[T] | None = None,
    ) -> "LazyFrames[T]":
        """Create LazyFrames from a list of numpy arrays.

        Args:
            frames: List of numpy arrays to stack
            compressed: Whether to LZ4 compress the frames
            dtype: Override dtype (defaults to first frame's dtype)

        Returns:
            LazyFrames instance
        """
        dtype = dtype if dtype is not None else frames[0].dtype
        shape = tuple(frames[0].shape)
        encode = lz4.block.compress if compressed else (lambda x: x)
        return cls(
            frames=[encode(frame) for frame in frames],
            _shape=shape,
            _dtype=dtype,
            _compressed=compressed,
        )


class FrameStack(gym.ObservationWrapper):
    """Observation wrapper that stacks the observations in a rolling manner.

    For example, if the number of stacks is 4, then the returned observation contains
    the most recent 4 observations. For environment 'Pendulum-v1', the original observation
    is an array with shape [3], so if we stack 4 observations, the processed observation
    has shape [4, 3].

    Note:
        - To be memory efficient, the stacked observations are wrapped by :class:`LazyFrame`.
        - The observation space must be :class:`Box` type. If one uses :class:`Dict`
          as observation space, it should apply :class:`FlattenObservation` wrapper first.
          - After :meth:`reset` is called, the frame buffer will be filled with the initial observation. I.e. the observation returned by :meth:`reset` will consist of ``num_stack`-many identical frames,

    Example:
        >>> import gym
        >>> env = gym.make('CarRacing-v1')
        >>> env = FrameStack(env, 4)
        >>> env.observation_space
        Box(4, 96, 96, 3)
        >>> obs = env.reset()
        >>> obs.shape
        (4, 96, 96, 3)
    """

    def __init__(
        self,
        env: gym.Env,
        num_stack: int,
        lz4_compress: bool = False,
    ):
        """Observation wrapper that stacks the observations in a rolling manner.

        Args:
            env (Env): The environment to apply the wrapper
            num_stack (int): The number of frames to stack
            lz4_compress (bool): Use lz4 to compress the frames internally
        """
        super().__init__(env)
        self.num_stack = num_stack
        self.lz4_compress = lz4_compress

        self.frames: deque = deque(maxlen=num_stack)

        low = np.repeat(self.observation_space.low[np.newaxis, ...], num_stack, axis=0)  # type: ignore[attr-defined]
        high = np.repeat(self.observation_space.high[np.newaxis, ...], num_stack, axis=0)  # type: ignore[attr-defined]
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)  # type: ignore[arg-type,attr-defined]

    def observation(self, observation):
        """Converts the wrappers current frames to lazy frames.

        Args:
            observation: Ignored

        Returns:
            :class:`LazyFrames` object for the wrapper's frame buffer,  :attr:`self.frames`
        """
        assert len(self.frames) == self.num_stack, (len(self.frames), self.num_stack)
        return LazyFrames.from_frames(
            list(self.frames), compressed=self.lz4_compress, dtype=self.observation_space.dtype
        )

    def step(self, action):
        """Steps through the environment, appending the observation to the frame buffer.

        Args:
            action: The action to step through the environment with

        Returns:
            Stacked observations, reward, terminated, truncated, and information from the environment
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(observation)
        return self.observation(None), reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment with kwargs.

        Args:
            **kwargs: The kwargs for the environment reset

        Returns:
            The stacked observations
        """
        obs, info = self.env.reset(**kwargs)

        [self.frames.append(obs) for _ in range(self.num_stack)]

        return self.observation(None), info
