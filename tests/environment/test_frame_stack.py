"""Comprehensive tests for LazyFrames and FrameStack."""

import pytest
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box

from mario_rl.environment.frame_stack import FrameStack
from mario_rl.environment.frame_stack import LazyFrames

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def uint8_frames() -> list[np.ndarray]:
    """Create sample uint8 frames for testing."""
    return [np.full((84, 84), i * 10, dtype=np.uint8) for i in range(4)]


@pytest.fixture
def float32_frames() -> list[np.ndarray]:
    """Create sample float32 frames for testing."""
    return [np.full((64, 64), i * 0.1, dtype=np.float32) for i in range(4)]


@pytest.fixture
def frames_3d() -> list[np.ndarray]:
    """Create sample 3D frames (with channel dimension) for testing."""
    return [np.full((84, 84, 3), i * 20, dtype=np.uint8) for i in range(4)]


class SimpleEnv(gym.Env):
    """Minimal environment for testing FrameStack wrapper."""

    def __init__(
        self,
        obs_shape: tuple[int, ...] = (84, 84),
        dtype: type = np.uint8,
    ) -> None:
        super().__init__()
        self.obs_shape = obs_shape
        self._dtype = dtype
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=dtype)
        self.action_space = gym.spaces.Discrete(4)
        self._step_count = 0

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        """Reset environment to initial state."""
        self._step_count = 0
        return np.zeros(self.obs_shape, dtype=self._dtype), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step with incrementing observation values."""
        self._step_count += 1
        obs: np.ndarray = np.full(self.obs_shape, self._step_count * 10, dtype=self._dtype)
        return obs, 1.0, False, False, {}


@pytest.fixture
def simple_env() -> SimpleEnv:
    """Create a simple gymnasium environment for testing."""
    return SimpleEnv()


# =============================================================================
# LazyFrames - Initialization
# =============================================================================


def test_lazyframes_init_stores_frames_uncompressed(uint8_frames: list[np.ndarray]) -> None:
    """LazyFrames should store frames without compression when compressed=False."""
    lazy = LazyFrames.from_frames(uint8_frames, compressed=False)

    assert len(lazy) == 4
    assert lazy.shape == (4, 84, 84)
    assert lazy.dtype == np.uint8


def test_lazyframes_init_compresses_frames(uint8_frames: list[np.ndarray]) -> None:
    """LazyFrames should compress frames when compressed=True."""
    lazy = LazyFrames.from_frames(uint8_frames, compressed=True)

    assert len(lazy) == 4
    assert lazy.shape == (4, 84, 84)
    # Should still work - compression is internal detail


def test_lazyframes_init_preserves_float32_dtype(float32_frames: list[np.ndarray]) -> None:
    """LazyFrames should preserve float32 dtype."""
    lazy = LazyFrames.from_frames(float32_frames, compressed=False)

    assert lazy.dtype == np.float32
    assert lazy.shape == (4, 64, 64)


def test_lazyframes_init_handles_3d_frames(frames_3d: list[np.ndarray]) -> None:
    """LazyFrames should handle 3D frames (H, W, C)."""
    lazy = LazyFrames.from_frames(frames_3d, compressed=False)

    assert lazy.shape == (4, 84, 84, 3)


# =============================================================================
# LazyFrames - Length
# =============================================================================


@pytest.mark.parametrize("num_frames", [1, 2, 4, 8])
def test_lazyframes_len_returns_frame_count(num_frames: int) -> None:
    """len(LazyFrames) should return the number of stacked frames."""
    frames = [np.zeros((32, 32), dtype=np.uint8) for _ in range(num_frames)]
    lazy = LazyFrames.from_frames(frames, compressed=False)

    assert len(lazy) == num_frames


# =============================================================================
# LazyFrames - Indexing
# =============================================================================


def test_lazyframes_getitem_single_index_returns_frame(uint8_frames: list[np.ndarray]) -> None:
    """Indexing with int should return a single frame as numpy array."""
    lazy = LazyFrames.from_frames(uint8_frames, compressed=False)

    frame = lazy[0]

    assert isinstance(frame, np.ndarray)
    assert frame.shape == (84, 84)
    np.testing.assert_array_equal(frame, uint8_frames[0])


def test_lazyframes_getitem_compressed_decompresses_correctly(
    uint8_frames: list[np.ndarray],
) -> None:
    """Indexing compressed LazyFrames should decompress correctly."""
    lazy = LazyFrames.from_frames(uint8_frames, compressed=True)

    frame = lazy[0]

    np.testing.assert_array_equal(frame, uint8_frames[0])


def test_lazyframes_getitem_negative_index(uint8_frames: list[np.ndarray]) -> None:
    """Negative indexing should work correctly."""
    lazy = LazyFrames.from_frames(uint8_frames, compressed=False)

    np.testing.assert_array_equal(lazy[-1], uint8_frames[-1])


def test_lazyframes_getitem_slice_returns_stacked_array(
    uint8_frames: list[np.ndarray],
) -> None:
    """Slicing should return a stacked numpy array."""
    lazy = LazyFrames.from_frames(uint8_frames, compressed=False)

    frames = lazy[:]

    assert isinstance(frames, np.ndarray)
    assert frames.shape == (4, 84, 84)


def test_lazyframes_getitem_partial_slice(uint8_frames: list[np.ndarray]) -> None:
    """Partial slicing should return correct subset."""
    lazy = LazyFrames.from_frames(uint8_frames, compressed=False)

    frames = lazy[1:3]

    assert frames.shape == (2, 84, 84)
    np.testing.assert_array_equal(frames[0], uint8_frames[1])
    np.testing.assert_array_equal(frames[1], uint8_frames[2])


def test_lazyframes_getitem_step_slice(uint8_frames: list[np.ndarray]) -> None:
    """Slicing with step should work correctly."""
    lazy = LazyFrames.from_frames(uint8_frames, compressed=False)

    frames = lazy[::2]

    assert frames.shape == (2, 84, 84)
    np.testing.assert_array_equal(frames[0], uint8_frames[0])
    np.testing.assert_array_equal(frames[1], uint8_frames[2])


# =============================================================================
# LazyFrames - Array Conversion
# =============================================================================


def test_lazyframes_array_conversion(uint8_frames: list[np.ndarray]) -> None:
    """np.array(LazyFrames) should return stacked frames."""
    lazy = LazyFrames.from_frames(uint8_frames, compressed=False)

    arr = np.array(lazy)

    assert arr.shape == (4, 84, 84)
    assert arr.dtype == np.uint8


def test_lazyframes_array_dtype_conversion(uint8_frames: list[np.ndarray]) -> None:
    """np.array(LazyFrames, dtype=...) should convert dtype."""
    lazy = LazyFrames.from_frames(uint8_frames, compressed=False)

    arr = np.array(lazy, dtype=np.float32)

    assert arr.dtype == np.float32


def test_lazyframes_array_compressed_roundtrip(uint8_frames: list[np.ndarray]) -> None:
    """Converting compressed LazyFrames to array should preserve values."""
    lazy = LazyFrames.from_frames(uint8_frames, compressed=True)

    arr = np.array(lazy)

    for i in range(4):
        np.testing.assert_array_equal(arr[i], uint8_frames[i])


# =============================================================================
# LazyFrames - Equality
# =============================================================================


def test_lazyframes_eq_with_matching_array(uint8_frames: list[np.ndarray]) -> None:
    """Equality with matching array should return True for all elements."""
    lazy = LazyFrames.from_frames(uint8_frames, compressed=False)
    expected = np.stack(uint8_frames, axis=0)

    result = lazy == expected

    assert isinstance(result, np.ndarray)
    assert result.all()


def test_lazyframes_eq_with_different_array(uint8_frames: list[np.ndarray]) -> None:
    """Equality with different array should return False for differing elements."""
    lazy = LazyFrames.from_frames(uint8_frames, compressed=False)
    different = np.ones((4, 84, 84), dtype=np.uint8) * 255

    result = lazy == different

    assert not result.all()


# =============================================================================
# LazyFrames - Compression Roundtrip
# =============================================================================


@pytest.mark.parametrize("dtype", [np.uint8, np.float32, np.float64])
def test_lazyframes_compression_preserves_dtype(dtype: type) -> None:
    """Compression roundtrip should preserve data for different dtypes."""
    if np.issubdtype(dtype, np.integer):
        frames = [np.random.randint(0, 255, (64, 64), dtype=dtype) for _ in range(4)]
    else:
        frames = [np.random.rand(64, 64).astype(dtype) for _ in range(4)]

    lazy = LazyFrames.from_frames(frames, compressed=True)

    for i, original in enumerate(frames):
        np.testing.assert_array_equal(lazy[i], original)


@pytest.mark.parametrize(
    "shape",
    [(32, 32), (64, 64), (84, 84), (96, 96, 3), (64, 64, 1)],
)
def test_lazyframes_compression_preserves_shape(shape: tuple[int, ...]) -> None:
    """Compression roundtrip should preserve different frame shapes."""
    frames = [np.random.randint(0, 255, shape, dtype=np.uint8) for _ in range(4)]

    lazy = LazyFrames.from_frames(frames, compressed=True)

    for i, original in enumerate(frames):
        np.testing.assert_array_equal(lazy[i], original)


# =============================================================================
# LazyFrames - Edge Cases
# =============================================================================


def test_lazyframes_single_frame() -> None:
    """LazyFrames should work with a single frame."""
    frames = [np.zeros((32, 32), dtype=np.uint8)]
    lazy = LazyFrames.from_frames(frames, compressed=False)

    assert len(lazy) == 1
    assert lazy.shape == (1, 32, 32)


def test_lazyframes_large_frames_compressed() -> None:
    """LazyFrames should handle large frames with compression."""
    frames = [np.random.randint(0, 255, (256, 256), dtype=np.uint8) for _ in range(4)]

    lazy = LazyFrames.from_frames(frames, compressed=True)

    for i in range(4):
        np.testing.assert_array_equal(lazy[i], frames[i])


# =============================================================================
# FrameStack - Initialization
# =============================================================================


def test_framestack_creates_correct_observation_space(simple_env: SimpleEnv) -> None:
    """FrameStack should create observation space with stacked dimensions."""
    wrapper = FrameStack(simple_env, num_stack=4, lz4_compress=False)

    assert wrapper.observation_space.shape == (4, 84, 84)
    assert wrapper.observation_space.dtype == np.uint8


@pytest.mark.parametrize("num_stack", [2, 4, 8])
def test_framestack_num_stack_affects_observation_space(num_stack: int) -> None:
    """Observation space first dimension should match num_stack."""
    env = SimpleEnv()
    wrapper = FrameStack(env, num_stack=num_stack, lz4_compress=False)

    assert wrapper.observation_space.shape[0] == num_stack


# =============================================================================
# FrameStack - Reset
# =============================================================================


def test_framestack_reset_returns_lazyframes(simple_env: SimpleEnv) -> None:
    """Reset should return a LazyFrames observation."""
    wrapper = FrameStack(simple_env, num_stack=4, lz4_compress=False)

    obs, info = wrapper.reset()

    assert isinstance(obs, LazyFrames)
    assert len(obs) == 4


def test_framestack_reset_fills_buffer_with_initial_obs(simple_env: SimpleEnv) -> None:
    """Reset should fill frame buffer with repeated initial observation."""
    wrapper = FrameStack(simple_env, num_stack=4, lz4_compress=False)

    obs, _ = wrapper.reset()

    # All frames should be identical (initial observation repeated)
    for i in range(4):
        np.testing.assert_array_equal(obs[i], obs[0])


def test_framestack_reset_returns_info_dict(simple_env: SimpleEnv) -> None:
    """Reset should return info dictionary from underlying env."""
    wrapper = FrameStack(simple_env, num_stack=4, lz4_compress=False)

    obs, info = wrapper.reset()

    assert isinstance(info, dict)


# =============================================================================
# FrameStack - Step
# =============================================================================


def test_framestack_step_returns_lazyframes(simple_env: SimpleEnv) -> None:
    """Step should return a LazyFrames observation."""
    wrapper = FrameStack(simple_env, num_stack=4, lz4_compress=False)
    wrapper.reset()

    obs, reward, terminated, truncated, info = wrapper.step(0)

    assert isinstance(obs, LazyFrames)


def test_framestack_step_returns_correct_values(simple_env: SimpleEnv) -> None:
    """Step should return correct reward, terminated, truncated, info."""
    wrapper = FrameStack(simple_env, num_stack=4, lz4_compress=False)
    wrapper.reset()

    obs, reward, terminated, truncated, info = wrapper.step(0)

    assert reward == 1.0
    assert terminated is False
    assert truncated is False
    assert isinstance(info, dict)


def test_framestack_step_shifts_buffer(simple_env: SimpleEnv) -> None:
    """Each step should shift the frame buffer, adding new observation at end."""
    wrapper = FrameStack(simple_env, num_stack=4, lz4_compress=False)
    wrapper.reset()

    # Take a step - new observation will be different from reset observation
    obs, _, _, _, _ = wrapper.step(0)

    # Last frame should be from step (value 10), first frames from reset (value 0)
    assert obs[-1].max() == 10  # Step 1 produces value 10
    assert obs[0].max() == 0  # Still initial observation


def test_framestack_multiple_steps_maintain_buffer_size(simple_env: SimpleEnv) -> None:
    """Buffer should maintain num_stack size after multiple steps."""
    wrapper = FrameStack(simple_env, num_stack=4, lz4_compress=False)
    wrapper.reset()

    for _ in range(10):
        obs, _, _, _, _ = wrapper.step(0)
        assert len(obs) == 4


def test_framestack_buffer_contains_last_n_observations(simple_env: SimpleEnv) -> None:
    """After multiple steps, buffer should contain last num_stack observations."""
    wrapper = FrameStack(simple_env, num_stack=4, lz4_compress=False)
    wrapper.reset()

    # Take 6 steps (observations will have values 10, 20, 30, 40, 50, 60)
    for _ in range(6):
        obs, _, _, _, _ = wrapper.step(0)

    # Buffer should contain observations from steps 3, 4, 5, 6 (values 30, 40, 50, 60)
    expected_values = [30, 40, 50, 60]
    for i, expected_val in enumerate(expected_values):
        assert obs[i].max() == expected_val


# =============================================================================
# FrameStack - Compression
# =============================================================================


def test_framestack_compression_enabled(simple_env: SimpleEnv) -> None:
    """FrameStack with compression should return compressed LazyFrames."""
    wrapper = FrameStack(simple_env, num_stack=4, lz4_compress=True)

    obs, _ = wrapper.reset()

    assert isinstance(obs, LazyFrames)
    assert obs.is_compressed is True


def test_framestack_compression_disabled(simple_env: SimpleEnv) -> None:
    """FrameStack without compression should return uncompressed LazyFrames."""
    wrapper = FrameStack(simple_env, num_stack=4, lz4_compress=False)

    obs, _ = wrapper.reset()

    assert obs.is_compressed is False


def test_framestack_compressed_obs_decompresses_correctly(simple_env: SimpleEnv) -> None:
    """Compressed observations should decompress to correct values."""
    wrapper = FrameStack(simple_env, num_stack=4, lz4_compress=True)

    obs, _ = wrapper.reset()
    arr = np.array(obs)

    # All frames should be zeros (initial observation)
    assert arr.shape == (4, 84, 84)
    np.testing.assert_array_equal(arr, np.zeros((4, 84, 84), dtype=np.uint8))


# =============================================================================
# Integration
# =============================================================================


def test_framestack_full_episode_flow(simple_env: SimpleEnv) -> None:
    """Test complete episode flow with reset and multiple steps."""
    wrapper = FrameStack(simple_env, num_stack=4, lz4_compress=True)

    obs, _ = wrapper.reset()
    assert isinstance(obs, LazyFrames)

    for _ in range(10):
        obs, reward, terminated, truncated, info = wrapper.step(0)
        assert isinstance(obs, LazyFrames)
        assert len(obs) == 4
        # Should be convertible to numpy
        arr = np.array(obs)
        assert arr.shape == (4, 84, 84)


def test_framestack_observation_matches_space(simple_env: SimpleEnv) -> None:
    """Observations should be compatible with observation_space."""
    wrapper = FrameStack(simple_env, num_stack=4, lz4_compress=False)

    obs, _ = wrapper.reset()
    arr = np.array(obs)

    assert arr.shape == wrapper.observation_space.shape
    assert np.all(arr >= wrapper.observation_space.low)
    assert np.all(arr <= wrapper.observation_space.high)
