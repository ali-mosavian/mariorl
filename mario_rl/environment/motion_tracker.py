"""
Motion-based entity tracker for detecting enemies and moving objects.

Detects entities by finding motion that is independent of the scrolling background.
When the screen scrolls (Mario moves), the background shifts uniformly. Enemies and
other entities move independently and can be detected by compensating for scroll
and finding remaining motion.

This approach works without RAM hacking - purely from visual observation.

Template matching allows tracking specific entity types by saving their sprite
patterns and matching new detections against the library.
"""

from __future__ import annotations

from collections import deque
from dataclasses import field
from dataclasses import dataclass

import cv2
import numpy as np

# Default size for normalizing templates for comparison
TEMPLATE_NORMALIZE_SIZE = (16, 16)


@dataclass(frozen=True)
class Entity:
    """A detected moving entity."""

    x: int  # Screen x position (pixels)
    y: int  # Screen y position (pixels)
    vx: float  # Velocity x (pixels/frame)
    vy: float  # Velocity y (pixels/frame)
    width: int  # Bounding box width
    height: int  # Bounding box height

    @property
    def center(self) -> tuple[int, int]:
        """Get center position."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    def distance_to(self, x: int, y: int) -> float:
        """Calculate distance to a point."""
        cx, cy = self.center
        return float(np.sqrt((cx - x) ** 2 + (cy - y) ** 2))

    def is_approaching(self, target_x: int) -> bool:
        """Check if entity is moving toward a target x position."""
        cx = self.x + self.width // 2
        # Entity is approaching if it's moving toward target
        if cx > target_x:
            return self.vx < 0  # Entity is to the right, moving left
        elif cx < target_x:
            return self.vx > 0  # Entity is to the left, moving right
        return False


@dataclass(frozen=True)
class TrackedEntity(Entity):
    """An entity with tracking information."""

    entity_id: int | None = None  # Persistent ID from tracking


class KalmanTrack:
    """A Kalman filter tracker with constant-acceleration motion model.

    State vector: [x, y, vx, vy, ax, ay]
    Measurement: [x, y] (center position from detection)

    The acceleration model allows tracking through:
    - Jumps (ay captures gravity)
    - Direction changes (ax/ay can flip signs)
    - Speed changes (acceleration/deceleration)
    """

    def __init__(
        self,
        entity_id: int,
        x: float,
        y: float,
        width: int,
        height: int,
        process_noise: float = 1.0,
        measurement_noise: float = 3.0,
        accel_decay: float = 0.9,
    ):
        self.entity_id = entity_id
        self.width = width
        self.height = height
        self.frames_tracked = 1
        self.frames_lost = 0
        self.accel_decay = accel_decay

        # State: [x, y, vx, vy, ax, ay]
        self.state = np.array([x, y, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # State transition matrix (constant acceleration model)
        # x' = x + vx + 0.5*ax
        # vx' = vx + ax
        # ax' = ax * decay
        dt = 1.0
        self.F = np.array(
            [
                [1, 0, dt, 0, 0.5 * dt**2, 0],  # x
                [0, 1, 0, dt, 0, 0.5 * dt**2],  # y
                [0, 0, 1, 0, dt, 0],  # vx
                [0, 0, 0, 1, 0, dt],  # vy
                [0, 0, 0, 0, accel_decay, 0],  # ax
                [0, 0, 0, 0, 0, accel_decay],  # ay
            ],
            dtype=np.float64,
        )

        # Measurement matrix (we only observe position)
        self.H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], dtype=np.float64)

        # Covariance matrix (uncertainty in state)
        self.P = np.eye(6, dtype=np.float64) * 10.0

        # Process noise (different for position, velocity, acceleration)
        self.Q = np.diag(
            [
                process_noise * 0.5,  # x position
                process_noise * 0.5,  # y position
                process_noise * 1.0,  # vx
                process_noise * 1.0,  # vy
                process_noise * 0.5,  # ax
                process_noise * 0.5,  # ay
            ]
        )

        # Measurement noise
        self.R = np.eye(2, dtype=np.float64) * measurement_noise

    def predict(self) -> tuple[float, float]:
        """Predict next state (call before update each frame).

        Returns:
            Predicted (x, y) center position
        """
        # State prediction
        self.state = self.F @ self.state

        # Covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q

        return (self.state[0], self.state[1])

    def update(self, x: float, y: float, width: int, height: int) -> None:
        """Update state with new measurement.

        Args:
            x: Measured center x position
            y: Measured center y position
            width: Detection width
            height: Detection height
        """
        z = np.array([x, y], dtype=np.float64)

        # Innovation (measurement residual)
        y_residual = z - self.H @ self.state

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.state = self.state + K @ y_residual

        # Covariance update
        identity = np.eye(6)
        self.P = (identity - K @ self.H) @ self.P

        # Update size (simple averaging)
        self.width = int(0.7 * self.width + 0.3 * width)
        self.height = int(0.7 * self.height + 0.3 * height)

        self.frames_tracked += 1
        self.frames_lost = 0

    def mark_lost(self) -> None:
        """Mark this track as having no detection this frame."""
        self.frames_lost += 1
        # Increase uncertainty when lost
        self.P *= 1.1

    @property
    def x(self) -> int:
        """Get top-left x position."""
        return int(self.state[0] - self.width / 2)

    @property
    def y(self) -> int:
        """Get top-left y position."""
        return int(self.state[1] - self.height / 2)

    @property
    def vx(self) -> float:
        """Get x velocity."""
        return float(self.state[2])

    @property
    def vy(self) -> float:
        """Get y velocity."""
        return float(self.state[3])

    @property
    def ax(self) -> float:
        """Get x acceleration."""
        return float(self.state[4])

    @property
    def ay(self) -> float:
        """Get y acceleration."""
        return float(self.state[5])

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Get bounding box as (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)

    @property
    def center(self) -> tuple[float, float]:
        """Get center position."""
        return (self.state[0], self.state[1])


class ParticleTrack:
    """A particle filter tracker with constant-acceleration motion model.

    Uses Sequential Monte Carlo (particle filter) with stratified resampling.
    State: [x, y, vx, vy, ax, ay] for each particle

    The acceleration model allows tracking through:
    - Jumps (ay captures gravity)
    - Direction changes (ax/ay can flip signs)
    - Speed changes (acceleration/deceleration)
    """

    def __init__(
        self,
        entity_id: int,
        x: float,
        y: float,
        width: int,
        height: int,
        n_particles: int = 50,
        process_noise_pos: float = 2.0,
        process_noise_vel: float = 0.5,
        process_noise_accel: float = 0.3,
        measurement_noise: float = 5.0,
        accel_decay: float = 0.9,
    ):
        self.entity_id = entity_id
        self.width = width
        self.height = height
        self.frames_tracked = 1
        self.frames_lost = 0
        self.n_particles = n_particles
        self.accel_decay = accel_decay

        # Noise parameters
        self.process_noise_pos = process_noise_pos
        self.process_noise_vel = process_noise_vel
        self.process_noise_accel = process_noise_accel
        self.measurement_noise = measurement_noise

        # Particles: each row is [x, y, vx, vy, ax, ay]
        self.particles = np.zeros((n_particles, 6), dtype=np.float64)
        self.particles[:, 0] = x  # All start at initial position
        self.particles[:, 1] = y
        # Initialize with small velocity/accel spread
        self.particles[:, 2] = np.random.randn(n_particles) * 0.5  # vx
        self.particles[:, 3] = np.random.randn(n_particles) * 0.5  # vy
        self.particles[:, 4] = np.random.randn(n_particles) * 0.1  # ax
        self.particles[:, 5] = np.random.randn(n_particles) * 0.1  # ay

        # Weights (uniform initially)
        self.weights = np.ones(n_particles) / n_particles

    def predict(self) -> tuple[float, float]:
        """Predict next state by propagating particles with acceleration model.

        Returns:
            Predicted (x, y) center position (weighted mean)
        """
        n = self.n_particles
        noise_scale = 1.0 if self.frames_lost == 0 else 1.5

        # Get current state components
        x = self.particles[:, 0]
        y = self.particles[:, 1]
        vx = self.particles[:, 2]
        vy = self.particles[:, 3]
        ax = self.particles[:, 4]
        ay = self.particles[:, 5]

        # Constant acceleration motion model
        # x' = x + vx + 0.5*ax
        # vx' = vx + ax
        # ax' = ax * decay
        self.particles[:, 0] = x + vx + 0.5 * ax + np.random.randn(n) * self.process_noise_pos * noise_scale
        self.particles[:, 1] = y + vy + 0.5 * ay + np.random.randn(n) * self.process_noise_pos * noise_scale
        self.particles[:, 2] = vx + ax + np.random.randn(n) * self.process_noise_vel
        self.particles[:, 3] = vy + ay + np.random.randn(n) * self.process_noise_vel
        self.particles[:, 4] = ax * self.accel_decay + np.random.randn(n) * self.process_noise_accel
        self.particles[:, 5] = ay * self.accel_decay + np.random.randn(n) * self.process_noise_accel

        return self._weighted_mean_position()

    def update(self, x: float, y: float, width: int, height: int) -> None:
        """Update particle weights based on position measurement.

        Args:
            x: Measured center x position
            y: Measured center y position
            width: Detection width
            height: Detection height
        """
        # Compute position likelihood for each particle (Gaussian)
        dx = self.particles[:, 0] - x
        dy = self.particles[:, 1] - y
        distances_sq = dx**2 + dy**2
        likelihoods = np.exp(-distances_sq / (2 * self.measurement_noise**2))

        # Update weights
        self.weights *= likelihoods
        weight_sum = np.sum(self.weights)
        if weight_sum > 1e-10:
            self.weights /= weight_sum
        else:
            self.weights = np.ones(self.n_particles) / self.n_particles

        # Adaptive resampling when ESS is low
        ess = 1.0 / np.sum(self.weights**2)
        if ess < self.n_particles * 0.5:
            self._stratified_resample()

        # Update size
        self.width = int(0.7 * self.width + 0.3 * width)
        self.height = int(0.7 * self.height + 0.3 * height)

        self.frames_tracked += 1
        self.frames_lost = 0

    def _stratified_resample(self) -> None:
        """Stratified resampling to maintain particle diversity."""
        n = self.n_particles
        cumsum = np.cumsum(self.weights)
        positions = (np.arange(n) + np.random.rand(n)) / n
        indices = np.searchsorted(cumsum, positions)
        indices = np.clip(indices, 0, n - 1)

        self.particles = self.particles[indices].copy()

        # Add small noise to maintain diversity
        self.particles[:, 0] += np.random.randn(n) * 0.5
        self.particles[:, 1] += np.random.randn(n) * 0.5

        self.weights = np.ones(n) / n

    def _weighted_mean_position(self) -> tuple[float, float]:
        """Compute weighted mean position of particles."""
        x = np.sum(self.weights * self.particles[:, 0])
        y = np.sum(self.weights * self.particles[:, 1])
        return (x, y)

    def mark_lost(self) -> None:
        """Mark this track as having no detection this frame."""
        self.frames_lost += 1
        spread_factor = min(1.0 + self.frames_lost * 0.3, 3.0)
        n = self.n_particles
        self.particles[:, 0] += np.random.randn(n) * self.process_noise_pos * spread_factor * 0.5
        self.particles[:, 1] += np.random.randn(n) * self.process_noise_pos * spread_factor * 0.5

    @property
    def x(self) -> int:
        """Get top-left x position."""
        cx, _ = self._weighted_mean_position()
        return int(cx - self.width / 2)

    @property
    def y(self) -> int:
        """Get top-left y position."""
        _, cy = self._weighted_mean_position()
        return int(cy - self.height / 2)

    @property
    def vx(self) -> float:
        """Get x velocity (weighted mean)."""
        return float(np.sum(self.weights * self.particles[:, 2]))

    @property
    def vy(self) -> float:
        """Get y velocity (weighted mean)."""
        return float(np.sum(self.weights * self.particles[:, 3]))

    @property
    def ax(self) -> float:
        """Get x acceleration (weighted mean)."""
        return float(np.sum(self.weights * self.particles[:, 4]))

    @property
    def ay(self) -> float:
        """Get y acceleration (weighted mean)."""
        return float(np.sum(self.weights * self.particles[:, 5]))

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Get bounding box as (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)

    @property
    def center(self) -> tuple[float, float]:
        """Get center position (weighted mean)."""
        return self._weighted_mean_position()

    def get_particle_spread(self) -> float:
        """Get standard deviation of particle positions (uncertainty measure)."""
        cx, cy = self._weighted_mean_position()
        dx = self.particles[:, 0] - cx
        dy = self.particles[:, 1] - cy
        return float(np.sqrt(np.mean(dx**2 + dy**2)))


# Type alias for tracks (can be either Kalman or Particle)
Track = KalmanTrack | ParticleTrack


@dataclass
class TrackerPool:
    """Pool of trackers for persistent entity tracking.

    Supports two tracking modes:
    - "kalman": Kalman filter with constant-acceleration model
    - "particle": Particle filter with constant-acceleration model

    Features:
    - Predict entity positions when detections are missing
    - Smooth noisy detections
    - Estimate velocity and acceleration
    - Maintain persistent entity IDs across frames

    Uses simple distance-based association (no appearance matching).
    """

    max_trackers: int = 20
    max_lost_frames: int = 8  # Frames to keep lost tracks
    max_association_distance: float = 50.0  # Max distance for association
    process_noise: float = 1.5
    measurement_noise: float = 5.0

    # Tracker type: "kalman" or "particle"
    tracker_type: str = "particle"
    n_particles: int = 50

    tracks: list[Track] = field(default_factory=list)
    next_id: int = 0

    def reset(self) -> None:
        """Clear all tracks."""
        self.tracks = []
        self.next_id = 0

    @property
    def trackers(self) -> list[Track]:
        """Alias for tracks (backwards compatibility)."""
        return self.tracks

    def _create_track(
        self,
        entity_id: int,
        x: float,
        y: float,
        width: int,
        height: int,
    ) -> Track:
        """Create a track of the configured type."""
        if self.tracker_type == "particle":
            return ParticleTrack(
                entity_id=entity_id,
                x=x,
                y=y,
                width=width,
                height=height,
                n_particles=self.n_particles,
                process_noise_pos=self.process_noise,
                process_noise_vel=self.process_noise * 0.3,
                process_noise_accel=self.process_noise * 0.2,
                measurement_noise=self.measurement_noise,
            )
        else:
            return KalmanTrack(
                entity_id=entity_id,
                x=x,
                y=y,
                width=width,
                height=height,
                process_noise=self.process_noise,
                measurement_noise=self.measurement_noise,
            )

    def update(self, detections: list[tuple[int, int, int, int]]) -> list[Track]:
        """Update all tracks with new detections using distance-based association.

        Args:
            detections: Bounding boxes from motion detection as (x, y, w, h)

        Returns:
            List of active Track objects
        """
        # Step 1: Predict all tracks
        predicted_centers = []
        for track in self.tracks:
            pred_x, pred_y = track.predict()
            predicted_centers.append((pred_x, pred_y))

        # Step 2: Convert detections to centers
        detection_centers = []
        for x, y, dw, dh in detections:
            cx = x + dw / 2
            cy = y + dh / 2
            detection_centers.append((cx, cy, dw, dh))

        # Step 3: Associate detections with tracks using distance only
        track_matched = [False] * len(self.tracks)
        detection_matched = [False] * len(detections)

        # Build distance matrix and sort matches
        matches = []
        for t_idx, (px, py) in enumerate(predicted_centers):
            for d_idx, (dx, dy, _dw, _dh) in enumerate(detection_centers):
                dist = np.sqrt((px - dx) ** 2 + (py - dy) ** 2)
                if dist <= self.max_association_distance:
                    matches.append((dist, t_idx, d_idx))

        matches.sort(key=lambda m: m[0])  # Sort by distance

        # Greedy matching
        for _dist, t_idx, d_idx in matches:
            if track_matched[t_idx] or detection_matched[d_idx]:
                continue

            # Match found - update track
            dx, dy, dw, dh = detection_centers[d_idx]
            self.tracks[t_idx].update(dx, dy, int(dw), int(dh))
            track_matched[t_idx] = True
            detection_matched[d_idx] = True

        # Step 4: Mark unmatched tracks as lost
        for t_idx, matched in enumerate(track_matched):
            if not matched:
                self.tracks[t_idx].mark_lost()

        # Step 5: Remove tracks that have been lost too long
        self.tracks = [t for t in self.tracks if t.frames_lost < self.max_lost_frames]

        # Step 6: Create new tracks for unmatched detections
        for d_idx, matched in enumerate(detection_matched):
            if not matched and len(self.tracks) < self.max_trackers:
                x, y, dw, dh = detections[d_idx]
                if dw < 8 or dh < 8:
                    continue

                cx, cy = x + dw / 2, y + dh / 2
                new_track = self._create_track(
                    entity_id=self.next_id,
                    x=cx,
                    y=cy,
                    width=dw,
                    height=dh,
                )
                self.next_id += 1
                self.tracks.append(new_track)

        return self.tracks


class MotionEntityTracker:
    """
    Track entities by detecting motion independent of screen scroll.

    The algorithm:
    1. Track recent frames and Mario's x position
    2. Measure actual background scroll using phase correlation
    3. Shift previous frame to compensate for scroll
    4. Use EDGE DETECTION to find position changes (ignores palette cycling)
    5. Track blob positions across frames with Kalman/Particle filters

    This detects Goombas, Koopas, Piranha Plants, Bullet Bills, etc.
    without needing access to game RAM.

    Edge detection is key: it ignores palette cycling (question blocks shimmer)
    because edges stay in place even when colors change.
    """

    def __init__(
        self,
        history_len: int = 4,
        motion_threshold: int = 30,
        min_blob_area: int = 64,
        max_blob_area: int = 2000,
        mario_exclusion_width: int = 32,
        mario_screen_x: int = 70,
        edge_margin: int = 8,
        max_aspect_ratio: float = 4.0,
        hud_height: int = 40,
        max_trackers: int = 20,
        tracker_lost_frames: int = 8,
        tracker_association_dist: float = 50.0,
        tracker_type: str = "particle",
        n_particles: int = 50,
    ):
        """
        Initialize the motion tracker.

        Args:
            history_len: Number of frames to keep in history
            motion_threshold: Pixel intensity change threshold (used for Canny)
            min_blob_area: Minimum blob area to consider as entity (pixels)
            max_blob_area: Maximum blob area (larger = background artifacts)
            mario_exclusion_width: Width around Mario to exclude (avoid self-detection)
            mario_screen_x: Mario's typical screen X position (NES has Mario at ~70px)
            edge_margin: Pixels from screen edge to exclude (scroll artifacts)
            max_aspect_ratio: Maximum width/height ratio (filter noise artifacts)
            hud_height: Height of HUD region at top of screen to exclude
            max_trackers: Maximum number of trackers to maintain
            tracker_lost_frames: Remove tracker after this many frames without detection
            tracker_association_dist: Max distance to associate detection with track
            tracker_type: "particle" or "kalman"
            n_particles: Number of particles for particle filter
        """
        self.history_len = history_len
        self.motion_threshold = motion_threshold
        self.min_blob_area = min_blob_area
        self.max_blob_area = max_blob_area
        self.mario_exclusion_width = mario_exclusion_width
        self.mario_screen_x = mario_screen_x
        self.edge_margin = edge_margin
        self.max_aspect_ratio = max_aspect_ratio
        self.hud_height = hud_height

        # Frame history (grayscale for efficiency)
        self.frame_history: deque[np.ndarray] = deque(maxlen=history_len)
        self.x_pos_history: deque[int] = deque(maxlen=history_len)

        # Tracker pool for persistent entity tracking
        self.tracker_pool = TrackerPool(
            max_trackers=max_trackers,
            max_lost_frames=tracker_lost_frames,
            max_association_distance=tracker_association_dist,
            tracker_type=tracker_type,
            n_particles=n_particles,
        )

    def reset(self) -> None:
        """Reset tracker state (call on episode reset)."""
        self.frame_history.clear()
        self.x_pos_history.clear()
        self.tracker_pool.reset()

    def update(
        self,
        frame: np.ndarray,
        mario_x: int,
        mario_y: int | None = None,
    ) -> list[TrackedEntity]:
        """
        Process a new frame and detect moving entities.

        Args:
            frame: RGB frame from the environment, ideally full 256x240 NES resolution
            mario_x: Mario's world x position (from info dict)
            mario_y: Mario's screen y position (optional, for exclusion)

        Returns:
            List of detected TrackedEntity objects with entity IDs
        """
        # Convert RGB to grayscale for optical flow (keep full resolution)
        if frame.ndim == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        elif frame.ndim == 3 and frame.shape[2] == 1:
            gray = frame[:, :, 0]
        else:
            gray = frame

        self.frame_history.append(gray.copy())
        self.x_pos_history.append(mario_x)

        # Need at least 2 frames
        if len(self.frame_history) < 2:
            return []

        # Detect motion entities using optical flow
        raw_entities = self._detect_motion_entities(mario_y)

        # Update trackers
        detection_bboxes = [(e.x, e.y, e.width, e.height) for e in raw_entities]
        active_tracks = self.tracker_pool.update(detection_bboxes)

        # Build entity list from tracks (smoothed positions and velocities)
        entities = []
        for t in active_tracks:
            entities.append(
                TrackedEntity(
                    x=t.x,
                    y=t.y,
                    vx=t.vx,
                    vy=t.vy,
                    width=t.width,
                    height=t.height,
                    entity_id=t.entity_id,
                )
            )

        return entities

    def _detect_motion_entities(self, mario_y: int | None) -> list[TrackedEntity]:
        """Detect entities using ORB feature matching with RANSAC.

        Uses ORB features to robustly estimate background scroll:
        1. Detect ORB features in both frames
        2. Match features using brute-force Hamming distance
        3. Use RANSAC to find dominant translation (background scroll)
        4. Features that are RANSAC outliers = sprite candidates
        5. Group nearby sprite features into entities

        ORB is fast and robust for NES pixel art.
        """
        curr_frame = self.frame_history[-1]
        prev_frame = self.frame_history[-2]
        h, w = curr_frame.shape[:2]

        # Create ORB detector
        orb = cv2.ORB_create(nfeatures=1000, scoreType=cv2.ORB_FAST_SCORE)  # type: ignore[attr-defined]

        # Detect and compute descriptors
        kp1, des1 = orb.detectAndCompute(prev_frame, None)
        kp2, des2 = orb.detectAndCompute(curr_frame, None)

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return []

        # Match features using brute-force Hamming distance
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        if len(matches) < 10:
            return []

        # Extract matched point pairs
        prev_pts = np.float32([kp1[m.queryIdx].pt for m in matches])  # type: ignore[arg-type]
        curr_pts = np.float32([kp2[m.trainIdx].pt for m in matches])  # type: ignore[arg-type]

        # Use RANSAC to find the dominant translation (background scroll)
        # estimateAffinePartial2D finds rotation + translation + scale
        # For side-scrolling, this should be mostly X translation
        transform, inlier_mask = cv2.estimateAffinePartial2D(  # type: ignore[call-overload]
            prev_pts, curr_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0
        )

        if transform is None or inlier_mask is None:
            return []

        inlier_mask = inlier_mask.ravel().astype(bool)

        # Background scroll from the transformation matrix
        # transform is [[cos, -sin, tx], [sin, cos, ty]]
        bg_dx = transform[0, 2]
        bg_dy = transform[1, 2]

        # Outliers are sprite candidates (motion doesn't match background)
        outlier_mask = ~inlier_mask

        # Compute actual motion for all points
        motion_vectors = curr_pts - prev_pts

        # Get sprite features (outliers) and their motions relative to background
        sprite_pts = curr_pts[outlier_mask]  # type: ignore[index]
        sprite_motions = motion_vectors[outlier_mask] - np.array([bg_dx, bg_dy])  # type: ignore[index]

        if len(sprite_pts) == 0:
            return []

        # Create motion mask by marking sprite feature locations
        motion_mask = np.zeros((h, w), dtype=np.uint8)
        for pt in sprite_pts:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < w and 0 <= y < h:
                # Mark a small region around each feature
                cv2.circle(motion_mask, (x, y), 8, 255, -1)  # type: ignore[call-overload]

        # Morphological operations to group nearby features into blobs
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (16, 16))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel_close)  # type: ignore[assignment]

        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel_open)  # type: ignore[assignment]

        # === EXCLUSION ZONES (native 256x240 coordinates) ===

        # Exclude screen edges
        motion_mask[:, : self.edge_margin] = 0
        motion_mask[:, w - self.edge_margin :] = 0

        # Exclude Mario's region
        mario_left = max(0, self.mario_screen_x - self.mario_exclusion_width // 2)
        mario_right = min(w, self.mario_screen_x + self.mario_exclusion_width // 2)

        if mario_y is not None:
            mario_top = max(0, mario_y - 24)
            mario_bottom = min(h, mario_y + 24)
            motion_mask[mario_top:mario_bottom, mario_left:mario_right] = 0
        else:
            # Default exclusion in middle of screen
            motion_mask[100:200, mario_left:mario_right] = 0

        # Exclude HUD region (top of screen)
        motion_mask[: self.hud_height, :] = 0

        # Find contours (blobs)
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        entities: list[TrackedEntity] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_blob_area <= area <= self.max_blob_area:
                x, y, bw, bh = cv2.boundingRect(contour)

                # Filter tiny detections (< 6x6 at full resolution)
                if bw < 6 or bh < 6:
                    continue

                # Filter extreme aspect ratios
                aspect_ratio = bw / max(bh, 1)
                if aspect_ratio > self.max_aspect_ratio:
                    continue
                if bh / max(bw, 1) > self.max_aspect_ratio:
                    continue

                # Compute velocity from sprite features within this blob
                vx, vy = 0.0, 0.0
                count = 0
                for i, pt in enumerate(sprite_pts):
                    px, py = pt[0], pt[1]
                    if x <= px < x + bw and y <= py < y + bh:
                        vx += sprite_motions[i, 0]
                        vy += sprite_motions[i, 1]
                        count += 1
                if count > 0:
                    vx /= count
                    vy /= count

                entities.append(
                    TrackedEntity(
                        x=x,
                        y=y,
                        vx=float(vx),
                        vy=float(vy),
                        width=bw,
                        height=bh,
                    )
                )

        return entities

    def get_active_tracker_count(self) -> int:
        """Get the number of active trackers."""
        return len(self.tracker_pool.trackers)


def render_entity_overlay(
    frame_bgr: np.ndarray,
    entities: list[Entity] | list[TrackedEntity],
    mario_screen_x: int = 70,
    mario_screen_y: int | None = None,
    scale: int = 1,
    tracker_pool: TrackerPool | None = None,
    show_particle_spread: bool = True,
) -> np.ndarray:
    """
    Draw entity tracking overlay on a frame.

    Draws bounding boxes and velocity vectors for detected entities.
    Color indicates danger level based on distance to Mario.

    Args:
        frame_bgr: BGR frame to draw on (will be modified)
        entities: List of detected entities
        mario_screen_x: Mario's screen X position
        mario_screen_y: Mario's screen Y position (optional)
        scale: Scale factor for drawing (if frame is scaled up)
        tracker_pool: TrackerPool for drawing particle spread (optional)
        show_particle_spread: Whether to draw particle spread circles

    Returns:
        Frame with overlay drawn
    """
    mario_y = mario_screen_y if mario_screen_y is not None else 150

    for entity in entities:
        cx, cy = entity.center

        # Distance to Mario
        dist = entity.distance_to(mario_screen_x, mario_y)

        # Color based on danger level
        if dist < 25:
            color = (0, 0, 255)  # Red - imminent danger
        elif dist < 50:
            color = (0, 128, 255)  # Orange - caution
        elif dist < 80:
            color = (0, 200, 255)  # Yellow - attention
        else:
            color = (0, 255, 0)  # Green - safe distance

        # Draw bounding box
        x1, y1 = entity.x * scale, entity.y * scale
        x2, y2 = (entity.x + entity.width) * scale, (entity.y + entity.height) * scale
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

        # Draw center dot
        cx_scaled, cy_scaled = int(cx * scale), int(cy * scale)
        cv2.circle(frame_bgr, (cx_scaled, cy_scaled), 4, color, -1)

        # Draw velocity vector (if entity is moving)
        if abs(entity.vx) > 0.5 or abs(entity.vy) > 0.5:
            vel_scale = 5 * scale
            vx_draw = int(entity.vx * vel_scale)
            vy_draw = int(entity.vy * vel_scale)
            end_point = (cx_scaled + vx_draw, cy_scaled + vy_draw)
            cv2.arrowedLine(frame_bgr, (cx_scaled, cy_scaled), end_point, color, 2)

        # Draw entity ID
        if isinstance(entity, TrackedEntity) and entity.entity_id is not None:
            id_text = f"#{entity.entity_id}"
            cv2.putText(
                frame_bgr,
                id_text,
                (x2 + 2, y1 + 12 * scale),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4 * scale,
                (255, 255, 0),
                1,
            )

        # Draw particle spread if using particle filter tracking
        if show_particle_spread and tracker_pool is not None:
            if isinstance(entity, TrackedEntity) and entity.entity_id is not None:
                for track in tracker_pool.tracks:
                    if track.entity_id == entity.entity_id:
                        if isinstance(track, ParticleTrack):
                            spread = track.get_particle_spread()
                            spread_radius = int(spread * scale)
                            if spread_radius > 2:
                                cv2.circle(
                                    frame_bgr,
                                    (cx_scaled, cy_scaled),
                                    spread_radius,
                                    color,
                                    1,
                                    cv2.LINE_AA,
                                )
                        break

        # Draw danger indicator if approaching Mario
        if entity.is_approaching(mario_screen_x):
            cv2.putText(
                frame_bgr,
                "!",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6 * scale,
                (0, 0, 255),
                2,
            )

    return frame_bgr


def render_danger_hud(
    frame_bgr: np.ndarray,
    entities: list[Entity] | list[TrackedEntity],
    mario_screen_x: int = 70,
    hud_y: int = 20,
    template_count: int | None = None,
) -> np.ndarray:
    """
    Draw a danger level HUD showing threat status.

    Args:
        frame_bgr: BGR frame to draw on
        entities: List of detected entities
        mario_screen_x: Mario's screen X position
        hud_y: Y position for HUD
        template_count: Number of templates learned (optional, shown if provided)

    Returns:
        Frame with HUD drawn
    """
    # Find closest threat
    if entities:
        closest = min(entities, key=lambda e: e.distance_to(mario_screen_x, 150))
        closest_dist = closest.distance_to(mario_screen_x, 150)
        approaching = closest.is_approaching(mario_screen_x)

        # Determine threat level text
        if closest_dist < 25:
            threat_text = "DANGER!"
            threat_color = (0, 0, 255)
        elif closest_dist < 50 and approaching:
            threat_text = "THREAT APPROACHING"
            threat_color = (0, 128, 255)
        elif closest_dist < 80:
            threat_text = f"Entity nearby ({closest_dist:.0f}px)"
            threat_color = (0, 200, 255)
        else:
            threat_text = f"Clear ({len(entities)} entities)"
            threat_color = (0, 255, 0)
    else:
        threat_text = "No entities detected"
        threat_color = (128, 128, 128)

    # Add template count if provided
    if template_count is not None:
        threat_text += f" | {template_count} templates"

    # Draw background rectangle for readability
    text_size = cv2.getTextSize(threat_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.rectangle(
        frame_bgr,
        (5, hud_y - 15),
        (15 + text_size[0], hud_y + 5),
        (0, 0, 0),
        -1,
    )

    # Draw text
    cv2.putText(
        frame_bgr,
        threat_text,
        (10, hud_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        threat_color,
        1,
    )

    return frame_bgr
