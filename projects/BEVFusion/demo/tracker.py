"""Training-free 3D multi-object tracking for BEVFusion detections.

The tracker operates in the nuScenes global frame so ego motion is removed
before association.  It combines a Kalman motion model, class-specific gates,
two-stage high/low-score matching, and short coasting through missed frames.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    from threadpoolctl import ThreadpoolController
    _BLAS = ThreadpoolController()
except ImportError:
    _BLAS = None


TRACKING_CLASSES = (
    'bicycle', 'bus', 'car', 'motorcycle', 'pedestrian', 'trailer', 'truck')
VEHICLE_CLASSES = {'bus', 'car', 'trailer', 'truck'}
CYCLE_CLASSES = {'bicycle', 'motorcycle'}
CLASS_GATES_M = {
    'bicycle': 3.0,
    'bus': 5.0,
    'car': 4.0,
    'motorcycle': 3.0,
    'pedestrian': 2.0,
    'trailer': 5.0,
    'truck': 5.0,
}
_IDENTITY_7 = np.eye(7)
_DIAG_7 = np.diag_indices(7)
# Measured state components: x, y, z, yaw, then vx, vy when available.
_OBSERVED = np.array([0, 1, 2, 3, 4, 5])
_MEASUREMENT_NOISE = np.diag([0.8, 0.8, 0.5, 0.2, 2.0, 2.0])
NMS_RADII_M = {
    'bicycle': 0.75,
    'bus': 2.0,
    'car': 1.5,
    'motorcycle': 0.75,
    'pedestrian': 0.5,
    'trailer': 2.0,
    'truck': 2.0,
}


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def transform_matrix(info: dict) -> np.ndarray:
    """Return the homogeneous LiDAR-to-global transform for one sample."""
    return (np.asarray(info['ego2global'], dtype=np.float64)
            @ np.asarray(info['lidar_points']['lidar2ego'], dtype=np.float64))


def transform_detection(det: dict, matrix: np.ndarray) -> dict:
    """Transform a bottom-centered LiDAR detection into the global frame."""
    box = np.asarray(det['bbox_3d'], dtype=np.float64).copy()
    xyz = matrix[:3, :3] @ box[:3] + matrix[:3, 3]
    frame_yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    box[:3] = xyz
    box[6] = wrap_angle(box[6] + frame_yaw)
    if len(box) >= 9:
        box[7:9] = matrix[:2, :2] @ box[7:9]
    result = dict(det)
    result['bbox_3d'] = box
    return result


def inverse_transform_box(
        box: Iterable[float], matrix: np.ndarray) -> np.ndarray:
    """Transform a global bottom-centered box into the current LiDAR frame."""
    box = np.asarray(box, dtype=np.float64).copy()
    inverse = np.linalg.inv(matrix)
    box[:3] = inverse[:3, :3] @ box[:3] + inverse[:3, 3]
    frame_yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    box[6] = wrap_angle(box[6] - frame_yaw)
    if len(box) >= 9:
        box[7:9] = inverse[:2, :2] @ box[7:9]
    return box


def class_nms(detections: list[dict]) -> list[dict]:
    """Suppress near-duplicate centers independently for each class."""
    kept = []
    for name in TRACKING_CLASSES:
        candidates = sorted(
            (d for d in detections if d['class_name'] == name),
            key=lambda d: float(d.get('score', 0.0)), reverse=True)
        if not candidates:
            continue
        radius = NMS_RADII_M[name]
        centers = np.array(
            [det['bbox_3d'][:2] for det in candidates], dtype=np.float64)
        distances = np.linalg.norm(
            centers[:, None, :] - centers[None, :, :], axis=2)
        kept_idx = []
        for idx in range(len(candidates)):
            if kept_idx and np.any(distances[idx, kept_idx] < radius):
                continue
            kept_idx.append(idx)
        kept.extend(candidates[idx] for idx in kept_idx)
    return kept


@dataclass
class Track:
    """One filtered global-frame object track."""

    track_id: str
    class_name: str
    state: np.ndarray
    covariance: np.ndarray
    size: np.ndarray
    score: float
    model: str
    timestamp: float
    hits: int = 1
    misses: int = 0
    missed_seconds: float = 0.0
    age: int = 1
    confirmed: bool = False
    moving: bool = False
    acceleration: float = 0.0
    history: list[tuple[float, float, float]] = field(default_factory=list)
    last_speed: float = 0.0

    @classmethod
    def create(cls, track_id: str, det: dict, timestamp: float,
               model: str) -> 'Track':
        box = np.asarray(det['bbox_3d'], dtype=np.float64)
        velocity = box[7:9] if len(box) >= 9 else np.zeros(2)
        state = np.array([
            box[0], box[1], box[2], box[6],
            velocity[0], velocity[1], 0.0,
        ], dtype=np.float64)
        covariance = np.diag([2.0, 2.0, 1.0, 0.5, 4.0, 4.0, 1.0])
        speed = float(np.linalg.norm(velocity))
        return cls(
            track_id=track_id,
            class_name=det['class_name'],
            state=state,
            covariance=covariance,
            size=box[3:6].copy(),
            score=float(det.get('score', 0.0)),
            model=model,
            timestamp=float(timestamp),
            moving=speed >= 0.3,
            history=[(float(timestamp), float(box[0]), float(box[1]))],
            last_speed=speed,
        )

    @property
    def speed(self) -> float:
        return math.hypot(self.state[4], self.state[5])

    @property
    def bbox(self) -> np.ndarray:
        return np.concatenate((
            self.state[:3], self.size, self.state[3:4], self.state[4:6]))

    def predict(self, timestamp: float) -> None:
        dt = max(0.0, float(timestamp) - self.timestamp)
        if dt <= 0:
            return
        x, y, _z, yaw, vx, vy, yaw_rate = self.state
        if self.model in {'ctrv', 'bicycle'} and abs(yaw_rate) > 1e-3:
            angle = yaw_rate * dt
            c, s = math.cos(angle), math.sin(angle)
            new_vx, new_vy = c * vx - s * vy, s * vx + c * vy
            self.state[0] = x + 0.5 * (vx + new_vx) * dt
            self.state[1] = y + 0.5 * (vy + new_vy) * dt
            self.state[4:6] = new_vx, new_vy
        else:
            self.state[0] = x + vx * dt
            self.state[1] = y + vy * dt
        self.state[3] = wrap_angle(yaw + yaw_rate * dt)

        f = _IDENTITY_7.copy()
        f[0, 4] = dt
        f[1, 5] = dt
        f[3, 6] = dt
        noise = 1.2 if self.class_name == 'pedestrian' else (
            2.0 if self.model == 'bicycle' else 0.8)
        covariance = f @ self.covariance @ f.T
        dt2 = dt * dt
        covariance[_DIAG_7] += (
            noise * dt2, noise * dt2, 0.2 * dt2,
            0.15 * dt2, noise * dt, noise * dt, 0.2 * dt)
        self.covariance = covariance
        self.timestamp = float(timestamp)
        self.age += 1
        self.misses += 1
        self.missed_seconds += dt

    def update(self, det: dict) -> None:
        box = np.asarray(det['bbox_3d'], dtype=np.float64)
        has_velocity = len(box) >= 9 and np.isfinite(box[7:9]).all()
        # The measurement observes state components directly, so H is a row
        # selection: H P H^T, P H^T, and H P become index slices.
        size = 6 if has_velocity else 4
        observed = _OBSERVED[:size]
        measurement = np.concatenate(
            (box[:3], box[6:7], box[7:9]))[:size]
        old_yaw = float(self.state[3])
        innovation = measurement - self.state[observed]
        innovation[3] = wrap_angle(innovation[3])
        p_ht = self.covariance[:, observed]
        s = p_ht[observed] + _MEASUREMENT_NOISE[:size, :size]
        gain = p_ht @ np.linalg.inv(s)
        old_speed = self.speed
        self.state += gain @ innovation
        self.state[3] = wrap_angle(self.state[3])
        self.covariance = self.covariance - gain @ self.covariance[observed]
        self.size = 0.8 * self.size + 0.2 * box[3:6]
        self.score = 0.7 * float(det.get('score', 0.0)) + 0.3 * self.score
        self.hits += 1
        self.misses = 0
        self.missed_seconds = 0.0
        self.confirmed = self.confirmed or self.hits >= 2
        dt = max(self.timestamp - self.history[-1][0], 1e-3)
        observed_yaw_rate = wrap_angle(box[6] - old_yaw) / dt
        self.state[6] = 0.8 * self.state[6] + 0.2 * observed_yaw_rate
        self.acceleration = (self.speed - old_speed) / dt
        self.last_speed = self.speed
        if self.moving:
            self.moving = self.speed >= 0.15
        else:
            self.moving = self.speed >= 0.30
        self.history.append(
            (self.timestamp, float(self.state[0]), float(self.state[1])))
        cutoff = self.timestamp - 2.0
        self.history = [point for point in self.history if point[0] >= cutoff]

    def mark_coasted(self) -> None:
        self.score *= 0.85
        self.history.append(
            (self.timestamp, float(self.state[0]), float(self.state[1])))
        cutoff = self.timestamp - 2.0
        self.history = [point for point in self.history if point[0] >= cutoff]

    def forecast(
            self, horizon: float = 3.0,
            step: float = 0.5) -> list[list[float]]:
        """Roll the motion model forward without modifying the track."""
        result = []
        x, y = float(self.state[0]), float(self.state[1])
        vx, vy, omega = map(float, self.state[4:7])
        elapsed = 0.0
        while elapsed + step <= horizon + 1e-9:
            if self.model in {'ctrv', 'bicycle'} and abs(omega) > 1e-3:
                angle = omega * step
                c, s = math.cos(angle), math.sin(angle)
                nvx, nvy = c * vx - s * vy, s * vx + c * vy
                x += 0.5 * (vx + nvx) * step
                y += 0.5 * (vy + nvy) * step
                vx, vy = nvx, nvy
            else:
                x += vx * step
                y += vy * step
            elapsed += step
            result.append([elapsed, x, y])
        return result

    def as_dict(self) -> dict:
        velocity_heading = (
            math.atan2(self.state[5], self.state[4])
            if self.speed > 1e-3 else self.state[3])
        heading_delta = wrap_angle(velocity_heading - self.state[3])
        direction = 'forward'
        if abs(heading_delta) > 3.0 * math.pi / 4.0:
            direction = 'reverse'
        elif abs(heading_delta) > math.pi / 4.0:
            direction = 'sideways'
        return {
            'tracking_id': self.track_id,
            'class_name': self.class_name,
            'bbox_3d': self.bbox.tolist(),
            'score': float(self.score),
            'speed_mps': self.speed,
            'yaw_rate_rps': float(self.state[6]),
            'acceleration_mps2': float(self.acceleration),
            'moving': bool(self.moving),
            'direction': direction,
            'history': [list(item) for item in self.history],
            'forecast': self.forecast(),
            'coasted': self.misses > 0,
        }


class MultiObjectTracker:
    """Class-aware global-frame tracker with two-stage association."""

    def __init__(self, high_score: float = 0.4, low_score: float = 0.05,
                 min_hits: int = 2, max_age_s: float = 3.0,
                 motion_model: str = 'poly', output_coasted: bool = True,
                 birth_score: float = 0.05):
        self.high_score = float(high_score)
        # Unmatched detections at or above this score start tracks. It is
        # separate from ``high_score``, which only splits association stages.
        self.birth_score = float(birth_score)
        self.low_score = float(low_score)
        self.min_hits = int(min_hits)
        self.max_age_s = float(max_age_s)
        self.motion_model = motion_model
        self.output_coasted = bool(output_coasted)
        self.tracks: list[Track] = []
        self._next_id = 1

    def _model_for(self, name: str) -> str:
        if self.motion_model == 'cv':
            return 'cv'
        if name in VEHICLE_CLASSES:
            return 'ctrv'
        if name in CYCLE_CLASSES:
            return 'bicycle'
        return 'cv'

    def _cost_matrix(self, track_indices: list[int],
                     detections: list[dict]) -> np.ndarray:
        """Distance + 0.25 * |yaw diff| + 0.5 * mean |log size ratio|.

        Pairs of different classes or beyond the class center gate get 1e6.
        """
        tracks = [self.tracks[idx] for idx in track_indices]
        track_xy = np.array([track.state[:2] for track in tracks])
        track_yaw = np.array([track.state[3] for track in tracks])
        track_size = np.array([track.size[:2] for track in tracks])
        track_class = np.array([track.class_name for track in tracks])
        gates = np.array([CLASS_GATES_M[name] for name in track_class])
        boxes = np.array(
            [det['bbox_3d'][:7] for det in detections], dtype=np.float64)
        det_class = np.array([det['class_name'] for det in detections])

        distance = np.linalg.norm(
            track_xy[:, None, :] - boxes[None, :, :2], axis=2)
        yaw_diff = track_yaw[:, None] - boxes[None, :, 6]
        yaw_cost = np.abs((yaw_diff + math.pi) % (2.0 * math.pi) - math.pi)
        size_cost = np.mean(np.abs(np.log(
            np.maximum(track_size, 0.1)[:, None, :]
            / np.maximum(boxes[:, 3:5], 0.1)[None, :, :])), axis=2)
        costs = distance + 0.25 * yaw_cost + 0.5 * size_cost
        valid = ((track_class[:, None] == det_class[None, :])
                 & (distance <= gates[:, None]))
        return np.where(valid, costs, 1e6)

    def _associate(self, track_indices: list[int],
                   detections: list[dict]
                   ) -> tuple[list[tuple[int, int]], list[int]]:
        if not track_indices or not detections:
            return [], list(range(len(detections)))
        costs = self._cost_matrix(track_indices, detections)
        rows, cols = linear_sum_assignment(costs)
        matches = []
        matched_dets = set()
        for row, col in zip(rows, cols):
            if costs[row, col] >= 1e5:
                continue
            matches.append((track_indices[row], int(col)))
            matched_dets.add(int(col))
        return matches, [
            idx for idx in range(len(detections)) if idx not in matched_dets]

    def step(self, detections: list[dict], timestamp: float) -> list[dict]:
        """Advance the tracker and return confirmed/coasted tracks."""
        if _BLAS is None:
            return self._step(detections, timestamp)
        # 7x7 Kalman algebra is slower with multithreaded BLAS.
        with _BLAS.limit(limits=1, user_api='blas'):
            return self._step(detections, timestamp)

    def _step(self, detections: list[dict], timestamp: float) -> list[dict]:
        timestamp = float(timestamp)
        detections = class_nms([
            d for d in detections
            if d.get('class_name') in TRACKING_CLASSES
            and float(d.get('score', 0.0)) >= self.low_score
        ])
        for track in self.tracks:
            track.predict(timestamp)

        high = [
            det for det in detections
            if det.get('score', 0.0) >= self.high_score]
        low = [
            det for det in detections
            if det.get('score', 0.0) < self.high_score]
        available = list(range(len(self.tracks)))
        matched_high, unmatched_high = self._associate(available, high)
        matched_tracks = set()
        for track_idx, det_idx in matched_high:
            self.tracks[track_idx].update(high[det_idx])
            matched_tracks.add(track_idx)

        remaining_tracks = [
            idx for idx in available if idx not in matched_tracks]
        matched_low, unmatched_low = self._associate(remaining_tracks, low)
        for track_idx, det_idx in matched_low:
            self.tracks[track_idx].update(low[det_idx])
            matched_tracks.add(track_idx)

        for idx, track in enumerate(self.tracks):
            if idx not in matched_tracks:
                track.mark_coasted()

        births = [high[idx] for idx in unmatched_high]
        births += [low[idx] for idx in unmatched_low]
        for det in births:
            if float(det.get('score', 0.0)) < self.birth_score:
                continue
            track = Track.create(
                str(self._next_id), det, timestamp,
                self._model_for(det['class_name']))
            track.confirmed = self.min_hits <= 1
            self._next_id += 1
            self.tracks.append(track)

        self.tracks = [
            track for track in self.tracks
            if track.missed_seconds <= self.max_age_s
        ]
        return [
            track.as_dict() for track in self.tracks
            if track.confirmed and track.missed_seconds <= self.max_age_s
            and (self.output_coasted or track.misses == 0)
        ]


class ConstantVelocityTracker:
    """CenterPoint-style baseline: predict with velocity, match greedily.

    Rules:
        - Input: same class filter, score threshold, and class NMS as
          :class:`MultiObjectTracker`.
        - Predict: each track moves by its last detector velocity times the
          elapsed time. There is no filtering; state is the last detection.
        - Associate: per class, detections in descending score order take the
          nearest unclaimed predicted track within ``CLASS_GATES_M``.
        - Create: every unmatched detection starts a new track.
        - Update: a matched track copies the detection box and velocity.
        - Delete: a track unmatched for more than ``max_age_s`` is removed.
        - Output: only tracks matched in the current frame.
    """

    def __init__(self, low_score: float = 0.05, max_age_s: float = 1.5):
        self.low_score = float(low_score)
        self.max_age_s = float(max_age_s)
        self.tracks: list[dict] = []
        self._next_id = 1

    def step(self, detections: list[dict], timestamp: float) -> list[dict]:
        timestamp = float(timestamp)
        detections = class_nms([
            d for d in detections
            if d.get('class_name') in TRACKING_CLASSES
            and float(d.get('score', 0.0)) >= self.low_score
        ])
        for track in self.tracks:
            dt = max(0.0, timestamp - track['timestamp'])
            track['box'][:2] += track['box'][7:9] * dt
            track['timestamp'] = timestamp
            track['missed_seconds'] += dt

        claimed = set()
        outputs = []
        for det in sorted(detections, key=lambda d: float(d['score']),
                          reverse=True):
            box = np.asarray(det['bbox_3d'], dtype=np.float64).copy()
            if len(box) < 9:
                box = np.r_[box[:7], 0.0, 0.0]
            gate = CLASS_GATES_M[det['class_name']]
            best, best_distance = None, gate
            for index, track in enumerate(self.tracks):
                if (index in claimed
                        or track['class_name'] != det['class_name']):
                    continue
                distance = float(np.linalg.norm(track['box'][:2] - box[:2]))
                if distance <= best_distance:
                    best, best_distance = index, distance
            if best is None:
                track = {
                    'track_id': str(self._next_id),
                    'class_name': det['class_name'],
                    'history': [],
                }
                self._next_id += 1
                self.tracks.append(track)
                best = len(self.tracks) - 1
            track = self.tracks[best]
            claimed.add(best)
            track.update({
                'box': box,
                'score': float(det['score']),
                'timestamp': timestamp,
                'missed_seconds': 0.0,
            })
            track['history'].append((timestamp, float(box[0]), float(box[1])))
            track['history'] = [
                item for item in track['history']
                if item[0] >= timestamp - 2.0]
            outputs.append(self._as_dict(track))

        self.tracks = [
            track for track in self.tracks
            if track['missed_seconds'] <= self.max_age_s]
        return outputs

    @staticmethod
    def _as_dict(track: dict) -> dict:
        box = track['box']
        speed = float(np.linalg.norm(box[7:9]))
        forecast = [
            [t, float(box[0] + box[7] * t), float(box[1] + box[8] * t)]
            for t in np.arange(0.5, 3.01, 0.5)]
        return {
            'tracking_id': track['track_id'],
            'class_name': track['class_name'],
            'bbox_3d': box.tolist(),
            'score': track['score'],
            'speed_mps': speed,
            'yaw_rate_rps': 0.0,
            'acceleration_mps2': 0.0,
            'moving': speed >= 0.2,
            'direction': 'forward',
            'history': [list(item) for item in track['history']],
            'forecast': forecast,
            'coasted': False,
        }
