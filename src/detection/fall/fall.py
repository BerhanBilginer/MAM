from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import cv2
import numpy as np
from numpy import typing as npt

from src.detection.utils.pose import PoseDetection


ANGLE_THRESH: float = 30.0
"""Trunk angle threshold (degrees) for fall detection."""

BENDING_ANGLE_THRESH: float = 45.0
"""Trunk angle threshold (degrees) for bending detection."""

ASPECT_TOL: float = 0.3
"""Aspect ratio tolerance fraction for fall detection."""

DELTA_Y_FRAC: float = 0.2
"""Delta Y fraction of bbox height for fall/sitting detection."""

SMALL_TOL: float = 15.0
"""Tolerance for sitting trunk angle (degrees from vertical)."""

SHOULDER_TOL: float = 20.0
"""Tolerance for shoulder horizontal angle for sitting (degrees from horizontal)."""

LIE_ANGLE_THRESH: float = 25.0
"""Trunk ≤ 25° from horizontal ⇒ lying."""

LIE_ASPECT_FRAC: float = 0.5
"""h/shoulder_dist ≤ 0.5 x ideal ⇒ lying."""


KEYPOINT_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1),
    (1, 3),
    (0, 2),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (11, 12),
    (5, 11),
    (6, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


@dataclass
class FallDetectorConfig:
    """Configuration for the FallDetector."""

    angle_thresh: float = ANGLE_THRESH
    bending_angle_thresh: float = BENDING_ANGLE_THRESH
    aspect_tol: float = ASPECT_TOL
    delta_y_frac: float = DELTA_Y_FRAC
    small_tol: float = SMALL_TOL
    shoulder_tol: float = SHOULDER_TOL
    lie_angle_thresh: float = LIE_ANGLE_THRESH
    lie_aspect_frac: float = LIE_ASPECT_FRAC


@dataclass
class PersonFallResult:
    """Result of fall detection for a single person."""

    bbox: tuple[float, float, float, float]
    pose_class: str  # "normal", "bending", "lying"
    sitting: bool
    label: str  # display label e.g. "NORMAL", "LYING/FALL", "BENDING", "SITTING"
    color: tuple[int, int, int]  # BGR color for annotation
    keypoints: dict[int, tuple[float, float]]


@dataclass
class FallDetectionResult:
    """Result of fall detection for a frame."""

    frame_status: str  # "normal", "anomaly", "bending", "falling"
    person_results: list[PersonFallResult] = field(default_factory=list)


class FallDetector:
    """Standalone fall detector that works with PoseDetection objects."""

    def __init__(self, config: FallDetectorConfig | None = None) -> None:
        self.config = config or FallDetectorConfig()
        self._ideal_aspect: float | None = None

    def _get_body_angle(self, kps: dict[int, tuple[float, float]]) -> float | None:
        """Calculate the angle of the trunk axis relative to the horizontal axis."""
        if not all(idx in kps for idx in (5, 6, 11, 12)):
            return None

        x5, y5 = kps[5]
        x6, y6 = kps[6]
        x11, y11 = kps[11]
        x12, y12 = kps[12]
        mid_sh = ((x5 + x6) * 0.5, (y5 + y6) * 0.5)
        mid_hp = ((x11 + x12) * 0.5, (y11 + y12) * 0.5)
        dx, dy = mid_hp[0] - mid_sh[0], mid_hp[1] - mid_sh[1]
        return float(abs(np.degrees(np.arctan2(dy, dx))))

    def _get_shoulder_angle(self, kps: dict[int, tuple[float, float]]) -> float | None:
        """Calculate the angle of the shoulder axis relative to the horizontal."""
        if not all(idx in kps for idx in (5, 6)):
            return None

        x5, y5 = kps[5]
        x6, y6 = kps[6]
        return float(abs(np.degrees(np.arctan2(y6 - y5, x6 - x5))))

    def _pose_classification(
        self, kps: dict[int, tuple[float, float]], h: float
    ) -> tuple[str, float, float, float]:
        """Classify pose as lying, bending, or normal."""
        angle = self._get_body_angle(kps) or 90.0

        kp5, kp6 = kps[5], kps[6]
        shoulder_dist = max(abs(kp6[0] - kp5[0]), 1)
        aspect = h / shoulder_dist

        if self._ideal_aspect is None:
            self._ideal_aspect = aspect

        ideal_aspect = self._ideal_aspect
        dyn_aspect = ideal_aspect * (1 + self.config.aspect_tol)

        kp11, kp12 = kps[11], kps[12]
        mid_sh_y = (kp5[1] + kp6[1]) * 0.5
        mid_hp_y = (kp11[1] + kp12[1]) * 0.5
        delta_y = abs(mid_hp_y - mid_sh_y)

        aspect_vote = aspect < dyn_aspect
        delta_vote = delta_y < self.config.delta_y_frac * h

        is_lying = (angle <= self.config.lie_angle_thresh) and (
            aspect <= ideal_aspect * self.config.lie_aspect_frac
        )

        is_bending = (
            not is_lying
            and angle <= self.config.bending_angle_thresh
            and (delta_vote or aspect_vote)
        )

        if is_lying:
            pose_class = "lying"
        elif is_bending:
            pose_class = "bending"
        else:
            pose_class = "normal"

        return pose_class, angle, aspect, delta_y

    def _sitting_detection(self, kps: dict[int, tuple[float, float]], h: float) -> bool:
        """Detect sitting position."""
        if not all(idx in kps for idx in (5, 6, 11, 12)):
            return False

        kp5, kp6 = kps[5], kps[6]
        kp11, kp12 = kps[11], kps[12]
        mid_sh_y = (kp5[1] + kp6[1]) * 0.5
        mid_hp_y = (kp11[1] + kp12[1]) * 0.5
        delta_y = abs(mid_hp_y - mid_sh_y)

        dyn_delta_thr = self.config.delta_y_frac * h
        if delta_y > dyn_delta_thr:
            return False

        body_ang = self._get_body_angle(kps)
        if body_ang is not None and body_ang >= (90.0 - self.config.small_tol):
            return True

        sh_ang = self._get_shoulder_angle(kps)
        return sh_ang is not None and sh_ang <= self.config.shoulder_tol

    def process(self, detections: Iterable[PoseDetection]) -> FallDetectionResult:
        """Run fall detection on a list of pose detections.

        Args:
            detections: Iterable of PoseDetection objects from pose.py.

        Returns:
            FallDetectionResult with frame status and per-person results.
        """
        frame_status = "normal"
        person_results: list[PersonFallResult] = []

        for det in detections:
            kps = det.keypoints
            if not all(idx in kps for idx in (5, 6, 11, 12)):
                continue

            x1, y1, x2, y2 = det.bbox
            h = y2 - y1

            pose_class, angle, aspect, delta_y = self._pose_classification(kps, h)
            sitting = self._sitting_detection(kps, h)

            if pose_class == "lying":
                color, label = (0, 0, 255), "LYING/FALL"
                frame_status = "falling"
            elif pose_class == "bending":
                color, label = (255, 140, 0), "BENDING"
                if frame_status not in ("falling",):
                    frame_status = "bending"
            elif sitting:
                color, label = (255, 165, 0), "SITTING"
                if frame_status not in ("falling", "bending"):
                    frame_status = "anomaly"
            else:
                color, label = (0, 255, 0), "NORMAL"

            person_results.append(
                PersonFallResult(
                    bbox=det.bbox,
                    pose_class=pose_class,
                    sitting=sitting,
                    label=label,
                    color=color,
                    keypoints=kps,
                )
            )

        return FallDetectionResult(frame_status=frame_status, person_results=person_results)

    def reset(self) -> None:
        """Reset internal state (e.g., ideal aspect ratio)."""
        self._ideal_aspect = None


def annotate_fall_results(
    frame: npt.NDArray[np.uint8],
    result: FallDetectionResult,
) -> None:
    """Draw fall detection annotations on a frame in-place."""
    for pr in result.person_results:
        x1, y1, x2, y2 = map(int, pr.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), pr.color, 2)
        cv2.putText(
            frame,
            pr.label,
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            pr.color,
            2,
        )

        for kp_x, kp_y in pr.keypoints.values():
            cv2.circle(frame, (int(kp_x), int(kp_y)), 4, (255, 255, 0), -1)

        for a_idx, b_idx in KEYPOINT_CONNECTIONS:
            if a_idx in pr.keypoints and b_idx in pr.keypoints:
                pt_a = pr.keypoints[a_idx]
                pt_b = pr.keypoints[b_idx]
                cv2.line(
                    frame,
                    (int(pt_a[0]), int(pt_a[1])),
                    (int(pt_b[0]), int(pt_b[1])),
                    (255, 255, 0),
                    2,
                )

    status_color = {
        "normal": (0, 255, 0),
        "anomaly": (255, 165, 0),
        "bending": (255, 140, 0),
        "falling": (0, 0, 255),
    }.get(result.frame_status, (255, 255, 255))

    cv2.putText(
        frame,
        f"Status: {result.frame_status.upper()}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2,
    )
