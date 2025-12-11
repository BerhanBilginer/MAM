from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from collections import deque

import cv2
import numpy as np
from numpy import typing as npt

from src.detection.utils.pose import PoseDetection


DEFAULT_ENTROPY_THRESHOLD: float = 10.0
"""Base entropy threshold above which the scene is considered PANIC."""

DEFAULT_HEATMAP_DECAY: float = 0.5
"""Decay factor for cumulative heatmap (0-1). Higher = longer memory."""

DEFAULT_GAUSSIAN_SIGMA: int = 15
"""Sigma for Gaussian blur applied to heatmap points."""


@dataclass
class PanicDetectorConfig:
    """Configuration for PanicDetector."""

    entropy_threshold: float = DEFAULT_ENTROPY_THRESHOLD
    heatmap_decay: float = DEFAULT_HEATMAP_DECAY
    gaussian_sigma: int = DEFAULT_GAUSSIAN_SIGMA

    # Motion bileşeni & kişi sayısı guard
    motion_weight: float = 1.0
    """How strongly per-person motion contributes to the heatmap."""

    min_people_for_panic: int = 2
    """Do not output PANIC if fewer people than this are in the scene."""

    # Entropy delta + rolling window
    rolling_window_size: int = 10
    """Number of frames in the rolling entropy window."""

    delta_entropy_threshold: float = 0.5
    """Minimum positive change in entropy considered as a spike."""

    variance_threshold: float = 0.3
    """Entropy variance threshold - high variance indicates rapid changes."""

    consecutive_spikes_for_panic: int = 3
    """Number of consecutive delta spikes needed to trigger panic."""


@dataclass
class PanicDetectionResult:
    """Result of panic detection for a frame."""

    is_panic: bool
    entropy: float               # current frame entropy
    rolling_entropy: float       # rolling mean entropy
    delta_entropy: float         # current - previous entropy
    entropy_variance: float      # variance in rolling window
    spike_count: int             # consecutive spike count
    heatmap: npt.NDArray[np.float32] | None = None


class PanicDetector:
    """Detects panic situations based on cumulative movement entropy."""

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        config: PanicDetectorConfig | None = None,
    ) -> None:
        self.config = config or PanicDetectorConfig()
        self.frame_width = frame_width
        self.frame_height = frame_height

        self._cumulative_heatmap: npt.NDArray[np.float32] = np.zeros(
            (frame_height, frame_width), dtype=np.float32
        )
        self._prev_centers: dict[int, tuple[float, float]] = {}

        # Entropy history & previous entropy
        self._entropy_history: deque[float] = deque(maxlen=self.config.rolling_window_size)
        self._prev_entropy: float | None = None
        self._spike_count: int = 0

    @staticmethod
    def _get_bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
        """Calculate center point of a bounding box."""
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    def _update_heatmap(self, detections: list[PoseDetection]) -> None:
        """Update cumulative heatmap based on proximity + motion."""

        # Kümülatif bellek decay
        self._cumulative_heatmap *= self.config.heatmap_decay

        if not detections:
            self._prev_centers.clear()
            return

        centers: list[tuple[float, float]] = []
        track_ids: list[int] = []

        for i, det in enumerate(detections):
            cx, cy = self._get_bbox_center(det.bbox)

            track_id = getattr(det, "track_id", None)
            if track_id is None:
                track_id = getattr(det, "id", i)

            centers.append((cx, cy))
            track_ids.append(int(track_id))

        current_frame_heatmap = np.zeros_like(self._cumulative_heatmap)

        # Kişi bazlı skor hesaplama
        for idx, (cx_i, cy_i) in enumerate(centers):
            tid = track_ids[idx]

            # Motion bileşeni
            prev_center = self._prev_centers.get(tid)
            if prev_center is not None:
                dx = cx_i - prev_center[0]
                dy = cy_i - prev_center[1]
                motion = float(np.hypot(dx, dy))
            else:
                motion = 0.0

            self._prev_centers[tid] = (cx_i, cy_i)

            # Proximity bileşeni
            proximity_score = 0.0
            for jdx, (cx_j, cy_j) in enumerate(centers):
                if idx == jdx:
                    continue

                dist = float(np.hypot(cx_i - cx_j, cy_i - cy_j))
                if dist < 1.0:
                    dist = 1.0

                proximity_score += 1.0 / dist

            score = proximity_score + self.config.motion_weight * motion

            cx_int = int(np.clip(cx_i, 0, self.frame_width - 1))
            cy_int = int(np.clip(cy_i, 0, self.frame_height - 1))

            current_frame_heatmap[cy_int, cx_int] += score

        if self.config.gaussian_sigma > 0:
            current_frame_heatmap = cv2.GaussianBlur(
                current_frame_heatmap,
                (0, 0),
                self.config.gaussian_sigma,
            )

        self._cumulative_heatmap += current_frame_heatmap

        valid_ids = set(track_ids)
        self._prev_centers = {
            tid: c for tid, c in self._prev_centers.items() if tid in valid_ids
        }

    def _calculate_entropy(self) -> float:
        """Calculate Shannon entropy of the cumulative heatmap."""
        heatmap = self._cumulative_heatmap.copy()

        total = float(heatmap.sum())
        if total <= 0.0:
            return 0.0

        prob = heatmap / total
        prob = prob[prob > 0.0]

        entropy = -np.sum(prob * np.log2(prob))
        return float(entropy)

    def process(self, detections: Iterable[PoseDetection]) -> PanicDetectionResult:
        """Process detections and determine if scene is in PANIC state."""
        detections_list = list(detections)

        self._update_heatmap(detections_list)

        # Entropy hesapları
        entropy = self._calculate_entropy()

        self._entropy_history.append(entropy)
        if len(self._entropy_history) > 1:
            rolling_entropy = float(np.mean(self._entropy_history))
            entropy_variance = float(np.var(self._entropy_history))
        else:
            rolling_entropy = entropy
            entropy_variance = 0.0

        if self._prev_entropy is None:
            delta_entropy = 0.0
        else:
            delta_entropy = entropy - self._prev_entropy

        self._prev_entropy = entropy

        # Ani değişim (spike) tespiti
        is_spike = abs(delta_entropy) > self.config.delta_entropy_threshold
        if is_spike:
            self._spike_count += 1
        else:
            self._spike_count = 0

        # Panic kararı - ani değişimlere odaklan
        if len(detections_list) < self.config.min_people_for_panic:
            is_panic = False
        else:
            # Ardışık spike'lar veya yüksek varyans = PANIC
            is_panic = (
                self._spike_count >= self.config.consecutive_spikes_for_panic
                or entropy_variance > self.config.variance_threshold
            )

        return PanicDetectionResult(
            is_panic=is_panic,
            entropy=entropy,
            rolling_entropy=rolling_entropy,
            delta_entropy=delta_entropy,
            entropy_variance=entropy_variance,
            spike_count=self._spike_count,
            heatmap=self._cumulative_heatmap.copy(),
        )

    def get_heatmap_image(self) -> npt.NDArray[np.uint8]:
        """Generate a standalone colored heatmap image."""
        heatmap = self._cumulative_heatmap.copy()

        if heatmap.max() > 0:
            heatmap_normalized = (heatmap / heatmap.max() * 255).astype(np.uint8)
        else:
            heatmap_normalized = np.zeros_like(heatmap, dtype=np.uint8)

        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        return heatmap_colored

    def reset(self) -> None:
        """Reset cumulative heatmap and tracking state."""
        self._cumulative_heatmap.fill(0)
        self._prev_centers.clear()
        self._entropy_history.clear()
        self._prev_entropy = None
        self._spike_count = 0


def draw_video_frame(
    frame: npt.NDArray[np.uint8],
    detections: Iterable[PoseDetection],
    result: PanicDetectionResult,
) -> npt.NDArray[np.uint8]:
    """Draw bounding boxes and status info on video frame."""
    output = frame.copy()
    detections_list = list(detections)

    # Bounding box'ları çiz
    for det in detections_list:
        x1, y1, x2, y2 = map(int, det.bbox)
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Sol üstte durum kutusu
    status_text = "PANIC!" if result.is_panic else "Normal"
    status_color = (0, 0, 255) if result.is_panic else (0, 255, 0)

    # Arka plan
    cv2.rectangle(output, (5, 5), (250, 155), (0, 0, 0), thickness=-1)

    cv2.putText(
        output,
        f"Status: {status_text}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2,
    )

    cv2.putText(
        output,
        f"Delta: {result.delta_entropy:+.2f}",
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    cv2.putText(
        output,
        f"Variance: {result.entropy_variance:.3f}",
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    cv2.putText(
        output,
        f"Spikes: {result.spike_count}",
        (10, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    cv2.putText(
        output,
        f"Entropy: {result.entropy:.2f}",
        (10, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    cv2.putText(
        output,
        f"People: {len(detections_list)}",
        (10, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    return output
