from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.detection.utils.detection import Detection
from src.detection.utils.pose import PoseDetection
from src.detection.panic.panic import (
    YoloPoseFlowFusionPanic,
    FusionPanicConfig,
    FusionPanicResult,
)

if TYPE_CHECKING:
    from numpy import typing as npt


class PanicDetector:
    """Wrapper for panic detection that works with PoseDetection objects."""

    def __init__(self, frame_width: int, frame_height: int, fps: float = 30.0) -> None:
        self.config = FusionPanicConfig()
        self.detector = YoloPoseFlowFusionPanic(fps=fps, config=self.config)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self._last_result: FusionPanicResult | None = None

    def process(self, detections: list[PoseDetection]) -> FusionPanicResult:
        """Process pose detections and return panic detection result."""
        # This method will be called with frame in the main loop
        # For now, return a dummy result
        return FusionPanicResult(is_panic=False, score=0.0, metrics={})

    def update(
        self, frame: npt.NDArray[np.uint8], detections: list[PoseDetection]
    ) -> FusionPanicResult:
        """Update panic detector with new frame and detections."""
        # Convert PoseDetection to Detection
        simple_detections = [
            Detection(
                bbox=det.bbox,
                confidence=det.confidence,
                class_id=det.class_id,
                class_name=det.class_name,
                track_id=None,
            )
            for det in detections
        ]
        result = self.detector.update(frame, simple_detections)
        self._last_result = result
        return result

    def get_heatmap_image(self) -> npt.NDArray[np.uint8]:
        """Generate optical flow magnitude heatmap visualization."""
        # Create heatmap from optical flow data
        heatmap = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        
        # If we have flow magnitude data, visualize it
        if self.detector._last_flow_mag is not None:
            mag = self.detector._last_flow_mag
            
            # Normalize magnitude to 0-255 range for visualization
            mag_normalized = np.clip(mag * 10, 0, 255).astype(np.uint8)
            
            # Apply colormap (JET: blue=low, red=high motion)
            flow_colored = cv2.applyColorMap(mag_normalized, cv2.COLORMAP_JET)
            
            # Resize to original frame size
            heatmap = cv2.resize(flow_colored, (self.frame_width, self.frame_height))
        
        # Draw status overlay
        if self._last_result is not None:
            status = "PANIC!" if self._last_result.is_panic else "Normal"
            color = (0, 0, 255) if self._last_result.is_panic else (0, 255, 0)
            
            # Semi-transparent background for text
            overlay = heatmap.copy()
            cv2.rectangle(overlay, (8, 8), (520, 140), (0, 0, 0), thickness=-1)
            cv2.addWeighted(overlay, 0.7, heatmap, 0.3, 0, heatmap)
            
            cv2.putText(
                heatmap,
                f"Status: {status}",
                (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
            )
            cv2.putText(
                heatmap,
                f"Score: {self._last_result.score:.2f}",
                (16, 64),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )
            
            m = self._last_result.metrics
            cv2.putText(
                heatmap,
                f"v_p95: {m.get('v_p95', 0):.2f}  H_dir: {m.get('h_dir', 0):.2f}",
                (16, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                heatmap,
                f"People: {int(m.get('people', 0))}",
                (16, 114),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
            )
        else:
            cv2.putText(
                heatmap,
                "Optical Flow Heatmap",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
        
        return heatmap


def draw_video_frame(
    frame: npt.NDArray[np.uint8],
    detections: list[PoseDetection],
    result: FusionPanicResult,
) -> npt.NDArray[np.uint8]:
    """Draw bounding boxes and panic status on the video frame."""
    annotated = frame.copy()

    # Draw bounding boxes for each detection
    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        color = (0, 0, 255) if result.is_panic else (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        if det.confidence is not None:
            label = f"{det.class_name}: {det.confidence:.2f}"
            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    # Draw panic status
    status_text = "PANIC!" if result.is_panic else "Normal"
    status_color = (0, 0, 255) if result.is_panic else (0, 255, 0)
    cv2.putText(
        annotated,
        f"Status: {status_text}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2,
    )

    # Draw score
    cv2.putText(
        annotated,
        f"Score: {result.score:.2f}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    # Draw metrics
    if result.metrics:
        metrics_text = f"People: {int(result.metrics.get('people', 0))}"
        cv2.putText(
            annotated,
            metrics_text,
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    return annotated
