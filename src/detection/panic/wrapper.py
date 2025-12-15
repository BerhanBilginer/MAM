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
from src.detection.panic.convlstm_model import PanicConvLSTMDetector

if TYPE_CHECKING:
    from numpy import typing as npt


class PanicDetector:
    """Wrapper for panic detection that works with PoseDetection objects."""

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        fps: float = 30.0,
        *,
        convlstm_model_path: str | None = None,
        convlstm_device: str | None = None,
        convlstm_threshold: float | None = None,
        convlstm_sequence_length: int = 16,
        convlstm_image_size: int = 96,
    ) -> None:
        self.config = FusionPanicConfig()

        self._convlstm = None
        if convlstm_model_path is not None:
            self._convlstm = PanicConvLSTMDetector(
                model_path=convlstm_model_path,
                device=convlstm_device or "cpu",
                threshold=convlstm_threshold if convlstm_threshold is not None else 0.1,
                sequence_length=int(convlstm_sequence_length),
                image_size=(int(convlstm_image_size), int(convlstm_image_size)),
            )

        self.detector = YoloPoseFlowFusionPanic(fps=fps, config=self.config)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self._last_result: FusionPanicResult | None = None

        self._prev_gray_small = None
        self._last_flow_mag = None

    @staticmethod
    def _create_bbox_heatmap(detections: list[PoseDetection], image_size: tuple[int, int]) -> np.ndarray:
        h, w = image_size
        heatmap = np.zeros((h, w), dtype=np.float32)

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                heatmap[y1:y2, x1:x2] += 1.0

        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap

    @staticmethod
    def _to_gray_small(frame_bgr: npt.NDArray[np.uint8], size: tuple[int, int]) -> npt.NDArray[np.uint8]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, size)

    def process(self, detections: list[PoseDetection]) -> FusionPanicResult:
        """Process pose detections and return panic detection result."""
        # This method will be called with frame in the main loop
        # For now, return a dummy result
        return FusionPanicResult(is_panic=False, score=0.0, metrics={})

    def update(
        self, frame: npt.NDArray[np.uint8], detections: list[PoseDetection]
    ) -> FusionPanicResult:
        """Update panic detector with new frame and detections."""

        if self._convlstm is not None:
            h, w = self._convlstm.image_size
            gray_small = self._to_gray_small(frame, (w, h))

            if self._prev_gray_small is None:
                self._prev_gray_small = gray_small
                metrics = {
                    "people": float(len(detections)),
                    "reconstruction_error": 0.0,
                    "threshold": float(self._convlstm.threshold),
                }
                result = FusionPanicResult(is_panic=False, score=0.0, metrics=metrics)
                self._last_result = result
                return result

            flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray_small,
                gray_small,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            self._prev_gray_small = gray_small

            flow_mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).astype(np.float32)
            flow_angle = np.arctan2(flow[..., 1], flow[..., 0]).astype(np.float32)
            self._last_flow_mag = flow_mag

            heatmap = self._create_bbox_heatmap(detections, (h, w))
            out = self._convlstm.add_frame(flow_mag, flow_angle, heatmap)
            if out is None:
                metrics = {
                    "people": float(len(detections)),
                    "reconstruction_error": 0.0,
                    "threshold": float(self._convlstm.threshold),
                }
                result = FusionPanicResult(is_panic=False, score=0.0, metrics=metrics)
                self._last_result = result
                return result

            is_panic, err = out
            metrics = {
                "people": float(len(detections)),
                "reconstruction_error": float(err),
                "threshold": float(self._convlstm.threshold),
            }
            result = FusionPanicResult(is_panic=bool(is_panic), score=float(err), metrics=metrics)
            self._last_result = result
            return result

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
        mag = self._last_flow_mag if self._last_flow_mag is not None else self.detector._last_flow_mag
        if mag is not None:
            
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

            if "reconstruction_error" in m:
                cv2.putText(
                    heatmap,
                    f"Err: {m.get('reconstruction_error', 0.0):.6g}  Thr: {m.get('threshold', 0.0):.6g}",
                    (16, 138),
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
    score_text = f"Score: {result.score:.2f}"
    if result.metrics and "reconstruction_error" in result.metrics:
        score_text = f"Score: {float(result.score):.6g}"
    cv2.putText(
        annotated,
        score_text,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    # Draw metrics
    if result.metrics:
        m = result.metrics
        metrics_text = f"People: {int(m.get('people', 0))}"
        cv2.putText(
            annotated,
            metrics_text,
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        if "reconstruction_error" in m:
            cv2.putText(
                annotated,
                f"Err: {m.get('reconstruction_error', 0.0):.6g}  Thr: {m.get('threshold', 0.0):.6g}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

    return annotated
