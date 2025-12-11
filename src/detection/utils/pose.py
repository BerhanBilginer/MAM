from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy import typing as npt
from ultralytics import YOLO


DEFAULT_ALLOWED_CLASS_NAMES: set[str] = {"person"}
DEFAULT_CONF_THRESHOLD: float = 0.25
DEFAULT_KP_CONF_THRESHOLD: float = 0.3


@dataclass(frozen=True)
class PoseDetection:
    """Container holding a single person detection and filtered keypoints."""

    bbox: tuple[float, float, float, float]
    confidence: float | None
    class_id: int
    class_name: str
    keypoints: dict[int, tuple[float, float]]


def load_model(model_path: str | Path, device: str | int | None = None) -> YOLO:
    """Load a YOLO pose model and optionally move it to a device."""

    model = YOLO(str(model_path))
    if device is not None:
        model.to(device)
    return model


def _normalize_class_name(names: dict[int, str] | list[str] | None, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def detect_people(
    model: YOLO,
    frame: npt.NDArray[np.uint8],
    *,
    conf: float = DEFAULT_CONF_THRESHOLD,
    iou: float = 0.7,
    max_det: int = 100,
    allowed_class_names: Iterable[str] | None = None,
    keypoint_conf_threshold: float = DEFAULT_KP_CONF_THRESHOLD,
) -> list[PoseDetection]:
    """Run YOLO pose inference on a frame and return filtered detections."""

    results = model.predict(frame, conf=conf, iou=iou, max_det=max_det, verbose=False)
    detections: list[PoseDetection] = []

    if not results:
        return detections

    yolo_result = results[0]
    boxes = yolo_result.boxes
    if boxes is None or boxes.xyxy is None or boxes.xyxy.shape[0] == 0:
        return detections

    normalized_allowed = (
        {name.lower() for name in allowed_class_names}
        if allowed_class_names is not None
        else {name.lower() for name in DEFAULT_ALLOWED_CLASS_NAMES}
    )

    kp_xy = yolo_result.keypoints.xy if yolo_result.keypoints is not None else None
    kp_conf = (
        yolo_result.keypoints.conf
        if yolo_result.keypoints is not None and yolo_result.keypoints.conf is not None
        else None
    )

    for det_idx in range(boxes.xyxy.shape[0]):
        class_id = int(boxes.cls[det_idx].item()) if boxes.cls is not None else -1
        class_name = _normalize_class_name(yolo_result.names, class_id)

        if normalized_allowed and class_name.lower() not in normalized_allowed:
            continue

        bbox_tensor = boxes.xyxy[det_idx]
        bbox_np = (
            bbox_tensor.cpu().numpy() if hasattr(bbox_tensor, "cpu") else np.asarray(bbox_tensor)
        )
        x1, y1, x2, y2 = map(float, bbox_np.tolist())

        conf_tensor = boxes.conf[det_idx] if boxes.conf is not None else None
        conf_val = float(conf_tensor.item()) if conf_tensor is not None else None

        keypoints: dict[int, tuple[float, float]] = {}
        if kp_xy is not None:
            xy_tensor = kp_xy[det_idx]
            xy_np = xy_tensor.cpu().numpy() if hasattr(xy_tensor, "cpu") else np.asarray(xy_tensor)

            conf_array = None
            if kp_conf is not None:
                conf_tensor = kp_conf[det_idx]
                conf_array = (
                    conf_tensor.cpu().numpy()
                    if hasattr(conf_tensor, "cpu")
                    else np.asarray(conf_tensor)
                )

            for kp_idx, coords in enumerate(xy_np):
                x_kp, y_kp = float(coords[0]), float(coords[1])
                kp_conf_val = (
                    float(conf_array[kp_idx])
                    if conf_array is not None
                    else 1.0
                )
                if kp_conf_val < keypoint_conf_threshold or (x_kp == 0.0 and y_kp == 0.0):
                    continue
                keypoints[kp_idx] = (x_kp, y_kp)

        detections.append(
            PoseDetection(
                bbox=(x1, y1, x2, y2),
                confidence=conf_val,
                class_id=class_id,
                class_name=class_name,
                keypoints=keypoints,
            )
        )

    return detections


def extract_keypoints(detections: Iterable[PoseDetection]) -> list[dict[int, tuple[float, float]]]:
    """Utility that converts pose detections into the fall detection keypoint format."""

    return [det.keypoints for det in detections if det.keypoints]