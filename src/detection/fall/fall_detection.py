import logging
from typing import ClassVar, cast

import cv2
import numpy as np
from attrs import define, field, validators
from numpy import typing as npt

from breacheye.detection.inference.pipeline_modules.base import (
    BasePipelineModuleConstructDict,
    BasePipelineModuleInputs,
    BasePipelineModuleOutputs,
    PipelineModule,
    PipelineModuleType,
)
from breacheye.detection.models.base.results import Results

LOGGER = logging.getLogger(__name__)


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

KEYPOINT_CONF_THRESHOLD: float = 0.3
"""Minimum confidence for a keypoint to be considered valid."""

LIE_ANGLE_THRESH: float = 25.0
"trunk ≤ 25° from horizontal ⇒ lying"

LIE_ASPECT_FRAC: float = 0.5
"h/shoulder_dist ≤ 0.5 x ideal ⇒ lying"


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


class FallDetectionPipelineModuleConstructDict(BasePipelineModuleConstructDict):
    """Configuration for constructing the fall detection pipeline module."""

    fps: int
    """Frames per second used for processing input data."""


@define
class FallDetectionPipelineModuleInputs(BasePipelineModuleInputs):
    """Inputs for the fall detection pipeline module."""

    results: Results = field(init=True, validator=validators.instance_of(Results))
    """Inference results to process in the module."""


@define
class FallDetectionPipelineModuleOutputs(BasePipelineModuleOutputs):
    """Outputs of the fall detection pipeline module."""

    results: Results = field(init=True, validator=validators.instance_of(Results))
    """Processed inference results (passed through)."""

    fall_status: str = field(
        init=True, metadata={"choices": ["normal", "anomaly", "bending", "falling"]}
    )
    """Current fall status for the frame."""

    annotated_image: npt.NDArray[np.uint8] | None = field(default=None)
    """Annotated image with fall detection results."""

    keypoints_list: list[dict[int, tuple[float, float]]] | None = field(default=None)
    """List of per-person key-points in pixel coordinates. Each dict maps keypoint index to (x,y)."""

    processing_time: float = field(default=0.0, validator=validators.instance_of(float))
    """Processing time in seconds."""

    def __repr__(self) -> str:
        """Return string representation including results, fall status, annotated image shape, and processing time.

        Returns:
            String representation of the object.
        """
        img_shape = self.annotated_image.shape if self.annotated_image is not None else None
        return (
            "FallDetectionPipelineModuleOutputs("
            f"results={self.results}, fall_status={self.fall_status}, "
            f"annotated_image.shape={img_shape}, processing_time={self.processing_time})"
        )


@define
class FallDetectionPipelineModule(
    PipelineModule[FallDetectionPipelineModuleInputs, FallDetectionPipelineModuleOutputs]
):
    """Pipeline module that detects falls in a video stream."""

    module_type: ClassVar[PipelineModuleType] = PipelineModuleType.FALL_DETECTION
    """Identifier for the fall detection pipeline module."""

    fps: int = field(default=10, validator=validators.instance_of(int))
    """Frame rate (frames per second) used for processing. (to be removed if not used)"""

    angle_thresh: float = field(default=ANGLE_THRESH, validator=validators.instance_of(float))
    """Trunk angle threshold (degrees) for fall detection."""

    aspect_tol: float = field(default=ASPECT_TOL, validator=validators.instance_of(float))
    """Aspect ratio tolerance fraction for fall detection."""

    delta_y_frac: float = field(default=DELTA_Y_FRAC, validator=validators.instance_of(float))
    """Delta Y fraction of bbox height for fall/sitting detection."""

    small_tol: float = field(default=SMALL_TOL, validator=validators.instance_of(float))
    """Tolerance for sitting trunk angle (degrees from vertical)."""

    shoulder_tol: float = field(default=SHOULDER_TOL, validator=validators.instance_of(float))
    """Tolerance for shoulder horizontal angle for sitting (degrees from horizontal)."""

    keypoint_conf_threshold: float = field(
        default=KEYPOINT_CONF_THRESHOLD, validator=validators.instance_of(float)
    )
    """Minimum confidence for a keypoint to be considered valid."""

    lie_angle_thresh: float = field(
        default=LIE_ANGLE_THRESH, validator=validators.instance_of(float)
    )
    """Angle threshold for lying detection."""

    lie_aspect_frac: float = field(default=LIE_ASPECT_FRAC, validator=validators.instance_of(float))
    """Aspect ratio threshold for lying detection."""

    bending_angle_thresh: float = field(
        default=BENDING_ANGLE_THRESH, validator=validators.instance_of(float)
    )
    """Trunk angle threshold (degrees) for bending detection."""

    def __attrs_post_init__(self) -> None:
        """Post-initialization hook for attrs.

        Logs the initialization parameters of the FallDetection module.
        """
        LOGGER.info(
            f"FallDetection initialised | angle≤{self.angle_thresh:.1f}, "
            f"aspect_tol={self.aspect_tol:.2f}, Δy_frac={self.delta_y_frac:.2f}, "
            f"sit(vert/horiz)=±{self.small_tol:.1f}/±{self.shoulder_tol:.1f}, "
            f"kp_conf≥{self.keypoint_conf_threshold:.2f}, lie≤{self.lie_angle_thresh:.1f}°, "
            f"lie_aspect≤{self.lie_aspect_frac:.2f}x, bend≤{self.bending_angle_thresh:.1f}°"
        )

    def get_valid_predecessors(self) -> set[PipelineModuleType] | None:
        """Return set of valid predecessor module types.

        Returns:
            Set containing only DEFAULT as this module requires results from the default module
        """
        return {PipelineModuleType.DEFAULT}

    def get_body_angle(self, pts_list: list[tuple[float, float] | None]) -> float | None:
        """Calculate the angle of the trunk axis (shoulder midpoint to hip midpoint) relative to the horizontal axis.

        Args:
            pts_list: List of keypoints, where each keypoint is a tuple of (x, y) coordinates.

        Returns:
            Angle in degrees, or None if required keypoints are missing.
        """
        # Ensure critical keypoints are present
        if any(pts_list[idx] is None for idx in (5, 6, 11, 12)):
            return None

        x5, y5 = cast(tuple[float, float], pts_list[5])
        x6, y6 = cast(tuple[float, float], pts_list[6])
        x11, y11 = cast(tuple[float, float], pts_list[11])
        x12, y12 = cast(tuple[float, float], pts_list[12])
        mid_sh = ((x5 + x6) * 0.5, (y5 + y6) * 0.5)
        mid_hp = ((x11 + x12) * 0.5, (y11 + y12) * 0.5)
        dx, dy = mid_hp[0] - mid_sh[0], mid_hp[1] - mid_sh[1]
        return float(abs(np.degrees(np.arctan2(dy, dx))))

    def get_shoulder_angle(self, pts_list: list[tuple[float, float] | None]) -> float | None:
        """Calculate the angle of the left-right shoulder axis relative to the horizontal.

        Args:
            pts_list: List of keypoints, where each keypoint is a tuple of (x, y) coordinates.

        Returns:
            Angle in degrees, or None if required keypoints are missing.
        """
        if any(pts_list[idx] is None for idx in (5, 6)):
            return None

        x5, y5 = cast(tuple[float, float], pts_list[5])
        x6, y6 = cast(tuple[float, float], pts_list[6])
        return float(abs(np.degrees(np.arctan2(y6 - y5, x6 - x5))))

    def pose_classification(
        self, pts_list: list[tuple[float, float] | None], h: float, state: dict[str, float | None]
    ) -> tuple[str, float, float, float, int]:
        """Pose classification: distinguishes between lying, bending, and normal poses.

        Args:
            pts_list: List of keypoints, where each keypoint is a tuple of (x, y) coordinates.
            h: Height of the bounding box.
            state: State dictionary containing tracking information.

        Returns:
            Tuple of (pose_class, angle, aspect_ratio, delta_y, vote_count).
            pose_class can be: 'lying', 'bending', or 'normal'
        """
        angle = self.get_body_angle(pts_list) or 90.0

        # Safe casts because caller ensures required keypoints exist
        kp5 = cast(tuple[float, float], pts_list[5])
        kp6 = cast(tuple[float, float], pts_list[6])
        shoulder_dist = max(abs(kp6[0] - kp5[0]), 1)
        aspect = h / shoulder_dist

        if state.get("ideal_aspect") is None:
            state["ideal_aspect"] = aspect

        ideal_aspect = state["ideal_aspect"] or aspect
        dyn_aspect = ideal_aspect * (1 + self.aspect_tol)
        kp11 = cast(tuple[float, float], pts_list[11])
        kp12 = cast(tuple[float, float], pts_list[12])
        mid_sh_y = (kp5[1] + kp6[1]) * 0.5
        mid_hp_y = (kp11[1] + kp12[1]) * 0.5
        delta_y = abs(mid_hp_y - mid_sh_y)

        angle_vote = angle < self.angle_thresh
        aspect_vote = aspect < dyn_aspect
        delta_vote = delta_y < self.delta_y_frac * h

        # Explicit lying check - very strict criteria for lying
        is_lying = (angle <= self.lie_angle_thresh) and (
            aspect <= ideal_aspect * self.lie_aspect_frac
        )

        # Bending check - moderate angle but not lying
        is_bending = (
            not is_lying and angle <= self.bending_angle_thresh and (delta_vote or aspect_vote)
        )

        votes = int(angle_vote) + int(aspect_vote) + int(delta_vote)

        if is_lying:
            pose_class = "lying"
        elif is_bending:
            pose_class = "bending"
        else:
            pose_class = "normal"

        return pose_class, angle, aspect, delta_y, votes

    def sitting_detection(
        self,
        pts_list: list[tuple[float, float] | None],
        dyn_delta_thr: float,
        small_tol: float,
        shoulder_tol: float,
    ) -> bool:
        """Sitting position detection.

        1) delta_y < dyn_delta_thr
        2) trunk is vertical (angle >= 90° - small_tol)
           or shoulder axis is horizontal (angle <= shoulder_tol)

        Args:
            pts_list: List of keypoints, where each keypoint is a tuple of (x, y) coordinates.
            dyn_delta_thr: Dynamic threshold for vertical distance between shoulder and hip.
            small_tol: Tolerance angle for vertical trunk position (degrees).
            shoulder_tol: Tolerance angle for horizontal shoulder position (degrees).

        Returns:
            True if sitting position is detected, False otherwise.
        """
        if any(pts_list[idx] is None for idx in (5, 6, 11, 12)):
            return False

        kp5 = cast(tuple[float, float], pts_list[5])
        kp6 = cast(tuple[float, float], pts_list[6])
        kp11 = cast(tuple[float, float], pts_list[11])
        kp12 = cast(tuple[float, float], pts_list[12])
        mid_sh_y = (kp5[1] + kp6[1]) * 0.5
        mid_hp_y = (kp11[1] + kp12[1]) * 0.5
        delta_y = abs(mid_hp_y - mid_sh_y)
        if delta_y > dyn_delta_thr:
            return False

        # Is trunk nearly vertical?
        body_ang = self.get_body_angle(pts_list)
        if body_ang is not None and body_ang >= (90.0 - small_tol):
            return True

        # Is shoulder axis nearly horizontal?
        sh_ang = self.get_shoulder_angle(pts_list)
        return sh_ang is not None and sh_ang <= shoulder_tol

    def process(
        self, inputs: FallDetectionPipelineModuleInputs
    ) -> FallDetectionPipelineModuleOutputs:
        """Run the fall detection pipeline on given inputs.

        Args:
            inputs: Pipeline inputs containing inference results.

        Returns:
            Pipeline outputs with results, fall status, annotated image, and keypoints.
        """
        results: Results = inputs.results

        # Work on a copy of the original image to draw annotations.
        annotated_image = results.original_image.copy()

        frame_fall_status: str = "normal"  # Default status for the entire frame.
        keypoints_list: list[dict[int, tuple[float, float]]] = []

        # Convenience references
        boxes = results.bounding_boxes.xyxy
        class_ids = results.bounding_boxes.cls
        names = results.bounding_boxes.names

        # Iterate over each detected object
        for det_idx in range(boxes.shape[0]):
            class_id: int = int(class_ids[det_idx].item())
            label_name: str = names.get(class_id, "") if isinstance(names, dict) else str(class_id)

            # Focus only on person detections
            if label_name.lower() not in {"person", "kisi"}:
                continue

            # Bounding box coordinates
            x1, y1, x2, y2 = boxes[det_idx].tolist()
            h: float = y2 - y1

            # Extract keypoints (xy) and, if present, confidence scores
            pts_xy = results.keypoints.xy[det_idx]
            # Move to CPU & numpy for easier manipulation
            pts_xy_np = pts_xy.cpu().numpy() if hasattr(pts_xy, "cpu") else np.asarray(pts_xy)

            conf_scores = None
            if results.keypoints.matrix.shape[-1] >= 3:
                conf_t = results.keypoints.matrix[det_idx, :, 2]
                conf_scores = conf_t.cpu().numpy() if hasattr(conf_t, "cpu") else np.asarray(conf_t)

            pts_list: list[tuple[float, float] | None] = []
            for kp_idx, (coord_x, coord_y) in enumerate(pts_xy_np):
                # Determine confidence for this keypoint (if available)
                conf_val = conf_scores[kp_idx] if conf_scores is not None else 1.0
                if conf_val >= self.keypoint_conf_threshold and (coord_x, coord_y) != (0.0, 0.0):
                    pts_list.append((float(coord_x), float(coord_y)))
                else:
                    pts_list.append(None)

            # Skip if critical keypoints are missing
            if any(pt is None for pt in [pts_list[5], pts_list[6], pts_list[11], pts_list[12]]):
                continue

            # Perform pose classification
            state: dict[str, float | None] = {"ideal_aspect": None}
            pose_class, angle, aspect, delta_y, votes = self.pose_classification(pts_list, h, state)

            # Perform sitting detection
            dyn_delta_thr = self.delta_y_frac * h
            sitting = self.sitting_detection(
                pts_list, dyn_delta_thr, self.small_tol, self.shoulder_tol
            )

            # Determine label & colour for visualization
            if pose_class == "lying":
                person_color, person_label = (0, 0, 255), "LYING/FALL"
                frame_fall_status = "falling"
            elif pose_class == "bending":
                person_color, person_label = (255, 140, 0), "BENDING"
                if frame_fall_status not in ["falling"]:
                    frame_fall_status = "bending"
            elif sitting:
                person_color, person_label = (255, 165, 0), "SITTING"
                if frame_fall_status not in ["falling", "bending"]:
                    frame_fall_status = "anomaly"
            else:
                person_color, person_label = (0, 255, 0), "NORMAL"

            # Draw bounding box and label
            cv2.rectangle(
                annotated_image,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                person_color,
                2,
            )
            cv2.putText(
                annotated_image,
                person_label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                person_color,
                2,
            )

            # Debug overlay text (angle, aspect, etc.)
            # debug_lines = [
            #    f"A:{angle:.1f}",
            #    f"Asp:{aspect:.2f}",
            #    f"Dy:{delta_y:.1f}",
            #    f"V:{votes}",
            #    f"Sit:{int(sitting)}",
            # ]
            # for idx_txt, txt in enumerate(debug_lines):
            #    tx, ty = int(x1 + 5), int(y2 - 5 - idx_txt * 20)
            #    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            #    cv2.rectangle(
            #        annotated_image,
            #        (tx, ty - th - 2),
            #        (tx + tw, ty + 2),
            #        (255, 255, 255),
            #        -1,
            #    )
            #    cv2.putText(
            #        annotated_image,
            #        txt,
            #        (tx, ty),
            #        cv2.FONT_HERSHEY_SIMPLEX,
            #        0.5,
            #        person_color,
            #        1,
            #    )

            # Draw keypoints and skeleton
            for kp in pts_list:
                if kp is not None:
                    cv2.circle(annotated_image, (int(kp[0]), int(kp[1])), 4, (255, 255, 0), -1)

            for a, b in KEYPOINT_CONNECTIONS:
                if (
                    a < len(pts_list)
                    and b < len(pts_list)
                    and pts_list[a] is not None
                    and pts_list[b] is not None
                ):
                    pt_a = cast(tuple[float, float], pts_list[a])
                    pt_b = cast(tuple[float, float], pts_list[b])
                    x0, y0 = pt_a
                    x1_p, y1_p = pt_b
                    cv2.line(
                        annotated_image,
                        (int(x0), int(y0)),
                        (int(x1_p), int(y1_p)),
                        (255, 255, 0),
                        2,
                    )

            # Collect keypoints into list for output (excluding None)
            kp_dict: dict[int, tuple[float, float]] = {}
            for kp_idx, pt in enumerate(pts_list):
                if pt is not None:
                    kp_dict[kp_idx] = (float(pt[0]), float(pt[1]))
            keypoints_list.append(kp_dict)

        # Make sure frame_fall_status is lowercase
        frame_fall_status = frame_fall_status.lower()

        return FallDetectionPipelineModuleOutputs(
            results=results,
            fall_status=frame_fall_status,
            annotated_image=annotated_image,
            keypoints_list=keypoints_list,
            processing_time=0.0,
        )

    def cleanup(self) -> None:
        """Cleanup module state after processing is complete."""
        LOGGER.debug("Cleaning up FallDetection module")
