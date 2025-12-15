from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import cv2

from src.detection.utils.pose import (
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_KP_CONF_THRESHOLD,
    PoseDetection,
    detect_people,
    load_model,
)

if TYPE_CHECKING:
    from numpy import typing as npt
    import numpy as np


def parse_source(src: str) -> str | int:
    """Cast the source argument to int when possible for webcam indices."""
    try:
        return int(src)
    except ValueError:
        return src


def open_capture(src: str | int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video source: {src}")
    return cap


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MAM: Multi-module detection system.")

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--fall",
        action="store_true",
        help="Run fall detection module.",
    )
    mode_group.add_argument(
        "--panic",
        action="store_true",
        help="Run panic detection module.",
    )
    mode_group.add_argument(
        "--navigation",
        action="store_true",
        help="Run navigation module.",
    )

    parser.add_argument(
        "--source",
        default="0",
        help="Video source: camera index or path to video file (default: 0).",
    )
    parser.add_argument(
        "--model",
        default="yolov8n-pose.pt",
        help="YOLO pose model path or name (default: yolov8n-pose.pt).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run inference on, e.g., 'cpu', 'cuda:0'.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF_THRESHOLD,
        help=f"Detection confidence threshold (default: {DEFAULT_CONF_THRESHOLD}).",
    )
    parser.add_argument(
        "--kp-conf",
        type=float,
        default=DEFAULT_KP_CONF_THRESHOLD,
        help=f"Keypoint confidence threshold (default: {DEFAULT_KP_CONF_THRESHOLD}).",
    )

    parser.add_argument(
        "--panic-convlstm-model",
        default=None,
        help="Optional ConvLSTM panic model checkpoint path (.pt). If provided, panic decision uses ConvLSTM only.",
    )
    parser.add_argument(
        "--panic-convlstm-device",
        default=None,
        help="Device for ConvLSTM panic model (defaults to --device).",
    )
    parser.add_argument(
        "--panic-convlstm-seq-len",
        type=int,
        default=16,
        help="Sequence length for ConvLSTM panic model (must match training).",
    )
    parser.add_argument(
        "--panic-convlstm-image-size",
        type=int,
        default=96,
        help="Image size for ConvLSTM features (must match training).",
    )
    parser.add_argument(
        "--panic-convlstm-threshold",
        type=float,
        default=None,
        help="Optional override for ConvLSTM anomaly threshold (defaults to checkpoint threshold).",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional output video path (panic mode only).",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable GUI windows (useful for Colab/headless).",
    )
    return parser


def run_fall_detection(
    cap: cv2.VideoCapture,
    model,
    args: argparse.Namespace,
) -> None:
    """Run fall detection loop."""
    from src.detection.fall.fall import FallDetector, annotate_fall_results

    fall_detector = FallDetector()
    window_name = "MAM - Fall Detection"
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_people(
            model,
            frame,
            conf=args.conf,
            keypoint_conf_threshold=args.kp_conf,
        )

        fall_result = fall_detector.process(detections)
        annotate_fall_results(frame, fall_result)

        if fall_result.frame_status != "normal":
            print(f"[ALERT] Frame status: {fall_result.frame_status.upper()}")

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def run_panic_detection(
    cap: cv2.VideoCapture,
    model,
    args: argparse.Namespace,
) -> None:
    """Run panic detection with two windows: video with bboxes and heatmap."""
    from src.detection.panic.wrapper import PanicDetector, draw_video_frame

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
        print(f"[WARNING] Could not detect FPS, using default {fps}")

    convlstm_device = args.panic_convlstm_device if args.panic_convlstm_device is not None else args.device
    panic_detector = PanicDetector(
        frame_width,
        frame_height,
        fps=fps,
        convlstm_model_path=args.panic_convlstm_model,
        convlstm_device=convlstm_device,
        convlstm_threshold=args.panic_convlstm_threshold,
        convlstm_sequence_length=args.panic_convlstm_seq_len,
        convlstm_image_size=args.panic_convlstm_image_size,
    )
    video_window = "MAM - Panic Detection (Video)"
    heatmap_window = "MAM - Panic Detection (Heatmap)"
    if not args.no_display:
        print("Press 'q' to quit.")
    if args.panic_convlstm_model:
        print(
            "[INFO] ConvLSTM mode enabled: first decision available after "
            f"{int(args.panic_convlstm_seq_len)} frames (sequence buffer)."
        )
    else:
        print(f"[INFO] Warmup period: {panic_detector.config.warmup_seconds}s")

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))
        print(f"[INFO] Writing output video to: {args.output}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_people(
            model,
            frame,
            conf=args.conf,
            keypoint_conf_threshold=args.kp_conf,
        )

        panic_result = panic_detector.update(frame, detections)

        video_frame = draw_video_frame(frame, detections, panic_result)
        heatmap_frame = panic_detector.get_heatmap_image()

        if panic_result.is_panic:
            print(f"[ALERT] PANIC detected! Score: {panic_result.score:.2f}")

        if writer is not None:
            writer.write(video_frame)

        if not args.no_display:
            cv2.imshow(video_window, video_frame)
            cv2.imshow(heatmap_window, heatmap_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if writer is not None:
        writer.release()


def run_navigation(
    cap: cv2.VideoCapture,
    model,
    args: argparse.Namespace,
) -> None:
    """Run navigation module loop."""
    window_name = "MAM - Navigation"
    print("[INFO] Navigation module is not implemented yet.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_people(
            model,
            frame,
            conf=args.conf,
            keypoint_conf_threshold=args.kp_conf,
        )

        cv2.putText(
            frame,
            "Navigation (Not Implemented)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 165, 255),
            2,
        )

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cap = open_capture(parse_source(args.source))
    model = load_model(Path(args.model), device=args.device)

    try:
        if args.fall:
            run_fall_detection(cap, model, args)
        elif args.panic:
            run_panic_detection(cap, model, args)
        elif args.navigation:
            run_navigation(cap, model, args)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()