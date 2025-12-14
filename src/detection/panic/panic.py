from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
from numpy import typing as npt

from src.detection.utils.detection import Detection


# ----------------------------
# Robust stats helpers
# ----------------------------
def median_and_mad(x: list[float]) -> tuple[float, float]:
    if not x:
        return 0.0, 0.0
    arr = np.asarray(x, dtype=np.float32)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return med, mad

def robust_z(x: float, med: float, mad: float, eps: float = 1e-6) -> float:
    # 1.4826 * MAD ~ std for normal dist
    return (x - med) / (1.4826 * mad + eps)


# ----------------------------
# Config / Result
# ----------------------------
@dataclass
class FusionPanicConfig:
    # Flow params
    flow_resize_width: int = 320           # downscale for CPU speed
    flow_win_seconds: float = 1.0          # metrics computed per ~1 second window
    flow_use_farneback: bool = True

    # Baseline / decision
    warmup_seconds: float = 15.0           # only learn baseline stats (disabled if embeddings used)
    min_people: int = 3
    debounce_windows: int = 2              # consecutive windows above threshold

    # Fusion weights (score = sum(w_i * z_i))
    w_v_p95: float = 0.60                  # flow speed high-tail
    w_h_dir: float = 0.40                  # direction entropy

    score_threshold: float = 4.0           # tune
    use_person_roi_flow: bool = True       # ROI flow features on people bboxes

    # Embedding-based anomaly detection
    use_embeddings: bool = False           # enable embedding-based anomaly detection
    embedding_db_path: str = "data/embeddings/normal_embeddings.json"
    embedding_window_size: int = 30        # frames per embedding
    embedding_weight: float = 0.5          # weight for anomaly score in fusion
    anomaly_threshold: float = 1.0         # anomaly score threshold (higher = more strict)

    # Visualization
    overlay_alpha: float = 0.35


@dataclass
class FusionPanicResult:
    is_panic: bool
    score: float
    metrics: dict[str, float]


# ----------------------------
# Main Detector
# ----------------------------
class YoloPoseFlowFusionPanic:
    """
    Training-free panic detector:
      - optical flow features (global + optional person-ROI)
      - pose/tracking-based speed/accel + run-like ratio
      - robust baseline (median/MAD) from warmup
      - online robust z-score fusion + debounce
    """

    def __init__(self, fps: float, config: Optional[FusionPanicConfig] = None) -> None:
        self.cfg = config or FusionPanicConfig()
        self.fps = float(fps)

        self._prev_gray_small: Optional[npt.NDArray[np.uint8]] = None
        self._prev_ts: Optional[float] = None

        # windowing
        self._win_frames_target = max(1, int(round(self.cfg.flow_win_seconds * self.fps)))
        self._win_buf = []  # list of per-frame metrics dicts

        # baseline collection (skip warmup if using embeddings)
        self._warmup_frames = 0 if self.cfg.use_embeddings else int(round(self.cfg.warmup_seconds * self.fps))
        self._seen_frames = 0
        self._baseline_values: dict[str, list[float]] = {
            "v_p95": [],
            "h_dir": [],
        }
        self._baseline_stats: dict[str, tuple[float, float]] = {}  # key -> (median, mad)
        self._baseline_ready = self.cfg.use_embeddings  # ready immediately if using embeddings

        # pose tracking state
        self._prev_centers: dict[int, tuple[float, float]] = {}
        self._prev_speeds: dict[int, float] = {}

        # debounce
        self._panic_streak = 0
        
        # store last flow for visualization
        self._last_flow_mag: Optional[np.ndarray] = None
        
        # embedding-based detection
        self._embedding_db = None
        self._embedding_extractor = None
        if self.cfg.use_embeddings:
            self._init_embeddings()

    def _init_embeddings(self) -> None:
        """Initialize embedding database and extractor."""
        try:
            from src.detection.panic.embeddings import EmbeddingDatabase, EmbeddingExtractor
            
            db_path = Path(self.cfg.embedding_db_path)
            if not db_path.exists():
                print(f"[WARNING] Embedding database not found: {db_path}")
                print(f"[WARNING] Embedding-based detection disabled")
                self.cfg.use_embeddings = False
                return
            
            self._embedding_db = EmbeddingDatabase(db_path)
            self._embedding_db.load()
            self._embedding_extractor = EmbeddingExtractor(window_size=self.cfg.embedding_window_size)
            print(f"[INFO] Loaded embedding database with {len(self._embedding_db.embeddings)} embeddings")
            
            # Initialize baseline stats with dummy values when using embeddings
            # (z-scores won't be used much, anomaly score is primary)
            self._baseline_stats = {
                "v_p95": (1.0, 0.5),  # (median, mad)
                "h_dir": (1.0, 0.5),
            }
        except Exception as e:
            print(f"[ERROR] Failed to initialize embeddings: {e}")
            self.cfg.use_embeddings = False

    # ---- Flow feature extraction
    def _to_gray_small(self, frame_bgr: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        h, w = frame_bgr.shape[:2]
        if w != self.cfg.flow_resize_width:
            scale = self.cfg.flow_resize_width / float(w)
            new_h = max(1, int(round(h * scale)))
            frame_bgr = cv2.resize(frame_bgr, (self.cfg.flow_resize_width, new_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return gray

    def _compute_flow(self, prev: npt.NDArray[np.uint8], curr: npt.NDArray[np.uint8]) -> tuple[np.ndarray, np.ndarray]:
        # Farneback dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev, curr, None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0
        )
        fx, fy = flow[..., 0], flow[..., 1]
        mag = np.hypot(fx, fy).astype(np.float32)
        ang = (np.arctan2(fy, fx) + np.pi).astype(np.float32)  # 0..2pi
        return mag, ang

    def _direction_entropy(self, angles: np.ndarray, bins: int = 16) -> float:
        # histogram on [0, 2pi)
        hist, _ = np.histogram(angles, bins=bins, range=(0.0, 2.0 * np.pi))
        p = hist.astype(np.float32)
        s = float(p.sum())
        if s <= 0:
            return 0.0
        p /= s
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))

    def _roi_mask_from_dets_small(
        self,
        dets: list[Detection],
        small_shape: tuple[int, int],
        orig_shape: tuple[int, int],
        pad: float = 0.10,
    ) -> Optional[np.ndarray]:
        if not dets:
            return None
        sh, sw = small_shape
        oh, ow = orig_shape
        sx = sw / float(ow)
        sy = sh / float(oh)

        mask = np.zeros((sh, sw), dtype=np.uint8)
        for det in dets:
            x1, y1, x2, y2 = det.bbox
            # pad bbox
            bw = (x2 - x1)
            bh = (y2 - y1)
            x1p = x1 - pad * bw
            y1p = y1 - pad * bh
            x2p = x2 + pad * bw
            y2p = y2 + pad * bh

            rx1 = int(np.clip(x1p * sx, 0, sw - 1))
            ry1 = int(np.clip(y1p * sy, 0, sh - 1))
            rx2 = int(np.clip(x2p * sx, 0, sw - 1))
            ry2 = int(np.clip(y2p * sy, 0, sh - 1))
            if rx2 <= rx1 or ry2 <= ry1:
                continue
            mask[ry1:ry2, rx1:rx2] = 255
        if mask.sum() == 0:
            return None
        return mask

    # ---- Pose features
    @staticmethod
    def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) * 0.5, (y1 + y2) * 0.5

    def _tracking_metrics(self, dets: list[Detection]) -> dict[str, float]:
        """Track person movement for additional context (optional, not used in score)."""
        if not dets:
            self._prev_centers.clear()
            self._prev_speeds.clear()
            return {"avg_speed": 0.0}

        speeds = []

        for i, det in enumerate(dets):
            tid = det.track_id if det.track_id is not None else i
            tid = int(tid)

            cx, cy = self._bbox_center(det.bbox)
            prev = self._prev_centers.get(tid)

            if prev is None:
                speed = 0.0
            else:
                dx = cx - prev[0]
                dy = cy - prev[1]
                speed = float(np.hypot(dx, dy) * self.fps)  # px/sec

            self._prev_centers[tid] = (cx, cy)
            self._prev_speeds[tid] = speed
            speeds.append(speed)

        # prune disappeared ids
        valid_ids = {det.track_id if det.track_id is not None else idx for idx, det in enumerate(dets)}
        self._prev_centers = {k: v for k, v in self._prev_centers.items() if k in valid_ids}
        self._prev_speeds = {k: v for k, v in self._prev_speeds.items() if k in valid_ids}

        avg_speed = float(np.mean(speeds)) if speeds else 0.0
        return {"avg_speed": avg_speed}

    # ---- Main update
    def update(
        self,
        frame_bgr: npt.NDArray[np.uint8],
        detections: Iterable[Detection],
    ) -> FusionPanicResult:
        dets = list(detections)
        self._seen_frames += 1

        gray_small = self._to_gray_small(frame_bgr)

        # If first frame, cannot compute flow yet
        if self._prev_gray_small is None:
            self._prev_gray_small = gray_small
            track_m = self._tracking_metrics(dets)
            metrics = {"v_p95": 0.0, "h_dir": 0.0, **track_m}
            return FusionPanicResult(False, 0.0, metrics)

        mag, ang = self._compute_flow(self._prev_gray_small, gray_small)
        self._last_flow_mag = mag  # Store for visualization
        self._prev_gray_small = gray_small

        # global flow features
        v_p95_global = float(np.percentile(mag, 95))
        h_dir_global = self._direction_entropy(ang, bins=16)

        # optional: person ROI flow (more sensitive to crowd motion rather than background)
        if self.cfg.use_person_roi_flow and dets:
            mask = self._roi_mask_from_dets_small(dets, mag.shape, frame_bgr.shape[:2])
            if mask is not None:
                mm = mag[mask > 0]
                aa = ang[mask > 0]
                if mm.size > 50:
                    v_p95 = float(np.percentile(mm, 95))
                else:
                    v_p95 = v_p95_global
                if aa.size > 50:
                    h_dir = self._direction_entropy(aa, bins=16)
                else:
                    h_dir = h_dir_global
            else:
                v_p95, h_dir = v_p95_global, h_dir_global
        else:
            v_p95, h_dir = v_p95_global, h_dir_global

        track_m = self._tracking_metrics(dets)

        # accumulate per-frame metrics into window buffer
        self._win_buf.append(
            {"v_p95": v_p95, "h_dir": h_dir, "people": len(dets)}
        )

        # If window not complete yet, return current instantaneous (not decision)
        if len(self._win_buf) < self._win_frames_target:
            metrics = {"v_p95": v_p95, "h_dir": h_dir, **track_m}
            return FusionPanicResult(False, 0.0, metrics)

        # window metrics (aggregate)
        win = self._win_buf
        self._win_buf = []  # reset window

        v_p95_w = float(np.mean([m["v_p95"] for m in win]))
        h_dir_w = float(np.mean([m["h_dir"] for m in win]))
        people_w = int(np.round(np.mean([m["people"] for m in win])))

        metrics_w = {
            "v_p95": v_p95_w,
            "h_dir": h_dir_w,
            "people": float(people_w),
        }

        # Warmup: collect baseline
        if (not self._baseline_ready) and (self._seen_frames <= self._warmup_frames):
            for k in ("v_p95", "h_dir"):
                self._baseline_values[k].append(metrics_w[k])
            return FusionPanicResult(False, 0.0, metrics_w)

        # Finish baseline once
        if not self._baseline_ready:
            for k in ("v_p95", "h_dir"):
                med, mad = median_and_mad(self._baseline_values[k])
                # if MAD ~0 (very stable), give tiny mad to avoid infinite z
                if mad < 1e-6:
                    mad = max(1e-3, 0.05 * abs(med) + 1e-3)
                self._baseline_stats[k] = (med, mad)
            self._baseline_ready = True

        # If not enough people, force normal
        if people_w < self.cfg.min_people:
            self._panic_streak = 0
            return FusionPanicResult(False, 0.0, metrics_w)

        # Robust z-scores
        z_v = robust_z(v_p95_w, *self._baseline_stats["v_p95"])
        z_h = robust_z(h_dir_w, *self._baseline_stats["h_dir"])

        # Base score from z-scores
        base_score = (
            self.cfg.w_v_p95 * z_v
            + self.cfg.w_h_dir * z_h
        )

        # Anomaly-based score (if enabled)
        anomaly_score = 0.0
        if self.cfg.use_embeddings and self._embedding_extractor is not None:
            # Add frame features to embedding extractor
            roi_flow = v_p95_w  # use same as global for now
            embedding = self._embedding_extractor.add_frame_features(
                flow_mag=v_p95_w,
                direction_entropy=h_dir_w,
                people_count=people_w,
                roi_flow=roi_flow,
            )
            
            if embedding is not None and self._embedding_db is not None:
                # Compute anomaly score (deviation from normal patterns)
                anomaly_metrics = self._embedding_db.compute_anomaly_score(embedding)
                anomaly_score = anomaly_metrics.get("anomaly_score", 0.0)
                
                # Add to metrics
                metrics_w.update({
                    "anomaly_score": anomaly_score,
                    "reconstruction_error": anomaly_metrics.get("reconstruction_error", 0.0),
                    "mahalanobis_distance": anomaly_metrics.get("mahalanobis_distance", 0.0),
                    "nn_distance": anomaly_metrics.get("nearest_neighbor_distance", 0.0),
                    "max_z_score": anomaly_metrics.get("max_z_score", 0.0),
                })

        # Fuse base score with anomaly score
        # Higher anomaly score = more deviation from normal = more panic-like
        if self.cfg.use_embeddings:
            score = (1 - self.cfg.embedding_weight) * base_score + self.cfg.embedding_weight * (anomaly_score * 5)
        else:
            score = base_score

        # Debounce
        if score > self.cfg.score_threshold:
            self._panic_streak += 1
        else:
            self._panic_streak = 0

        is_panic = self._panic_streak >= self.cfg.debounce_windows

        metrics_w.update({"z_v": float(z_v), "z_h": float(z_h), "base_score": float(base_score)})
        return FusionPanicResult(is_panic=is_panic, score=float(score), metrics=metrics_w)

    # ---- Visualization helper
    def draw_overlay(
        self,
        frame_bgr: npt.NDArray[np.uint8],
        result: FusionPanicResult,
    ) -> npt.NDArray[np.uint8]:
        out = frame_bgr.copy()

        # status box (larger if anomaly metrics available)
        box_height = 165 if "anomaly_score" in result.metrics else 140
        cv2.rectangle(out, (8, 8), (520, box_height), (0, 0, 0), thickness=-1)

        status = "PANIC!" if result.is_panic else "Normal"
        color = (0, 0, 255) if result.is_panic else (0, 255, 0)

        cv2.putText(out, f"Status: {status}", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(out, f"Score: {result.score:.2f} (thr={self.cfg.score_threshold:.2f})", (16, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        m = result.metrics
        cv2.putText(out, f"v_p95: {m.get('v_p95',0):.2f}  H_dir: {m.get('h_dir',0):.2f}  people: {int(m.get('people',0))}",
                    (16, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # z-scores if available
        if "z_v" in m:
            cv2.putText(out, f"z_v: {m['z_v']:.2f}  z_h: {m['z_h']:.2f}",
                        (16, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        
        # Anomaly score if available
        if "anomaly_score" in m:
            cv2.putText(out, f"Anomaly: {m['anomaly_score']:.3f}  Recon: {m.get('reconstruction_error',0):.2f}",
                        (16, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 200, 255), 1)

        return out


if __name__ == "__main__":
    import argparse
    from src.detection.utils.detection import load_model, detect_people

    parser = argparse.ArgumentParser(description="YoloPoseFlowFusionPanic test runner")
    parser.add_argument("--model", required=True, help="Path to YOLO pose model")
    parser.add_argument("--source", required=True, help="Path to video file")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--device", default="cpu", help="Device (cpu, cuda:0, etc.)")
    parser.add_argument("--track", action="store_true", help="Enable tracking for better motion analysis")
    parser.add_argument("--output", default=None, help="Optional output video path")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = load_model(args.model, device=args.device)

    print(f"Opening video: {args.source}")
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
        print(f"Warning: Could not detect FPS, using default {fps}")
    else:
        print(f"Video FPS: {fps}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video resolution: {width}x{height}, Total frames: {total_frames}")

    # Check if embeddings exist
    import os
    embedding_path = "data/embeddings/normal_motion_embeddings.json"
    use_embeddings = os.path.exists(embedding_path)
    
    if use_embeddings:
        print(f"\n✓ Found embeddings: {embedding_path}")
        print("  Enabling anomaly detection mode")
    else:
        print(f"\n✗ No embeddings found at: {embedding_path}")
        print("  Using base detection only")
    
    config = FusionPanicConfig(
        use_embeddings=use_embeddings,
        embedding_db_path=embedding_path,
        embedding_window_size=30,
        embedding_weight=0.5,
    )
    detector = YoloPoseFlowFusionPanic(fps=fps, config=config)

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"Output will be saved to: {args.output}")

    window_name = "YoloPoseFlowFusionPanic - Press 'q' to quit"
    frame_idx = 0

    print("\nProcessing video...")
    if use_embeddings:
        print("Warmup: DISABLED (using embeddings for baseline)")
    else:
        print(f"Warmup period: {config.warmup_seconds}s ({int(config.warmup_seconds * fps)} frames)")
    print(f"Score threshold: {config.score_threshold}")
    print(f"Min people: {config.min_people}")
    print("-" * 60)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            detections = detect_people(
                model,
                frame,
                conf=args.conf,
                track=args.track,
            )

            result = detector.update(frame, detections)
            annotated = detector.draw_overlay(frame, result)

            if result.is_panic:
                print(f"[FRAME {frame_idx}] PANIC DETECTED! Score: {result.score:.2f}")

            cv2.imshow(window_name, annotated)

            if writer:
                writer.write(annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nUser interrupted.")
                break

            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames...")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"\nProcessing complete. Total frames: {frame_idx}")
