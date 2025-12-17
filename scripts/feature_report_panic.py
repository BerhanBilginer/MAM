"""
Feature engineering report script for panic detection.

Extracts window-based scalar features from videos, computes baseline statistics,
and evaluates panic onset detection performance.

Usage:
    python scripts/feature_report_panic.py \
        --train "data/dataset/train/*.mp4" \
        --val "data/dataset/val/*.mp4" \
        --test "data/dataset/test/test.mp4" \
        --output-csv "runs/feature_report.csv" \
        --output-summary "runs/feature_summary.json"
"""

import argparse
import json
import csv
import sys
from pathlib import Path
from glob import glob
from typing import Optional
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.utils.pose import load_model, detect_people
from src.detection.panic.features import (
    to_gray_small,
    compute_flow,
    roi_mask_from_dets,
    create_bbox_heatmap,
    direction_entropy,
    direction_coherence,
)


def median_and_mad(x: list[float]) -> tuple[float, float]:
    """Compute median and MAD for robust statistics."""
    if not x:
        return 0.0, 0.0
    arr = np.asarray(x, dtype=np.float32)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return med, mad


def robust_z(x: float, med: float, mad: float, eps: float = 1e-6) -> float:
    """Compute robust z-score using median and MAD."""
    return (x - med) / (1.4826 * mad + eps)


def extract_frame_features(
    frame_bgr: np.ndarray,
    prev_gray_small: Optional[np.ndarray],
    detections: list,
    image_size: int = 96,
) -> tuple[Optional[dict], np.ndarray]:
    """
    Extract scalar features from a single frame.
    
    Returns:
        (features_dict, gray_small) where features_dict is None for first frame
    """
    h_orig, w_orig = frame_bgr.shape[:2]
    size = (image_size, image_size)
    
    gray_small = to_gray_small(frame_bgr, size)
    
    if prev_gray_small is None:
        return None, gray_small
    
    # Compute flow
    flow_x, flow_y, flow_mag, flow_ang = compute_flow(prev_gray_small, gray_small, remove_global_motion=True)
    
    # Global features
    v_p95_global = float(np.percentile(flow_mag, 95))
    v_p99_global = float(np.percentile(flow_mag, 99))
    entropy_global = direction_entropy(flow_ang, flow_mag, bins=16, mag_eps=0.05, weighted=False)
    coh_global = direction_coherence(flow_ang, flow_mag, mag_eps=0.05)
    
    # ROI features
    roi_mask = roi_mask_from_dets(detections, size, (h_orig, w_orig), pad=0.10)
    
    if roi_mask is not None and roi_mask.any():
        mag_roi = flow_mag[roi_mask]
        ang_roi = flow_ang[roi_mask]
        v_p95_roi = float(np.percentile(mag_roi, 95))
        v_p99_roi = float(np.percentile(mag_roi, 99))
        entropy_roi = direction_entropy(ang_roi, mag_roi, bins=16, mag_eps=0.05, weighted=False)
        coh_roi = direction_coherence(ang_roi, mag_roi, mag_eps=0.05)
    else:
        v_p95_roi = 0.0
        v_p99_roi = 0.0
        entropy_roi = 0.0
        coh_roi = 0.0
    
    # People count and bbox area
    people_count = len(detections)
    
    total_bbox_area = 0.0
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        total_bbox_area += (x2 - x1) * (y2 - y1)
    
    frame_area = h_orig * w_orig
    bbox_area_ratio = total_bbox_area / frame_area if frame_area > 0 else 0.0
    
    features = {
        'v_p95_global': v_p95_global,
        'v_p99_global': v_p99_global,
        'entropy_global': entropy_global,
        'coh_global': coh_global,
        'v_p95_roi': v_p95_roi,
        'v_p99_roi': v_p99_roi,
        'entropy_roi': entropy_roi,
        'coh_roi': coh_roi,
        'people_count': float(people_count),
        'bbox_area_ratio': bbox_area_ratio,
    }
    
    return features, gray_small


def process_video(
    video_path: str,
    model,
    split: str,
    image_size: int = 96,
    window_frames: int = 25,
    stride_frames: int = 12,
    fps: float = 25.0,
) -> list[dict]:
    """
    Process a single video and extract windowed features.
    
    Returns:
        List of window feature dictionaries
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARNING] Cannot open video: {video_path}")
        return []
    
    windows = []
    frame_buffer = []
    prev_gray_small = None
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect people
        detections = detect_people(model, frame, conf=0.3, keypoint_conf_threshold=0.3)
        
        # Extract frame features
        frame_features, prev_gray_small = extract_frame_features(
            frame, prev_gray_small, detections, image_size
        )
        
        if frame_features is not None:
            frame_buffer.append(frame_features)
        
        frame_idx += 1
        
        # Check if window is ready
        if len(frame_buffer) >= window_frames:
            # Compute window aggregates
            window_features = {}
            
            # Average each feature over the window
            for key in frame_buffer[0].keys():
                values = [f[key] for f in frame_buffer]
                window_features[key] = float(np.mean(values))
            
            # Temporal derivatives (if we have previous window)
            if windows:
                prev_window = windows[-1]
                window_features['dv_p95'] = window_features['v_p95_global'] - prev_window['v_p95_global']
                window_features['dentropy'] = window_features['entropy_global'] - prev_window['entropy_global']
                window_features['dcoh'] = window_features['coh_global'] - prev_window['coh_global']
            else:
                window_features['dv_p95'] = 0.0
                window_features['dentropy'] = 0.0
                window_features['dcoh'] = 0.0
            
            # Metadata
            t_start_sec = (frame_idx - window_frames) / fps
            t_end_sec = frame_idx / fps
            
            window_features['split'] = split
            window_features['video_path'] = video_path
            window_features['window_idx'] = len(windows)
            window_features['t_start_sec'] = t_start_sec
            window_features['t_end_sec'] = t_end_sec
            
            windows.append(window_features)
            
            # Slide window
            frame_buffer = frame_buffer[stride_frames:]
    
    cap.release()
    return windows


def main():
    parser = argparse.ArgumentParser(description="Feature engineering report for panic detection")
    parser.add_argument('--train', nargs='+', default=[], help='Train video paths or globs')
    parser.add_argument('--val', nargs='+', default=[], help='Val video paths or globs')
    parser.add_argument('--test', nargs='+', default=[], help='Test video paths or globs')
    parser.add_argument('--model', default='yolo11x-pose.pt', help='YOLO pose model')
    parser.add_argument('--image-size', type=int, default=96, help='Feature grid size')
    parser.add_argument('--fps', type=float, default=25.0, help='Video FPS')
    parser.add_argument('--window', type=int, default=25, help='Window size in frames')
    parser.add_argument('--stride', type=int, default=12, help='Window stride in frames')
    parser.add_argument('--onset-sec', type=float, default=10.0, help='Panic onset time for test video')
    parser.add_argument('--output-csv', default='runs/feature_report.csv', help='Output CSV path')
    parser.add_argument('--output-summary', default='runs/feature_summary.json', help='Output summary JSON')
    parser.add_argument('--device', default=None, help='Device for YOLO model')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    
    # Load YOLO model
    print(f"Loading YOLO model: {args.model}")
    model = load_model(Path(args.model), device=args.device)
    
    # Expand globs
    train_videos = []
    for pattern in args.train:
        train_videos.extend(glob(pattern))
    
    val_videos = []
    for pattern in args.val:
        val_videos.extend(glob(pattern))
    
    test_videos = []
    for pattern in args.test:
        test_videos.extend(glob(pattern))
    
    print(f"Train videos: {len(train_videos)}")
    print(f"Val videos: {len(val_videos)}")
    print(f"Test videos: {len(test_videos)}")
    
    # Process all videos
    all_windows = []
    
    for video_path in train_videos:
        print(f"Processing train: {video_path}")
        windows = process_video(
            video_path, model, 'train',
            image_size=args.image_size,
            window_frames=args.window,
            stride_frames=args.stride,
            fps=args.fps,
        )
        all_windows.extend(windows)
    
    for video_path in val_videos:
        print(f"Processing val: {video_path}")
        windows = process_video(
            video_path, model, 'val',
            image_size=args.image_size,
            window_frames=args.window,
            stride_frames=args.stride,
            fps=args.fps,
        )
        all_windows.extend(windows)
    
    for video_path in test_videos:
        print(f"Processing test: {video_path}")
        windows = process_video(
            video_path, model, 'test',
            image_size=args.image_size,
            window_frames=args.window,
            stride_frames=args.stride,
            fps=args.fps,
        )
        all_windows.extend(windows)
    
    print(f"Total windows: {len(all_windows)}")
    
    # Write CSV
    if all_windows:
        fieldnames = list(all_windows[0].keys())
        with open(args.output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_windows)
        print(f"Wrote CSV: {args.output_csv}")
    
    # Compute baseline statistics from train+val
    normal_windows = [w for w in all_windows if w['split'] in ['train', 'val']]
    test_windows = [w for w in all_windows if w['split'] == 'test']
    
    feature_keys = [
        'v_p95_global', 'v_p99_global', 'entropy_global', 'coh_global',
        'v_p95_roi', 'v_p99_roi', 'entropy_roi', 'coh_roi',
        'people_count', 'bbox_area_ratio',
        'dv_p95', 'dentropy', 'dcoh',
    ]
    
    baseline_stats = {}
    for key in feature_keys:
        values = [w[key] for w in normal_windows if key in w]
        if values:
            med, mad = median_and_mad(values)
            baseline_stats[key] = {'median': med, 'mad': mad}
        else:
            baseline_stats[key] = {'median': 0.0, 'mad': 0.0}
    
    # Compute z-scores for all windows
    for w in all_windows:
        for key in feature_keys:
            if key in w and key in baseline_stats:
                med = baseline_stats[key]['median']
                mad = baseline_stats[key]['mad']
                w[f'{key}_z'] = robust_z(w[key], med, mad)
    
    # Evaluate panic detection on test video
    feature_summary = []
    
    for key in feature_keys:
        if key not in baseline_stats:
            continue
        
        # Normal rate (train+val)
        normal_z_values = [w[f'{key}_z'] for w in normal_windows if f'{key}_z' in w]
        normal_rate = sum(1 for z in normal_z_values if z > 3) / len(normal_z_values) if normal_z_values else 0.0
        
        # Test pre-onset and post-onset
        pre_onset_windows = [w for w in test_windows if w['t_end_sec'] < args.onset_sec]
        post_onset_windows = [w for w in test_windows if w['t_start_sec'] >= args.onset_sec]
        
        pre_z_values = [w[f'{key}_z'] for w in pre_onset_windows if f'{key}_z' in w]
        post_z_values = [w[f'{key}_z'] for w in post_onset_windows if f'{key}_z' in w]
        
        pre_rate = sum(1 for z in pre_z_values if z > 3) / len(pre_z_values) if pre_z_values else 0.0
        panic_rate = sum(1 for z in post_z_values if z > 3) / len(post_z_values) if post_z_values else 0.0
        
        separation = panic_rate - normal_rate
        
        # Latency: first window after onset with z > 3
        latency = None
        for w in sorted(post_onset_windows, key=lambda x: x['t_start_sec']):
            if f'{key}_z' in w and w[f'{key}_z'] > 3:
                latency = w['t_start_sec'] - args.onset_sec
                break
        
        feature_summary.append({
            'feature': key,
            'normal_rate': normal_rate,
            'pre_rate': pre_rate,
            'panic_rate': panic_rate,
            'separation': separation,
            'latency_sec': latency,
            'baseline_median': baseline_stats[key]['median'],
            'baseline_mad': baseline_stats[key]['mad'],
        })
    
    # Sort by separation (descending)
    feature_summary.sort(key=lambda x: x['separation'], reverse=True)
    
    # Write summary JSON
    summary_output = {
        'baseline_stats': baseline_stats,
        'feature_ranking': feature_summary,
        'config': {
            'image_size': args.image_size,
            'window_frames': args.window,
            'stride_frames': args.stride,
            'fps': args.fps,
            'onset_sec': args.onset_sec,
        }
    }
    
    with open(args.output_summary, 'w') as f:
        json.dump(summary_output, f, indent=2)
    
    print(f"Wrote summary: {args.output_summary}")
    
    # Print top features
    print("\n=== Top Features by Panic Separation ===")
    for item in feature_summary[:10]:
        print(f"{item['feature']:20s} | sep={item['separation']:6.3f} | panic={item['panic_rate']:5.3f} | normal={item['normal_rate']:5.3f} | latency={item['latency_sec']}")


if __name__ == '__main__':
    main()
