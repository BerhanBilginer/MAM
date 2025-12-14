#!/usr/bin/env python3
"""
Extract normal motion embeddings for panic detection (anomaly detection approach).

This script extracts embeddings from NORMAL motion videos only.
Panic is detected as deviation from these normal patterns at runtime.

Usage:
    python scripts/extract_panic_embeddings.py \
        --videos data/normal_videos/*.mp4 \
        --output data/embeddings/normal_embeddings.json
    
    python scripts/extract_panic_embeddings.py \
        --videos data/more_normal_videos/*.mp4 \
        --output data/embeddings/normal_embeddings.json \
        --append
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.panic.panic import YoloPoseFlowFusionPanic, FusionPanicConfig
from src.detection.panic.embeddings import EmbeddingExtractor, EmbeddingDatabase
from src.detection.utils.detection import load_model, detect_people


def extract_embeddings_from_video(
    video_path: str,
    model,
    extractor: EmbeddingExtractor,
    config: FusionPanicConfig,
) -> list:
    """Extract normal motion embeddings from a single video."""
    print(f"\nProcessing: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open video")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  FPS: {fps}, Total frames: {total_frames}")
    
    # Create panic detector for feature extraction
    panic_detector = YoloPoseFlowFusionPanic(fps=fps, config=config)
    
    embeddings = []
    frame_idx = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # Detect people
            detections = detect_people(model, frame, conf=0.25, track=False)
            
            # Update panic detector to get features
            result = panic_detector.update(frame, detections)
            
            # Extract features for embedding
            metrics = result.metrics
            flow_mag = metrics.get("v_p95", 0.0)
            direction_entropy = metrics.get("h_dir", 0.0)
            people_count = int(metrics.get("people", 0))
            
            # For ROI flow, use the same as global for now
            # (could be enhanced to track separately)
            roi_flow = flow_mag
            
            # Add to extractor
            embedding = extractor.add_frame_features(
                flow_mag=flow_mag,
                direction_entropy=direction_entropy,
                people_count=people_count,
                roi_flow=roi_flow,
            )
            
            if embedding is not None:
                # Always label as "normal" for anomaly detection
                embedding.label = "normal"
                embeddings.append(embedding)
            
            if frame_idx % 100 == 0:
                print(f"  Processed {frame_idx}/{total_frames} frames, {len(embeddings)} embeddings")
    
    finally:
        cap.release()
    
    # Finalize remaining frames
    final_embedding = extractor.finalize(label="normal")
    if final_embedding is not None:
        embeddings.append(final_embedding)
    
    print(f"  Extracted {len(embeddings)} normal embeddings from {frame_idx} frames")
    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Extract panic detection embeddings from videos")
    parser.add_argument(
        "--videos",
        nargs="+",
        required=True,
        help="Video file paths (supports glob patterns)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file path for embeddings",
    )
    parser.add_argument(
        "--model",
        default="yolov8n-pose.pt",
        help="YOLO pose model path",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=30,
        help="Number of frames per embedding window",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing database instead of overwriting",
    )
    
    args = parser.parse_args()
    
    # Expand glob patterns
    video_paths = []
    for pattern in args.videos:
        matches = list(Path().glob(pattern))
        if matches:
            video_paths.extend([str(p) for p in matches if p.is_file()])
        else:
            # Try as direct path
            if Path(pattern).is_file():
                video_paths.append(pattern)
    
    if not video_paths:
        print("ERROR: No video files found")
        return 1
    
    print(f"Found {len(video_paths)} video(s) to process")
    print(f"Extracting NORMAL motion patterns for anomaly detection")
    print(f"Output: {args.output}")
    
    # Load model
    print(f"\nLoading model: {args.model}")
    model = load_model(args.model, device=args.device)
    
    # Create or load database
    db = EmbeddingDatabase(args.output)
    if args.append and Path(args.output).exists():
        print(f"Loading existing database...")
        db.load()
        print(f"Existing embeddings: {len(db.embeddings)}")
    
    # Create extractor
    extractor = EmbeddingExtractor(window_size=args.window_size)
    config = FusionPanicConfig()
    
    # Process each video
    all_embeddings = []
    for video_path in video_paths:
        embeddings = extract_embeddings_from_video(
            video_path,
            model,
            extractor,
            config,
        )
        all_embeddings.extend(embeddings)
    
    # Add to database
    for emb in all_embeddings:
        db.add_embedding(emb)
    
    # Save
    db.save()
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total embeddings extracted: {len(all_embeddings)}")
    print(f"Total embeddings in database: {len(db.embeddings)}")
    
    # Count by label
    label_counts = {}
    for emb in db.embeddings:
        label_counts[emb.label] = label_counts.get(emb.label, 0) + 1
    
    print(f"Label distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    
    print(f"\nEmbeddings saved to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
