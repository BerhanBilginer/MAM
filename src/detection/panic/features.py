"""
Unified feature extraction module for panic detection.

This module provides a single source of truth for feature computation
used by both training scripts and inference wrappers to avoid train/inference mismatch.

ConvLSTM uses a fixed 5-channel feature set (P3):
- flow_x, flow_y: Optical flow components
- flow_mag: Flow magnitude
- bbox_heatmap: Human bounding box density
- flow_divergence: Spatial flow derivatives (captures sudden velocity changes)

Additional scalar features (entropy, coherence, etc.) are computed for analysis
but do NOT go into the ConvLSTM model.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional
import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy import typing as npt
    from src.detection.utils.pose import PoseDetection


# 7-keypoint set indices for COCO pose (adjust based on your YOLO pose model)
# Typical COCO: 0=nose, 5=L_shoulder, 6=R_shoulder, 11=L_hip, 12=R_hip, 15=L_ankle, 16=R_ankle
KEYPOINT_7_SET = [0, 5, 6, 11, 12, 15, 16]  # head/neck + shoulders + hips + ankles


def to_gray_small(frame_bgr: npt.NDArray[np.uint8], size: tuple[int, int] = (96, 96)) -> npt.NDArray[np.uint8]:
    """Convert BGR frame to grayscale and resize to small grid."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)


def compute_flow(
    prev_gray_small: npt.NDArray[np.uint8],
    gray_small: npt.NDArray[np.uint8],
    remove_global_motion: bool = True,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Compute optical flow using Farneback method.
    
    Returns:
        (flow_x, flow_y, magnitude, angle)
        All arrays are float32 with same shape as input.
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray_small,
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
    
    flow_x = flow[..., 0].astype(np.float32)
    flow_y = flow[..., 1].astype(np.float32)
    
    if remove_global_motion:
        flow_x -= np.median(flow_x)
        flow_y -= np.median(flow_y)
    
    magnitude = np.sqrt(flow_x**2 + flow_y**2).astype(np.float32)
    angle = np.arctan2(flow_y, flow_x).astype(np.float32)
    
    return flow_x, flow_y, magnitude, angle


def compute_divergence(flow_x: npt.NDArray[np.float32], flow_y: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """
    Compute flow divergence using finite differences: d/dx(flow_x) + d/dy(flow_y).
    
    Returns:
        Divergence field (same shape as input).
    """
    # Simple central differences
    dx_fx = np.zeros_like(flow_x)
    dy_fy = np.zeros_like(flow_y)
    
    # d/dx(flow_x)
    dx_fx[:, 1:-1] = (flow_x[:, 2:] - flow_x[:, :-2]) / 2.0
    dx_fx[:, 0] = flow_x[:, 1] - flow_x[:, 0]
    dx_fx[:, -1] = flow_x[:, -1] - flow_x[:, -2]
    
    # d/dy(flow_y)
    dy_fy[1:-1, :] = (flow_y[2:, :] - flow_y[:-2, :]) / 2.0
    dy_fy[0, :] = flow_y[1, :] - flow_y[0, :]
    dy_fy[-1, :] = flow_y[-1, :] - flow_y[-2, :]
    
    divergence = (dx_fx + dy_fy).astype(np.float32)
    return divergence


def direction_entropy(
    angles: npt.NDArray[np.float32],
    mags: Optional[npt.NDArray[np.float32]] = None,
    bins: int = 16,
    mag_eps: float = 0.05,
    weighted: bool = False,
) -> float:
    """
    Compute entropy of flow direction distribution.
    
    Args:
        angles: Flow angles in radians [-pi, pi]
        mags: Flow magnitudes (optional, for filtering/weighting)
        bins: Number of histogram bins
        mag_eps: Magnitude threshold for filtering low-motion pixels
        weighted: If True, weight histogram by magnitude
    
    Returns:
        Entropy value (higher = more chaotic motion)
    """
    if mags is not None:
        mask = mags > mag_eps
        if not mask.any():
            return 0.0
        angles_filtered = angles[mask]
        mags_filtered = mags[mask] if weighted else None
    else:
        angles_filtered = angles.flatten()
        mags_filtered = None
    
    if len(angles_filtered) == 0:
        return 0.0
    
    # Histogram
    if weighted and mags_filtered is not None:
        hist, _ = np.histogram(angles_filtered, bins=bins, range=(-np.pi, np.pi), weights=mags_filtered)
    else:
        hist, _ = np.histogram(angles_filtered, bins=bins, range=(-np.pi, np.pi))
    
    # Normalize to probability
    hist = hist.astype(np.float32)
    hist_sum = hist.sum()
    if hist_sum == 0:
        return 0.0
    prob = hist / hist_sum
    
    # Entropy
    prob = prob[prob > 0]
    entropy = -np.sum(prob * np.log2(prob))
    return float(entropy)


def direction_coherence(
    angles: npt.NDArray[np.float32],
    mags: Optional[npt.NDArray[np.float32]] = None,
    mag_eps: float = 0.05,
) -> float:
    """
    Compute direction coherence as magnitude of mean unit vector.
    
    Returns:
        Coherence in [0, 1] where 1 = all motion in same direction, 0 = random
    """
    if mags is not None:
        mask = mags > mag_eps
        if not mask.any():
            return 0.0
        angles_filtered = angles[mask]
    else:
        angles_filtered = angles.flatten()
    
    if len(angles_filtered) == 0:
        return 0.0
    
    # Mean unit vector
    cos_mean = np.mean(np.cos(angles_filtered))
    sin_mean = np.mean(np.sin(angles_filtered))
    coherence = np.sqrt(cos_mean**2 + sin_mean**2)
    
    return float(coherence)


def roi_mask_from_dets(
    detections: list,
    small_shape: tuple[int, int],
    orig_shape: tuple[int, int],
    pad: float = 0.10,
) -> Optional[npt.NDArray[np.bool_]]:
    """
    Create ROI mask from detections projected to small grid.
    
    Args:
        detections: List of detections with .bbox attribute
        small_shape: (H, W) of small grid
        orig_shape: (H, W) of original frame
        pad: Padding ratio around bboxes
    
    Returns:
        Boolean mask or None if no detections
    """
    if not detections:
        return None
    
    h_small, w_small = small_shape
    h_orig, w_orig = orig_shape
    
    mask = np.zeros((h_small, w_small), dtype=np.bool_)
    
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        
        # Add padding
        w_box = x2 - x1
        h_box = y2 - y1
        x1 = max(0, x1 - pad * w_box)
        y1 = max(0, y1 - pad * h_box)
        x2 = min(w_orig, x2 + pad * w_box)
        y2 = min(h_orig, y2 + pad * h_box)
        
        # Project to small grid
        x1_s = int(x1 * w_small / w_orig)
        y1_s = int(y1 * h_small / h_orig)
        x2_s = int(x2 * w_small / w_orig)
        y2_s = int(y2 * h_small / h_orig)
        
        x1_s = max(0, x1_s)
        y1_s = max(0, y1_s)
        x2_s = min(w_small, x2_s)
        y2_s = min(h_small, y2_s)
        
        if x2_s > x1_s and y2_s > y1_s:
            mask[y1_s:y2_s, x1_s:x2_s] = True
    
    return mask


def create_bbox_heatmap(
    detections: list,
    small_shape: tuple[int, int],
    orig_shape: tuple[int, int],
) -> npt.NDArray[np.float32]:
    """
    Create normalized bbox occupancy heatmap.
    
    Returns:
        Heatmap in [0, 1] with shape small_shape
    """
    h_small, w_small = small_shape
    h_orig, w_orig = orig_shape
    
    heatmap = np.zeros((h_small, w_small), dtype=np.float32)
    
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        
        # Project to small grid
        x1_s = int(x1 * w_small / w_orig)
        y1_s = int(y1 * h_small / h_orig)
        x2_s = int(x2 * w_small / w_orig)
        y2_s = int(y2 * h_small / h_orig)
        
        x1_s = max(0, x1_s)
        y1_s = max(0, y1_s)
        x2_s = min(w_small, x2_s)
        y2_s = min(h_small, y2_s)
        
        if x2_s > x1_s and y2_s > y1_s:
            heatmap[y1_s:y2_s, x1_s:x2_s] += 1.0
    
    # Normalize
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    return heatmap


def create_keypoint_heatmaps(
    detections_pose: list,
    small_shape: tuple[int, int],
    orig_shape: tuple[int, int],
    kp_ids: list[int] = KEYPOINT_7_SET,
    sigma: float = 2.0,
) -> npt.NDArray[np.float32]:
    """
    Create Gaussian heatmaps for specified keypoints.
    
    Args:
        detections_pose: List of PoseDetection objects with .keypoints
        small_shape: (H, W) of output grid
        orig_shape: (H, W) of original frame
        kp_ids: List of keypoint indices to include
        sigma: Gaussian sigma for heatmap blobs
    
    Returns:
        Array of shape (K, H, W) where K = len(kp_ids)
    """
    h_small, w_small = small_shape
    h_orig, w_orig = orig_shape
    K = len(kp_ids)
    
    heatmaps = np.zeros((K, h_small, w_small), dtype=np.float32)
    
    # Create Gaussian kernel
    kernel_size = int(6 * sigma)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_half = kernel_size // 2
    
    y_grid, x_grid = np.ogrid[-kernel_half:kernel_half+1, -kernel_half:kernel_half+1]
    gaussian = np.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
    gaussian = gaussian.astype(np.float32)
    
    for det in detections_pose:
        if not hasattr(det, 'keypoints') or det.keypoints is None:
            continue
        
        for k_idx, kp_id in enumerate(kp_ids):
            if kp_id >= len(det.keypoints):
                continue
            
            kp = det.keypoints[kp_id]
            x, y, conf = kp
            
            # Skip low-confidence keypoints
            if conf < 0.3:
                continue
            
            # Project to small grid
            x_s = int(x * w_small / w_orig)
            y_s = int(y * h_small / h_orig)
            
            if 0 <= x_s < w_small and 0 <= y_s < h_small:
                # Add Gaussian blob
                y1 = max(0, y_s - kernel_half)
                y2 = min(h_small, y_s + kernel_half + 1)
                x1 = max(0, x_s - kernel_half)
                x2 = min(w_small, x_s + kernel_half + 1)
                
                ky1 = kernel_half - (y_s - y1)
                ky2 = kernel_half + (y2 - y_s)
                kx1 = kernel_half - (x_s - x1)
                kx2 = kernel_half + (x2 - x_s)
                
                heatmaps[k_idx, y1:y2, x1:x2] += gaussian[ky1:ky2, kx1:kx2]
    
    # Normalize each channel
    for k in range(K):
        if heatmaps[k].max() > 0:
            heatmaps[k] = heatmaps[k] / heatmaps[k].max()
    
    return heatmaps


def build_convlstm_features(
    frame_bgr: npt.NDArray[np.uint8],
    detections_pose: list,
    prev_gray_small: Optional[npt.NDArray[np.uint8]],
    *,
    image_size: int = 96,
    vmax: float = 10.0,
    dmax: Optional[float] = None,
) -> tuple[Optional[npt.NDArray[np.float32]], npt.NDArray[np.uint8]]:
    """
    Build ConvLSTM input features (P3: 5 channels).
    
    Channel order:
        0: flow_x - normalized to [-1, 1]
        1: flow_y - normalized to [-1, 1]
        2: flow_mag - normalized to [0, 1]
        3: bbox_heatmap - normalized to [0, 1]
        4: flow_divergence - normalized to [-1, 1]
    
    Args:
        frame_bgr: Current frame (H, W, 3)
        detections_pose: List of PoseDetection objects
        prev_gray_small: Previous grayscale small frame (or None for first frame)
        image_size: Grid size (square)
        vmax: Flow normalization max value (px/frame)
        dmax: Divergence normalization max (if None, uses per-frame p99)
    
    Returns:
        (features, gray_small) where features is (5, H, W) or None if first frame
    """
    h_orig, w_orig = frame_bgr.shape[:2]
    size = (image_size, image_size)
    
    gray_small = to_gray_small(frame_bgr, size)
    
    if prev_gray_small is None:
        return None, gray_small
    
    # Compute flow
    flow_x, flow_y, flow_mag, flow_ang = compute_flow(prev_gray_small, gray_small, remove_global_motion=True)
    
    # Bbox heatmap
    bbox_heatmap = create_bbox_heatmap(detections_pose, size, (h_orig, w_orig))
    
    # Normalize flow components
    flow_x_norm = np.clip(flow_x, -vmax, vmax) / vmax
    flow_y_norm = np.clip(flow_y, -vmax, vmax) / vmax
    flow_mag_norm = np.clip(flow_mag, 0, vmax) / vmax
    
    # Compute and normalize divergence
    divergence = compute_divergence(flow_x, flow_y)
    
    if dmax is None:
        # Use per-frame p99 normalization
        div_abs = np.abs(divergence)
        div_scale = np.percentile(div_abs, 99) if div_abs.max() > 0 else 1.0
        div_scale = max(div_scale, 1e-6)
    else:
        div_scale = dmax
    
    divergence_norm = np.clip(divergence / div_scale, -1.0, 1.0)
    
    # Stack P3 features: flow_x, flow_y, flow_mag, bbox_heatmap, divergence
    features = np.stack([
        flow_x_norm,
        flow_y_norm,
        flow_mag_norm,
        bbox_heatmap,
        divergence_norm,
    ], axis=0).astype(np.float32)
    
    return features, gray_small
