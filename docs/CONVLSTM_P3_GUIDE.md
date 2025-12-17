# ConvLSTM Panic Detection - P3 Feature Guide

## Overview

The ConvLSTM panic detection system uses a **fixed 5-channel feature set (P3)** for anomaly-based panic detection. This document describes the feature engineering, training workflow, and evaluation process.

---

## 1. Feature Set (P3 - FINAL)

### ConvLSTM Input Tensor Shape

```
[Batch, Time, Channels=5, Height=96, Width=96]
```

### Channel Order and Content

| Channel | Feature | Normalization | Purpose |
|---------|---------|---------------|---------|
| 0 | `flow_x` | `clip(-vmax, vmax) / vmax` → [-1, 1] | Horizontal motion component |
| 1 | `flow_y` | `clip(-vmax, vmax) / vmax` → [-1, 1] | Vertical motion component |
| 2 | `flow_mag` | `clip(0, vmax) / vmax` → [0, 1] | Motion magnitude (speed) |
| 3 | `bbox_heatmap` | [0, 1] | Human bounding box density |
| 4 | `flow_divergence` | `clip(-dmax, dmax) / dmax` → [-1, 1] | Spatial flow derivatives (sudden velocity changes) |

### Feature Computation Details

**Optical Flow:**
- Method: Farneback
- Global motion removal: median subtraction
- `vmax = 10.0` px/frame (default)

**Divergence:**
- Computed as: `∂flow_x/∂x + ∂flow_y/∂y`
- Uses central finite differences
- Normalization: per-frame p99 or fixed `dmax`

**Bbox Heatmap:**
- Projects all detected person bboxes to 96×96 grid
- Overlapping regions accumulate
- Normalized to [0, 1]

---

## 2. Features NOT in ConvLSTM Input

The following scalar features are computed for **analysis only** (e.g., feature report) but **do NOT go into the ConvLSTM model**:

❌ `entropy_global`, `entropy_roi`  
❌ `coherence_global`, `coherence_roi`  
❌ `dentropy`, `dcoh`  
❌ `people_count`  
❌ `bbox_area_ratio`  
❌ `dv_p95` (scalar derivative - divergence channel captures this spatially)

These features can be used for:
- Feature engineering analysis
- Baseline panic detection (non-ConvLSTM)
- Interpretability and debugging

---

## 3. Training Workflow

### Command

```bash
python scripts/train_convlstm_panic.py \
  --videos "data/dataset/train/*.mp4" \
  --val-videos "data/dataset/val/*.mp4" \
  --output "models/convlstm_panic.pt" \
  --model "weights/yolo11x-pose.pt" \
  --device cuda:0 \
  --epochs 50 \
  --batch-size 8 \
  --sequence-length 16 \
  --image-size 96 \
  --vmax 10.0 \
  --hdf5-cache "data/cache/sequences.h5" \
  --num-workers 4
```

### Key Parameters

- `--sequence-length 16`: Temporal window (16 frames ≈ 0.64s at 25 FPS)
- `--image-size 96`: Spatial resolution (96×96 grid)
- `--vmax 10.0`: Flow normalization max (px/frame)
- `--stride`: Auto-set to `sequence_length // 2` (50% overlap)

### Checkpoint Contents

The saved model includes:
```python
{
    'model_state_dict': ...,
    'threshold': float,      # p95 of val reconstruction errors
    'vmax': float,           # Flow normalization
    'sequence_length': int,  # Temporal window
    'image_size': int,       # Spatial grid size
    'epoch': int,
    'train_loss': float,
    'val_loss': float,
}
```

### Threshold Strategy

- **p95 threshold**: Soft panic detection (saved in checkpoint)
- **p99 threshold**: Hard panic detection (can be computed from val errors)
- Both values should be logged during training for later tuning

---

## 4. Inference Workflow

### Command

```bash
python main.py --panic \
  --source "data/dataset/test/test.mp4" \
  --model "weights/yolo11x-pose.pt" \
  --panic-convlstm-model "models/convlstm_panic.pt" \
  --output "runs/test_output.mp4" \
  --no-display
```

### Threshold Override

If you want to test different thresholds without retraining:

```bash
python main.py --panic \
  --source "data/dataset/test/test.mp4" \
  --model "weights/yolo11x-pose.pt" \
  --panic-convlstm-model "models/convlstm_panic.pt" \
  --panic-convlstm-threshold 0.0001 \
  --output "runs/test_output.mp4" \
  --no-display
```

### Feature Consistency

The inference wrapper (`src/detection/panic/wrapper.py`) uses the **same** `build_convlstm_features()` function as training, ensuring no train/test mismatch.

---

## 5. Feature Engineering Report

Before training, analyze which scalar features best separate normal vs panic behavior.

### Command

```bash
python scripts/feature_report_panic.py \
  --train "data/dataset/train/*.mp4" \
  --val "data/dataset/val/*.mp4" \
  --test "data/dataset/test/test.mp4" \
  --model "weights/yolo11x-pose.pt" \
  --image-size 96 \
  --fps 25 \
  --window 25 \
  --stride 12 \
  --onset-sec 10.0 \
  --output-csv "runs/feature_report.csv" \
  --output-summary "runs/feature_summary.json" \
  --device cuda:0
```

### Output

**CSV** (`runs/feature_report.csv`):
- One row per window
- All scalar features + robust z-scores

**JSON** (`runs/feature_summary.json`):
- Baseline statistics (median, MAD)
- Feature ranking by panic separation
- Onset detection latency

### Interpretation

Top features by separation indicate which signals best distinguish panic from normal behavior. This helps:
- Validate that flow-based features are informative
- Identify complementary signals for future work
- Debug if ConvLSTM is not learning useful patterns

---

## 6. Evaluation Metrics

### Reconstruction Error

- **Normal behavior**: Low reconstruction error (model has seen similar patterns)
- **Panic behavior**: High reconstruction error (anomaly)

### Key Metrics

1. **Val p95/p99 thresholds**: From training, indicates model sensitivity
2. **Onset latency**: Time from panic start (e.g., 10.0s) to first alert
3. **False alarm rate**: Number of alerts in pre-onset period (0-10s)
4. **Sustained detection**: Consecutive alerts after onset

### Example Console Output

```
Loaded ConvLSTM model from models/convlstm_panic.pt
Channels: 5 (P3: flow_x, flow_y, flow_mag, bbox_heatmap, divergence)
Threshold: 2.96816e-05
Image size: 96x96

[INFO] ConvLSTM mode enabled: first decision available after 16 frames.
[INFO] Writing output video to: runs/test_output.mp4

Frame 250 (10.0s):
[ALERT] PANIC detected! Score: 1.18  ← Onset latency = 0.0s

Frame 262 (10.48s):
[ALERT] PANIC detected! Score: 1.57  ← Sustained detection

Frame 275 (11.0s):
[ALERT] PANIC detected! Score: 1.82
```

---

## 7. Colab Training Workflow

### Setup

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/YOUR_USERNAME/MAM.git
%cd MAM

!pip install -r requirements.txt

!wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11x-pose.pt -P weights/
```

### Upload Dataset

```python
!unzip /content/drive/MyDrive/dataset.zip -d data/
```

Expected structure:
```
data/dataset/
  train/*.mp4
  val/*.mp4
  test/test.mp4
```

### Feature Report

```python
!python scripts/feature_report_panic.py \
  --train "data/dataset/train/*.mp4" \
  --val "data/dataset/val/*.mp4" \
  --test "data/dataset/test/test.mp4" \
  --model "weights/yolo11x-pose.pt" \
  --output-csv "runs/feature_report.csv" \
  --output-summary "runs/feature_summary.json" \
  --device cuda:0
```

### Training

```python
!python scripts/train_convlstm_panic.py \
  --videos "data/dataset/train/*.mp4" \
  --val-videos "data/dataset/val/*.mp4" \
  --output "models/convlstm_panic.pt" \
  --model "weights/yolo11x-pose.pt" \
  --device cuda:0 \
  --epochs 50 \
  --batch-size 8 \
  --sequence-length 16 \
  --image-size 96 \
  --vmax 10.0 \
  --hdf5-cache "data/cache/sequences.h5" \
  --num-workers 2
```

### Evaluation

```python
!python main.py --panic \
  --source "data/dataset/test/test.mp4" \
  --model "weights/yolo11x-pose.pt" \
  --panic-convlstm-model "models/convlstm_panic.pt" \
  --output "runs/test_output.mp4" \
  --no-display
```

### Download Results

```python
from google.colab import files

files.download("models/convlstm_panic.pt")
files.download("runs/feature_summary.json")
files.download("runs/test_output.mp4")
```

---

## 8. Troubleshooting

### Issue: "No sequences extracted"

**Cause**: YOLO not detecting people or incorrect video paths.

**Fix**:
```bash
# Test YOLO detection
python -c "
import cv2
from src.detection.utils.pose import load_model, detect_people
model = load_model('weights/yolo11x-pose.pt')
cap = cv2.VideoCapture('data/dataset/train/video.mp4')
ret, frame = cap.read()
dets = detect_people(model, frame, conf=0.25)
print(f'Detected {len(dets)} people')
"
```

### Issue: "CUDA out of memory"

**Fix**: Reduce batch size or use float16 caching:
```bash
--batch-size 4 --cache-float16
```

### Issue: "High false alarm rate"

**Cause**: Threshold too low.

**Fix**: Override with higher threshold:
```bash
--panic-convlstm-threshold 0.0001
```

Or retrain with more normal data.

### Issue: "High onset latency"

**Cause**: Model not sensitive enough or test video has gradual panic onset.

**Fix**:
- Check feature report: are flow features separating well?
- Try lower threshold (trade-off with false alarms)
- Ensure training data covers diverse normal scenarios

---

## 9. Next Steps

1. **Train baseline model** with P3 features
2. **Evaluate on test video** and measure metrics
3. **Tune threshold** if needed (p95 vs p99 vs custom)
4. **Deploy** with selected threshold
5. **Monitor** false positives on real-world data
6. **Retrain periodically** with new normal data

### Future Work

- **Keypoint features**: Add pose keypoint heatmaps (would increase to 12 channels)
- **Multi-scale**: Train on multiple image sizes (64, 96, 128)
- **Temporal ablation**: Test different sequence lengths (8, 16, 32)
- **Ensemble**: Combine ConvLSTM with baseline flow detector

---

## 10. References

- **Feature module**: `src/detection/panic/features.py`
- **Training script**: `scripts/train_convlstm_panic.py`
- **Feature report**: `scripts/feature_report_panic.py`
- **Inference wrapper**: `src/detection/panic/wrapper.py`
- **ConvLSTM model**: `src/detection/panic/convlstm_model.py`

---

## Summary

**P3 Feature Set**: 5 channels (flow_x, flow_y, flow_mag, bbox_heatmap, divergence)  
**Training**: Normal videos only, autoencoder learns normal patterns  
**Inference**: High reconstruction error = anomaly = panic  
**Threshold**: p95 from validation set (adjustable)  
**Consistency**: Single `build_convlstm_features()` for train and inference
