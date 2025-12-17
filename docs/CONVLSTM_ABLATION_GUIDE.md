# ConvLSTM Feature Pack Ablation Study Guide

This guide documents the feature engineering workflow, ablation study commands, and expected outputs for the ConvLSTM panic detection system.

## Overview

The system now supports **4 feature packs** (p1-p4) with increasing complexity:

- **p1**: `flow_x`, `flow_y`, `bbox_heatmap` (3 channels)
- **p2**: p1 + `flow_magnitude` (4 channels)
- **p3**: p2 + `flow_divergence` (5 channels)
- **p4**: p3 + `keypoint_heatmaps_7` (12 channels)

All features use a unified module (`src/detection/panic/features.py`) to ensure train/inference consistency.

---

## 1. Feature Report (Scalar Analysis)

Before training ConvLSTM models, run the feature report script to understand which scalar features best separate normal vs panic behavior.

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

### Parameters

- `--train`, `--val`, `--test`: Video paths or globs
- `--image-size`: Feature grid size (default: 96)
- `--fps`: Video FPS (default: 25)
- `--window`: Window size in frames (default: 25 = 1 second)
- `--stride`: Window stride in frames (default: 12 = 0.5s overlap)
- `--onset-sec`: Panic onset time for test video (default: 10.0)

### Outputs

**CSV** (`runs/feature_report.csv`):
- Rows: one per window
- Columns: `split`, `video_path`, `window_idx`, `t_start_sec`, `t_end_sec`, all features, z-scores

**JSON** (`runs/feature_summary.json`):
- Baseline statistics (median, MAD) for each feature
- Feature ranking by panic separation (`panic_rate - normal_rate`)
- Latency (seconds after onset to first z>3 detection)

### Expected Output

Top features by separation (example):

```
Feature              | sep   | panic | normal | latency
---------------------|-------|-------|--------|--------
v_p95_roi            | 0.850 | 0.920 | 0.070  | 0.48
entropy_roi          | 0.720 | 0.780 | 0.060  | 0.96
dv_p95               | 0.680 | 0.710 | 0.030  | 1.20
v_p99_global         | 0.620 | 0.650 | 0.030  | 0.72
```

**Interpretation**: Features with high `separation` and low `latency` are good panic indicators.

---

## 2. Training ConvLSTM Models (Ablation)

Train 4 models (one per pack) on the same normal train/val data.

### Base Command Template

```bash
python scripts/train_convlstm_panic.py \
  --videos "data/dataset/train/*.mp4" \
  --val-videos "data/dataset/val/*.mp4" \
  --output "models/convlstm_PACK.pt" \
  --model "weights/yolo11x-pose.pt" \
  --device cuda:0 \
  --epochs 50 \
  --batch-size 8 \
  --sequence-length 16 \
  --image-size 96 \
  --vmax 10.0 \
  --pack PACK \
  --hdf5-cache "data/cache/sequences_PACK.h5" \
  --num-workers 4
```

### Pack-Specific Commands

#### p1: Flow + BBox (3ch)

```bash
python scripts/train_convlstm_panic.py \
  --videos "data/dataset/train/*.mp4" \
  --val-videos "data/dataset/val/*.mp4" \
  --output "models/convlstm_p1.pt" \
  --model "weights/yolo11x-pose.pt" \
  --device cuda:0 \
  --epochs 50 \
  --batch-size 8 \
  --sequence-length 16 \
  --image-size 96 \
  --vmax 10.0 \
  --pack p1 \
  --hdf5-cache "data/cache/sequences_p1.h5" \
  --num-workers 4
```

#### p2: + Magnitude (4ch)

```bash
python scripts/train_convlstm_panic.py \
  --videos "data/dataset/train/*.mp4" \
  --val-videos "data/dataset/val/*.mp4" \
  --output "models/convlstm_p2.pt" \
  --model "weights/yolo11x-pose.pt" \
  --device cuda:0 \
  --epochs 50 \
  --batch-size 8 \
  --sequence-length 16 \
  --image-size 96 \
  --vmax 10.0 \
  --pack p2 \
  --hdf5-cache "data/cache/sequences_p2.h5" \
  --num-workers 4
```

#### p3: + Divergence (5ch)

```bash
python scripts/train_convlstm_panic.py \
  --videos "data/dataset/train/*.mp4" \
  --val-videos "data/dataset/val/*.mp4" \
  --output "models/convlstm_p3.pt" \
  --model "weights/yolo11x-pose.pt" \
  --device cuda:0 \
  --epochs 50 \
  --batch-size 8 \
  --sequence-length 16 \
  --image-size 96 \
  --vmax 10.0 \
  --pack p3 \
  --hdf5-cache "data/cache/sequences_p3.h5" \
  --num-workers 4
```

#### p4: + Keypoints (12ch)

```bash
python scripts/train_convlstm_panic.py \
  --videos "data/dataset/train/*.mp4" \
  --val-videos "data/dataset/val/*.mp4" \
  --output "models/convlstm_p4.pt" \
  --model "weights/yolo11x-pose.pt" \
  --device cuda:0 \
  --epochs 50 \
  --batch-size 8 \
  --sequence-length 16 \
  --image-size 96 \
  --vmax 10.0 \
  --pack p4 \
  --hdf5-cache "data/cache/sequences_p4.h5" \
  --num-workers 4
```

### Training Output

Each model will save:
- `models/convlstm_pX.pt` checkpoint with:
  - `model_state_dict`
  - `threshold` (95th percentile of val reconstruction errors)
  - `vmax`, `pack`, `sequence_length`, `image_size`

---

## 3. Evaluation on Test Video

After training all 4 models, evaluate each on the panic test video.

### Command Template

```bash
python main.py --panic \
  --source "data/dataset/test/test.mp4" \
  --model "weights/yolo11x-pose.pt" \
  --panic-convlstm-model "models/convlstm_PACK.pt" \
  --output "runs/test_PACK.mp4" \
  --no-display
```

### Pack-Specific Evaluation

```bash
# p1
python main.py --panic \
  --source "data/dataset/test/test.mp4" \
  --model "weights/yolo11x-pose.pt" \
  --panic-convlstm-model "models/convlstm_p1.pt" \
  --output "runs/test_p1.mp4" \
  --no-display

# p2
python main.py --panic \
  --source "data/dataset/test/test.mp4" \
  --model "weights/yolo11x-pose.pt" \
  --panic-convlstm-model "models/convlstm_p2.pt" \
  --output "runs/test_p2.mp4" \
  --no-display

# p3
python main.py --panic \
  --source "data/dataset/test/test.mp4" \
  --model "weights/yolo11x-pose.pt" \
  --panic-convlstm-model "models/convlstm_p3.pt" \
  --output "runs/test_p3.mp4" \
  --no-display

# p4
python main.py --panic \
  --source "data/dataset/test/test.mp4" \
  --model "weights/yolo11x-pose.pt" \
  --panic-convlstm-model "models/convlstm_p4.pt" \
  --output "runs/test_p4.mp4" \
  --no-display
```

### Metrics to Collect

For each pack, record:

1. **Val threshold** (from checkpoint, 95th percentile)
2. **Onset latency**: Time from 10.0s to first `[ALERT] PANIC detected!` in console
3. **False alarm rate**: Count of `[ALERT]` messages in 0-10s range
4. **Sustained detection**: Number of consecutive alerts after onset

### Example Console Output

```
Loaded ConvLSTM model from models/convlstm_p3.pt
Pack: p3 (5 channels)
Threshold: 2.96816e-05
Image size: 96x96
[INFO] ConvLSTM mode enabled: first decision available after 16 frames (sequence buffer).
[INFO] Writing output video to: runs/test_p3.mp4
[ALERT] PANIC detected! Score: 1.18    # <- onset latency = this timestamp - 10.0s
[ALERT] PANIC detected! Score: 1.57
[ALERT] PANIC detected! Score: 1.82
...
```

---

## 4. Ablation Results Summary

Create a table comparing all packs:

| Pack | Channels | Val Threshold | Onset Latency (s) | False Alarms (0-10s) | Sustained Alerts |
|------|----------|---------------|-------------------|----------------------|------------------|
| p1   | 3        | ?             | ?                 | ?                    | ?                |
| p2   | 4        | ?             | ?                 | ?                    | ?                |
| p3   | 5        | ?             | ?                 | ?                    | ?                |
| p4   | 12       | ?             | ?                 | ?                    | ?                |

**Selection Criteria**:
- **Best pack**: Lowest onset latency + fewest false alarms + most sustained alerts
- **Trade-off**: p4 has more channels (richer features) but may overfit or be slower

---

## 5. Colab Workflow

### Setup

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone repo
!git clone https://github.com/YOUR_USERNAME/MAM.git
%cd MAM

# Install dependencies
!pip install -r requirements.txt

# Download YOLO weights
!wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11x-pose.pt -P weights/
```

### Upload Dataset

Upload `dataset.zip` to Drive, then:

```python
!unzip /content/drive/MyDrive/dataset.zip -d data/
```

Expected structure:
```
data/dataset/
  train/
    *.mp4
  val/
    *.mp4
  test/
    test.mp4
```

### Run Feature Report

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

### Train All Packs

```python
for pack in ["p1", "p2", "p3", "p4"]:
    !python scripts/train_convlstm_panic.py \
      --videos "data/dataset/train/*.mp4" \
      --val-videos "data/dataset/val/*.mp4" \
      --output f"models/convlstm_{pack}.pt" \
      --model "weights/yolo11x-pose.pt" \
      --device cuda:0 \
      --epochs 50 \
      --batch-size 8 \
      --pack {pack} \
      --hdf5-cache f"data/cache/sequences_{pack}.h5" \
      --num-workers 2
```

### Evaluate All Packs

```python
for pack in ["p1", "p2", "p3", "p4"]:
    !python main.py --panic \
      --source "data/dataset/test/test.mp4" \
      --model "weights/yolo11x-pose.pt" \
      --panic-convlstm-model f"models/convlstm_{pack}.pt" \
      --output f"runs/test_{pack}.mp4" \
      --no-display
```

### Download Results

```python
from google.colab import files

# Download models
for pack in ["p1", "p2", "p3", "p4"]:
    files.download(f"models/convlstm_{pack}.pt")

# Download feature report
files.download("runs/feature_summary.json")

# Download test videos
for pack in ["p1", "p2", "p3", "p4"]:
    files.download(f"runs/test_{pack}.mp4")
```

---

## 6. Expected Timeline

- **Feature report**: ~10-20 min (depends on video count)
- **Training per pack**: ~30-60 min (50 epochs, GPU)
- **Evaluation per pack**: ~1-2 min
- **Total ablation**: ~3-4 hours

---

## 7. Troubleshooting

### Issue: "No sequences extracted"

**Cause**: YOLO model not detecting people or video paths incorrect.

**Fix**:
```bash
# Check video paths
ls data/dataset/train/*.mp4

# Test YOLO detection on one frame
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

**Fix**: Reduce batch size or use HDF5 cache with float16:

```bash
--batch-size 4 \
--cache-float16
```

### Issue: "Threshold too low/high"

**Cause**: Val data distribution mismatch.

**Fix**: Override threshold manually:

```bash
--panic-convlstm-threshold 0.0001
```

---

## 8. Next Steps After Ablation

1. **Select best pack** based on metrics
2. **Fine-tune threshold** if needed
3. **Deploy to production** with selected pack
4. **Monitor false positives** on real-world data
5. **Retrain periodically** with new normal data

---

## References

- Feature module: `src/detection/panic/features.py`
- Training script: `scripts/train_convlstm_panic.py`
- Feature report: `scripts/feature_report_panic.py`
- Inference wrapper: `src/detection/panic/wrapper.py`
