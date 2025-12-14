# ConvLSTM-Based Panic Detection

ConvLSTM Autoencoder kullanarak anomaly detection yaklaşımı ile panic tespiti.

## Genel Bakış

**Yaklaşım**: ConvLSTM Autoencoder sadece normal motion pattern'lerini öğrenir. Panic, yüksek reconstruction error olarak tespit edilir.

### Avantajlar

- ✅ **Temporal patterns**: LSTM sayesinde zaman içindeki değişimleri öğrenir
- ✅ **Spatial-temporal features**: Optical flow + pose bilgisini birleştirir
- ✅ **End-to-end learning**: Manuel feature engineering gerektirmez
- ✅ **Sadece normal video**: Panic video toplamaya gerek yok
- ✅ **Deep learning**: Daha karmaşık pattern'leri yakalayabilir

### Mimari

```
Input: [B, T=16, C=3, H=64, W=64]
  ├── Channel 0: Flow X (horizontal motion)
  ├── Channel 1: Flow Y (vertical motion)
  └── Channel 2: Pose heatmap (people density)

Encoder (ConvLSTM):
  ├── Layer 1: 3 → 32 channels
  ├── Layer 2: 32 → 64 channels
  └── Layer 3: 64 → 32 channels

Decoder (ConvLSTM):
  ├── Layer 1: 32 → 64 channels
  ├── Layer 2: 64 → 32 channels
  └── Layer 3: 32 → 3 channels

Output: Reconstructed [B, T=16, C=3, H=64, W=64]

Loss: MSE(Input, Output)
```

## Kurulum

```bash
pip install torch tqdm
```

veya

```bash
pip install -r requirements.txt
```

## Kullanım

### 1. Normal Videolar Topla

YouTube downloader kullanarak normal kalabalık videoları topla:

```bash
python scripts/download_youtube_videos.py \
    --url "NORMAL_CROWD_PLAYLIST_URL" \
    --output data/normal_videos \
    --label normal \
    --quality 720p
```

### 2. Model Training

Normal videolardan ConvLSTM modelini eğit:

```bash
python scripts/train_convlstm_panic.py \
    --videos "data/normal_videos/*.mp4" \
    --output models/convlstm_panic.pt \
    --model yolov8n-pose.pt \
    --device cpu \
    --epochs 50 \
    --batch-size 4 \
    --sequence-length 16 \
    --image-size 64
```

**Parametreler:**
- `--videos`: Normal video dosyaları (glob pattern)
- `--output`: Model çıktı yolu (.pt)
- `--model`: YOLO pose model
- `--device`: Training device (cpu, cuda:0)
- `--epochs`: Epoch sayısı (50-100 önerilen)
- `--batch-size`: Batch size (GPU'ya göre ayarla)
- `--sequence-length`: Sequence uzunluğu (16 frame önerilen)
- `--image-size`: Input image boyutu (64x64 önerilen)

**Training süreci:**
1. Her videodan sequence'ler çıkarılır (16 frame)
2. Her frame: optical flow + pose heatmap
3. Autoencoder normal pattern'leri öğrenir
4. Reconstruction error minimize edilir
5. Threshold otomatik hesaplanır (95th percentile)

### 3. Model Kullanımı

#### Programatik Kullanım

```python
from src.detection.panic.convlstm_model import PanicConvLSTMDetector

# Detector oluştur
detector = PanicConvLSTMDetector(
    model_path="models/convlstm_panic.pt",
    device="cpu",
    threshold=0.1,  # training'den gelen threshold
    sequence_length=16,
    image_size=(64, 64),
)

# Her frame için
for frame in video:
    # Optical flow hesapla
    flow_mag, flow_angle = compute_optical_flow(frame)
    
    # Pose heatmap oluştur
    pose_heatmap = create_pose_heatmap(detections)
    
    # Panic tespiti
    result = detector.add_frame(flow_mag, flow_angle, pose_heatmap)
    
    if result is not None:
        is_panic, error = result
        if is_panic:
            print(f"PANIC! Reconstruction error: {error:.4f}")
```

## Training İpuçları

### 1. Data Collection

**Yeterli veri:**
- Minimum 10-20 normal video
- Her video 1-2 dakika
- Farklı açılar ve aydınlatma
- Farklı kalabalık yoğunlukları

**Video özellikleri:**
- Normal yürüyüş hareketleri
- Sakin kalabalık akışı
- Düzenli hareket pattern'leri

### 2. Hyperparameter Tuning

**Sequence length:**
- 16 frame: Hızlı, kısa pattern'ler
- 32 frame: Daha uzun temporal pattern'ler
- Trade-off: Memory vs temporal coverage

**Image size:**
- 64x64: Hızlı, düşük memory
- 128x128: Daha detaylı, yavaş
- Önerilen: 64x64 (yeterli)

**Hidden dimensions:**
- [32, 64, 32]: Hafif model
- [64, 128, 64]: Daha güçlü
- [32, 64, 128, 64, 32]: Çok derin

**Learning rate:**
- 0.001: Başlangıç
- ReduceLROnPlateau ile otomatik ayarlama

### 3. Threshold Calibration

Training sonunda otomatik hesaplanan threshold:
- 95th percentile of reconstruction errors
- Normal videoların %95'i bu threshold'un altında
- Daha strict: 90th percentile
- Daha loose: 98th percentile

Manuel ayarlama:
```python
detector.threshold = 0.15  # Daha strict
detector.threshold = 0.05  # Daha loose
```

## Performans Optimizasyonu

### GPU Kullanımı

```bash
python scripts/train_convlstm_panic.py \
    --videos "data/normal_videos/*.mp4" \
    --output models/convlstm_panic.pt \
    --device cuda:0 \
    --batch-size 16 \
    --epochs 100
```

### Model Boyutu

**Hafif model (CPU için):**
```python
model = PanicConvLSTMAutoencoder(
    input_channels=3,
    hidden_dims=[16, 32, 16],
    kernel_size=(3, 3),
)
```

**Güçlü model (GPU için):**
```python
model = PanicConvLSTMAutoencoder(
    input_channels=3,
    hidden_dims=[64, 128, 64],
    kernel_size=(3, 3),
)
```

## Embeddings vs ConvLSTM

### JSON Embeddings (Mevcut)

**Avantajlar:**
- ✅ Training gerektirmiyor
- ✅ Hızlı başlangıç
- ✅ Anlaşılır (statistical features)
- ✅ Hafif (JSON file)

**Dezavantajlar:**
- ❌ Manuel feature engineering
- ❌ Temporal patterns sınırlı
- ❌ Spatial-temporal ilişkiler yok

### ConvLSTM (Yeni)

**Avantajlar:**
- ✅ Otomatik feature learning
- ✅ Güçlü temporal modeling
- ✅ Spatial-temporal features
- ✅ End-to-end learning

**Dezavantajlar:**
- ❌ Training gerekiyor
- ❌ GPU önerilen
- ❌ Daha büyük model

## Troubleshooting

### Training loss düşmüyor

- Daha fazla data toplayın
- Learning rate azaltın
- Model kapasitesini artırın (hidden_dims)
- Daha fazla epoch

### Reconstruction error çok yüksek

- Threshold'u artırın
- Model daha fazla eğitilmeli
- Daha fazla normal video ekleyin

### Memory error

- Batch size azaltın
- Image size küçültün (64 → 32)
- Sequence length azaltın (16 → 8)

### Çok yavaş inference

- Model boyutunu küçültün
- Image size azaltın
- CPU yerine GPU kullanın

## Örnek Workflow

```bash
# 1. Normal videolar topla
python scripts/download_youtube_videos.py \
    --url "PLAYLIST_URL" \
    --output data/normal_videos \
    --label normal

# 2. Model eğit
python scripts/train_convlstm_panic.py \
    --videos "data/normal_videos/*.mp4" \
    --output models/convlstm_panic.pt \
    --epochs 50 \
    --batch-size 4

# 3. Test et (panic.py'ye entegre edilecek)
python -m src.detection.panic.panic \
    --model weights/yolo11x.pt \
    --source data/test.mp4 \
    --use-convlstm \
    --convlstm-model models/convlstm_panic.pt
```

## Model Dosyası

Training sonunda oluşan `.pt` dosyası şunları içerir:
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state
- `epoch`: Training epoch
- `train_loss`: Training loss
- `val_loss`: Validation loss
- `threshold`: Optimal anomaly threshold

## İleri Seviye

### Custom Architecture

```python
from src.detection.panic.convlstm_model import PanicConvLSTMAutoencoder

model = PanicConvLSTMAutoencoder(
    input_channels=3,
    hidden_dims=[64, 128, 256, 128, 64],  # Daha derin
    kernel_size=(5, 5),  # Daha büyük receptive field
    num_layers=5,
)
```

### Data Augmentation

Training script'e eklenebilir:
- Random crop
- Random flip
- Brightness/contrast adjustment
- Temporal augmentation (speed up/down)

### Ensemble

Birden fazla model kullan:
```python
models = [
    PanicConvLSTMDetector("models/model1.pt"),
    PanicConvLSTMDetector("models/model2.pt"),
    PanicConvLSTMDetector("models/model3.pt"),
]

errors = [model.add_frame(...)[1] for model in models]
avg_error = np.mean(errors)
is_panic = avg_error > threshold
```

## Notlar

- ConvLSTM training GPU'da çok daha hızlı
- Minimum 500-1000 sequence önerilen
- Overfitting'i önlemek için validation kullanın
- Threshold'u validation set üzerinde ayarlayın
- Production'da model.eval() modunda kullanın
