# Panic Detection with Anomaly Detection

Bu dokümantasyon, panic detection modülünde **anomaly detection** yaklaşımının nasıl kullanılacağını açıklar.

## Genel Bakış

**Anomaly Detection Yaklaşımı**: Sadece **normal motion pattern'leri** öğreniyoruz. Panic, bu normal pattern'lerden sapma olarak tespit ediliyor.

Bu yaklaşım iki aşamadan oluşur:

1. **Offline Feature Extraction**: SADECE normal videolardan embedding'ler çıkarılır
2. **Online Anomaly Detection**: Runtime'da mevcut motion pattern, normal pattern'lerden ne kadar sapıyor hesaplanır
   - Düşük sapma = Normal hareket
   - Yüksek sapma = Anomali = **PANIC**

## Adım 1: Normal Videolar Topla

**ÖNEMLİ**: Sadece **normal motion** videoları toplayın. Panic videoları GEREKMEZ!

```
data/
└── normal_videos/         # Normal kalabalık hareketleri
    ├── normal_001.mp4     # Alışveriş merkezi
    ├── normal_002.mp4     # Havaalanı
    ├── normal_003.mp4     # Tren istasyonu
    └── ...
```

**Normal video örnekleri:**
- Alışveriş merkezlerinde yürüyen insanlar
- Havaalanı/tren istasyonunda normal akış
- Etkinliklerde sakin kalabalık
- Caddelerde normal yaya trafiği

## Adım 2: Normal Embedding'leri Çıkar

Sadece normal videolardan embedding çıkarın:

```bash
python scripts/extract_panic_embeddings.py \
    --videos "data/normal_videos/*.mp4" \
    --output data/embeddings/normal_embeddings.json \
    --model yolov8n-pose.pt \
    --device cpu \
    --window-size 30
```

Daha fazla normal video eklemek için (append mode):

```bash
python scripts/extract_panic_embeddings.py \
    --videos "data/more_normal_videos/*.mp4" \
    --output data/embeddings/normal_embeddings.json \
    --model yolov8n-pose.pt \
    --device cpu \
    --window-size 30 \
    --append
```

### Parametreler:

- `--videos`: Video dosya yolları (glob pattern destekler)
- `--output`: Çıktı JSON dosyası
- `--model`: YOLO pose model yolu
- `--device`: İnference cihazı (`cpu`, `cuda:0`, vb.)
- `--window-size`: Her embedding için frame sayısı (varsayılan: 30)
- `--append`: Mevcut database'e ekle (yeni database oluşturmak yerine)

## Adım 3: Embedding-Based Detection'ı Aktifleştirme

### Programatik olarak:

```python
from src.detection.panic.panic import YoloPoseFlowFusionPanic, FusionPanicConfig

config = FusionPanicConfig(
    use_embeddings=True,
    embedding_db_path="data/embeddings/normal_embeddings.json",
    embedding_window_size=30,
    embedding_weight=0.5,  # 0-1 arası, anomaly skorunun ağırlığı
    anomaly_threshold=1.0,  # Anomaly threshold (yüksek = daha strict)
)

detector = YoloPoseFlowFusionPanic(fps=30.0, config=config)
```

### main.py ile:

`main.py` dosyasında config'i güncelleyin veya yeni bir argüman ekleyin.

## Çıkarılan Feature'lar

Her embedding şu feature'ları içerir:

### Flow-based features:
- `flow_mag_mean`: Ortalama optical flow magnitude
- `flow_mag_std`: Flow magnitude standart sapması
- `flow_mag_p50/p95/p99`: Flow magnitude percentile'ları

### Direction entropy features:
- `direction_entropy_mean`: Ortalama hareket yönü entropisi
- `direction_entropy_std`: Entropi standart sapması
- `direction_entropy_max`: Maksimum entropi

### People count features:
- `people_count_mean`: Ortalama insan sayısı
- `people_count_std`: İnsan sayısı standart sapması
- `people_count_max`: Maksimum insan sayısı

### Motion density features:
- `roi_flow_mean`: Person ROI'larındaki ortalama flow
- `roi_flow_std`: ROI flow standart sapması
- `roi_flow_p95`: ROI flow 95. percentile

### Temporal features:
- `flow_acceleration`: Flow magnitude'deki değişim hızı
- `entropy_acceleration`: Entropi'deki değişim hızı

## Anomaly Metrics

Runtime'da her embedding için şu anomaly metrikleri hesaplanır:

- `reconstruction_error`: Normal pattern'den L2 uzaklığı
- `mahalanobis_distance`: Normalize edilmiş uzaklık (std ile)
- `nearest_neighbor_distance`: En yakın normal pattern'e uzaklık
- `knn_distance`: 5 en yakın normal pattern'e ortalama uzaklık
- `max_z_score`: Feature'lar arasında maksimum z-score
- `anomaly_score`: Composite anomaly skoru (0-∞, yüksek = daha anomalous)

## Score Fusion

Final panic score şu şekilde hesaplanır:

```
final_score = (1 - w) * base_score + w * (anomaly_score * 5)
```

Burada:
- `base_score`: Robust z-score tabanlı skor (optical flow + entropy)
- `anomaly_score`: Normal pattern'lerden sapma skoru
- `w`: `embedding_weight` parametresi (0-1 arası)

**Yüksek anomaly score = Normal pattern'den uzak = PANIC**

## Tuning Parametreleri

### `embedding_weight` (0-1):
- **0.0**: Sadece z-score tabanlı detection
- **0.5**: Dengeli fusion (önerilen)
- **1.0**: Sadece embedding-based detection

### `score_threshold`:
- Varsayılan: 4.0
- Embedding kullanırken 3.0-5.0 arası test edin

### `embedding_window_size`:
- Varsayılan: 30 frame (~1 saniye @ 30fps)
- Daha uzun window = daha stabil ama daha yavaş tepki
- Daha kısa window = daha hızlı tepki ama daha gürültülü

## Örnek Kullanım

```python
import cv2
from src.detection.panic.panic import YoloPoseFlowFusionPanic, FusionPanicConfig
from src.detection.utils.detection import load_model, detect_people

# Config
config = FusionPanicConfig(
    use_embeddings=True,
    embedding_db_path="data/embeddings/panic_embeddings.json",
    embedding_weight=0.5,
)

# Load model
model = load_model("yolov8n-pose.pt")

# Open video
cap = cv2.VideoCapture("test_video.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

# Create detector
detector = YoloPoseFlowFusionPanic(fps=fps, config=config)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect people
    detections = detect_people(model, frame)
    
    # Update panic detector
    result = detector.update(frame, detections)
    
    # Check result
    if result.is_panic:
        print(f"PANIC! Score: {result.score:.2f}")
        if "emb_score" in result.metrics:
            print(f"  Embedding score: {result.metrics['emb_score']:.3f}")
            print(f"  Panic similarity: {result.metrics['emb_panic_sim']:.3f}")
            print(f"  Normal similarity: {result.metrics['emb_normal_sim']:.3f}")

cap.release()
```

## Troubleshooting

### "Embedding database not found"
- Embedding dosyasının doğru yolda olduğundan emin olun
- `embedding_db_path` parametresini kontrol edin

### Çok fazla false positive
- `score_threshold` değerini artırın
- `embedding_weight` değerini azaltın
- Daha fazla normal video ile eğitin

### Çok az detection
- `score_threshold` değerini azaltın
- `embedding_weight` değerini artırın
- Daha fazla panic video ile eğitin

### Yavaş performans
- `embedding_window_size` değerini artırın (daha az sık comparison)
- GPU kullanın (`--device cuda:0`)
