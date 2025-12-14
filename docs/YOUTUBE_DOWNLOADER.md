# YouTube Video Downloader

YouTube'dan video indirmek için kullanışlı bir script.

## Kurulum

```bash
pip install yt-dlp
```

veya

```bash
pip install -r requirements.txt
```

## Kullanım

### Tek Video İndirme

```bash
python scripts/download_youtube_videos.py \
    --url "https://www.youtube.com/watch?v=VIDEO_ID" \
    --output data/panic_videos
```

### Etiketli İndirme

Dosya isimlerine otomatik prefix ekler:

```bash
python scripts/download_youtube_videos.py \
    --url "https://www.youtube.com/watch?v=VIDEO_ID" \
    --output data/panic_videos \
    --label panic
```

Çıktı: `panic_Video_Title_VIDEO_ID.mp4`

### Playlist İndirme

```bash
python scripts/download_youtube_videos.py \
    --url "https://www.youtube.com/playlist?list=PLAYLIST_ID" \
    --output data/normal_videos \
    --label normal
```

### URL Listesinden İndirme

`urls.txt` dosyası oluştur:
```
https://www.youtube.com/watch?v=VIDEO_ID_1
https://www.youtube.com/watch?v=VIDEO_ID_2
# Bu satır yorum satırı
https://www.youtube.com/watch?v=VIDEO_ID_3
```

İndir:
```bash
python scripts/download_youtube_videos.py \
    --file urls.txt \
    --output data/videos \
    --label training
```

### Kalite Seçimi

```bash
python scripts/download_youtube_videos.py \
    --url "URL" \
    --output data/videos \
    --quality 720p
```

Kalite seçenekleri:
- `360p`: Düşük kalite, küçük dosya
- `480p`: Orta kalite
- `720p`: HD (varsayılan, önerilen)
- `1080p`: Full HD
- `best`: En yüksek kalite

### Video Kırpma

Belirli bir zaman aralığını indir:

```bash
python scripts/download_youtube_videos.py \
    --url "URL" \
    --output data/videos \
    --start-time 00:01:30 \
    --end-time 00:05:00
```

Zaman formatları:
- `HH:MM:SS` (örn: `00:01:30`)
- Saniye cinsinden (örn: `90`)

### Format Seçimi

```bash
python scripts/download_youtube_videos.py \
    --url "URL" \
    --output data/videos \
    --format mp4
```

## Panic Detection İçin Önerilen Workflow

### 1. Panic Videoları Topla

```bash
# Playlist'ten toplu indirme
python scripts/download_youtube_videos.py \
    --url "https://www.youtube.com/playlist?list=PANIC_PLAYLIST_ID" \
    --output data/panic_videos \
    --label panic \
    --quality 720p
```

### 2. Normal Videoları Topla

```bash
# Normal kalabalık videoları
python scripts/download_youtube_videos.py \
    --url "https://www.youtube.com/playlist?list=NORMAL_PLAYLIST_ID" \
    --output data/normal_videos \
    --label normal \
    --quality 720p
```

### 3. Embedding'leri Çıkar

```bash
# Normal videolar
python scripts/extract_panic_embeddings.py \
    --videos "data/normal_videos/*.mp4" \
    --label normal \
    --output data/embeddings/panic_embeddings.json

# Panic videolar
python scripts/extract_panic_embeddings.py \
    --videos "data/panic_videos/*.mp4" \
    --label panic \
    --output data/embeddings/panic_embeddings.json \
    --append
```

### 4. Modeli Çalıştır

```bash
python main.py --panic --source 0  # webcam
python main.py --panic --source test_video.mp4  # video file
```

## Örnek URL Kaynakları

### Panic Senaryoları
- Kalabalık panik videoları
- Acil durum tahliye videoları
- Stampede (izdiham) videoları
- Kaos senaryoları

### Normal Senaryoları
- Normal kalabalık videoları
- Alışveriş merkezi videoları
- Havaalanı/tren istasyonu videoları
- Etkinlik/konser videoları (sakin anlar)

## Troubleshooting

### "yt-dlp not installed"
```bash
pip install yt-dlp
```

### Video indirilemedi
- URL'nin doğru olduğundan emin olun
- Video'nun erişilebilir olduğunu kontrol edin
- Yaş kısıtlaması veya bölge kısıtlaması olabilir

### Yavaş indirme
- Daha düşük kalite seçin (`--quality 480p`)
- İnternet bağlantınızı kontrol edin

### Disk alanı yetersiz
- Gereksiz videoları silin
- Daha düşük kalite kullanın
- `--start-time` ve `--end-time` ile sadece gerekli kısmı indirin

## İpuçları

1. **Kalite vs Boyut**: 720p çoğu panic detection görevi için yeterlidir
2. **Veri Çeşitliliği**: Farklı açılar, aydınlatma ve kalabalık yoğunluğu toplayın
3. **Etiketleme**: Dosya isimlerinde label kullanarak organizasyon kolaylaşır
4. **Kırpma**: Uzun videolardan sadece ilgili kısımları indirin
5. **Batch İndirme**: URL listesi kullanarak toplu indirme yapın

## Parametreler

| Parametre | Açıklama | Varsayılan |
|-----------|----------|------------|
| `--url` | YouTube video/playlist URL | - |
| `--file` | URL listesi dosyası | - |
| `--output` | Çıktı dizini | - |
| `--label` | Dosya ismi prefix | None |
| `--quality` | Video kalitesi | 720p |
| `--format` | Video formatı | mp4 |
| `--start-time` | Başlangıç zamanı | None |
| `--end-time` | Bitiş zamanı | None |

## Notlar

- Script, yt-dlp kütüphanesini kullanır (youtube-dl'in daha hızlı ve güncel versiyonu)
- Playlist indirirken her video için ilerleme gösterilir
- Hata durumunda diğer videolar indirilmeye devam eder
- İndirilen dosyalar otomatik olarak mp4 formatına dönüştürülür
