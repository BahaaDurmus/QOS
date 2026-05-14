# QoS-Based Video Streaming Optimization in Wireless Mesh Networks

> **Kablosuz Örgüsel Ağlarda QoS Tabanlı Video Akışı Optimizasyonu**

---

## Proje Yapısı

```
QOS-master/
│
├── streaming/                  ← Aşama 1: Video Akışı Altyapısı (Bu repo)
│   ├── stream_server.py        # UDP video streaming sunucusu
│   ├── stream_client.py        # UDP istemci + VSR anlık onarım
│   ├── run_streaming_demo.py   # Tek komutla sunucu+istemci başlatıcı
│   └── test_stream_no_model.py # Modelsiz bağlantı testi
│
├── restoration/                ← Aşama 4: Video Restorasyonu (Arkadaş)
│   ├── mp4_restorer.py         # VSR modeli ile video onarım aracı
│   ├── vsr_infer.py            # Model inference motoru
│   ├── make_corrupted_video.py # Bozuk video oluşturucu (test için)
│   ├── video_inference.py      # Video üzerinde inference
│   ├── inference_final.py      # Final inference scripti
│   ├── evaluate_metrics.py     # PSNR/SSIM metrik hesaplama
│   ├── extract_wmn_dataset.py  # Veri seti hazırlama
│   └── opencv_demo.py          # Basit OpenCV demo
│
└── .gitignore
```

---

## Aşamalar

| # | Aşama | Durum | Klasör |
|---|-------|-------|--------|
| 1 | **Video Streaming Altyapısı** (UDP soket) | ✅ Tamamlandı | `streaming/` |
| 2 | **Wireless Mesh Network Simülasyonu** (Paket kaybı) | 🔄 Yapılıyor | `streaming/` |
| 3 | **QoS Metrikleri** (Bant genişliği, gecikme, jitter) | ⏳ Bekliyor | — |
| 4 | **Video Restorasyonu** (VSR / Super-Resolution) | ✅ Tamamlandı | `restoration/` |

---

## Aşama 1 – Video Streaming Altyapısı

### Nasıl Çalışır?

```
[Video Dosyası]
      │
      ▼
[stream_server.py]  →  UDP Paketleri  →  [stream_client.py]
  Kareyi oku               Ağ               Parçaları birleştir
  JPEG sıkıştır        (Aşama 2'de          Bozuk kare tespiti
  Chunk'a böl          paket kaybı           VSR modeli ile onarım
  UDP ile gönder        eklenir)             Video'ya yaz + CSV log
```

### Protokol Detayları

Her UDP paketi **12 bayt header** + JPEG verisi içerir:

```
[frame_id: 4B] [total_chunks: 2B] [chunk_id: 2B] [data_len: 4B] | JPEG data
```

- **Max chunk boyutu:** 60 000 bayt (UDP limit altında güvenli)
- **Bitiş sinyali:** `frame_id = 0xFFFFFFFF`
- **Bozuk kare tespiti:** Eksik chunk → piksel bölgesi siyah blok olarak doldurulur

### Kurulum

```bash
pip install opencv-python torch torchvision Pillow numpy tqdm
```

### Kullanım

#### Seçenek 1 – Tek Komutla Demo (Önerilen)
```bash
cd streaming
python run_streaming_demo.py \
    --video ../input.mp4 \
    --model ../vsr_model_sharp_v2_ep10.pth \
    --fps 25 \
    --display
```

#### Seçenek 2 – İki Ayrı Terminal

**Terminal 1 (Sunucu):**
```bash
cd streaming
python stream_server.py --video ../input.mp4 --host 127.0.0.1 --port 9999 --fps 25
```

**Terminal 2 (İstemci):**
```bash
cd streaming
python stream_client.py --port 9999 --model ../vsr_model_sharp_v2_ep10.pth --display --out ../output/stream
```

#### Seçenek 3 – Modelsiz Bağlantı Testi
```bash
cd streaming
python test_stream_no_model.py
# Beklenen çıktı: 60/60 kare, %0 kayıp, [BASARILI]
```

### Çıktılar

```
output/stream/
├── raw_stream_20260513_123456.mp4       # Ham akış (bozuk kareler dahil)
└── restored_stream_20260513_123456.mp4  # VSR ile onarılmış akış
```

> Not: `.csv` istatistik dosyaları `.gitignore`'da dışlanmıştır.

### Parametreler

#### stream_server.py
| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `--video` | — | Kaynak .mp4 dosyası |
| `--webcam` | — | Webcam indeksi (0, 1, ...) |
| `--host` | `127.0.0.1` | Hedef IP |
| `--port` | `9999` | UDP hedef portu |
| `--fps` | `25.0` | Gönderim hızı |
| `--quality` | `80` | JPEG kalitesi (1-100) |
| `--loop` | kapalı | Videoyu döngüye sok |

#### stream_client.py
| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `--port` | `9999` | Dinlenecek UDP portu |
| `--model` | — | VSR checkpoint (.pth) |
| `--no-vsr` | kapalı | VSR modelini devre dışı bırak |
| `--out` | `output/stream` | Çıktı klasörü |
| `--display` | kapalı | OpenCV canlı pencere |
| `--out-fps` | `25.0` | Çıktı video FPS |

---

## Aşama 4 – Video Restorasyonu

> Bu aşama `restoration/` klasöründe bulunmaktadır.  
> Detaylı kullanım için o klasördeki kodlara bakınız.

**Model:** 15 karelik kayan pencere kullanan VSR (Video Super-Resolution) ağı  
**Mimari:** `ResidualBlock × 8` + `Conv2d` (45 giriş kanalı → 3 çıkış kanalı)

```bash
cd restoration

# Bozuk video oluştur
python make_corrupted_video.py --video input.mp4 --out-video output/corrupted.mp4 --packet-loss-prob 0.05

# Video onarımı
python vsr_infer.py --video output/corrupted.mp4 --out-video output/restored.mp4 --ckpt vsr_model.pth

# Metrik değerlendirme (PSNR / SSIM)
python evaluate_metrics.py
```

---

## Notlar

- Model ağırlık dosyaları (`.pth`) ve video dosyaları (`.mp4`) `.gitignore`'a eklenmiştir.  
  Büyük dosyalar için **Git LFS** kullanmanız önerilir.
- Aşama 2 (WMN Simülasyonu) tamamlandığında `streaming/` klasörüne eklenecektir.

---

## Lisans

Bu proje akademik amaçlıdır.
