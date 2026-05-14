# QoS-Based Video Streaming Optimization in Wireless Mesh Networks

> **Kablosuz Örgüsel Ağlarda QoS Tabanlı Video Akışı Optimizasyonu**  
> Gerçek zamanlı UDP video akışı + WMN paket kaybı simülasyonu + Yapay Zeka VSR onarımı

---

## 📋 Proje Yapısı

```
QOS-master/
│
├── run_full_demo.py            ← ✅ TEK KOMUTLA TÜM SİSTEMİ BAŞLATIR
├── compare_models.py           ← İki VSR modelini PSNR/SSIM ile karşılaştır
├── make_web_frames.py          ← Videoları tarayıcı uyumlu frame'lere çevir
├── process_video.py            ← Offline: boz + onar + kaydet
│
├── streaming/                  ← Aşama 1-3: Akış + WMN + QoS
│   ├── stream_server.py        # UDP video streaming sunucusu
│   ├── stream_client.py        # UDP istemci + async VSR onarım
│   ├── wmn_simulator.py        # Wireless Mesh Network paket kaybı simülatörü
│   ├── qos_monitor.py          # QoS metrik izleyici + ABR karar mekanizması
│   └── run_streaming_demo.py   # Demo başlatıcı
│
├── restoration/                ← Aşama 4: Video Restorasyonu
│   ├── mp4_restorer.py         # VSR modeli ile video onarım
│   ├── vsr_infer.py            # Model inference motoru
│   ├── make_corrupted_video.py # Bozuk video oluşturucu
│   ├── evaluate_metrics.py     # PSNR/SSIM metrik hesaplama
│   └── extract_wmn_dataset.py  # Veri seti hazırlama
│
└── presentation/               ← Web Dashboard (Sunum Arayüzü)
    ├── index.html              # Ana dashboard — canlı ağ metrikleri
    ├── pipeline.html           # 4 aşama canlı kanıt sayfası
    ├── comparison.html         # Bozuk vs VSR onarımlı video karşılaştırması
    └── server.py               # HTTP sunucu (localhost:8080)
```

---

## ⚙️ Sistem Mimarisi — 4 Aşama

```
[input.mp4]
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  AŞAMA 1: UDP Video Sunucusu (stream_server.py)                 │
│  → Videoyu JPEG'e sıkıştırır, chunk'a böler, Port 9998'e gönderir│
└─────────────────────┬───────────────────────────────────────────┘
                      │ UDP Paketleri
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  AŞAMA 2: WMN Simülatörü (wmn_simulator.py)                     │
│  → Paket kaybı (%5-30) + gecikme (50ms) + jitter ekler          │
│  → Port 9998 → Port 9999 proxy                                  │
└─────────────────────┬───────────────────────────────────────────┘
                      │ Bozulmuş UDP Paketleri
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  AŞAMA 3: QoS Monitör (qos_monitor.py)                          │
│  → Paket kaybı, gecikme, jitter, bant genişliği ölçer           │
│  → Kalite kademesi belirler: HIGH / MEDIUM / LOW / CRITICAL      │
│  → LOW/CRITICAL → VSR modeli otomatik devreye girer             │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  AŞAMA 4: VSR İstemci (stream_client.py)                        │
│  → Bozuk kareleri alır, async thread'de VSR modeli çalıştırır  │
│  → 15 kare pencere → onarılmış kare üretir                      │
│  → Dashboard için JPEG frame'leri günceller                     │
└─────────────────────────────────────────────────────────────────┘
                      │
                      ▼
              [Web Dashboard :8080]
```

---

## 📦 Kurulum

### Gereksinimler

```bash
pip install torch torchvision opencv-python pillow numpy tqdm
```

### Gerekli Dosyalar (GitHub'a push'lanmaz — elle koyulmalı)

| Dosya | Açıklama |
|---|---|
| `model_sharp_ep10_fixed.pth` | Eğitilmiş VSR modeli (2.4 MB) |
| `input.mp4` | Test videosu |

> Model `.pth` dosyası ve videolar `.gitignore`'da tanımlı olduğundan repoda yer almaz.  
> Her klonlamadan sonra bu dosyaları proje kök dizinine manuel olarak kopyalayın.

---

## ▶️ Hızlı Başlangıç

### ADIM 1 — Sistemi Başlat

```bash
cd QOS-master
python run_full_demo.py --video input.mp4 --model model_sharp_ep10_fixed.pth --profile medium
```

**Beklenen çıktı:**
```
============================================================
 [SISTEM] KABLOSUZ AGLARDA QOS VIDEO AKISI - SUNUM DEMOSU
============================================================
>> WMN Simulatoru Baslatiliyor (Profil: medium)
>> Istemci Baslatiliyor (Port 9999)
>> Video Sunucusu Baslatiliyor (Gonderim -> 9998)
>> QoS Monitoru Baslatiliyor
>> Web Dashboard Baslatiliyor (Port 8080)

============================================================
[OK] TUM SISTEM CALISIYOR!
[*] Sunum Dashboard'u icin tarayicida acin: http://localhost:8080
============================================================

[WMN] Paket Kaybi : 5.0%
[WMN] Gecikme     : 50 ms (+- 20 ms jitter)
[QoS] Kalite Kademesi : LOW [VSR AKTIF]
```

> ⏳ Model yüklenene kadar **5-10 saniye** bekle. Terminalde `[Client] VSR async thread basladi` yazısını görünce hazır.

---

### ADIM 2 — Tarayıcıda Şu 3 Sekmeyi Aç

---

#### 📊 Sekme 1: Ana Dashboard
**→ `http://localhost:8080`**

Ne görürsün:
- **Sol panel:** Canlı ağ metrikleri (paket kaybı %, gecikme ms, jitter ms, bant genişliği kbps)
- **Sağ üst:** Kalite kademesi rozetleri (HIGH 🟢 / MEDIUM 🟡 / LOW 🔴 / CRITICAL ⚫)
- **Orta:** İki canlı video görüntüsü — sol bozuk akış, sağ VSR onarımlı
- **Alt grafikler:** Zaman serisi metrik grafikleri

> ⏳ İlk 10-15 saniye görüntüler "Bekleniyor..." gösterebilir — model ısınıyor.  
> Görüntüler geldikten sonra sol videoda siyah bantlar (paket kayıpları), sağda temiz görüntü görünür.

---

#### 🔬 Sekme 2: 4 Aşama Pipeline Kanıtı
**→ `http://localhost:8080/pipeline.html`**

Ne görürsün:
- **Aşama 1 (Mavi kutu):** "Gönderilen Paket" sayacı — her saniye artıyor
- **Aşama 2 (Kırmızı kutu):** "Düşürülen Paket" + "Kayıp Oranı %" — WMN gerçekten paket düşürüyor
- **Aşama 3 (Sarı kutu):** Kalite kademesi + VSR kararı (AKTİF / PASİF)
- **Aşama 4 (Yeşil kutu):** Model adı ✅ + onarılan kare sayısı + FPS
- **Alt terminal:** Zaman damgalı canlı sistem logu

> 🔐 Bu sayfa projenin yapay olmadığının kanıtıdır:  
> Model dosyası adı, inference sayısı, paket sayaçları — hepsi gerçek zamanlı.

---

#### 🎬 Sekme 3: Video Karşılaştırma
**→ `http://localhost:8080/comparison.html`**

> ⏳ **İlk açılışta 2-3 dakika yükleme ekranı görünür** — bu normaldir.  
> 300 bozuk + 300 onarılmış kare (600 JPEG) tarayıcıya yükleniyor.  
> Yükleme tamamlandığında ekran kaybolur ve oynatıcı çıkar.

Ne görürsün:
- **4 metrik kartı (üst):**
  - 🔴 Bozuk Video PSNR: `22.0 dB`
  - ✅ VSR Sonrası PSNR: `25.8 dB`
  - ⬆ Kazanım: `+3.79 dB`
  - 🔷 SSIM: `0.9137`
- **Sol panel:** Bozulmuş video — siyah yatay bantlar (paket kaybı)
- **Sağ panel:** VSR onarılmış — bantlar giderilmiş, temiz görüntü
- **Alt kontroller:** ▶ Oynat / ⏮ Başa Sar / Kare sürgüsü / FPS seçici

Kullanım:
1. Yükleme bitmesini bekle
2. "▶ Oynat" butonuna bas → her iki panel aynı anda oynar
3. İstersen sürgüyle belirli bir kareye git
4. FPS'i düşürerek yavaş izle (`5 fps` seçimi önerilen)

---

### ADIM 3 — Kapatmak

```bash
# Terminalde Ctrl+C bas
# Veya:
taskkill /F /IM python.exe   # Windows
```

---

## 🧪 Video Karşılaştırma Ön Hazırlığı

Karşılaştırma sayfası için bozuk + onarılmış video çiftini üret:

```bash
# ADIM A: Her iki modeli test et, kazananla 3 video üret (~8-9 dakika)
python compare_models.py --video input.mp4 --loss 0.25 --resize 0.5

# Çıktılar:
# output/A_corrupted.mp4    → %25 paket kayıplı bozulmuş video
# output/B_restored.mp4     → VSR ile onarılmış video
# output/C_comparison.mp4   → Yan yana karşılaştırma (VLC ile açılabilir)
# output/metrics_report.txt → PSNR/SSIM raporu

# ADIM B: Tarayıcı uyumlu JPEG frame'lere çevir (~30 saniye)
python make_web_frames.py

# Çıktı:
# output/frames/corr/0000.jpg ... 0299.jpg
# output/frames/rest/0000.jpg ... 0299.jpg
```

> Bu adımı bir kez yapman yeterli. `output/frames/` klasörü oluştuktan sonra comparison.html çalışır.

---

## 🔧 Ağ Profili Seçenekleri

```bash
# İyi ağ — %5 kayıp, 50ms gecikme
python run_full_demo.py --video input.mp4 --model model_sharp_ep10_fixed.pth --profile medium

# Kötü ağ — %15 kayıp, 80ms gecikme (sunum için dramatik)
python run_full_demo.py --video input.mp4 --model model_sharp_ep10_fixed.pth --profile poor

# Kritik ağ — %30 kayıp (çok kötü)
python run_full_demo.py --video input.mp4 --model model_sharp_ep10_fixed.pth --profile critical
```

| Profil | Paket Kaybı | Gecikme | VSR |
|--------|-------------|---------|-----|
| `medium` | %5-15 | 50ms | Aktif |
| `poor` | %15-25 | 80ms | Aktif |
| `critical` | %25-35 | 120ms | Aktif |

---

## 📊 Sonuçlar

| Metrik | Bozuk Video | VSR Onarım | Kazanım |
|--------|-------------|------------|---------|
| **PSNR** | 22.02 dB | 25.81 dB | **+3.79 dB** |
| **SSIM** | 0.7706 | 0.9137 | **+0.1431** |

> Her 6 dB ≈ 2× sinyal-gürültü oranı iyileşmesi.  
> +3.79 dB ≈ %56 oranında piksel kalitesi artışı.  
> SSIM 0.77 → 0.91: insan görsel algısında %18 iyileşme.

---

## ❓ Sorun Giderme

| Sorun | Çözüm |
|-------|-------|
| `Port already in use` | `taskkill /F /IM python.exe` → tekrar çalıştır |
| Dashboard açılmıyor | 10 saniye bekle, F5 yenile |
| Görüntüler hep siyah | VSR ısınıyor, 15-20 saniye bekle |
| comparison.html boş | Yükleme ekranı geçene kadar bekle (2-3 dk) |
| `Model bulunamadi` | `model_sharp_ep10_fixed.pth` proje kök klasöründe mi? |
| `Video bulunamadi` | `input.mp4` proje kök klasöründe mi? |

---

## 📜 Protokol Detayları

Her UDP paketi **12 bayt header** + JPEG verisi içerir:

```
[frame_id: 4B] [total_chunks: 2B] [chunk_id: 2B] [data_len: 4B] | JPEG data
```

- **Max chunk boyutu:** 60.000 bayt
- **Bitiş sinyali:** `frame_id = 0xFFFFFFFF`
- **Bozuk kare tespiti:** Eksik chunk → piksel bölgesi siyah blok
- **VSR pencere boyutu:** 15 ardışık kare → 1 onarılmış kare

---

## Lisans

Bu proje akademik amaçlıdır.
