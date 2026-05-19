---
marp: true
theme: default
paginate: true
size: 16:9
lang: tr
title: QoS Tabanlı Video Akışı — Sunum
description: Kablosuz Örgüsel Ağlarda QoS Tabanlı Video Akışı Optimizasyonu
---

<!-- _class: lead -->
# Kablosuz Örgüsel Ağlarda
# QoS Tabanlı Video Akışı Optimizasyonu

**Gerçek zamanlı UDP akışı · WMN simülasyonu · Yapay zeka ile video onarımı (VSR)**

*Adınız Soyadınız · Bölüm · Tarih: …*

---

## Sunum planı

1. Problem ve motivasyon
2. Proje hedefi
3. Sistem mimarisi (4 aşama)
4. Teknolojiler
5. Canlı demo
6. Ölçüm sonuçları
7. Sonuç ve gelecek çalışmalar

> Bu slaytı kendi sunum akışınıza göre düzenleyin.

---

## Problem

- Kablosuz mesh / WMN ortamlarında **paket kaybı**, **gecikme** ve **jitter** video kalitesini düşürür.
- Eksik UDP paketleri → JPEG karelerde **siyah bantlar** ve bozulmalar.
- Klasik çözümler: bitrate düşürme (ABR) — görüntü bulanıklaşır, detay kaybı artar.

**Soru:** Ağ kötüleşince hem veri tasarrufu hem de **kabul edilebilir görüntü** nasıl sağlanır?

---

## Proje hedefi

| Hedef | Açıklama |
|-------|----------|
| **Simülasyon** | WMN benzeri kayıp/gecikme ile gerçekçi test ortamı |
| **QoS izleme** | Kayıp, gecikme, bant genişliği → kalite kademesi |
| **VSR onarım** | 15 karelik pencere ile bozuk bölgeleri yapay zeka ile düzeltme |
| **Sunum** | Canlı web dashboard ile metrik ve görüntü izleme |

---

## Sistem mimarisi — genel bakış

```
[input.mp4] → UDP Sunucu :9998 → WMN Proxy → İstemci :9999 → VSR → Dashboard :8080
```

- Tek komut: `run_full_demo.py`
- Tüm bileşenler **paralel süreçler** olarak çalışır
- Metrikler `output/` altında JSON/CSV olarak toplanır

---

## Aşama 1 — Video sunucusu

**Dosya:** `streaming/stream_server.py`

- MP4 → JPEG sıkıştırma
- Kareler **60 KB** parçalara bölünür
- Her paket: **12 bayt header** + veri
- Port **9998** üzerinden UDP gönderimi

```
[frame_id][total_chunks][chunk_id][data_len] | JPEG verisi
```

---

## Aşama 2 — WMN simülatörü

**Dosya:** `streaming/wmn_simulator.py`

- UDP **proxy**: 9998 dinler → 9999’a iletir
- **Paket kaybı** (%5–30, profile göre)
- **Gecikme + jitter** (ör. 50 ms ± 20 ms)
- İsteğe bağlı: `wifi_simulator.py` (802.11n/ac/ax)

| Profil | Kayıp | Gecikme |
|--------|-------|---------|
| medium | ~%5 | 50 ms |
| poor | ~%15 | 150 ms |
| critical | ~%30 | 300 ms |

---

## Aşama 3 — QoS monitör

**Dosya:** `streaming/qos_monitor.py`

- `output/wmn_metrics.json` okur
- **ABR** kalite kademesi: HIGH / MEDIUM / LOW / CRITICAL
- LOW ve CRITICAL → **VSR aktif** kararı
- Dashboard ve terminalde canlı gösterim

---

## Aşama 4 — İstemci + VSR

**Dosya:** `streaming/stream_client.py`

- Eksik chunk → karede **siyah blok** (gerçek kayıp etkisi)
- **15 kare** kayan pencere → `VSRModel` (8 residual block)
- Onarım **async thread**’de; akış bloklanmaz
- `latest_raw.jpg` / `latest_restored.jpg` → canlı önizleme

**Model:** `model_sharp_ep10_fixed.pth` (~2.4 MB)

---

## VSR modeli (kısa)

- Giriş: 15 × 3 kanal = **45 kanal** (tek tensor)
- Çıkış: 1 onarılmış RGB kare
- Mimari: Conv → 8× ResidualBlock → Conv
- Eğitim verisi: WMN benzeri bozulmuş kareler (`restoration/`)

---

## Web dashboard (sunum arayüzü)

| Sayfa | URL | İçerik |
|-------|-----|--------|
| Ana panel | `http://localhost:8080` | Canlı metrikler + iki video |
| Pipeline | `/pipeline.html` | 4 aşama sayaçları + log |
| Karşılaştırma | `/comparison.html` | Bozuk vs onarılmış (frame’ler) |

Ön hazırlık: `compare_models.py` + `make_web_frames.py`

---

## Canlı demo — başlatma

```bash
cd QOS-master
pip install torch torchvision opencv-python pillow numpy tqdm

python run_full_demo.py --video input.mp4 --model model_sharp_ep10_fixed.pth --profile medium
```

Tarayıcı: **http://localhost:8080**

> Model yüklenene kadar 5–10 sn bekleyin: `[Client] VSR async thread basladi`

---

## Canlı demo — sunum akışı (öneri)

1. Terminalde paket kaybı ve **LOW [VSR AKTIF]** göster
2. Dashboard: sol **bozuk**, sağ **onarılmış** kare
3. `pipeline.html`: gönderilen / düşürülen paket sayacı
4. `comparison.html`: PSNR/SSIM kartları (önceden üretilmiş frame’ler)

**Dramatik profil:** `--profile poor` veya `critical`

---

## Ölçüm sonuçları (örnek)

| Metrik | Bozuk | VSR sonrası | Kazanım |
|--------|-------|-------------|---------|
| **PSNR** | 22.02 dB | 25.81 dB | **+3.79 dB** |
| **SSIM** | 0.7706 | 0.9137 | **+0.143** |

- ~6 dB ≈ 2× SNR iyileşmesi
- SSIM 0.77 → 0.91: algısal kalitede belirgin artış

> Kendi ölçümlerinizi `output/metrics_report.txt` ile güncelleyin.

---

## Proje klasör yapısı

```
QOS-master/
├── run_full_demo.py      # Tek komut demo
├── streaming/            # Akış + WMN + QoS + istemci
├── restoration/          # Offline onarım, metrik, veri seti
├── presentation/         # Web UI + bu sunum
└── output/               # Metrikler, videolar, frame’ler
```

---

## Katkılar / sınırlamalar

**Güçlü yönler**
- Uçtan uca çalışan prototip + web arayüzü
- Modüler mimari (simülatör / QoS / VSR ayrı)

**Sınırlamalar** *(düzenleyin)*
- Yerel loopback (gerçek donanım WMN değil)
- VSR gecikmesi (15 kare penceresi)
- Tek video / tek model ile test

---

## Gelecek çalışmalar

- [ ] Gerçek WMN test yatağı / Raspberry Pi mesh
- [ ] Daha hafif VSR veya donanım hızlandırma (GPU/TensorRT)
- [ ] QoS → sunucuya geri bildirim (bitrate/FPS kontrolü)
- [ ] Çoklu istemci ve multicast senaryoları

*Bu maddeleri kendi tezinize göre değiştirin.*

---

## Sorun giderme

| Sorun | Çözüm |
|-------|--------|
| Port meşgul | `taskkill /F /IM python.exe` |
| Görüntü yok | 15–20 sn VSR ısınması |
| comparison boş | `make_web_frames.py` çalıştırın |
| Model bulunamadı | `.pth` dosyası proje kökünde mi? |

---

<!-- _class: lead -->
# Teşekkürler

## Sorular?

**İletişim:** e-posta@universite.edu  
**Repo / demo:** `http://localhost:8080` (canlı sunumda)

---

## Ek: Referanslar ve notlar

- UDP video chunking protokolü: README § Protokol Detayları
- WiFi modu: `run_full_demo.py --wifi --standard 802.11ac`
- Bağlantı testi (modelsiz): `python streaming/test_stream_no_model.py`

*Kaynak makaleleri ve danışman notlarınızı buraya ekleyin.*
