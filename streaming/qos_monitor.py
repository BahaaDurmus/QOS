"""
qos_monitor.py  -  QoS Metrik Toplayici ve Analiz Modulu
=========================================================
QoS-Based Video Streaming Optimization in Wireless Mesh Networks
Asama 3: QoS Tabanli Mekanizma

Bu modul:
  1. WMN simulatorunden gelen metrikleri okur (JSON dosyasi veya UDP)
  2. Gercek zamanli olarak asagidaki QoS parametrelerini hesaplar/izler:
       - Throughput (Veri Hacmi) - kbps
       - Packet Loss Rate (Paket Kayip Orani) - %
       - Delay / Latency (Gecikme) - ms
       - Jitter (Titreme) - ms
       - PSNR / SSIM (Goruntu Kalitesi)
  3. QoS durumuna gore otomatik kalite kademesi secimi (ABR - Adaptive Bitrate):
       Iyi ag   → Yuksek kalite (JPEG 85, 720p)
       Orta ag  → Orta kalite  (JPEG 65, 480p)
       Kotu ag  → Dusuk kalite (JPEG 40, 360p)  ← VSR modeli bu asamada devreye girer!
  4. Metrikleri CSV'ye kaydeder (grafik icin)
  5. Sunum dashboard'una JSON besleme yapabilir

Kullanim:
  python qos_monitor.py --metrics output/wmn_metrics.json --watch
"""

import argparse
import csv
import json
import math
import time
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

try:
    from skimage.metrics import structural_similarity as ssim_func
    from skimage.metrics import peak_signal_noise_ratio as psnr_func
    SKIMAGE_OK = True
except ImportError:
    SKIMAGE_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# Kalite Kademeleri (ABR – Adaptive Bitrate)
# ─────────────────────────────────────────────────────────────────────────────

QUALITY_LEVELS = {
    "HIGH": {
        "label":       "Yuksek Kalite (720p)",
        "jpeg_quality": 85,
        "scale":        1.0,
        "description":  "Ag durumu iyi, tam cozunurluk",
    },
    "MEDIUM": {
        "label":       "Orta Kalite (480p)",
        "jpeg_quality": 65,
        "scale":        0.75,
        "description":  "Ag orta, kalite dusuruluyor",
    },
    "LOW": {
        "label":       "Dusuk Kalite (360p) + VSR",
        "jpeg_quality": 40,
        "scale":        0.50,
        "description":  "Ag kotu! Dusuk bitrate + VSR Super-Resolution aktif",
    },
    "CRITICAL": {
        "label":       "Kritik Kalite (240p) + VSR",
        "jpeg_quality": 20,
        "scale":        0.33,
        "description":  "Ag kritik! Minimal veri + VSR Super-Resolution aktif",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# QoS Karar Motoru
# ─────────────────────────────────────────────────────────────────────────────

def decide_quality(loss_pct: float, delay_ms: float, bandwidth_kbps: float) -> str:
    """
    Anlik QoS parametrelerine gore kalite kademesi secer.

    Kural tablosu:
      loss < 2%  AND delay < 30ms  AND bw > 5000 kbps → HIGH
      loss < 8%  AND delay < 80ms  AND bw > 1500 kbps → MEDIUM
      loss < 20% AND delay < 200ms AND bw > 400 kbps  → LOW
      else                                              → CRITICAL
    """
    if loss_pct < 2.0 and delay_ms < 30.0 and (bandwidth_kbps == 0 or bandwidth_kbps > 5000):
        return "HIGH"
    elif loss_pct < 8.0 and delay_ms < 80.0 and (bandwidth_kbps == 0 or bandwidth_kbps > 1500):
        return "MEDIUM"
    elif loss_pct < 20.0 and delay_ms < 200.0 and (bandwidth_kbps == 0 or bandwidth_kbps > 400):
        return "LOW"
    else:
        return "CRITICAL"


# ─────────────────────────────────────────────────────────────────────────────
# Goruntu Kalitesi Metrik Hesaplayici
# ─────────────────────────────────────────────────────────────────────────────

def compute_psnr(original: np.ndarray, restored: np.ndarray) -> float:
    """PSNR hesaplar (dB). Yuksek = daha iyi. Ideal: > 40dB."""
    if SKIMAGE_OK:
        return float(psnr_func(original, restored, data_range=255))
    mse = np.mean((original.astype(float) - restored.astype(float)) ** 2)
    if mse == 0:
        return 100.0
    return float(20 * math.log10(255.0 / math.sqrt(mse)))


def compute_ssim(original: np.ndarray, restored: np.ndarray) -> float:
    """SSIM hesaplar (0-1). Yuksek = daha iyi. Ideal: > 0.9."""
    if not SKIMAGE_OK:
        # Basit piksel korelasyon tahmini
        o = original.astype(float) / 255.0
        r = restored.astype(float) / 255.0
        return float(np.corrcoef(o.ravel(), r.ravel())[0, 1])
    if original.ndim == 3:
        return float(ssim_func(original, restored, channel_axis=2, data_range=255))
    return float(ssim_func(original, restored, data_range=255))


# ─────────────────────────────────────────────────────────────────────────────
# QoS Monitor Sinifi
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QoSSnapshot:
    timestamp:          float = 0.0
    loss_pct:           float = 0.0
    delay_ms:           float = 0.0
    jitter_ms:          float = 0.0
    bandwidth_kbps:     float = 0.0
    quality_level:      str   = "HIGH"
    vsr_active:         bool  = False
    psnr_before:        float = 0.0
    psnr_after:         float = 0.0
    ssim_before:        float = 0.0
    ssim_after:         float = 0.0


class QoSMonitor:
    """
    Gercek zamanli QoS izleme ve adaptasyon motoru.
    """

    HISTORY_SIZE = 300  # Son 300 olcum sakla (grafik icin)

    def __init__(self, metrics_path: Path, csv_out: Path | None = None):
        self.metrics_path = metrics_path
        self.csv_out      = csv_out
        self.history: deque[QoSSnapshot] = deque(maxlen=self.HISTORY_SIZE)
        self._prev_delay_ms: float | None = None

        if csv_out:
            csv_out.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = open(str(csv_out), "w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow([
                "timestamp", "loss_pct", "delay_ms", "jitter_ms",
                "bandwidth_kbps", "quality_level", "vsr_active",
                "psnr_before", "psnr_after", "ssim_before", "ssim_after"
            ])
        else:
            self._csv_file   = None
            self._csv_writer = None

    def read_wmn_metrics(self) -> dict | None:
        """WMN simulatorunun JSON dosyasindan metrikleri okur."""
        try:
            with open(str(self.metrics_path)) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def update(self, raw: dict) -> QoSSnapshot:
        """Ham WMN metriklerinden QoS snapshot olusturur."""
        loss       = float(raw.get("current_loss_rate", 0.0))
        delay      = float(raw.get("avg_delay_ms", 0.0))
        bandwidth  = float(raw.get("bandwidth_kbps", 0.0))

        # Jitter hesapla: gecen gecikme ile aradaki fark
        jitter = 0.0
        if self._prev_delay_ms is not None:
            jitter = abs(delay - self._prev_delay_ms)
        self._prev_delay_ms = delay

        level     = decide_quality(loss, delay, bandwidth)
        vsr_active = level in ("LOW", "CRITICAL")

        snap = QoSSnapshot(
            timestamp      = float(raw.get("timestamp", time.time())),
            loss_pct       = loss,
            delay_ms       = delay,
            jitter_ms      = jitter,
            bandwidth_kbps = bandwidth,
            quality_level  = level,
            vsr_active     = vsr_active,
        )
        self.history.append(snap)

        if self._csv_writer:
            self._csv_writer.writerow([
                datetime.fromtimestamp(snap.timestamp).strftime("%H:%M:%S"),
                f"{snap.loss_pct:.2f}",
                f"{snap.delay_ms:.1f}",
                f"{snap.jitter_ms:.1f}",
                f"{snap.bandwidth_kbps:.0f}",
                snap.quality_level,
                int(snap.vsr_active),
                f"{snap.psnr_before:.2f}",
                f"{snap.psnr_after:.2f}",
                f"{snap.ssim_before:.4f}",
                f"{snap.ssim_after:.4f}",
            ])
            if self._csv_file:
                self._csv_file.flush()

        return snap

    def add_image_metrics(
        self,
        snap: QoSSnapshot,
        original: np.ndarray,
        corrupted: np.ndarray,
        restored: np.ndarray,
    ) -> QoSSnapshot:
        """Goruntu kalite metriklerini snapshot'a ekler."""
        snap.psnr_before = compute_psnr(original, corrupted)
        snap.psnr_after  = compute_psnr(original, restored)
        snap.ssim_before = compute_ssim(original, corrupted)
        snap.ssim_after  = compute_ssim(original, restored)
        return snap

    def print_dashboard(self, snap: QoSSnapshot):
        """Terminale guzel bir QoS durumu tablosu yazdirir."""
        level_info = QUALITY_LEVELS.get(snap.quality_level, {})
        vsr_tag    = "[VSR AKTIF]" if snap.vsr_active else ""

        print("\n" + "─" * 60)
        print(f"  QoS MONITOR  |  {datetime.fromtimestamp(snap.timestamp).strftime('%H:%M:%S')}")
        print("─" * 60)
        print(f"  Paket Kaybi     : {snap.loss_pct:6.1f}%")
        print(f"  Gecikme         : {snap.delay_ms:6.1f} ms")
        print(f"  Jitter          : {snap.jitter_ms:6.1f} ms")
        print(f"  Bant Genisligi  : {snap.bandwidth_kbps:6.0f} kbps")
        print("─" * 60)
        print(f"  Kalite Kademesi : {snap.quality_level}  {vsr_tag}")
        print(f"  Aciklama        : {level_info.get('description', '')}")
        if snap.psnr_after > 0:
            print("─" * 60)
            print(f"  PSNR Oncesi (Bozuk)   : {snap.psnr_before:.2f} dB")
            print(f"  PSNR Sonrasi (VSR)    : {snap.psnr_after:.2f} dB  (+{snap.psnr_after - snap.psnr_before:.2f})")
            print(f"  SSIM Oncesi (Bozuk)   : {snap.ssim_before:.4f}")
            print(f"  SSIM Sonrasi (VSR)    : {snap.ssim_after:.4f}  (+{snap.ssim_after - snap.ssim_before:.4f})")
        print("─" * 60, flush=True)

    def get_summary_json(self) -> str:
        """Son 60 olcumun ozet istatistiklerini JSON olarak dondurur (dashboard icin)."""
        recent = list(self.history)[-60:]
        if not recent:
            return "{}"

        avg_loss  = sum(s.loss_pct       for s in recent) / len(recent)
        avg_delay = sum(s.delay_ms       for s in recent) / len(recent)
        avg_jitter= sum(s.jitter_ms      for s in recent) / len(recent)
        avg_bw    = sum(s.bandwidth_kbps for s in recent) / len(recent)
        vsr_pct   = sum(1 for s in recent if s.vsr_active) / len(recent) * 100
        level     = recent[-1].quality_level

        # PSNR/SSIM (eger hesaplandi ise)
        psnr_samples = [s for s in recent if s.psnr_after > 0]
        avg_psnr_b   = sum(s.psnr_before for s in psnr_samples) / max(1, len(psnr_samples))
        avg_psnr_a   = sum(s.psnr_after  for s in psnr_samples) / max(1, len(psnr_samples))
        avg_ssim_b   = sum(s.ssim_before for s in psnr_samples) / max(1, len(psnr_samples))
        avg_ssim_a   = sum(s.ssim_after  for s in psnr_samples) / max(1, len(psnr_samples))

        return json.dumps({
            "quality_level":   level,
            "avg_loss_pct":    round(avg_loss, 2),
            "avg_delay_ms":    round(avg_delay, 1),
            "avg_jitter_ms":   round(avg_jitter, 1),
            "avg_bw_kbps":    round(avg_bw, 0),
            "vsr_active_pct":  round(vsr_pct, 1),
            "psnr_before":     round(avg_psnr_b, 2),
            "psnr_after":      round(avg_psnr_a, 2),
            "ssim_before":     round(avg_ssim_b, 4),
            "ssim_after":      round(avg_ssim_a, 4),
            "history": [
                {
                    "t":    s.timestamp,
                    "loss": s.loss_pct,
                    "del":  s.delay_ms,
                    "bw":   s.bandwidth_kbps,
                    "lvl":  s.quality_level,
                }
                for s in recent
            ],
        }, indent=2)

    def close(self):
        if self._csv_file:
            self._csv_file.close()


# ─────────────────────────────────────────────────────────────────────────────
# CLI – Izleme modu
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="QoS Metrik Monitoru – Asama 3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--metrics", default="output/wmn_metrics.json", help="WMN metrik JSON dosyasi")
    ap.add_argument("--csv",     default="output/qos_log.csv",      help="Cikti CSV dosyasi")
    ap.add_argument("--watch",   action="store_true",                help="Surekli izleme modu")
    ap.add_argument("--interval",type=float, default=2.0,           help="Okuma araligi (saniye)")
    args = ap.parse_args()

    monitor = QoSMonitor(
        metrics_path = Path(args.metrics),
        csv_out      = Path(args.csv),
    )

    print(f"[QoS Monitor] Basladi – Metrik dosyasi: {args.metrics}")
    print(f"[QoS Monitor] CSV cikti: {args.csv}")

    if not args.watch:
        raw = monitor.read_wmn_metrics()
        if raw:
            snap = monitor.update(raw)
            monitor.print_dashboard(snap)
            print(monitor.get_summary_json())
        else:
            print(f"[QoS Monitor] Metrik dosyasi bulunamadi: {args.metrics}")
        monitor.close()
        return

    # Surekli izleme modu
    print("[QoS Monitor] Surekli izleme basliyor (Ctrl+C ile dur)...")
    try:
        while True:
            raw = monitor.read_wmn_metrics()
            if raw:
                snap = monitor.update(raw)
                monitor.print_dashboard(snap)
            else:
                print(f"[QoS Monitor] Bekleniyor: {args.metrics}", flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[QoS Monitor] Durduruldu.")
    finally:
        monitor.close()


if __name__ == "__main__":
    main()
