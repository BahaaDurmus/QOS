"""
wmn_simulator.py  -  Wireless Mesh Network (WMN) Simulatoru
============================================================
QoS-Based Video Streaming Optimization in Wireless Mesh Networks
Asama 2: Kablosuz Ag Simulasyonu

Bu modul, bir UDP PROXY olarak sunucu ile istemci arasinda calisir:

    [Server :9999] --> [WMN Simulator :9998] --> [Client :9999]

Gercek bir kablosuz ag gibi davranir:
  - Paket Kaybi  (Packet Loss)   : Her paketin %X ihtimalle dusturulmesi
  - Gecikme      (Delay)         : Paketlere rastgele gecikme eklenmesi (ms)
  - Titreme      (Jitter)        : Gecikmenin dalgalanmasi
  - Bant Daralt. (Throttle)      : Maksimum bant genisligi sinirlamasi (kbps)
  - Ani Bozulma  (Burst Loss)    : Gercek WMN'lerdeki ani paket kayip patlamalari

QoS Profilleri:
  --profile good      : Iyi baglanti  (%1 kayip, 10ms gecikme)
  --profile medium    : Orta baglanti (%5 kayip, 50ms gecikme)
  --profile poor      : Kotu baglanti (%15 kayip, 150ms gecikme)
  --profile critical  : Kritik       (%30 kayip, 300ms gecikme)

Kullanim:
  python wmn_simulator.py --listen 9998 --forward 127.0.0.1:9999 --profile medium
"""

import argparse
import socket
import struct
import threading
import time
import random
import json
import queue
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# QoS Profilleri
# ─────────────────────────────────────────────────────────────────────────────

QOS_PROFILES = {
    "perfect": {
        "packet_loss":   0.00,
        "delay_ms":      0.0,
        "jitter_ms":     0.0,
        "bandwidth_kbps": 0,        # Limitsiz
        "burst_loss_prob": 0.00,
        "burst_loss_len":  0,
    },
    "good": {
        "packet_loss":   0.01,      # %1  kayip
        "delay_ms":      10.0,
        "jitter_ms":     5.0,
        "bandwidth_kbps": 0,        # Limitsiz (sadece kayip/gecikme)
        "burst_loss_prob": 0.005,
        "burst_loss_len":  3,
    },
    "medium": {
        "packet_loss":   0.05,      # %5  kayip
        "delay_ms":      50.0,
        "jitter_ms":     20.0,
        "bandwidth_kbps": 0,        # Limitsiz (kayip orani yeterli)
        "burst_loss_prob": 0.02,
        "burst_loss_len":  5,
    },
    "poor": {
        "packet_loss":   0.15,      # %15 kayip
        "delay_ms":      150.0,
        "jitter_ms":     50.0,
        "bandwidth_kbps": 0,        # Limitsiz
        "burst_loss_prob": 0.05,
        "burst_loss_len":  8,
    },
    "critical": {
        "packet_loss":   0.30,      # %30 kayip
        "delay_ms":      300.0,
        "jitter_ms":     100.0,
        "bandwidth_kbps": 0,        # Limitsiz
        "burst_loss_prob": 0.10,
        "burst_loss_len":  12,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Metrik Veri Sinifi
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WMNMetrics:
    timestamp:         float = field(default_factory=time.time)
    packets_received:  int   = 0
    packets_forwarded: int   = 0
    packets_dropped:   int   = 0
    bytes_forwarded:   int   = 0
    avg_delay_ms:      float = 0.0
    current_loss_rate: float = 0.0
    bandwidth_kbps:    float = 0.0   # Anlik bant genisligi
    profile:           str   = "medium"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


# ─────────────────────────────────────────────────────────────────────────────
# WMN Simulatoru
# ─────────────────────────────────────────────────────────────────────────────

class WMNSimulator:
    """
    UDP Proxy olarak calisir:
      Client --> [listen_port] --> WMN Simulator --> [forward_host:forward_port] --> Server

    WMN etkileri paket iletimi sirasinda uygulanir.
    """

    # Metrik JSON'u yazmak icin kontrol portu
    METRICS_PATH = Path("output/wmn_metrics.json")

    def __init__(
        self,
        listen_port:  int,
        forward_host: str,
        forward_port: int,
        profile:      str    = "medium",
        log_interval: float  = 2.0,    # istatistik yazma araligi (saniye)
    ):
        self.listen_port  = listen_port
        self.forward_host = forward_host
        self.forward_port = forward_port
        self.log_interval = log_interval

        # Profil yukle
        if profile not in QOS_PROFILES:
            raise ValueError(f"Gecersiz profil: {profile}. Secenekler: {list(QOS_PROFILES)}")
        self.profile_name = profile
        self.params       = dict(QOS_PROFILES[profile])

        # Soketler
        self._sock_in  = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock_in.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
        self._sock_in.settimeout(1.0)
        self._sock_in.bind(("", listen_port))

        # Bant genisligi kontrolu
        self._bw_lock       = threading.Lock()
        self._bw_bucket     = float(self.params["bandwidth_kbps"] * 1000 / 8) if self.params["bandwidth_kbps"] > 0 else float("inf")
        self._bw_last_refill = time.time()

        # Burst kayip durumu
        self._in_burst     = False
        self._burst_remain = 0

        # Metrikler
        self._metrics = WMNMetrics(profile=profile)
        self._delays: list[float] = []

        # Durdurma
        self._stop = threading.Event()

        # Gecikmeli iletim kuyrugu: (gonderiminiz_zamani, data)
        self._delay_queue: queue.PriorityQueue = queue.PriorityQueue()

        print(f"[WMN] Simulasyon basladi: Profil='{profile}'")
        print(f"      Paket Kaybi   : {self.params['packet_loss']*100:.1f}%")
        print(f"      Gecikme       : {self.params['delay_ms']:.0f} ms (+- {self.params['jitter_ms']:.0f} ms jitter)")
        bw = self.params['bandwidth_kbps']
        print(f"      Bant Genisligi: {bw if bw > 0 else 'Limitsiz'} kbps")
        print(f"[WMN] Dinleniyor: 0.0.0.0:{listen_port}  -->  {forward_host}:{forward_port}")

    # ── Bant Genisligi Token Bucket ────────────────────────────────────────────
    def _check_bandwidth(self, num_bytes: int) -> bool:
        """Token bucket ile bant genisligi sinirlamasi. True = gecirebilir."""
        if self.params["bandwidth_kbps"] <= 0:
            return True
        with self._bw_lock:
            now    = time.time()
            refill = (now - self._bw_last_refill) * self.params["bandwidth_kbps"] * 1000 / 8
            self._bw_bucket = min(
                self._bw_bucket + refill,
                self.params["bandwidth_kbps"] * 1000 / 8,  # max = 1 saniyelik bant
            )
            self._bw_last_refill = now
            if self._bw_bucket >= num_bytes:
                self._bw_bucket -= num_bytes
                return True
            return False  # Bant doldu → paketin dusturulecek

    # ── Paket Kaybi Kararı ─────────────────────────────────────────────────────
    def _should_drop(self) -> bool:
        """Bu paketin dusturulup dusturulmeyecegine karar verir."""
        # Burst kayip durumu
        if self._in_burst:
            self._burst_remain -= 1
            if self._burst_remain <= 0:
                self._in_burst = False
            return True

        # Ani burst baslatma
        if random.random() < self.params["burst_loss_prob"]:
            self._in_burst     = True
            self._burst_remain = random.randint(1, self.params["burst_loss_len"])
            return True

        # Normal rastgele kayip
        return random.random() < self.params["packet_loss"]

    # ── Gecikme Hesapla ────────────────────────────────────────────────────────
    def _calc_delay(self) -> float:
        """ms cinsinden gecikme degerini hesaplar (Gaussian jitter ile)."""
        base    = self.params["delay_ms"]
        jitter  = self.params["jitter_ms"]
        delay   = max(0.0, random.gauss(base, jitter / 2))
        return delay / 1000.0  # saniyeye cevir

    # ── Gecikmeli Iletici Thread ───────────────────────────────────────────────
    def _delay_forwarder(self):
        """Priority queue'den vakti gelen paketleri iletir."""
        while not self._stop.is_set():
            try:
                send_at, data = self._delay_queue.get(timeout=0.01)
            except queue.Empty:
                continue
            now = time.time()
            if send_at > now:
                time.sleep(send_at - now)
            try:
                self._sock_out.sendto(data, (self.forward_host, self.forward_port))
                self._metrics.bytes_forwarded += len(data)
            except Exception:
                pass

    # ── Metrik Yazici Thread ───────────────────────────────────────────────────
    def _metrics_writer(self):
        """Belirli aralikla metrikleri JSON dosyasina yazar."""
        self.METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        t0          = time.time()
        bytes_prev  = 0
        while not self._stop.is_set():
            time.sleep(self.log_interval)
            now = time.time()

            # Anlik bant genisligi (kbps)
            bytes_now = self._metrics.bytes_forwarded
            dt        = now - t0
            if dt > 0:
                self._metrics.bandwidth_kbps = (bytes_now * 8) / dt / 1000

            # Kayip orani
            total = self._metrics.packets_received
            if total > 0:
                self._metrics.current_loss_rate = (
                    self._metrics.packets_dropped / total * 100
                )

            # Ortalama gecikme
            if self._delays:
                self._metrics.avg_delay_ms = (
                    sum(self._delays[-200:]) / len(self._delays[-200:]) * 1000
                )

            self._metrics.timestamp = now

            # Dosyaya yaz (sunumun dashboard'u bunu okur)
            with open(self.METRICS_PATH, "w") as f:
                json.dump(self._metrics.to_dict(), f, indent=2)

            print(
                f"[WMN] Iletilen: {self._metrics.packets_forwarded}  "
                f"Dusurulen: {self._metrics.packets_dropped}  "
                f"Kayip: {self._metrics.current_loss_rate:.1f}%  "
                f"Bant: {self._metrics.bandwidth_kbps:.0f} kbps  "
                f"Gecikme: {self._metrics.avg_delay_ms:.1f} ms",
                flush=True,
            )

    # ── Ana Dongu ──────────────────────────────────────────────────────────────
    def run(self):
        # Thread'leri baslat
        t_fwd     = threading.Thread(target=self._delay_forwarder, daemon=True)
        t_metrics = threading.Thread(target=self._metrics_writer,  daemon=True)
        t_fwd.start()
        t_metrics.start()

        try:
            while not self._stop.is_set():
                try:
                    data, addr = self._sock_in.recvfrom(65536 + 32)
                except socket.timeout:
                    continue

                self._metrics.packets_received += 1

                # Bant genisligi kontrolu
                if not self._check_bandwidth(len(data)):
                    self._metrics.packets_dropped += 1
                    continue

                # Paket kaybi kontrolu
                if self._should_drop():
                    self._metrics.packets_dropped += 1
                    continue

                # Gecikme hesapla ve kuyruga ekle
                delay = self._calc_delay()
                self._delays.append(delay)
                send_at = time.time() + delay
                self._delay_queue.put((send_at, data))
                self._metrics.packets_forwarded += 1

        except KeyboardInterrupt:
            print("\n[WMN] Kullanici tarafindan durduruldu.", flush=True)
        finally:
            self._stop.set()
            self._sock_in.close()
            self._sock_out.close()
            print("[WMN] Simulasyon bitti.", flush=True)
            # Son metrikleri yazdir
            total = self._metrics.packets_received
            loss  = self._metrics.packets_dropped / max(1, total) * 100
            print(f"      Toplam Paket    : {total}")
            print(f"      Iletilen        : {self._metrics.packets_forwarded}")
            print(f"      Dusurulen       : {self._metrics.packets_dropped}  ({loss:.1f}%)")

    def update_profile(self, profile: str):
        """Calisma zamaninda profil degistirme."""
        if profile not in QOS_PROFILES:
            return
        self.params = dict(QOS_PROFILES[profile])
        self.profile_name = profile
        print(f"[WMN] Profil degistirildi: {profile}", flush=True)

    def stop(self):
        self._stop.set()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="WMN UDP Simulatoru – Asama 2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--listen",   type=int,   default=9998,        help="Dinlenecek UDP portu (sunucudan bu porta gonderin)")
    ap.add_argument("--forward",  default="127.0.0.1:9999",        help="Iletilecek adres host:port (istemci bu portu dinler)")
    ap.add_argument("--profile",  default="medium",
                    choices=list(QOS_PROFILES),                     help="QoS profili")
    ap.add_argument("--log-interval", type=float, default=2.0,     help="Metrik yazma araligi (saniye)")
    args = ap.parse_args()

    host, port_str = args.forward.rsplit(":", 1)
    sim = WMNSimulator(
        listen_port  = args.listen,
        forward_host = host,
        forward_port = int(port_str),
        profile      = args.profile,
        log_interval = args.log_interval,
    )
    sim.run()


if __name__ == "__main__":
    main()
