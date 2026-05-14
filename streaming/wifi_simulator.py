"""
wifi_simulator.py  -  802.11 WiFi Kanal Simulatoru
===================================================
QoS-Based Video Streaming Optimization in Wireless Mesh Networks
Asama 2: Gercekci Kablosuz Ag Simulasyonu

Modeller:
  - Free Space Path Loss (FSPL) - mesafe bazli sinyal zayiflamasi
  - Log-normal Shadowing       - engel ve yansima etkisi
  - Rayleigh Fading            - cok yollu yayilim
  - SNR -> MCS secimi          - 802.11ac MCS tablosu
  - SNR -> PER egri            - gercekci paket hata orani
  - CSMA/CA Backoff            - WiFi erisim gecikmesi
  - Kanal girisim              - diger cihazlar
  - Hareket simulasyonu        - mesafe degisimi

Kullanim:
  python wifi_simulator.py --listen 9998 --forward 127.0.0.1:9999 --standard 802.11ac --freq 5 --distance 15
"""

import argparse, socket, threading, time, random, json, math, queue
from dataclasses import dataclass, field, asdict
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 802.11 MCS Tablosu (802.11ac, 1 uzaysal akis, 80 MHz kanal)
# SNR esigi (dB) ve teorik throughput (Mbps)
# ─────────────────────────────────────────────────────────────────────────────
MCS_TABLE_AC = [
    # (mcs_idx, min_snr_dB, modulation,    coding, throughput_mbps)
    (0,   5,  "BPSK",   "1/2",  32.5),
    (1,  10,  "QPSK",   "1/2",  65.0),
    (2,  15,  "QPSK",   "3/4",  97.5),
    (3,  20,  "16-QAM", "1/2", 130.0),
    (4,  25,  "16-QAM", "3/4", 195.0),
    (5,  30,  "64-QAM", "2/3", 260.0),
    (6,  32,  "64-QAM", "3/4", 292.5),
    (7,  35,  "64-QAM", "5/6", 325.0),
    (8,  40, "256-QAM", "3/4", 390.0),
    (9,  45, "256-QAM", "5/6", 433.3),
]

# 802.11n MCS (20 MHz, 1 stream)
MCS_TABLE_N = [
    (0,   5, "BPSK",   "1/2",  7.2),
    (1,  10, "QPSK",   "1/2", 14.4),
    (2,  15, "QPSK",   "3/4", 21.7),
    (3,  20, "16-QAM", "1/2", 28.9),
    (4,  25, "16-QAM", "3/4", 43.3),
    (5,  30, "64-QAM", "2/3", 57.8),
    (6,  35, "64-QAM", "3/4", 65.0),
    (7,  40, "64-QAM", "5/6", 72.2),
]

MCS_TABLES = {"802.11n": MCS_TABLE_N, "802.11ac": MCS_TABLE_AC, "802.11ax": MCS_TABLE_AC}

# ─────────────────────────────────────────────────────────────────────────────
# Fiziksel Kanal Modeli
# ─────────────────────────────────────────────────────────────────────────────

class WiFiChannel:
    """
    Gercekci 802.11 WiFi kanal modeli.
    """
    NOISE_FLOOR_DBM = -95.0   # Tipik WiFi gurultu zemini

    def __init__(self, freq_ghz: float = 5.0, standard: str = "802.11ac",
                 distance_m: float = 10.0, n_interferers: int = 3):
        self.freq_ghz      = freq_ghz
        self.standard      = standard
        self.distance_m    = distance_m         # Baslangic mesafesi
        self.n_interferers = n_interferers      # Kanal girisim kaynaklari
        self.mcs_table     = MCS_TABLES.get(standard, MCS_TABLE_AC)

        # Rayleigh fading durumu
        self._fading_phase = random.uniform(0, 2 * math.pi)
        self._fading_speed = 0.05   # Hz (yuruyus hizi)

        # Mesafe dalgalanmasi (hareket sim)
        self._dist_drift   = 0.0
        self._dist_speed   = 0.02   # m/s

    # ── ITU Indoor Path Loss (dB) ─────────────────────────────────────────────
    def _fspl_db(self, d: float) -> float:
        """
        ITU-R P.1238 Ic Mekan Path Loss Modeli:
        PL = 20*log10(f_MHz) + N*log10(d) + Lf - 28
        N = 28 (ofis/ic mekan uzaklik katsayisi)
        Lf = 0 (tek kat, engel yok)
        """
        f_mhz = self.freq_ghz * 1000
        N = 28  # ofis ic mekan katsayisi
        return 20 * math.log10(f_mhz) + N * math.log10(max(d, 0.5)) - 28

    # ── Log-normal Shadowing ─────────────────────────────────────────────────
    def _shadowing_db(self) -> float:
        """Engel ve yansima etkisi — sigma=8 dB tipik ic mekan"""
        return random.gauss(0, 8.0)

    # ── Rayleigh Fading ───────────────────────────────────────────────────────
    def _rayleigh_db(self) -> float:
        """Rayleigh fading: cok yollu yayilim — tipik +-10 dB"""
        t = time.time()
        self._fading_phase += self._fading_speed * 0.01
        # Iki bagimsiz Gaussian -> Rayleigh
        i = math.cos(2 * math.pi * self._fading_speed * t + self._fading_phase)
        q = math.sin(2 * math.pi * self._fading_speed * t)
        amp = math.sqrt(i**2 + q**2) / math.sqrt(2)
        return 20 * math.log10(max(amp, 1e-6))

    # ── Kanal Girisimi ────────────────────────────────────────────────────────
    def _interference_dbm(self) -> float:
        """Diger WiFi cihazlarindan girisim gucu (uzak/zayif cihazlar)"""
        if self.n_interferers == 0:
            return -999.0
        # Gercekci girisimci gucu: -85 ile -70 dBm arasi
        total = sum(10 ** (random.uniform(-90, -75) / 10)
                    for _ in range(self.n_interferers))
        return 10 * math.log10(total + 1e-20)

    # ── RSSI Hesapla ─────────────────────────────────────────────────────────
    def compute_rssi(self) -> float:
        """Anlik RSSI (dBm) — AP guc: +20 dBm (100 mW tipik)"""
        tx_power_dbm = 20.0
        # Mesafeyi hafifce degistir (hareket)
        self._dist_drift += random.gauss(0, self._dist_speed)
        self._dist_drift  = max(-5, min(5, self._dist_drift))
        d = max(1.0, self.distance_m + self._dist_drift)

        rssi = (tx_power_dbm
                - self._fspl_db(d)
                - abs(self._shadowing_db()) * 0.3   # hafifletilmis
                + self._rayleigh_db() * 0.5)
        return round(rssi, 1)

    # ── SNR Hesapla ───────────────────────────────────────────────────────────
    def compute_snr(self, rssi_dbm: float) -> float:
        interf = self._interference_dbm()
        noise  = 10 ** (self.NOISE_FLOOR_DBM / 10)
        intf   = 10 ** (interf / 10)
        signal = 10 ** (rssi_dbm / 10)
        snr_linear = signal / (noise + intf + 1e-30)
        return round(10 * math.log10(max(snr_linear, 1e-10)), 1)

    # ── MCS Sec ───────────────────────────────────────────────────────────────
    def select_mcs(self, snr_db: float) -> dict:
        """SNR'a gore en iyi MCS indeksini sec"""
        best = self.mcs_table[0]
        for entry in self.mcs_table:
            if snr_db >= entry[1]:
                best = entry
        return {
            "index":      best[0],
            "modulation": best[2],
            "coding":     best[3],
            "throughput": best[4],
        }

    # ── Paket Hata Orani (PER) ────────────────────────────────────────────────
    def compute_per(self, snr_db: float, mcs: dict) -> float:
        """
        SNR bazli PER — sigmoid egri modeli.
        MCS'e gore esik noktasi degisir.
        PER = 1 / (1 + exp(k*(SNR - SNR_threshold)))
        """
        snr_thresh = self.mcs_table[mcs["index"]][1]  # Bu MCS'in min SNR esigi
        k = 1.2  # egri dikligi
        per = 1.0 / (1.0 + math.exp(k * (snr_db - snr_thresh)))
        return round(min(per, 0.99), 4)

    # ── CSMA/CA Backoff Gecikmesi ─────────────────────────────────────────────
    def csma_ca_delay_ms(self, load_factor: float = 0.5) -> float:
        """
        802.11 CSMA/CA rastgele erteleme.
        Kanal yogunlugu artinca bekleme suresi uzar.
        CW = [CWmin, CWmax] = [15, 1023] slot
        Slot suresi: 9 us (5 GHz), 20 us (2.4 GHz)
        """
        slot_us = 9.0 if self.freq_ghz >= 5.0 else 20.0
        cw_min, cw_max = 15, 1023
        # Kanal yogunlugu artinca CW buyur
        cw = int(cw_min + (cw_max - cw_min) * load_factor)
        backoff_slots = random.randint(0, cw)
        difs_us = 3 * slot_us  # DIFS = 2 slot + SIFS
        total_us = difs_us + backoff_slots * slot_us
        # Kanal mesgulse ekstra bekleme
        if random.random() < load_factor * 0.3:
            total_us += random.uniform(500, 5000)  # us
        return total_us / 1000.0  # ms'e cevir

    # ── Tam Metrik Hesapla ────────────────────────────────────────────────────
    def step(self) -> dict:
        """Bir zaman adiminda tum WiFi metriklerini hesapla"""
        rssi = self.compute_rssi()
        snr  = self.compute_snr(rssi)
        mcs  = self.select_mcs(snr)
        per  = self.compute_per(snr, mcs)
        load = random.uniform(0.2, 0.8)  # Kanal yogunlugu tahmini
        csma = self.csma_ca_delay_ms(load)
        return {
            "rssi_dbm":         rssi,
            "snr_db":           snr,
            "mcs_index":        mcs["index"],
            "mcs_modulation":   mcs["modulation"],
            "mcs_coding":       mcs["coding"],
            "throughput_mbps":  mcs["throughput"],
            "per":              per,
            "csma_delay_ms":    round(csma, 2),
            "channel_load_pct": round(load * 100, 1),
            "freq_ghz":         self.freq_ghz,
            "standard":         self.standard,
            "distance_m":       round(self.distance_m + self._dist_drift, 1),
        }


# ─────────────────────────────────────────────────────────────────────────────
# WiFi Simulatoru (UDP Proxy)
# ─────────────────────────────────────────────────────────────────────────────

class WiFiSimulator:
    METRICS_PATH = Path("output/wmn_metrics.json")

    def __init__(self, listen_port: int, forward_host: str, forward_port: int,
                 standard: str = "802.11ac", freq_ghz: float = 5.0,
                 distance_m: float = 15.0, n_interferers: int = 3,
                 log_interval: float = 2.0):

        self.listen_port  = listen_port
        self.forward_host = forward_host
        self.forward_port = forward_port
        self.log_interval = log_interval

        self.channel = WiFiChannel(freq_ghz, standard, distance_m, n_interferers)

        self._sock_in  = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock_in.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
        self._sock_in.settimeout(1.0)
        self._sock_in.bind(("", listen_port))

        self._stop         = threading.Event()
        self._delay_queue  = queue.PriorityQueue()

        # Sayaçlar
        self._rx   = 0
        self._fwd  = 0
        self._drop = 0
        self._bytes_fwd = 0
        self._delays = []

        # Mevcut kanal durumu
        self._wifi_state = {}
        self._start_time = time.time()

        print(f"\n{'='*60}")
        print(f"  WiFi Simulatoru Baslatildi")
        print(f"  Standard  : {standard}")
        print(f"  Frekans   : {freq_ghz} GHz")
        print(f"  Mesafe    : {distance_m} m")
        print(f"  Girisimci : {n_interferers} cihaz")
        print(f"  Dinliyor  : 0.0.0.0:{listen_port} --> {forward_host}:{forward_port}")
        print(f"{'='*60}\n")

    def _delay_forwarder(self):
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
                self._bytes_fwd += len(data)
            except Exception:
                pass

    def _metrics_writer(self):
        self.METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        while not self._stop.is_set():
            time.sleep(self.log_interval)
            now = time.time()

            # Kanal adimi
            w = self.channel.step()
            self._wifi_state = w

            # QoS karar
            snr = w["snr_db"]
            if snr >= 35:   ql = "HIGH"
            elif snr >= 25: ql = "MEDIUM"
            elif snr >= 15: ql = "LOW"
            else:           ql = "CRITICAL"

            total = self._rx
            loss  = (self._drop / total * 100) if total > 0 else 0.0
            bw    = (self._bytes_fwd * 8 / (now - self._start_time) / 1000) if now > self._start_time else 0.0
            avg_d = (sum(self._delays[-200:]) / len(self._delays[-200:]) * 1000) if self._delays else 0.0

            metrics = {
                # Temel metrikler (dashboard uyumlulugu)
                "timestamp":         now,
                "packets_received":  self._rx,
                "packets_forwarded": self._fwd,
                "packets_dropped":   self._drop,
                "bytes_forwarded":   self._bytes_fwd,
                "avg_delay_ms":      round(avg_d, 1),
                "current_loss_rate": round(loss, 1),
                "bandwidth_kbps":    round(bw, 1),
                "quality_level":     ql,
                "profile":           "wifi",
                "jitter_ms":         round(w["csma_delay_ms"], 1),
                # WiFi'ye ozgu metrikler
                "wifi": w,
            }

            with open(self.METRICS_PATH, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)

            print(
                f"[WiFi] RSSI:{w['rssi_dbm']:+.0f}dBm  SNR:{w['snr_db']:.1f}dB  "
                f"MCS{w['mcs_index']}({w['mcs_modulation']})  "
                f"PER:{w['per']*100:.1f}%  "
                f"Kayip:{loss:.1f}%  BW:{bw:.0f}kbps  Kalite:{ql}",
                flush=True
            )

    def run(self):
        threading.Thread(target=self._delay_forwarder, daemon=True).start()
        threading.Thread(target=self._metrics_writer,  daemon=True).start()

        try:
            while not self._stop.is_set():
                try:
                    data, addr = self._sock_in.recvfrom(65536 + 32)
                except socket.timeout:
                    continue

                self._rx += 1

                # PER'e gore dusur
                w   = self._wifi_state or self.channel.step()
                per = w.get("per", 0.05)
                if random.random() < per:
                    self._drop += 1
                    continue

                # CSMA/CA + propagation gecikmesi
                delay_ms = w.get("csma_delay_ms", 5.0) + random.gauss(2, 1)
                delay_s  = max(0, delay_ms / 1000.0)
                self._delays.append(delay_s)
                self._delay_queue.put((time.time() + delay_s, data))
                self._fwd += 1

        except KeyboardInterrupt:
            print("\n[WiFi] Durduruldu.")
        finally:
            self._stop.set()
            self._sock_in.close()
            self._sock_out.close()

    def stop(self):
        self._stop.set()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="802.11 WiFi Kanal Simulatoru")
    ap.add_argument("--listen",       type=int,   default=9998)
    ap.add_argument("--forward",      default="127.0.0.1:9999")
    ap.add_argument("--standard",     default="802.11ac",
                    choices=["802.11n", "802.11ac", "802.11ax"])
    ap.add_argument("--freq",         type=float, default=5.0,
                    help="Frekans bandi: 2.4 veya 5 (GHz)")
    ap.add_argument("--distance",     type=float, default=15.0,
                    help="AP'ye mesafe (metre)")
    ap.add_argument("--interferers",  type=int,   default=3,
                    help="Kanal girisimci cihaz sayisi")
    ap.add_argument("--log-interval", type=float, default=2.0)
    args = ap.parse_args()

    host, port_str = args.forward.rsplit(":", 1)
    sim = WiFiSimulator(
        listen_port   = args.listen,
        forward_host  = host,
        forward_port  = int(port_str),
        standard      = args.standard,
        freq_ghz      = args.freq,
        distance_m    = args.distance,
        n_interferers = args.interferers,
        log_interval  = args.log_interval,
    )
    sim.run()


if __name__ == "__main__":
    main()
