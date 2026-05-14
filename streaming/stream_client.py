"""
stream_client.py  –  UDP Video Streaming Client + VSR Restoration
==================================================================
QoS-Based Video Streaming Optimization in Wireless Mesh Networks
Aşama 1: Video Akışı (Streaming) Altyapısı

Bu istemci:
  1. Belirlenen UDP portundan paket parçalarını (chunks) dinler.
  2. Parçaları birleştirerek tam JPEG karelerini elde eder.
  3. Eksik parça olan kareler bozuk (corrupted) olarak işaretlenir:
     - Aşama 2'de ağ simülatörü paketleri düşürdüğünde bu alan otomatik çalışır.
     - Eksik piksel bölgeleri siyah bloklar ile doldurulur (gerçek paket kayıbı etkisi).
  4. Alınan kareler 15 karelik kayan pencereye (sliding window) beslenir.
  5. VSRModel (mp4_restorer.py içindeki model mimarisi) anlık onarım yapar.
  6. Onarılmış ve orijinal kareler yan yana kaydedilir:
     - Çıktı video:  restored_stream_<timestamp>.mp4
     - İstatistik log: stream_stats_<timestamp>.csv

Kullanım:
  python stream_client.py --port 9999 --model vsr_model_sharp_v2_ep10.pth --out output/

Not: --no-vsr parametresi ile model devre dışı bırakılabilir (ham akış testi için).
"""

import argparse
import csv
import socket
import struct
import sys
import time
import threading
import queue
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import io

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

# ─────────────────────────────────────────────────────────────────────────────
# Sunucu ile aynı protokol sabitleri
# ─────────────────────────────────────────────────────────────────────────────
HEADER_FORMAT = "!IHHi"        # frame_id, total_chunks, chunk_id, data_len
HEADER_SIZE   = struct.calcsize(HEADER_FORMAT)
SOCKET_BUF    = 2 << 20        # 2 MB alım tamponu
RECV_TIMEOUT  = 5.0            # Saniye – bu süre geçince akış tamamlandı sayılır
CHUNK_TIMEOUT = 0.5            # Bir karenin tamamlanması için bekleme süresi (saniye)

# ─────────────────────────────────────────────────────────────────────────────
# VSR Model Mimarisi (mp4_restorer.py ile aynı)
# ─────────────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, n_feats: int = 64) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


class VSRModel(nn.Module):
    """15 giriş karesi → 1 onarılmış merkez kare (45 kanal giriş, 3 kanal çıkış)."""

    def __init__(self, in_ch: int = 45, feats: int = 64, n_blocks: int = 8) -> None:
        super().__init__()
        self.conv_first = nn.Conv2d(in_ch, feats, 3, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(feats) for _ in range(n_blocks)])
        self.conv_last  = nn.Conv2d(feats, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x = x.view(b, t * c, h, w)
        out = F.relu(self.conv_first(x))
        out = self.res_blocks(out)
        return self.conv_last(out)


# ─────────────────────────────────────────────────────────────────────────────
# VSR Yükleyici / Onarım Motoru
# ─────────────────────────────────────────────────────────────────────────────

def load_vsr_model(ckpt_path: Path, device: torch.device) -> nn.Module:
    """Checkpoint'ten model yükler; checkpoint yapısını otomatik algılar."""
    print(f"[VSR] Model yükleniyor: {ckpt_path}", flush=True)
    try:
        sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except TypeError:
        sd = torch.load(str(ckpt_path), map_location="cpu")

    if not isinstance(sd, dict):
        raise RuntimeError("Checkpoint bir state_dict değil.")

    # Hiperparametre çıkarımı
    w = sd.get("conv_first.weight")
    if w is None:
        raise RuntimeError("Checkpoint 'conv_first.weight' içermiyor.")
    in_ch  = int(w.shape[1])
    feats  = int(w.shape[0])
    n_blks = sum(1 for k in sd if k.startswith("res_blocks.") and ".conv.0.weight" in k)
    n_blks = n_blks if n_blks > 0 else 8

    model = VSRModel(in_ch=in_ch, feats=feats, n_blocks=n_blks)
    model.load_state_dict(sd, strict=True)
    model.eval()

    if device.type == "cuda":
        model = model.half()
        print("[VSR] Model FP16 modunda (GPU hızlandırması aktif).", flush=True)

    model.to(device)
    print(f"[VSR] Model hazır – in_ch={in_ch}, feats={feats}, blocks={n_blks}, device={device}", flush=True)
    return model


_to_tensor = T.Compose([T.ToTensor()])


def _frame_to_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    """BGR numpy → [0,1] RGB torch tensor (CHW)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return _to_tensor(Image.fromarray(rgb))  # (3, H, W)


def _tensor_to_bgr(t: torch.Tensor) -> np.ndarray:
    """[0,1] CHW torch tensor → BGR uint8 numpy."""
    arr = t.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    rgb = (arr * 255.0 + 0.5).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


@torch.no_grad()
def vsr_restore(model: nn.Module, window_tensors: list[torch.Tensor], device: torch.device) -> np.ndarray:
    """
    15 karelik pencereyi modele besler ve onarılmış merkez kareyi (BGR uint8) döndürür.

    Parameters
    ----------
    window_tensors : 15 adet (3, H, W) tensör
    """
    stack = torch.stack(window_tensors).unsqueeze(0)  # (1, 15, 3, H, W)
    if device.type == "cuda":
        stack = stack.half()
    stack = stack.to(device)

    output = model(stack).squeeze(0).cpu().float()  # (3, H, W)

    # Residual addition: model çıkışı + merkez kare
    center = window_tensors[7].float()
    output = (output + center).clamp(0, 1)

    return _tensor_to_bgr(output)


def apply_packet_loss_mask(frame_bgr: np.ndarray, missing_chunk_ids: set, total_chunks: int) -> np.ndarray:
    """
    Eksik parçaların karşılık geldiği piksel alanlarını siyah bloklar ile doldurur.
    Bu gerçek bir paket kaybının görsel etkisini simüle eder.
    """
    if not missing_chunk_ids:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    corrupted = frame_bgr.copy()
    pixels_per_chunk = (h * w * 3) // max(total_chunks, 1)  # yaklaşık piksel / chunk
    rows_per_chunk   = max(1, h // max(total_chunks, 1))

    for cid in missing_chunk_ids:
        y_start = cid * rows_per_chunk
        y_end   = min(y_start + rows_per_chunk, h)
        corrupted[y_start:y_end, :] = 0  # siyah bölge

    return corrupted


# ─────────────────────────────────────────────────────────────────────────────
# UDP Alıcı Thread
# ─────────────────────────────────────────────────────────────────────────────

class UDPReceiver(threading.Thread):
    """
    Ayrı bir thread olarak çalışır ve UDP soketinden gelen paket parçalarını toplar.
    Tamamlanan veya timeout'a uğrayan kareleri `frame_queue`'ye koyar.

    frame_queue elemanları: (frame_id, frame_bgr_or_None, is_corrupted: bool)
      - is_corrupted=True  → eksik parçalar var (paket kaybı yaşandı)
      - is_corrupted=False → kare tam olarak alındı
    """

    def __init__(self, sock: socket.socket, frame_queue: queue.Queue, timeout: float = CHUNK_TIMEOUT):
        super().__init__(daemon=True)
        self.sock         = sock
        self.frame_queue  = frame_queue
        self.timeout      = timeout
        self._stop_event  = threading.Event()

        # {frame_id → {chunk_id → bytes}}
        self._chunks: dict[int, dict[int, bytes]] = defaultdict(dict)
        # {frame_id → total_chunks}
        self._totals: dict[int, int] = {}
        # {frame_id → first_chunk_arrive_time}
        self._arrive_times: dict[int, float] = {}

    def _try_flush_old_frames(self, current_frame_id: int):
        """Timeout'a uğramış eski kareleri bozuk olarak teslim eder."""
        now = time.time()
        stale = [
            fid for fid, t in self._arrive_times.items()
            if fid < current_frame_id and (now - t) > self.timeout
        ]
        for fid in stale:
            self._deliver_frame(fid)

    def _deliver_frame(self, fid: int):
        """Bir kareyi (tam veya eksik) frame_queue'ye koyar."""
        chunks_got  = self._chunks.get(fid, {})
        total       = self._totals.get(fid, max(chunks_got.keys(), default=0) + 1)
        missing     = set(range(total)) - set(chunks_got.keys())
        is_corrupted = bool(missing)

        frame_bgr = None
        if chunks_got:
            # Mevcut parçaları sıraya diz; eksikleri boş bırak (sıfır doldurma)
            data_parts = []
            for cid in range(total):
                data_parts.append(chunks_got.get(cid, b"\x00"))  # eksik → tek null byte
            jpeg_data = b"".join(data_parts)

            try:
                arr = np.frombuffer(jpeg_data, dtype=np.uint8)
                decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if decoded is not None:
                    if is_corrupted:
                        decoded = apply_packet_loss_mask(decoded, missing, total)
                    frame_bgr = decoded
            except Exception:
                frame_bgr = None  # Çözümleme başarısız → tamamen bozuk kare

        self.frame_queue.put((fid, frame_bgr, is_corrupted))

        # Temizlik
        self._chunks.pop(fid, None)
        self._totals.pop(fid, None)
        self._arrive_times.pop(fid, None)

    def run(self):
        while not self._stop_event.is_set():
            try:
                raw, _ = self.sock.recvfrom(65536 + HEADER_SIZE)
            except socket.timeout:
                continue
            except OSError:
                break

            if len(raw) < HEADER_SIZE:
                continue

            header   = raw[:HEADER_SIZE]
            payload  = raw[HEADER_SIZE:]
            frame_id, total_chunks, chunk_id, data_len = struct.unpack(HEADER_FORMAT, header)

            # Bitiş sinyali
            if frame_id == 0xFFFFFFFF:
                print("[Receiver] Sunucu bitiş sinyali gönderdi.", flush=True)
                self._stop_event.set()
                self.frame_queue.put(None)  # sentinel
                return

            if total_chunks == 0:
                continue

            now = time.time()
            if frame_id not in self._arrive_times:
                self._arrive_times[frame_id] = now
            self._totals[frame_id] = total_chunks
            self._chunks[frame_id][chunk_id] = payload[:data_len]

            # Kare tamamlandı mı?
            if len(self._chunks[frame_id]) == total_chunks:
                self._deliver_frame(frame_id)

            # Eski kareleri temizle
            self._try_flush_old_frames(frame_id)

    def stop(self):
        self._stop_event.set()


# ─────────────────────────────────────────────────────────────────────────────
# Ana İstemci Sınıfı
# ─────────────────────────────────────────────────────────────────────────────

class StreamClient:
    """
    UDP Video Streaming İstemcisi.
    Gelen kareleri toplar, VSR modeli ile onarır, çıktı videosuna yazar.
    """

    WINDOW_SIZE = 15  # VSR modeli 15 kare bekliyor

    def __init__(
        self,
        port: int,
        model_path: Path | None,
        out_dir: Path,
        display: bool,
        no_vsr: bool,
        out_fps: float,
    ):
        self.port       = port
        self.out_dir    = out_dir
        self.display    = display
        self.no_vsr     = no_vsr
        self.out_fps    = out_fps
        self.model_name = model_path.name if model_path else "(model yok)"

        out_dir.mkdir(parents=True, exist_ok=True)

        # ── Model ─────────────────────────────────────────────────────────────
        self.model  = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not no_vsr:
            if model_path is None or not model_path.exists():
                print("[Client] UYARI: Model dosyası bulunamadı → VSR devre dışı.", flush=True)
                self.no_vsr = True
            else:
                self.model = load_vsr_model(model_path, self.device)

        # ── İstatistikler ──────────────────────────────────────────────────────
        self._total_frames     = 0
        self._corrupted_frames = 0
        self._restored_frames  = 0
        self._start_time       = 0.0
        self._latencies: list[float] = []  # çerçeve başına işlem süresi

        # ── Async VSR Thread ───────────────────────────────────────────────────
        # VSR islemi cok yavas (CPU'da 1-2sn/kare)
        # Ana dongu bloke olmamak icin async yapildi:
        #   - vsr_input_queue: ham kareleri alir
        #   - vsr_output: en son onarilmis kareyi saklar
        self._vsr_input_q: queue.Queue = queue.Queue(maxsize=2)
        self._vsr_lock = threading.Lock()
        # Her ikisi de AYNI kareye ait olmali (senkron karsilastirma icin)
        self._vsr_pair_ready = False  # ilk cift hazir mi

    def _vsr_worker(self):
        """Arka planda VSR onarimi yapan thread - ana akisi bloke etmez."""
        while True:
            try:
                item = self._vsr_input_q.get(timeout=5.0)
                if item is None:
                    break
                window, raw_small, out_dir, w, h = item
                try:
                    restored_small = vsr_restore(self.model, window, self.device)
                    # Gercek boyuta yuksel
                    raw_full      = cv2.resize(raw_small, (w, h))
                    restored_full = cv2.resize(restored_small, (w, h))
                    # Etikket ekle
                    cv2.putText(raw_full, "Ham Akis - Bozuk", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 2)
                    cv2.putText(restored_full, "VSR Yapay Zeka Onarimi", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 80), 2)
                    # ATOMIK yaz: once .tmp sonra rename (race condition yok)
                    self._atomic_write_jpg(out_dir / "latest_raw.jpg",      raw_full)
                    self._atomic_write_jpg(out_dir / "latest_restored.jpg", restored_full)
                    with self._vsr_lock:
                        self._vsr_pair_ready = True
                    self._restored_frames += 1
                except Exception as e:
                    print(f"[VSR Worker] Hata: {e}", flush=True)
            except queue.Empty:
                continue

    @staticmethod
    def _atomic_write_jpg(path: Path, img: np.ndarray, quality: int = 85):
        """JPEG'i guvenli yazar (Windows uyumlu)."""
        import tempfile, shutil
        # Ayni klasore gecici dosya olustur (rename sadece ayni sürücüde calısır)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix='.tmp.jpg')
        try:
            os.close(tmp_fd)
            cv2.imwrite(tmp_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            shutil.move(tmp_path, str(path))
        except Exception:
            try: os.unlink(tmp_path)
            except: pass

    # ── Video Writer ──────────────────────────────────────────────────────────
    def _make_writer(self, w: int, h: int, suffix: str):
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.out_dir / f"{suffix}_{ts}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, self.out_fps, (w, h))
        if not writer.isOpened():
            print(f"[Client] UYARI: VideoWriter acilamadi: {path} - video kaydedilmeyecek", flush=True)
            return None, path
        print(f"[Client] Video yazici: {path}", flush=True)
        return writer, path

    # ── İstatistik Logu ───────────────────────────────────────────────────────
    def _save_stats(self, rows: list[dict]):
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.out_dir / f"stream_stats_{ts}.csv"
        if not rows:
            return
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"[Client] İstatistik kaydedildi: {csv_path}", flush=True)

    # ── Ana döngü ─────────────────────────────────────────────────────────────
    def run(self):
        # Soket kurulumu
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SOCKET_BUF)
        sock.settimeout(RECV_TIMEOUT)
        sock.bind(("", self.port))
        print(f"[Client] UDP port {self.port} dinleniyor...", flush=True)

        frame_queue = queue.Queue(maxsize=120)
        receiver    = UDPReceiver(sock, frame_queue)
        receiver.start()

        # VSR arka plan thread'i baslat
        vsr_thread = None
        if not self.no_vsr and self.model is not None:
            vsr_thread = threading.Thread(target=self._vsr_worker, daemon=True)
            vsr_thread.start()
            print("[Client] VSR async thread basladi (arka planda onarim yapiliyor)", flush=True)

        # Kayan pencere
        window: list[torch.Tensor] = []
        # Boş/bozuk kare için placeholder (siyah)
        placeholder_bgr = None

        # Video writer'lar (ilk kare gelince oluşturulur)
        restored_writer = None
        raw_writer      = None
        restored_path   = None
        raw_path        = None

        stats_rows: list[dict] = []
        self._start_time = time.time()
        last_good_frame: np.ndarray | None = None  # Son iyi (siyah olmayan) kare

        try:
            while True:
                try:
                    item = frame_queue.get(timeout=RECV_TIMEOUT + 1)
                except queue.Empty:
                    print("[Client] Zaman aşımı – akış kesildi.", flush=True)
                    break

                if item is None:  # sentinel
                    break

                frame_id, frame_bgr, is_corrupted = item
                t_proc_start = time.time()

                self._total_frames += 1
                if is_corrupted:
                    self._corrupted_frames += 1

                # Kare None veya tamamen siyah → son iyi kareyi kullan (kara ekran yok)
                if frame_bgr is None:
                    if last_good_frame is not None:
                        frame_bgr = last_good_frame.copy()
                        # Bozuk oldugunu belirten overlay ekle
                        cv2.rectangle(frame_bgr, (0, 0), (frame_bgr.shape[1], 40), (0, 0, 0), -1)
                        cv2.putText(frame_bgr, "PAKET KAYBI - Son iyi kare gosteriliyor",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 60, 255), 2)
                    elif placeholder_bgr is None:
                        placeholder_bgr = np.zeros((360, 640, 3), dtype=np.uint8)
                        cv2.putText(placeholder_bgr, "Ag baglantisi bekleniyor...",
                            (80, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 255), 2)
                        frame_bgr = placeholder_bgr.copy()
                    else:
                        frame_bgr = placeholder_bgr.copy()
                else:
                    # Kare geldi ama tamamen siyah mi? (tum piksel 0)
                    if frame_bgr.mean() > 2.0:  # en az birakin parlaklik varsa iyi kare
                        last_good_frame = frame_bgr.copy()

                h, w = frame_bgr.shape[:2]

                if placeholder_bgr is None:
                    placeholder_bgr = np.zeros((h, w, 3), dtype=np.uint8)
                    if not self.no_vsr and restored_writer is None:
                        restored_writer, restored_path = self._make_writer(w, h, "restored_stream")
                    if raw_writer is None:
                        raw_writer, raw_path = self._make_writer(w, h, "raw_stream")

                # Ham videoyu yaz
                if raw_writer:
                    raw_writer.write(frame_bgr)

                # Kayan pencereye ekle (kucultulmus boyut ile)
                DISPLAY_W, DISPLAY_H = 480, 270  # VSR icin kucuk boyut
                small = cv2.resize(frame_bgr, (DISPLAY_W, DISPLAY_H))
                tensor = _frame_to_tensor(small)
                window.append(tensor)
                if len(window) > self.WINDOW_SIZE:
                    window.pop(0)

                # Async VSR: bozuk karelerde veya periyodik gonder
                if not self.no_vsr and self.model is not None and len(window) == self.WINDOW_SIZE:
                    if is_corrupted or self._total_frames % 6 == 0:
                        if not self._vsr_input_q.full():
                            self._vsr_input_q.put_nowait((list(window), small.copy(), self.out_dir, w, h))

                # Model yoksa: yazilimsal fallback (hizli, akarken goster)
                elif not self.no_vsr and self.model is None and len(window) == self.WINDOW_SIZE:
                    if is_corrupted:
                        mask = cv2.inRange(small, np.array([0, 0, 0]), np.array([8, 8, 8]))
                        if mask.any():
                            restored_small = cv2.inpaint(small, mask, 3, cv2.INPAINT_TELEA)
                            raw_full  = cv2.resize(small, (w, h))
                            rest_full = cv2.resize(restored_small, (w, h))
                            cv2.putText(raw_full,  "Ham Akis - Bozuk",    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 2)
                            cv2.putText(rest_full, "Yazilimsal Onarim",   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 80), 2)
                            self._atomic_write_jpg(self.out_dir / "latest_raw.jpg",      raw_full)
                            self._atomic_write_jpg(self.out_dir / "latest_restored.jpg", rest_full)
                            self._restored_frames += 1

                # Onarilmis videoyu yaz (ham kare ile)
                if restored_writer:
                    restored_writer.write(frame_bgr)  # ham kare (senkron kayit)

                # Web dashboard icin: JPEG ciftini her 2 karede bir guncelle (raw only)
                # VSR ciftini VSR worker atar; burada sadece ham guncelleniyor
                if self._total_frames % 2 == 0 and not self.no_vsr:
                    # Eger VSR cift hazirsa ekstra guncelleme gerekmez
                    # Degilse sadece ham'i goster (restored bosken)
                    with self._vsr_lock:
                        pair_ready = self._vsr_pair_ready
                    if not pair_ready:
                        raw_disp = cv2.resize(frame_bgr, (480, 270))
                        cv2.putText(raw_disp, "Ham Akis - Bekleniyor...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2)
                        self._atomic_write_jpg(self.out_dir / "latest_raw.jpg", raw_disp)

                # İstatistik satırı
                proc_ms = (time.time() - t_proc_start) * 1000
                self._latencies.append(proc_ms)
                stats_rows.append({
                    "frame_id"      : frame_id,
                    "is_corrupted"  : int(is_corrupted),
                    "vsr_applied"   : int(not self.no_vsr and len(window) == self.WINDOW_SIZE),
                    "proc_ms"       : round(proc_ms, 2),
                    "elapsed_s"     : round(time.time() - self._start_time, 3),
                })

                # Her 10 karede bir client_stats.json guncelle (pipeline sayfasi icin)
                if self._total_frames % 10 == 0:
                    elapsed = max(0.1, time.time() - self._start_time)
                    loss_rate = self._corrupted_frames / max(1, self._total_frames) * 100
                    stats = {
                        "stage": "VSR Istemci",
                        "model_name": self.model_name,
                        "model_active": self.model is not None,
                        "total_frames": self._total_frames,
                        "corrupted_frames": self._corrupted_frames,
                        "restored_frames": self._restored_frames,
                        "loss_rate_pct": round(loss_rate, 1),
                        "fps": round(self._total_frames / elapsed, 1),
                        "window_size": self.WINDOW_SIZE,
                        "timestamp": time.time(),
                    }
                    try:
                        import json as _json
                        stats_path = self.out_dir.parent / "client_stats.json"
                        stats_path.write_text(_json.dumps(stats, ensure_ascii=False), encoding="utf-8")
                    except Exception:
                        pass

                # Konsol özeti her 50 karede bir
                if self._total_frames % 50 == 0:
                    avg_lat = sum(self._latencies[-50:]) / 50
                    loss_rate = self._corrupted_frames / max(1, self._total_frames) * 100
                    print(
                        f"[Client] Kare: {self._total_frames}  "
                        f"Bozuk: {self._corrupted_frames} ({loss_rate:.1f}%)  "
                        f"Ort. İşlem: {avg_lat:.1f} ms",
                        flush=True,
                    )

        except KeyboardInterrupt:
            print("\n[Client] Kullanici tarafindan durduruldu.", flush=True)
        finally:
            receiver.stop()
            # VSR thread'i durdur
            if vsr_thread and vsr_thread.is_alive():
                try: self._vsr_input_q.put_nowait(None)
                except: pass
            if restored_writer:
                restored_writer.release()
            if raw_writer:
                raw_writer.release()
            sock.close()
            self._print_summary()
            self._save_stats(stats_rows)

    def _print_summary(self):
        elapsed  = max(1e-6, time.time() - self._start_time)
        fps_eff  = self._total_frames / elapsed
        avg_lat  = sum(self._latencies) / max(1, len(self._latencies))
        loss_pct = self._corrupted_frames / max(1, self._total_frames) * 100
        print("\n" + "="*55)
        print("  UDP STREAM – İSTEMCİ ÖZET RAPORU")
        print("="*55)
        print(f"  Toplam kare alındı      : {self._total_frames}")
        print(f"  Bozuk kare (paket kaybı): {self._corrupted_frames}  ({loss_pct:.1f}%)")
        print(f"  VSR ile onarılan        : {self._restored_frames}")
        print(f"  Etkin FPS               : {fps_eff:.2f}")
        print(f"  Ortalama işlem süresi   : {avg_lat:.1f} ms / kare")
        print(f"  Toplam süre             : {elapsed:.1f} s")
        print("="*55 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="UDP Video Streaming Client + VSR Restoration – QoS Projesi Aşama 1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    net = ap.add_argument_group("Ağ ayarları")
    net.add_argument("--port",  type=int, default=9999, help="Dinlenecek UDP portu")

    mdl = ap.add_argument_group("Model ayarları")
    mdl.add_argument(
        "--model",
        default=None,
        help="VSR model checkpoint (.pth). Örnek: vsr_model_sharp_v2_ep10.pth",
    )
    mdl.add_argument(
        "--no-vsr",
        action="store_true",
        help="VSR modelini devre dışı bırak (ham akış testi için)",
    )

    out = ap.add_argument_group("Çıktı ayarları")
    out.add_argument("--out",     default="output/stream", help="Çıktı klasörü")
    out.add_argument("--display", action="store_true", help="Canlı gösterim penceresi (OpenCV)")
    out.add_argument("--out-fps", type=float, default=25.0, help="Çıktı video FPS")

    args = ap.parse_args()

    model_path = Path(args.model) if args.model else None

    client = StreamClient(
        port       = args.port,
        model_path = model_path,
        out_dir    = Path(args.out),
        display    = args.display,
        no_vsr     = args.no_vsr,
        out_fps    = args.out_fps,
    )
    client.run()


if __name__ == "__main__":
    main()
