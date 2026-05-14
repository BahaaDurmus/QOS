"""
stream_server.py  –  UDP Video Streaming Server
================================================
QoS-Based Video Streaming Optimization in Wireless Mesh Networks
Aşama 1: Video Akışı (Streaming) Altyapısı

Bu sunucu:
  1. Belirtilen .mp4 dosyasını (veya webcam'i) açar.
  2. Her kareyi JPEG olarak sıkıştırır.
  3. Her JPEG'i 60 000 baytlık UDP paket parçalarına (chunks) böler.
  4. Her parçaya sabit uzunlukta bir başlık (header) ekler:
        [frame_id (4B)] [total_chunks (2B)] [chunk_id (2B)] [data_len (4B)]  → 12 bayt header
  5. Parçaları UDP soket üzerinden istemciye gönderir.

Kullanım:
  python stream_server.py --video input.mp4 --host 127.0.0.1 --port 9999 --fps 25 --quality 80

Not: Aşama 2'de bu sunucu, ağ simülatörüne bağlanarak paket kayıpları oluşturacaktır.
"""

import argparse
import socket
import struct
import time
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Sabitler
# ─────────────────────────────────────────────────────────────────────────────
HEADER_FORMAT  = "!IHHi"        # frame_id(uint32), total_chunks(uint16), chunk_id(uint16), data_len(int32)
HEADER_SIZE    = struct.calcsize(HEADER_FORMAT)   # 12 bayt
MAX_CHUNK_DATA = 60_000         # Bir paketteki maksimum veri boyutu (bayt)
CONTROL_PORT_OFFSET = 1         # Kontrol soket portu = --port + 1

# ─────────────────────────────────────────────────────────────────────────────
# Yardımcı fonksiyonlar
# ─────────────────────────────────────────────────────────────────────────────

def encode_frame(frame_bgr: np.ndarray, quality: int) -> bytes:
    """BGR frame → JPEG bytes."""
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding başarısız.")
    return buf.tobytes()


def split_into_chunks(data: bytes, chunk_size: int) -> list[bytes]:
    """Veriyi eşit büyüklükteki parçalara böler."""
    return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]


def build_packet(frame_id: int, total_chunks: int, chunk_id: int, chunk_data: bytes) -> bytes:
    """Header + veri birleştirerek tek UDP paketi oluşturur."""
    header = struct.pack(HEADER_FORMAT, frame_id, total_chunks, chunk_id, len(chunk_data))
    return header + chunk_data


# ─────────────────────────────────────────────────────────────────────────────
# Sunucu Sınıfı
# ─────────────────────────────────────────────────────────────────────────────

class StreamServer:
    """
    UDP Video Streaming Sunucusu.

    Parameters
    ----------
    host        : İstemci adresi
    port        : Hedef UDP portu
    fps         : Gönderim hızı (kare/saniye)
    quality     : JPEG kalite değeri (1-100)
    loop        : Videoyu döngüsel oynat
    """

    def __init__(self, host: str, port: int, fps: float, quality: int, loop: bool):
        self.host    = host
        self.port    = port
        self.fps     = fps
        self.quality = quality
        self.loop    = loop

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)  # 1 MB gönderim tamponu
        print(f"[Server] UDP soketi oluşturuldu → hedef: {host}:{port}")

        # Durum değişkenleri
        self._stop_event  = threading.Event()
        self._frame_count = 0
        self._bytes_sent  = 0
        self._start_time  = 0.0

    # ── İstatistik ────────────────────────────────────────────────────────────
    def _print_stats(self):
        elapsed = max(1e-6, time.time() - self._start_time)
        fps_eff  = self._frame_count / elapsed
        mbps     = (self._bytes_sent * 8) / elapsed / 1e6
        print(
            f"[Server] Gönderildi: {self._frame_count} kare | "
            f"Etkin FPS: {fps_eff:.2f} | "
            f"Bant Genişliği: {mbps:.2f} Mbps",
            flush=True,
        )

    # ── Tek kare gönderimi ────────────────────────────────────────────────────
    def _send_frame(self, frame_bgr: np.ndarray, frame_id: int) -> int:
        """
        Tek bir kareyi JPEG'e çevirip UDP paket parçaları halinde gönderir.
        Gönderilen toplam bayt sayısını döndürür.
        """
        jpeg_data    = encode_frame(frame_bgr, self.quality)
        chunks       = split_into_chunks(jpeg_data, MAX_CHUNK_DATA)
        total_chunks = len(chunks)
        sent_bytes   = 0

        for cid, chunk_data in enumerate(chunks):
            packet = build_packet(frame_id, total_chunks, cid, chunk_data)
            self.sock.sendto(packet, (self.host, self.port))
            sent_bytes += len(packet)

        return sent_bytes

    # ── Ana akış döngüsü ──────────────────────────────────────────────────────
    def stream_video(self, video_path: Path) -> None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"[Server] Video açılamadı: {video_path}")

        src_fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w         = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h         = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"[Server] Video: {video_path.name}")
        print(f"         Çözünürlük : {w}x{h}  Kaynak FPS: {src_fps:.2f}  Toplam Kare: {total}")
        print(f"         Gönderim FPS: {self.fps:.2f}  JPEG Kalite: {self.quality}")
        print(f"[Server] Akış başlıyor... (durdurmak için Ctrl+C)")

        frame_duration = 1.0 / self.fps
        frame_id       = 0
        self._start_time = time.time()

        try:
            while not self._stop_event.is_set():
                t_frame_start = time.time()

                ok, frame = cap.read()
                if not ok:
                    if self.loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = cap.read()
                        if not ok:
                            break
                        print("[Server] Video döngüye girdi.", flush=True)
                    else:
                        print("[Server] Video bitti.", flush=True)
                        break

                try:
                    sent = self._send_frame(frame, frame_id)
                    self._bytes_sent  += sent
                    self._frame_count += 1
                    frame_id          += 1
                except Exception as e:
                    print(f"[Server] Gönderim hatası kare {frame_id}: {e}", flush=True)

                # İstatistik her 100 karede bir
                if self._frame_count % 100 == 0:
                    self._print_stats()

                # FPS kontrolü
                elapsed_frame = time.time() - t_frame_start
                sleep_time    = frame_duration - elapsed_frame
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n[Server] Kullanıcı tarafından durduruldu.", flush=True)
        finally:
            cap.release()
            # Bitiş sinyali gönder (frame_id = 0xFFFFFFFF, total_chunks = 0)
            end_signal = struct.pack(HEADER_FORMAT, 0xFFFFFFFF, 0, 0, 0)
            try:
                self.sock.sendto(end_signal, (self.host, self.port))
            except Exception:
                pass
            self._print_stats()
            print("[Server] Akış tamamlandı. Soket kapatılıyor.", flush=True)
            self.sock.close()

    def stream_webcam(self, cam_index: int = 0) -> None:
        """Webcam kaynağından canlı akış yapar (opsiyonel)."""
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            raise RuntimeError(f"[Server] Webcam {cam_index} açılamadı.")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        print(f"[Server] Webcam akışı başlıyor (cam_index={cam_index})")
        frame_duration = 1.0 / self.fps
        frame_id       = 0
        self._start_time = time.time()

        try:
            while not self._stop_event.is_set():
                t0 = time.time()
                ok, frame = cap.read()
                if not ok:
                    continue
                try:
                    sent = self._send_frame(frame, frame_id)
                    self._bytes_sent  += sent
                    self._frame_count += 1
                    frame_id          += 1
                except Exception as e:
                    print(f"[Server] Gönderim hatası: {e}", flush=True)
                sleep = frame_duration - (time.time() - t0)
                if sleep > 0:
                    time.sleep(sleep)
        except KeyboardInterrupt:
            print("\n[Server] Kullanıcı tarafından durduruldu.", flush=True)
        finally:
            cap.release()
            self.sock.close()

    def stop(self):
        self._stop_event.set()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="UDP Video Streaming Server – QoS Projesi Aşama 1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = ap.add_argument_group("Kaynak (birini seçin)")
    src.add_argument("--video",  default=None, help="Akışı yapılacak .mp4 dosyası")
    src.add_argument("--webcam", type=int, default=None, help="Webcam indeksi (0, 1, ...)")

    net = ap.add_argument_group("Ağ ayarları")
    net.add_argument("--host",    default="127.0.0.1", help="İstemci IP adresi")
    net.add_argument("--port",    type=int, default=9999, help="UDP hedef portu")

    vid = ap.add_argument_group("Video ayarları")
    vid.add_argument("--fps",     type=float, default=25.0, help="Gönderim hızı (kare/saniye)")
    vid.add_argument("--quality", type=int,   default=80,   help="JPEG kalitesi (1-100)")
    vid.add_argument("--loop",    action="store_true",       help="Video bitince başa dön")

    args = ap.parse_args()

    if args.video is None and args.webcam is None:
        ap.error("--video veya --webcam parametrelerinden birini vermelisiniz.")

    server = StreamServer(
        host    = args.host,
        port    = args.port,
        fps     = args.fps,
        quality = args.quality,
        loop    = args.loop,
    )

    if args.webcam is not None:
        server.stream_webcam(cam_index=args.webcam)
    else:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"[Server] HATA: Video dosyası bulunamadı: {video_path}", file=sys.stderr)
            sys.exit(1)
        server.stream_video(video_path)


if __name__ == "__main__":
    main()
