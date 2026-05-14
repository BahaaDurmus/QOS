"""
run_streaming_demo.py  –  Tek Komutla UDP Akış Demosu
======================================================
QoS-Based Video Streaming Optimization in Wireless Mesh Networks
Aşama 1: Video Akışı (Streaming) Altyapısı – Entegrasyon Testi

Bu script:
  1. stream_server.py'yi ayrı bir process olarak başlatır (sunucu).
  2. stream_client.py'yi aynı makinede ayrı bir process olarak başlatır (istemci).
  3. Her iki process'in çıktısını terminale yansıtır.
  4. Ctrl+C ile her ikisini de düzgünce kapatır.

Kullanım:
  python run_streaming_demo.py --video input.mp4 --model vsr_model_sharp_v2_ep10.pth

Sadece bağlantı testi (VSR modeli olmadan):
  python run_streaming_demo.py --video input.mp4 --no-vsr
"""

import argparse
import subprocess
import sys
import time
import threading
import os
from pathlib import Path


def stream_output(proc, prefix: str):
    """Process stdout'unu prefix ile terminale yansıtır."""
    for line in iter(proc.stdout.readline, b""):
        print(f"[{prefix}] {line.decode('utf-8', errors='replace').rstrip()}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="UDP Streaming Demo – Sunucu + İstemci aynı anda başlatır",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--video",   required=True, help="Kaynak .mp4 dosyası")
    ap.add_argument("--model",   default=None,  help="VSR checkpoint (.pth)")
    ap.add_argument("--port",    type=int, default=9999, help="UDP portu")
    ap.add_argument("--fps",     type=float, default=25.0, help="Gönderim FPS")
    ap.add_argument("--quality", type=int,   default=80,   help="JPEG kalitesi (1-100)")
    ap.add_argument("--loop",    action="store_true",      help="Videoyu döngüsel oynat")
    ap.add_argument("--display", action="store_true",      help="OpenCV canlı pencere")
    ap.add_argument("--no-vsr",  action="store_true",      help="VSR modelini devre dışı bırak")
    ap.add_argument("--out",     default="output/stream",  help="Çıktı klasörü")
    args = ap.parse_args()

    # ── Dosya kontrolleri ─────────────────────────────────────────────────────
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"HATA: Video dosyası bulunamadı: {video_path}", file=sys.stderr)
        sys.exit(1)

    model_path = Path(args.model) if args.model else None
    if not args.no_vsr and model_path and not model_path.exists():
        print(f"HATA: Model dosyası bulunamadı: {model_path}", file=sys.stderr)
        sys.exit(1)

    base_dir = Path(__file__).parent
    python   = sys.executable

    # ── Sunucu komutu ─────────────────────────────────────────────────────────
    server_cmd = [
        python, str(base_dir / "stream_server.py"),
        "--video",   str(video_path),
        "--host",    "127.0.0.1",
        "--port",    str(args.port),
        "--fps",     str(args.fps),
        "--quality", str(args.quality),
    ]
    if args.loop:
        server_cmd.append("--loop")

    # ── İstemci komutu ────────────────────────────────────────────────────────
    client_cmd = [
        python, str(base_dir / "stream_client.py"),
        "--port",    str(args.port),
        "--out",     str(args.out),
        "--out-fps", str(args.fps),
    ]
    if model_path and not args.no_vsr:
        client_cmd += ["--model", str(model_path)]
    else:
        client_cmd.append("--no-vsr")

    if args.display:
        client_cmd.append("--display")

    # ── Process başlatma ──────────────────────────────────────────────────────
    print("="*60)
    print("  QoS UDP Streaming Demo – Aşama 1")
    print("="*60)
    print(f"  Video   : {video_path}")
    print(f"  Model   : {model_path if model_path and not args.no_vsr else 'Devre dışı (--no-vsr)'}")
    print(f"  Port    : {args.port}")
    print(f"  FPS     : {args.fps}")
    print(f"  Çıktı   : {args.out}")
    print("="*60)
    print("  Durdurmak için: Ctrl+C")
    print("="*60 + "\n")

    # İstemcinin başlamadan önce soketi hazırlaması için 1 saniye bekle
    print("[Demo] İstemci başlatılıyor...", flush=True)
    client_proc = subprocess.Popen(
        client_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    time.sleep(1.5)

    print("[Demo] Sunucu başlatılıyor...", flush=True)
    server_proc = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Çıktı thread'leri
    t_server = threading.Thread(target=stream_output, args=(server_proc, "SERVER"), daemon=True)
    t_client = threading.Thread(target=stream_output, args=(client_proc, "CLIENT"), daemon=True)
    t_server.start()
    t_client.start()

    try:
        # İkisi de bitene kadar bekle
        while True:
            sc = server_proc.poll()
            cc = client_proc.poll()
            if sc is not None and cc is not None:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[Demo] Ctrl+C – Processler kapatılıyor...", flush=True)
    finally:
        for proc in [server_proc, client_proc]:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        t_server.join(timeout=2)
        t_client.join(timeout=2)
        print("[Demo] Tüm processler kapatıldı. Çıktılar:", args.out, flush=True)


if __name__ == "__main__":
    main()
