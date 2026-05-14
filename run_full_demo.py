"""
run_full_demo.py  –  Tek Komutla Tüm Proje Demosu (Okul Sunumu İçin)
=====================================================================
Bu script sırasıyla şu bileşenleri başlatır:
1. Video Sunucusu (Port 9998'e gönderir)
2. WMN Simülatörü (Port 9998'i dinler, paket kaybı/gecikme uygular, 9999'a iletir)
3. İstemci (Port 9999'u dinler, VSR onarımı yapar, videoyu kaydeder/gösterir)
4. QoS Monitör (Ağ kalitesini sürekli izler)
5. Sunum Dashboard'u (Web arayüzünü sunar)

Kullanım:
  python run_full_demo.py --video input.mp4 --model vsr_model.pth --profile medium
"""

import argparse
import subprocess
import sys
import time
import threading
import os
from pathlib import Path


def stream_output(proc, prefix: str):
    for line in iter(proc.stdout.readline, b""):
        try:
            print(f"[{prefix}] {line.decode('utf-8', errors='replace').rstrip()}", flush=True)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description="QoS Projesi - Tam Sistem Demosu")
    ap.add_argument("--video", required=True, help="Kaynak video (.mp4)")
    ap.add_argument("--model", default=None, help="VSR modeli (.pth)")
    ap.add_argument("--profile", default="medium", choices=["perfect", "good", "medium", "poor", "critical"], help="Ağ zorluk profili")
    ap.add_argument("--display", action="store_true", help="Canlı video pencerelerini aç")
    args = ap.parse_args()

    video_path = Path(args.video).resolve()
    model_path = Path(args.model).resolve() if args.model else None

    if not video_path.exists():
        print(f"[HATA] Video bulunamadı: {video_path}")
        sys.exit(1)

    python = sys.executable
    base_dir = Path(__file__).parent
    
    # Çıktı klasörünü temizle/oluştur
    out_dir = base_dir / "output"
    out_dir.mkdir(exist_ok=True)
    
    # Eski metrikleri temizle (temiz başlasın)
    metrics_file = out_dir / "wmn_metrics.json"
    if metrics_file.exists():
        metrics_file.unlink()

    print("="*60)
    print(" [SISTEM] KABLOSUZ AGLARDA QOS VIDEO AKISI - SUNUM DEMOSU")
    print("="*60)
    
    # Encoding sorununu cözmek icin enviroment
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    processes = []
    threads = []

    try:
        # 1. WMN Simülatörü
        print(">> WMN Simulatoru Baslatiliyor (Profil: {})".format(args.profile))
        wmn_cmd = [python, "streaming/wmn_simulator.py", "--listen", "9998", "--forward", "127.0.0.1:9999", "--profile", args.profile]
        p_wmn = subprocess.Popen(wmn_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=base_dir, env=env)
        processes.append((p_wmn, "WMN"))
        time.sleep(1)

        # 2. İstemci
        print(">> Istemci Baslatiliyor (Port 9999)")
        client_cmd = [python, "streaming/stream_client.py", "--port", "9999", "--out", "output/stream"]
        if model_path:
            client_cmd.extend(["--model", str(model_path)])
        else:
            client_cmd.append("--no-vsr")
        if args.display:
            client_cmd.append("--display")
            
        p_client = subprocess.Popen(client_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=base_dir, env=env)
        processes.append((p_client, "ISTEMCI"))
        time.sleep(1)

        # 3. Sunucu
        print(">> Video Sunucusu Baslatiliyor (Gonderim -> 9998)")
        server_cmd = [python, "streaming/stream_server.py", "--video", str(video_path), "--host", "127.0.0.1", "--port", "9998", "--loop"]
        p_server = subprocess.Popen(server_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=base_dir, env=env)
        processes.append((p_server, "SUNUCU"))
        
        # 4. QoS Monitör (CLI Output)
        print(">> QoS Monitoru Baslatiliyor")
        qos_cmd = [python, "streaming/qos_monitor.py", "--metrics", "output/wmn_metrics.json", "--watch"]
        p_qos = subprocess.Popen(qos_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=base_dir, env=env)
        processes.append((p_qos, "QoS"))
        
        # 5. Sunum Dashboard
        print(">> Web Dashboard Baslatiliyor (Port 8080)")
        dash_cmd = [python, "presentation/server.py", "--port", "8080"]
        p_dash = subprocess.Popen(dash_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=base_dir, env=env)
        processes.append((p_dash, "WEB"))

        print("\n" + "="*60)
        print("[OK] TUM SISTEM CALISIYOR!")
        print("[*] Sunum Dashboard'u icin tarayicida acin: http://localhost:8080")
        print("[!] Kapatmak icin terminalde Ctrl+C'ye basin.")
        print("="*60 + "\n")

        for p, prefix in processes:
            t = threading.Thread(target=stream_output, args=(p, prefix), daemon=True)
            t.start()
            threads.append(t)

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[SISTEM] Kapatiliyor... Lutfen bekleyin.")
    finally:
        for p, _ in processes:
            try:
                p.terminate()
            except:
                pass
        print("[SISTEM] Basariyla kapatildi.")

if __name__ == "__main__":
    main()
