# -*- coding: utf-8 -*-
"""
test_stream_no_model.py  -  Modelsiz Baglanti Testi
===================================================
Bu script, bir .mp4 dosyasi vermeden de calisabilmek icin
test kareler uretip UDP uzerinden gonderir ve alir.
VSR modeli olmadan saf baglanti testini yapmak icin kullanin.

Kullanim:
  python test_stream_no_model.py
"""

import socket
import struct
import threading
import time
import queue
import numpy as np
import cv2

HEADER_FORMAT = "!IHHi"
HEADER_SIZE   = struct.calcsize(HEADER_FORMAT)
HOST          = "127.0.0.1"
PORT          = 19999
MAX_CHUNK     = 60_000
NUM_FRAMES    = 60   # Test için gönderilecek kare sayısı
FRAME_W, FRAME_H = 640, 360


def build_packet(frame_id, total, cid, data):
    hdr = struct.pack(HEADER_FORMAT, frame_id, total, cid, len(data))
    return hdr + data


def server_thread():
    """Test kareleri üretip UDP ile gönderir."""
    time.sleep(0.3)  # İstemci hazırlansın
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"[Server] {NUM_FRAMES} test karesi gönderiliyor...")
    for fid in range(NUM_FRAMES):
        # Renkli gradient kare oluştur
        frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        color = int((fid / NUM_FRAMES) * 255)
        frame[:, :, 0] = color
        frame[:, :, 2] = 255 - color
        cv2.putText(frame, f"Kare {fid}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        data = buf.tobytes()
        chunks = [data[i:i+MAX_CHUNK] for i in range(0, len(data), MAX_CHUNK)]
        total  = len(chunks)
        for cid, chunk in enumerate(chunks):
            pkt = build_packet(fid, total, cid, chunk)
            sock.sendto(pkt, (HOST, PORT))
        time.sleep(1/30)

    # Bitiş sinyali
    end = struct.pack(HEADER_FORMAT, 0xFFFFFFFF, 0, 0, 0)
    sock.sendto(end, (HOST, PORT))
    sock.close()
    print("[Server] Bitti.")


def client_thread(result: dict):
    """UDP paketleri alır ve kareleri çözer."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(3.0)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 << 20)
    sock.bind(("", PORT))

    chunks_buf = {}
    totals_buf = {}
    received_frames = 0

    print(f"[Client] Port {PORT} dinleniyor...")
    while True:
        try:
            raw, _ = sock.recvfrom(65536 + HEADER_SIZE)
        except socket.timeout:
            print("[Client] Zaman aşımı.")
            break

        if len(raw) < HEADER_SIZE:
            continue
        frame_id, total_chunks, chunk_id, data_len = struct.unpack(HEADER_FORMAT, raw[:HEADER_SIZE])
        payload = raw[HEADER_SIZE: HEADER_SIZE + data_len]

        if frame_id == 0xFFFFFFFF:
            print("[Client] Bitis sinyali alindi.")
            break

        if frame_id not in chunks_buf:
            chunks_buf[frame_id] = {}
            totals_buf[frame_id] = total_chunks
        chunks_buf[frame_id][chunk_id] = payload

        if len(chunks_buf[frame_id]) == total_chunks:
            jpeg = b"".join(chunks_buf[frame_id][i] for i in range(total_chunks))
            arr  = np.frombuffer(jpeg, dtype=np.uint8)
            img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                received_frames += 1
            del chunks_buf[frame_id]
            del totals_buf[frame_id]

    sock.close()
    result["received"] = received_frames
    print(f"[Client] Toplam alinan ve cozumlenen kare: {received_frames}/{NUM_FRAMES}")


if __name__ == "__main__":
    result = {}
    ct = threading.Thread(target=client_thread, args=(result,))
    st = threading.Thread(target=server_thread)
    ct.start()
    st.start()
    st.join()
    ct.join()

    received = result.get("received", 0)
    loss_rate = (NUM_FRAMES - received) / NUM_FRAMES * 100
    print("\n" + "="*45)
    print("  BAGLANTI TESTI SONUCU")
    print("="*45)
    print(f"  Gonderilen  : {NUM_FRAMES} kare")
    print(f"  Alinan      : {received} kare")
    print(f"  Kayip orani : {loss_rate:.1f}%")
    status = "[BASARILI]" if received >= NUM_FRAMES * 0.95 else "[SORUNLU]"
    print(f"  Durum       : {status}")
    print("="*45)
