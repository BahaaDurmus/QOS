import argparse
from pathlib import Path

import cv2
import numpy as np


def _degrade(
    frame_bgr: np.ndarray,
    scale: float,
    blur_ksize: int,
    noise_sigma: float,
    jpeg_quality: int,
    packet_loss_prob: float = 0.05,
    block_size: int = 32
) -> np.ndarray:
    h, w = frame_bgr.shape[:2]

    # Downscale + upscale (typical LQ simulation)
    if scale < 1.0:
        sw = max(2, int(round(w * scale)))
        sh = max(2, int(round(h * scale)))
        small = cv2.resize(frame_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
        frame_bgr = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)

    # Paket kaybı simülasyonu (Siyah makrobloklar)
    if packet_loss_prob > 0:
        # Görüntüyü bloklara bölüp rastgele bazı blokları siyaha boyayalım
        num_blocks_y = h // block_size
        num_blocks_x = w // block_size
        for by in range(num_blocks_y):
            for bx in range(num_blocks_x):
                if np.random.rand() < packet_loss_prob:
                    y_start = by * block_size
                    y_end = min(y_start + block_size, h)
                    x_start = bx * block_size
                    x_end = min(x_start + block_size, w)
                    frame_bgr[y_start:y_end, x_start:x_end] = 0

    # Blur
    if blur_ksize > 0:
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        frame_bgr = cv2.GaussianBlur(frame_bgr, (k, k), 0)

    # Additive gaussian noise
    if noise_sigma > 0:
        noise = np.random.normal(0.0, noise_sigma, frame_bgr.shape).astype(np.float32)
        frame_bgr = np.clip(frame_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # JPEG recompress artifacts
    jpeg_quality = int(np.clip(jpeg_quality, 1, 100))
    ok, enc = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if ok:
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        if dec is not None:
            frame_bgr = dec

    return frame_bgr



def corrupt_video(
    input_video: Path,
    output_video: Path,
    start_sec: float,
    duration_sec: float,
    scale: float,
    blur_ksize: int,
    noise_sigma: float,
    jpeg_quality: int,
    out_fps: float | None,
    packet_loss_prob: float = 0.05,
    block_size: int = 32
) -> None:
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Video acilamadi: {input_video}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 25.0

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if w <= 0 or h <= 0:
            raise RuntimeError("Video boyutu okunamadi.")

        use_fps = float(out_fps) if out_fps and out_fps > 0 else fps

        start_frame = max(0, int(round(start_sec * fps)))
        end_frame = int(round((start_sec + duration_sec) * fps)) if duration_sec > 0 else 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        output_video.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video), fourcc, use_fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Video writer acilamadi: {output_video}")

        try:
            from tqdm import tqdm
            idx = start_frame
            total_frames = end_frame - start_frame if end_frame else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            pbar = tqdm(total=total_frames, desc="Videoyu Bozuyor")
            while True:
                if end_frame and idx >= end_frame:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                out = _degrade(
                    frame,
                    scale=scale,
                    blur_ksize=blur_ksize,
                    noise_sigma=noise_sigma,
                    jpeg_quality=jpeg_quality,
                    packet_loss_prob=packet_loss_prob,
                    block_size=block_size,
                )
                writer.write(out)
                idx += 1
                pbar.update(1)
            pbar.close()
        finally:
            writer.release()
    finally:
        cap.release()


def main() -> None:
    ap = argparse.ArgumentParser(description="Create a corrupted (LQ) version of a clean video.")
    ap.add_argument("--video", required=True, help="Input clean video path")
    ap.add_argument("--out-video", required=True, help="Output corrupted video path")
    ap.add_argument("--start-sec", type=float, default=0.0)
    ap.add_argument("--duration-sec", type=float, default=300.0, help="Default: 5 minutes")
    ap.add_argument("--scale", type=float, default=1.0, help="Downscale ratio (default: 1.0)")
    ap.add_argument("--blur-ksize", type=int, default=0, help="Gaussian blur kernel size (odd). 0 disables.")
    ap.add_argument("--noise-sigma", type=float, default=0.0, help="Gaussian noise sigma in [0..255]. 0 disables.")
    ap.add_argument("--jpeg-quality", type=int, default=100, help="JPEG quality 1..100 (lower = worse)")
    ap.add_argument("--out-fps", type=float, default=0.0, help="Override output FPS (0 keeps input FPS)")
    ap.add_argument("--packet-loss-prob", type=float, default=0.05, help="Probability of black macroblocks")
    ap.add_argument("--block-size", type=int, default=32, help="Size of the black blocks")
    args = ap.parse_args()

    corrupt_video(
        input_video=Path(args.video),
        output_video=Path(args.out_video),
        start_sec=float(args.start_sec),
        duration_sec=float(args.duration_sec),
        scale=float(args.scale),
        blur_ksize=int(args.blur_ksize),
        noise_sigma=float(args.noise_sigma),
        jpeg_quality=int(args.jpeg_quality),
        out_fps=(float(args.out_fps) if float(args.out_fps) > 0 else None),
        packet_loss_prob=float(args.packet_loss_prob),
        block_size=int(args.block_size)
    )
    print("Bitti (corrupt).")


if __name__ == "__main__":
    main()

