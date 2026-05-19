"""
Tek video yukle → boz → VSR → web frame + metrik (comparison sayfasi).
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compare_models import (  # noqa: E402
    add_label,
    corrupt,
    evaluate,
    load_model,
    make_videos,
    psnr,
    ssim_simple,
)
from make_web_frames import extract  # noqa: E402

DEFAULT_MODEL = ROOT / "model_sharp_ep10_fixed.pth"
DEFAULT_VIDEO = ROOT / "input.mp4"
JOB_FILE = ROOT / "output" / "comparison_job.json"
SOURCE_FILE = ROOT / "output" / "video_source.json"


def _write_job(status: str, progress: int = 0, message: str = "", error: str | None = None):
    JOB_FILE.parent.mkdir(parents=True, exist_ok=True)
    JOB_FILE.write_text(
        json.dumps(
            {"status": status, "progress": progress, "message": message, "error": error},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def process_video(
    video_path: Path,
    *,
    model_path: Path | None = None,
    loss: float = 0.25,
    resize: float = 0.5,
    max_frames: int = 150,
    out_dir: Path | None = None,
) -> dict:
    """Bozuk + VSR ciktilarini uretir; model_comparison.json yazar."""
    video_path = Path(video_path).resolve()
    out_dir = Path(out_dir or ROOT / "output")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(model_path or DEFAULT_MODEL).resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"Video bulunamadi: {video_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model bulunamadi: {model_path}")

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _write_job("processing", 5, "Video okunuyor...")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Video acilamadi")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = max(64, int(src_w * resize))
    H = max(64, int(src_h * resize))
    vsr_w = min(W, 480)
    vsr_h = min(H, 270)

    frames_clean, frames_corr = [], []
    np.random.seed(42)
    while len(frames_clean) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (W, H))
        frames_clean.append(frame)
        frames_corr.append(corrupt(frame, loss))
    cap.release()

    if len(frames_clean) < 20:
        raise RuntimeError("Video cok kisa (en az 20 kare gerekli)")

    _write_job("processing", 20, f"{len(frames_clean)} kare — model yukleniyor...")
    model, _, _, _ = load_model(model_path, device)
    label = model_path.name

    _write_job("processing", 35, "VSR onarimi calisiyor (biraz surebilir)...")
    t0 = time.time()
    def progress_cb(curr, tot):
        p = 35 + int((curr / tot) * 35)
        _write_job("processing", p, f"VSR onarimi calisiyor: Kare {curr}/{tot}")
    restored, avg_psnr, avg_ssim = evaluate(
        model, device, frames_clean, frames_corr, vsr_w, vsr_h, label, progress_cb=progress_cb
    )
    elapsed = time.time() - t0

    base_psnr = float(np.mean([psnr(c, r) for c, r in zip(frames_clean[7:], frames_corr[7:])]))
    base_ssim = float(np.mean([ssim_simple(c, r) for c, r in zip(frames_clean[7:], frames_corr[7:])]))
    psnr_gain = float(avg_psnr - base_psnr)
    ssim_gain = float(avg_ssim - base_ssim)

    _write_job("processing", 70, "MP4 dosyalari yaziliyor...")
    corr_labeled = [
        add_label(f, f"Bozuk (%{loss * 100:.0f} paket kaybi)", (60, 60, 255)) for f in frames_corr
    ]
    rest_labeled = [add_label(f, f"VSR — {label}", (60, 220, 60)) for f in restored]
    make_videos(out_dir, fps, frames_corr, restored, label, corr_labeled, rest_labeled)

    _write_job("processing", 85, "Web kareleri hazirlaniyor...")
    frames_dir = out_dir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)
    n_corr, fps_out = extract(out_dir / "A_corrupted.mp4", frames_dir / "corr", "Bozuk")
    n_rest, _ = extract(out_dir / "B_restored.mp4", frames_dir / "rest", "Onarilmis")
    is_default = video_path.resolve() == DEFAULT_VIDEO.resolve()
    source_label = "input.mp4" if is_default else video_path.name
    meta = {
        "fps": round(fps_out, 2),
        "n_frames": min(n_corr, n_rest),
        "source": source_label,
        "is_default": is_default,
    }
    (frames_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    SOURCE_FILE.write_text(
        json.dumps(
            {
                "active": source_label,
                "is_default": is_default,
                "path": str(video_path),
                "type": "default" if is_default else "upload",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    metrics = {
        label: {
            "path": str(model_path),
            "psnr": float(avg_psnr),
            "ssim": float(avg_ssim),
            "base_psnr": base_psnr,
            "base_ssim": base_ssim,
            "psnr_gain": psnr_gain,
            "ssim_gain": ssim_gain,
            "frames": len(frames_clean),
            "elapsed_sec": round(elapsed, 1),
        },
        "winner": label,
        "upload": {"video": video_path.name, "loss": loss, "resize": resize},
    }
    (out_dir / "model_comparison.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _write_job("done", 100, f"Tamamlandi ({len(frames_clean)} kare, {elapsed:.0f}s)")
    return metrics


def run_job(video_path: Path, **kwargs):
    try:
        return process_video(video_path, **kwargs)
    except Exception as e:
        _write_job("error", 0, "Hata", str(e))
        raise


def run_default_video(**kwargs):
    if not DEFAULT_VIDEO.exists():
        raise FileNotFoundError(
            f"Varsayilan video yok: {DEFAULT_VIDEO}\n"
            "input.mp4 dosyasini proje kokune koyun."
        )
    return run_job(DEFAULT_VIDEO, **kwargs)
