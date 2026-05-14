"""
make_web_frames.py
------------------
mp4v kodekli videoları tarayıcının okuyabileceği JPEG frame dizisine çevirir.
output/frames/corr/  -> bozuk video kareleri
output/frames/rest/  -> onarılmış video kareleri
"""
import cv2, json, sys
from pathlib import Path

def extract(src_path: Path, dst_dir: Path, label: str, step: int = 1):
    dst_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        print(f"[HATA] Acilamadi: {src_path}")
        return 0, 0
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    saved = 0
    idx   = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            cv2.imwrite(str(dst_dir / f"{saved:04d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 82])
            saved += 1
        idx += 1
    cap.release()
    print(f"  {label}: {saved} kare -> {dst_dir}")
    return saved, fps / step

if __name__ == "__main__":
    base = Path("output")
    corr_src = base / "A_corrupted.mp4"
    rest_src = base / "B_restored.mp4"

    if not corr_src.exists() or not rest_src.exists():
        print("HATA: output/A_corrupted.mp4 veya output/B_restored.mp4 bulunamadi.")
        print("Once: python compare_models.py --video input.mp4 --loss 0.25")
        sys.exit(1)

    frames_dir = base / "frames"
    print("Kareler ayiklaniyor...")
    n_corr, fps = extract(corr_src, frames_dir / "corr", "Bozuk")
    n_rest, _   = extract(rest_src, frames_dir / "rest", "Onarilmis")

    meta = {"fps": round(fps, 2), "n_frames": min(n_corr, n_rest)}
    (frames_dir / "meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nHazir! {meta['n_frames']} kare @ {meta['fps']} fps")
    print("Simdi sistemi baslatip comparison.html'yi ac.")
