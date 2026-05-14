"""
process_video.py  --  Offline Video Bozma + VSR Onarim
=======================================================
Kullanim:
  python process_video.py --video input.mp4 --model vsr_ep30_v2.pth

Cikti (output/ klasorune):
  output/corrupted_<isim>.mp4   <- Paket kaybi efektli bozulmus video
  output/restored_<isim>.mp4   <- VSR yapay zeka ile onarilmis video

Parametreler:
  --loss   Paket kayip orani (0.0-1.0, varsayilan 0.15 = %15)
  --fps    Cikti FPS (varsayilan: kaynak ile ayni)
  --resize Kucultme carpani (0.5 = yariya indir, varsayilan 1.0)
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

# ─── Model Mimarisi (stream_client ile ayni) ────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, n=64):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(n,n,3,padding=1), nn.ReLU(True), nn.Conv2d(n,n,3,padding=1))
    def forward(self, x): return x + self.conv(x)

class VSRModel(nn.Module):
    def __init__(self, in_ch=45, feats=64, n_blocks=8):
        super().__init__()
        self.conv_first = nn.Conv2d(in_ch, feats, 3, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(feats) for _ in range(n_blocks)])
        self.conv_last  = nn.Conv2d(feats, 3, 3, padding=1)
    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b, t*c, h, w)
        out = F.relu(self.conv_first(x))
        out = self.res_blocks(out)
        return self.conv_last(out)


_to_tensor = T.Compose([T.ToTensor()])

def frame_to_tensor(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return _to_tensor(Image.fromarray(rgb))

def tensor_to_bgr(t):
    arr = t.detach().float().cpu().clamp(0,1).permute(1,2,0).numpy()
    return cv2.cvtColor((arr*255+0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)

@torch.no_grad()
def vsr_restore(model, window, device):
    stack = torch.stack(window).unsqueeze(0).to(device)
    center = window[7].float().to(device)
    out = model(stack).squeeze(0).cpu().float()
    out = (out + center.cpu()).clamp(0,1)
    return tensor_to_bgr(out)


# ─── Paket Kaybi Simülasyonu ────────────────────────────────────────────────

def corrupt_frame(frame, loss_rate=0.15, block_h=None):
    """Kareye rastgele siyah yatay bantlar ekler (paket kaybi etkisi)."""
    h, w = frame.shape[:2]
    corrupted = frame.copy()
    if block_h is None:
        block_h = max(8, h // 16)  # 16 bolume ayir

    n_blocks = h // block_h
    for i in range(n_blocks):
        if np.random.random() < loss_rate:
            y1 = i * block_h
            y2 = min(y1 + block_h, h)
            corrupted[y1:y2, :] = 0  # siyah blok
    return corrupted


# ─── Ana Islem ──────────────────────────────────────────────────────────────

def process_video(video_path: Path, model_path: Path, loss_rate: float,
                  resize: float, out_dir: Path) -> tuple[Path, Path]:

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    corr_path = out_dir / f"corrupted_{stem}.mp4"
    rest_path = out_dir / f"restored_{stem}.mp4"

    # ── Modeli yukle ──────────────────────────────────────────────────────────
    print(f"[1/3] Model yukleniyor: {model_path.name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(str(model_path), map_location="cpu")
    w0 = sd.get("conv_first.weight")
    in_ch  = int(w0.shape[1])
    feats  = int(w0.shape[0])
    n_blks = sum(1 for k in sd if k.startswith("res_blocks.") and ".conv.0.weight" in k) or 8
    model  = VSRModel(in_ch=in_ch, feats=feats, n_blocks=n_blks)
    model.load_state_dict(sd)
    model.eval().to(device)
    print(f"    -> in_ch={in_ch}, feats={feats}, blocks={n_blks}, device={device}")

    # ── Videoyu oku ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Video acilamadi: {video_path}")

    src_fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frm = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_w = int(src_w * resize)
    out_h = int(src_h * resize)
    # VSR icin kucuk boyut (CPU hizi)
    vsr_w = min(out_w, 480)
    vsr_h = min(out_h, 270)

    print(f"[2/3] Video isleniyor: {src_w}x{src_h} @{src_fps:.1f}fps  -> {out_w}x{out_h}")
    print(f"    Toplam kare: {total_frm}  |  Paket kaybi: %{loss_rate*100:.0f}  |  VSR boyutu: {vsr_w}x{vsr_h}")
    print(f"    Cikti: {corr_path.name}  +  {rest_path.name}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    corr_writer = cv2.VideoWriter(str(corr_path), fourcc, src_fps, (out_w, out_h))
    rest_writer = cv2.VideoWriter(str(rest_path), fourcc, src_fps, (out_w, out_h))

    if not corr_writer.isOpened() or not rest_writer.isOpened():
        raise RuntimeError("VideoWriter acilamadi!")

    window = []
    WINDOW = 15
    frame_buf   = []   # (corrupted_full, corrupted_small) tampon — window dolana kadar
    t0 = time.time()
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if resize != 1.0:
            frame = cv2.resize(frame, (out_w, out_h))

        # Boz
        corrupted = corrupt_frame(frame, loss_rate)
        small     = cv2.resize(corrupted, (vsr_w, vsr_h))

        # Pencereye ekle
        tensor = frame_to_tensor(small)
        window.append(tensor)
        frame_buf.append(corrupted)

        if len(window) > WINDOW:
            window.pop(0)
            oldest_corr = frame_buf.pop(0)
        else:
            oldest_corr = None

        # Pencere dolunca islemeye basla
        if len(window) == WINDOW:
            with torch.no_grad():
                restored_small = vsr_restore(model, window, device)
            restored_full = cv2.resize(restored_small, (out_w, out_h))

            # Etiket ekle
            label_corr = oldest_corr.copy() if oldest_corr is not None else corrupted.copy()
            cv2.putText(label_corr, f"Bozuk (%{loss_rate*100:.0f} paket kaybi)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 60, 255), 2)
            cv2.putText(restored_full, "VSR Yapay Zeka Onarimi", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 80), 2)

            corr_writer.write(label_corr)
            rest_writer.write(restored_full)
            processed += 1

        # Ilerleme
        if processed % 30 == 0 and processed > 0:
            elapsed = time.time() - t0
            fps_eff = processed / elapsed
            eta     = (total_frm - processed) / max(fps_eff, 0.01)
            print(f"    Kare: {processed}/{total_frm}  |  {fps_eff:.1f} fps  |  Kalan: {eta:.0f}s", flush=True)

    cap.release()
    corr_writer.release()
    rest_writer.release()

    elapsed = time.time() - t0
    print(f"\n[3/3] Tamamlandi! {processed} kare, {elapsed:.1f}s")
    print(f"    Bozuk video : {corr_path}")
    print(f"    Onarilmis  : {rest_path}")
    return corr_path, rest_path


def main():
    ap = argparse.ArgumentParser(description="Offline Video Bozma + VSR Onarim")
    ap.add_argument("--video",  required=True,           help="Kaynak video (.mp4)")
    ap.add_argument("--model",  required=True,           help="VSR model dosyasi (.pth)")
    ap.add_argument("--loss",   type=float, default=0.15, help="Paket kayip orani (0.0-1.0)")
    ap.add_argument("--resize", type=float, default=1.0,  help="Boyut carpani (0.5 = yariya indir)")
    ap.add_argument("--out",    default="output",         help="Cikti klasoru")
    args = ap.parse_args()

    video_path = Path(args.video)
    model_path = Path(args.model)

    if not video_path.exists():
        print(f"HATA: Video bulunamadi: {video_path}")
        sys.exit(1)
    if not model_path.exists():
        print(f"HATA: Model bulunamadi: {model_path}")
        sys.exit(1)

    process_video(video_path, model_path, args.loss, args.resize, Path(args.out))


if __name__ == "__main__":
    main()
