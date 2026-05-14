"""
compare_models.py  --  Iki VSR Modelini Karsilastir + En Iyi Modelle Cikti Uret
================================================================================
Yaptiklarim:
  1. Input videoyu bozar (%20 paket kaybi)
  2. Her iki modeli de test eder
  3. PSNR + SSIM ile hangisinin daha iyi oldugunu olcer
  4. Kazanan model ile uc video uretir:
       output/A_corrupted.mp4    <- Bozulmus (orijinal bant)
       output/B_restored.mp4    <- VSR onarilmis
       output/C_comparison.mp4  <- Yan yana karsilastirma (sunum icin)
  5. Metrikleri ekrana yazar ve metrics_report.txt'e kaydeder

Kullanim:
  python compare_models.py --video input.mp4
"""

import argparse, sys, time, json, math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T


# ─── VSR Model ───────────────────────────────────────────────────────────────

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
        b,t,c,h,w = x.shape
        x = x.view(b,t*c,h,w)
        return self.conv_last(self.res_blocks(F.relu(self.conv_first(x))))

_to_t = T.Compose([T.ToTensor()])
def to_t(bgr): return _to_t(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
def to_bgr(t):
    arr = t.detach().float().cpu().clamp(0,1).permute(1,2,0).numpy()
    return cv2.cvtColor((arr*255+0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)

@torch.no_grad()
def restore(model, window, device):
    """VSR modelini calistir. Ciktiyi SADECE hasarli (siyah) bölgelere uygula."""
    stack = torch.stack(window).unsqueeze(0).to(device)
    # Residual EKLEME - model tam kare cikitiyor
    out = model(stack).squeeze(0).cpu().float()
    return to_bgr(out.clamp(0, 1))

def apply_mask_restoration(corrupted_bgr, restored_bgr, threshold=8):
    """
    Hasarli bölgeleri (siyah bloklar) VSR ciktisiyla doldur.
    Saglikli piksellerI degistirme - PSNR'i korur.
    """
    # Siyah piksel maskesi: her kanal threshold alti
    mask = np.all(corrupted_bgr <= threshold, axis=2).astype(np.uint8)
    # Maskeyi genislet (kenar pikselleri de kap)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    # Birles: mask=1 -> restored, mask=0 -> corrupted (original)
    mask3 = mask[:, :, np.newaxis]
    result = corrupted_bgr * (1 - mask3) + restored_bgr * mask3
    return result.astype(np.uint8)

def load_model(path, device):
    sd = torch.load(str(path), map_location="cpu")
    w  = sd["conv_first.weight"]
    in_ch = int(w.shape[1]); feats = int(w.shape[0])
    n_blks = sum(1 for k in sd if ".conv.0.weight" in k and "res_blocks" in k) or 8
    m = VSRModel(in_ch=in_ch, feats=feats, n_blocks=n_blks)
    m.load_state_dict(sd); m.eval().to(device)
    return m, in_ch, feats, n_blks


# ─── Metrik Hesaplama ────────────────────────────────────────────────────────

def psnr(a, b):
    mse = np.mean((a.astype(float) - b.astype(float))**2)
    return 100.0 if mse == 0 else 20*math.log10(255/math.sqrt(mse))

def ssim_simple(a, b):
    a, b = a.astype(float)/255, b.astype(float)/255
    mu_a, mu_b = a.mean(), b.mean()
    sig_a = ((a - mu_a)**2).mean()
    sig_b = ((b - mu_b)**2).mean()
    sig_ab = ((a-mu_a)*(b-mu_b)).mean()
    C1, C2 = 0.0001, 0.0009
    return ((2*mu_a*mu_b+C1)*(2*sig_ab+C2)) / ((mu_a**2+mu_b**2+C1)*(sig_a+sig_b+C2))


# ─── Bozma ───────────────────────────────────────────────────────────────────

def corrupt(frame, rate=0.20):
    h, w = frame.shape[:2]
    out  = frame.copy()
    bh   = max(8, h//16)
    for i in range(h//bh):
        if np.random.random() < rate:
            y1, y2 = i*bh, min((i+1)*bh, h)
            out[y1:y2] = 0
    return out


# ─── Tek Model Degerlendirme ─────────────────────────────────────────────────

def evaluate(model, device, frames_clean, frames_corr, vsr_w, vsr_h, label):
    """Tum kareleri isleterek PSNR/SSIM hesaplar. Onarilmis kareleri dondurur."""
    W, H = frames_clean[0].shape[1], frames_clean[0].shape[0]
    WINDOW = 15
    window = []
    restored_frames = []
    psnr_vals, ssim_vals = [], []

    for i, (clean, corr) in enumerate(zip(frames_clean, frames_corr)):
        small  = cv2.resize(corr, (vsr_w, vsr_h))
        tensor = to_t(small)
        window.append(tensor)
        if len(window) > WINDOW: window.pop(0)

        if len(window) == WINDOW:
            # VSR ciktisi al, sadece hasarli bolgelere uygula
            vsr_small = restore(model, window, device)
            vsr_full  = cv2.resize(vsr_small, (W, H))
            res_full  = apply_mask_restoration(corr, vsr_full)
        else:
            res_full = corr.copy()

        restored_frames.append(res_full)

        if len(window) == WINDOW:
            psnr_vals.append(psnr(clean, res_full))
            ssim_vals.append(ssim_simple(clean, res_full))

        if (i+1) % 30 == 0:
            avg_p = sum(psnr_vals[-30:])/max(1,len(psnr_vals[-30:]))
            print(f"  [{label}] Kare {i+1}/{len(frames_clean)}  PSNR: {avg_p:.2f}dB", flush=True)

    return restored_frames, np.mean(psnr_vals), np.mean(ssim_vals)


# ─── Karsilastirma Videosu ────────────────────────────────────────────────────

def add_label(frame, text, color=(255,255,255)):
    out = frame.copy()
    cv2.rectangle(out, (0,0),(out.shape[1],36),(0,0,0),-1)
    cv2.putText(out, text, (8,26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    return out

def make_videos(out_dir, fps, frames_corr, frames_best, best_label,
                frames_corr_labeled, frames_best_labeled):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    H, W   = frames_corr[0].shape[:2]

    # A: Sadece bozuk
    wa = cv2.VideoWriter(str(out_dir/"A_corrupted.mp4"), fourcc, fps, (W,H))
    for f in frames_corr_labeled: wa.write(f)
    wa.release()

    # B: Sadece onarilmis
    wb = cv2.VideoWriter(str(out_dir/"B_restored.mp4"), fourcc, fps, (W,H))
    for f in frames_best_labeled: wb.write(f)
    wb.release()

    # C: Yan yana karsilastirma (2x genislik)
    wc = cv2.VideoWriter(str(out_dir/"C_comparison.mp4"), fourcc, fps, (W*2, H))
    for corr, rest in zip(frames_corr_labeled, frames_best_labeled):
        wc.write(np.hstack([corr, rest]))
    wc.release()

    print(f"\n  A_corrupted.mp4   -> Bozuk video")
    print(f"  B_restored.mp4    -> {best_label} ile onarilmis")
    print(f"  C_comparison.mp4  -> Yan yana karsilastirma\n")


# ─── Ana Fonksiyon ───────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",  default="input.mp4")
    ap.add_argument("--model1", default="vsr_ep30_v2.pth",            help="Model 1 (ep30 Vimeo)")
    ap.add_argument("--model2", default="model_sharp_ep10_fixed.pth",  help="Model 2 (sharp ep10)")
    ap.add_argument("--loss",   type=float, default=0.20,             help="Paket kayip orani")
    ap.add_argument("--resize", type=float, default=0.5,              help="Boyut carpani (hiz icin)")
    ap.add_argument("--out",    default="output",                      help="Cikti klasoru")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}\n")

    # ── 1. Video yukle ──────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(src_w * args.resize); H = int(src_h * args.resize)
    vsr_w = min(W, 480); vsr_h = min(H, 270)

    print(f"Video: {src_w}x{src_h}  ->  {W}x{H} @{fps:.0f}fps")
    print(f"VSR boyutu: {vsr_w}x{vsr_h}  |  Paket kaybi: %{args.loss*100:.0f}\n")

    frames_clean, frames_corr = [], []
    np.random.seed(42)  # Tekrar uretebilmek icin sabit seed
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (W, H))
        frames_clean.append(frame)
        frames_corr.append(corrupt(frame, args.loss))
    cap.release()
    print(f"Toplam kare: {len(frames_clean)}\n")

    # ── 2. Her iki modeli test et ────────────────────────────────────────────
    results = {}
    for path_str, label in [(args.model1, "Vimeo ep30"), (args.model2, "Sharp ep10")]:
        path = Path(path_str)
        if not path.exists():
            print(f"[ATLA] {path_str} bulunamadi.")
            continue
        print(f"=== {label} test ediliyor: {path_str} ===")
        model, in_ch, feats, n_blks = load_model(path, device)
        print(f"  Mimari: in_ch={in_ch}, feats={feats}, blocks={n_blks}")
        t0 = time.time()
        restored, avg_psnr, avg_ssim = evaluate(
            model, device, frames_clean, frames_corr, vsr_w, vsr_h, label)
        elapsed = time.time() - t0
        # Baz: bozuk videonun skoru
        base_psnr = np.mean([psnr(c, r) for c, r in zip(frames_clean[7:], frames_corr[7:])])
        base_ssim = np.mean([ssim_simple(c, r) for c, r in zip(frames_clean[7:], frames_corr[7:])])
        psnr_gain = avg_psnr - base_psnr
        ssim_gain = avg_ssim - base_ssim
        print(f"  Bozuk PSNR:    {base_psnr:.2f} dB")
        print(f"  Onarilmis PSNR:{avg_psnr:.2f} dB  (+{psnr_gain:.2f} dB kazanim)")
        print(f"  Bozuk SSIM:    {base_ssim:.4f}")
        print(f"  Onarilmis SSIM:{avg_ssim:.4f}  (+{ssim_gain:.4f} kazanim)")
        print(f"  Sure: {elapsed:.1f}s\n")
        results[label] = {
            "path": path_str, "restored": restored,
            "psnr": avg_psnr, "ssim": avg_ssim,
            "base_psnr": base_psnr, "base_ssim": base_ssim,
            "psnr_gain": psnr_gain, "ssim_gain": ssim_gain,
        }

    if not results:
        print("Hicbir model bulunamadi!"); sys.exit(1)

    # ── 3. Kazanani sec (PSNR + SSIM ortalamasi) ────────────────────────────
    best_label = max(results, key=lambda k: results[k]["psnr"] + results[k]["ssim"]*10)
    best       = results[best_label]
    print("=" * 50)
    print(f"  KAZANAN MODEL: {best_label}")
    print(f"  PSNR: {best['psnr']:.2f} dB  |  SSIM: {best['ssim']:.4f}")
    print("=" * 50)

    # ── 4. Videolari uret ───────────────────────────────────────────────────
    print("\nVideolar olusturuluyor...")
    corr_labeled = [add_label(f, f"Bozuk (%{args.loss*100:.0f} paket kaybi)", (60,60,255))
                    for f in frames_corr]
    best_labeled = [add_label(f, f"VSR Onarim - {best_label}", (60,220,60))
                    for f in best["restored"]]
    make_videos(out_dir, fps, frames_corr, best["restored"],
                best_label, corr_labeled, best_labeled)

    # ── 5. Rapor kaydet ─────────────────────────────────────────────────────
    report = []
    report.append("=" * 60)
    report.append("  QoS VIDEO STREAMING - MODEL KARSILASTIRMA RAPORU")
    report.append("=" * 60)
    report.append(f"  Video   : {args.video}")
    report.append(f"  Kayip   : %{args.loss*100:.0f}")
    report.append("")
    for label, r in results.items():
        status = " *** KAZANAN ***" if label == best_label else ""
        report.append(f"  Model: {label}{status}")
        report.append(f"    Bozuk PSNR:     {r['base_psnr']:.2f} dB")
        report.append(f"    Onarilmis PSNR: {r['psnr']:.2f} dB  (+{r['psnr_gain']:.2f})")
        report.append(f"    Bozuk SSIM:     {r['base_ssim']:.4f}")
        report.append(f"    Onarilmis SSIM: {r['ssim']:.4f}  (+{r['ssim_gain']:.4f})")
        report.append("")
    report.append(f"  Cikti Dosyalari:")
    report.append(f"    output/A_corrupted.mp4   -> Bozuk video")
    report.append(f"    output/B_restored.mp4    -> {best_label} onarimi")
    report.append(f"    output/C_comparison.mp4  -> Yan yana karsilastirma")
    report_txt = "\n".join(report)
    print("\n" + report_txt)
    (out_dir / "metrics_report.txt").write_text(report_txt, encoding="utf-8")

    # JSON da kaydet (dashboard icin)
    json_out = {k: {kk: vv for kk, vv in v.items() if kk != "restored"}
                for k, v in results.items()}
    json_out["winner"] = best_label
    (out_dir / "model_comparison.json").write_text(
        json.dumps(json_out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRapor kaydedildi: output/metrics_report.txt")

if __name__ == "__main__":
    main()
