import argparse
from pathlib import Path

import cv2
import numpy as np
import time
import traceback
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


class VSR15To1(nn.Module):
    """
    Matches checkpoint keys:
      conv_first.{weight,bias}
      res_blocks.{i}.conv.{0,2}.{weight,bias}
      conv_last.{weight,bias}

    Input:  (N, 45, H, W)  = 15 frames concatenated on channel dim (RGB)
    Output: (N,  3, H, W)  = enhanced center frame
    """

    def __init__(self, in_channels: int = 45, hidden: int = 64, num_blocks: int = 8) -> None:
        super().__init__()
        self.conv_first = nn.Conv2d(in_channels, hidden, 3, 1, 1)
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden) for _ in range(num_blocks)])
        self.conv_last = nn.Conv2d(hidden, 3, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_first(x)
        for b in self.res_blocks:
            x = b(x)
        return self.conv_last(x)


def _load_model(ckpt_path: Path, device: str) -> nn.Module:
    print(f"[load] checkpoint okunuyor: {ckpt_path}", flush=True)
    sd = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(sd, dict):
        raise RuntimeError("Checkpoint dict degil; desteklenmiyor.")

    # Infer block count from keys
    block_ids = set()
    for k in sd.keys():
        if k.startswith("res_blocks.") and ".conv.0.weight" in k:
            try:
                block_ids.add(int(k.split(".")[1]))
            except Exception:
                pass
    num_blocks = (max(block_ids) + 1) if block_ids else 8

    w = sd.get("conv_first.weight")
    if w is None or not hasattr(w, "shape"):
        raise RuntimeError("Checkpoint conv_first.weight icermiyor.")
    in_channels = int(w.shape[1])
    hidden = int(w.shape[0])

    model = VSR15To1(in_channels=in_channels, hidden=hidden, num_blocks=num_blocks)
    model.load_state_dict(sd, strict=True)
    model.eval()
    model.to(device)
    print(f"[load] model hazir (device={device}, blocks={num_blocks}, hidden={hidden}, in_ch={in_channels})", flush=True)
    return model


def _bgr_to_rgb_chw01(frame_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    return np.transpose(x, (2, 0, 1))  # CHW


def _rgb01_chw_to_bgr_uint8(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    rgb = np.transpose(x, (1, 2, 0))
    bgr = cv2.cvtColor((rgb * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return bgr


@torch.no_grad()
def enhance_frame_from_window(
    model: nn.Module,
    window_bgr: list[np.ndarray],
    device: str,
    add_center: bool,
) -> np.ndarray:
    if len(window_bgr) != 15:
        raise ValueError("Model 15 frame bekliyor.")

    chw = [_bgr_to_rgb_chw01(f) for f in window_bgr]
    inp = np.concatenate(chw, axis=0)  # (45,H,W)

    t = torch.from_numpy(inp).unsqueeze(0).to(device)  # (1,45,H,W)
    out = model(t).squeeze(0).detach().cpu().numpy()  # (3,H,W)

    if add_center:
        center = chw[len(chw) // 2]  # (3,H,W)
        out = out + center

    return _rgb01_chw_to_bgr_uint8(out)


def infer_on_images(
    model: nn.Module,
    input_dir: Path,
    output_dir: Path,
    device: str,
    add_center: bool,
) -> None:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in exts])
    if not paths:
        raise RuntimeError(f"Gorsel bulunamadi: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Okuma hatasi: {p}")
        window = [img] * 15
        out = enhance_frame_from_window(model, window, device=device, add_center=add_center)
        out_path = output_dir / p.name
        if not cv2.imwrite(str(out_path), out):
            raise RuntimeError(f"Yazma hatasi: {out_path}")


def _list_image_paths(input_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


@torch.no_grad()
def infer_on_frame_sequence(
    model: nn.Module,
    input_dir: Path,
    output_dir: Path,
    device: str,
    add_center: bool,
    write_mp4: bool,
    out_video: Path | None,
    out_fps: float,
    max_frames: int,
) -> None:
    """
    Process a sequential frames folder with a sliding 15-frame window.

    - Writes restored frames into output_dir (same filenames)
    - Optionally writes an mp4 using OpenCV VideoWriter (mp4v)
    """
    paths = _list_image_paths(input_dir)
    if not paths:
        raise RuntimeError(f"Gorsel bulunamadi: {input_dir}")
    if max_frames > 0:
        paths = paths[: int(max_frames)]

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load frames (kept in memory for speed / consistent windows)
    frames: list[np.ndarray] = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Okuma hatasi: {p}")
        frames.append(img)

    h, w = frames[0].shape[:2]

    writer: cv2.VideoWriter | None = None
    if write_mp4:
        if out_video is None:
            out_video = output_dir.with_suffix(".mp4")
        out_video.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video), fourcc, float(out_fps), (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Video writer acilamadi: {out_video}")

    try:
        n = len(frames)
        t0 = time.time()
        for idx, p in enumerate(paths):
            window = [frames[min(max(idx + off, 0), n - 1)] for off in range(-7, 8)]
            out = enhance_frame_from_window(model, window, device=device, add_center=add_center)
            out_path = output_dir / p.name
            if not cv2.imwrite(str(out_path), out):
                raise RuntimeError(f"Yazma hatasi: {out_path}")
            if writer is not None:
                writer.write(out)

            if (idx + 1) % 30 == 0 or (idx + 1) == n:
                elapsed = max(1e-6, time.time() - t0)
                fps_eff = (idx + 1) / elapsed
                print(f"[frames] {idx+1}/{n} - {fps_eff:.2f} fps", flush=True)
    finally:
        if writer is not None:
            writer.release()


def infer_on_video(
    model: nn.Module,
    input_video: Path,
    output_video: Path,
    device: str,
    add_center: bool,
    start_sec: float,
    duration_sec: float,
    out_fps: float | None,
) -> None:
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Video acilamadi: {input_video}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 25.0

        in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if in_w <= 0 or in_h <= 0:
            raise RuntimeError("Video boyutu okunamadi.")

        use_fps = float(out_fps) if out_fps and out_fps > 0 else fps

        start_frame = max(0, int(round(start_sec * fps)))
        end_frame = (
            int(round((start_sec + duration_sec) * fps))
            if duration_sec > 0
            else int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        )

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        output_video.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video), fourcc, use_fps, (in_w, in_h))
        if not writer.isOpened():
            raise RuntimeError(f"Video writer acilamadi: {output_video}")

        try:
            window: list[np.ndarray] = []
            frame_idx = start_frame
            written = 0
            t0 = time.time()

            # Prime window with up to 15 frames
            while len(window) < 15:
                ok, frame = cap.read()
                if not ok:
                    break
                window.append(frame)
                frame_idx += 1

            if not window:
                raise RuntimeError("Video bos veya okunamadi.")

            # If video segment shorter than 15, pad by repeating last
            while len(window) < 15:
                window.append(window[-1])

            # Main loop: produce one output per input frame in segment
            while True:
                if end_frame and (frame_idx - len(window) // 2) >= end_frame:
                    break

                out = enhance_frame_from_window(model, window, device=device, add_center=add_center)
                writer.write(out)
                written += 1

                if written % 120 == 0:
                    elapsed = max(1e-6, time.time() - t0)
                    fps_eff = written / elapsed
                    if end_frame:
                        total_out = max(0, end_frame - start_frame)
                        remaining = max(0, total_out - written)
                        eta_s = remaining / max(1e-6, fps_eff)
                        print(
                            f"[video] {written}/{total_out} frame "
                            f"({written / total_out * 100:.1f}%) - {fps_eff:.2f} fps - ETA {eta_s/60:.1f} dk"
                        )
                    else:
                        print(f"[video] {written} frame - {fps_eff:.2f} fps")

                ok, next_frame = cap.read()
                if not ok:
                    break
                window.pop(0)
                window.append(next_frame)
                frame_idx += 1
        finally:
            writer.release()
    finally:
        cap.release()


def main() -> None:
    ap = argparse.ArgumentParser(description="Run 15-frame-to-1 VSR model on images or a video.")
    ap.add_argument(
        "--ckpt",
        default=str(Path(__file__).with_name("vsr_vimeo_model_ep30.pth")),
        help="Path to .pth checkpoint (default: vsr_vimeo_model_ep30.pth next to script)",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--add-center",
        action="store_true",
        help="Add center input frame to model output (residual learning). Try both on/off.",
    )

    src = ap.add_argument_group("source (choose one)")
    src.add_argument("--images", help="Input images folder (process each image independently)")
    src.add_argument(
        "--frames",
        help="Input frames folder (sequential). Uses sliding 15-frame window and writes restored frames.",
    )
    src.add_argument(
        "--frames-root",
        help="Root folder that contains many frame subfolders (e.g. corrupted/00001, 00002, ...). "
        "Each immediate subfolder is processed like --frames.",
    )
    src.add_argument("--video", help="Input video path")

    out = ap.add_argument_group("output")
    out.add_argument("--out-images", help="Output folder for images (required with --images)")
    out.add_argument("--out-frames", help="Output folder for restored frames (required with --frames)")
    out.add_argument("--out-video", help="Output video path (required with --video)")
    out.add_argument(
        "--out-root",
        help="Output root folder for --frames-root (required with --frames-root). "
        "Subfolders will be created under this root with the same names.",
    )

    vid = ap.add_argument_group("video options")
    vid.add_argument("--start-sec", type=float, default=0.0, help="Start time in seconds (default: 0)")
    vid.add_argument("--duration-sec", type=float, default=15.0, help="Duration in seconds (default: 15)")
    vid.add_argument("--out-fps", type=float, default=0.0, help="Override output FPS (0 keeps input FPS)")
    vid.add_argument(
        "--limit-seqs",
        type=int,
        default=0,
        help="When using --frames-root, process only first N subfolders (0 = all).",
    )
    vid.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="When using --frames, process only first N frames (default: 120). Use 0 for all.",
    )

    args = ap.parse_args()

    try:
        ckpt = Path(args.ckpt)
        if not ckpt.exists():
            raise SystemExit(f"Checkpoint bulunamadi: {ckpt}")

        use_images = bool(args.images)
        use_frames = bool(args.frames)
        use_frames_root = bool(args.frames_root)
        use_video = bool(args.video)
        if sum([use_images, use_frames, use_frames_root, use_video]) != 1:
            raise SystemExit("Kaynak olarak sadece birini verin: --images veya --frames veya --frames-root veya --video.")

        device = str(args.device)
        model = _load_model(ckpt, device=device)

        if use_images:
            if not args.out_images:
                raise SystemExit("--images kullaninca --out-images zorunlu.")
            infer_on_images(
                model=model,
                input_dir=Path(args.images),
                output_dir=Path(args.out_images),
                device=device,
                add_center=bool(args.add_center),
            )
            print("Bitti (images).", flush=True)
        elif use_frames:
            if not args.out_frames:
                raise SystemExit("--frames kullaninca --out-frames zorunlu.")
            infer_on_frame_sequence(
                model=model,
                input_dir=Path(args.frames),
                output_dir=Path(args.out_frames),
                device=device,
                add_center=bool(args.add_center),
                write_mp4=False,
                out_video=None,
                out_fps=25.0,
                max_frames=int(args.max_frames),
            )
            print("Bitti (frames).", flush=True)
        elif use_frames_root:
            if not args.out_root:
                raise SystemExit("--frames-root kullaninca --out-root zorunlu.")
            root_in = Path(args.frames_root)
            if not root_in.exists():
                raise SystemExit(f"Girdi klasoru bulunamadi: {root_in}")
            root_out = Path(args.out_root)
            root_out.mkdir(parents=True, exist_ok=True)

            subdirs = sorted([p for p in root_in.iterdir() if p.is_dir()])
            if int(args.limit_seqs) > 0:
                subdirs = subdirs[: int(args.limit_seqs)]
            if not subdirs:
                raise SystemExit(f"Alt klasor bulunamadi: {root_in}")

            total = len(subdirs)
            for i, seq_dir in enumerate(subdirs, start=1):
                out_dir = root_out / seq_dir.name
                print(f"[seq] {i}/{total}: {seq_dir.name}", flush=True)
                infer_on_frame_sequence(
                    model=model,
                    input_dir=seq_dir,
                    output_dir=out_dir,
                    device=device,
                    add_center=bool(args.add_center),
                    write_mp4=False,
                    out_video=None,
                    out_fps=25.0,
                    max_frames=int(args.max_frames),
                )
            print("Bitti (frames-root).", flush=True)
        else:
            if not args.out_video:
                raise SystemExit("--video kullaninca --out-video zorunlu.")
            infer_on_video(
                model=model,
                input_video=Path(args.video),
                output_video=Path(args.out_video),
                device=device,
                add_center=bool(args.add_center),
                start_sec=float(args.start_sec),
                duration_sec=float(args.duration_sec),
                out_fps=(float(args.out_fps) if float(args.out_fps) > 0 else None),
            )
            print("Bitti (video).", flush=True)
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

