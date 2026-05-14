import argparse
from pathlib import Path

import cv2


def _open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Video acilamadi: {path}")
    return cap


def _get_frame_count(cap: cv2.VideoCapture) -> int:
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return n if n > 0 else 0


def build_basicframes_dataset(
    gt_video: Path,
    lq_video: Path,
    out_root: Path,
    clip_len: int = 15,
    ann_path: Path | None = None,
) -> tuple[int, int]:
    """
    Creates a Vimeo-90K-like folder structure compatible with MMagic BasicFramesDataset:

      dataset/original/00001/im1.png ... im{clip_len}.png
      dataset/corrupted/00001/im1.png ... im{clip_len}.png

    And an annotation file with lines:
      00001 15
      00002 15
      ...
    """
    if clip_len <= 0:
        raise ValueError("--clip-len pozitif olmali.")

    out_root = out_root.resolve()
    original_root = out_root / "original"
    corrupted_root = out_root / "corrupted"
    original_root.mkdir(parents=True, exist_ok=True)
    corrupted_root.mkdir(parents=True, exist_ok=True)

    if ann_path is None:
        ann_path = Path("content") / "train_list.txt"
    ann_path = ann_path.resolve()
    ann_path.parent.mkdir(parents=True, exist_ok=True)

    cap_gt = _open_video(gt_video)
    cap_lq = _open_video(lq_video)
    try:
        n_gt = _get_frame_count(cap_gt)
        n_lq = _get_frame_count(cap_lq)
        if n_gt and n_lq and n_gt != n_lq:
            raise RuntimeError(f"Kare sayilari esit degil. GT={n_gt}, LQ={n_lq}")

        written_frames = 0
        clip_idx = 0

        with ann_path.open("w", encoding="utf-8", newline="\n") as f_ann:
            while True:
                ok_gt, frame_gt = cap_gt.read()
                ok_lq, frame_lq = cap_lq.read()

                if ok_gt != ok_lq:
                    raise RuntimeError(
                        "Videolardan biri daha erken bitti. Kare sayilari eslesmiyor."
                    )
                if not ok_gt:
                    break

                written_frames += 1
                clip_idx = (written_frames - 1) // clip_len + 1
                within = (written_frames - 1) % clip_len + 1

                clip_name = f"{clip_idx:05d}"
                clip_dir_gt = original_root / clip_name
                clip_dir_lq = corrupted_root / clip_name
                if within == 1:
                    clip_dir_gt.mkdir(parents=True, exist_ok=True)
                    clip_dir_lq.mkdir(parents=True, exist_ok=True)

                out_name = f"im{within}.png"
                p_gt = clip_dir_gt / out_name
                p_lq = clip_dir_lq / out_name

                if not cv2.imwrite(str(p_gt), frame_gt):
                    raise RuntimeError(f"Yazma hatasi: {p_gt}")
                if not cv2.imwrite(str(p_lq), frame_lq):
                    raise RuntimeError(f"Yazma hatasi: {p_lq}")

            if written_frames == 0:
                raise RuntimeError("Hic kare okunamadi. Video dosyalarini kontrol et.")

            if written_frames % clip_len != 0:
                incomplete = written_frames % clip_len
                raise RuntimeError(
                    f"Toplam kare sayisi {written_frames}. {clip_len} ile bolunmuyor; "
                    f"son klip eksik ({incomplete}/{clip_len})."
                )

            total_clips = written_frames // clip_len
            for i in range(1, total_clips + 1):
                f_ann.write(f"{i:05d} {clip_len}\n")

        # Validate each clip has exact frame count on both sides
        for i in range(1, (written_frames // clip_len) + 1):
            clip_name = f"{i:05d}"
            gt_dir = original_root / clip_name
            lq_dir = corrupted_root / clip_name
            gt_frames = sorted(gt_dir.glob("im*.png"))
            lq_frames = sorted(lq_dir.glob("im*.png"))
            if len(gt_frames) != clip_len or len(lq_frames) != clip_len:
                raise RuntimeError(
                    f"Klip dogrulama hatasi: {clip_name} icin "
                    f"GT={len(gt_frames)}, LQ={len(lq_frames)} (beklenen {clip_len})."
                )

        return written_frames, written_frames // clip_len
    finally:
        cap_gt.release()
        cap_lq.release()


def build_basicframes_dataset_from_frames(
    gt_frames_dir: Path,
    lq_frames_dir: Path,
    out_root: Path,
    clip_len: int = 15,
    ann_path: Path | None = None,
    pattern: str = "frame_*.jpg",
    drop_last: bool = False,
) -> tuple[int, int]:
    """
    Same output as build_basicframes_dataset(), but uses pre-extracted frames from folders.
    """
    if clip_len <= 0:
        raise ValueError("--clip-len pozitif olmali.")

    out_root = out_root.resolve()
    original_root = out_root / "original"
    corrupted_root = out_root / "corrupted"
    original_root.mkdir(parents=True, exist_ok=True)
    corrupted_root.mkdir(parents=True, exist_ok=True)

    if ann_path is None:
        ann_path = Path("content") / "train_list.txt"
    ann_path = ann_path.resolve()
    ann_path.parent.mkdir(parents=True, exist_ok=True)

    gt_frames_dir = gt_frames_dir.resolve()
    lq_frames_dir = lq_frames_dir.resolve()
    gt_paths = sorted(gt_frames_dir.glob(pattern))
    lq_paths = sorted(lq_frames_dir.glob(pattern))

    if not gt_paths:
        raise RuntimeError(f"GT frame bulunamadi: {gt_frames_dir} (pattern={pattern})")
    if not lq_paths:
        raise RuntimeError(f"LQ frame bulunamadi: {lq_frames_dir} (pattern={pattern})")
    if len(gt_paths) != len(lq_paths):
        raise RuntimeError(f"Kare sayilari esit degil. GT={len(gt_paths)}, LQ={len(lq_paths)}")

    total = len(gt_paths)
    remainder = total % clip_len
    if remainder != 0:
        if not drop_last:
            raise RuntimeError(
                f"Toplam kare sayisi {total}. {clip_len} ile bolunmuyor; "
                f"son klip eksik ({remainder}/{clip_len})."
            )
        total = total - remainder
        gt_paths = gt_paths[:total]
        lq_paths = lq_paths[:total]
        print(
            f"Uyari: Toplam {len(gt_paths) + remainder} kare vardi; son {remainder} kare dusuruldu. "
            f"Kullanilan toplam: {total}"
        )

    with ann_path.open("w", encoding="utf-8", newline="\n") as f_ann:
        for idx0, (p_gt, p_lq) in enumerate(zip(gt_paths, lq_paths), start=1):
            clip_idx = (idx0 - 1) // clip_len + 1
            within = (idx0 - 1) % clip_len + 1
            clip_name = f"{clip_idx:05d}"
            clip_dir_gt = original_root / clip_name
            clip_dir_lq = corrupted_root / clip_name
            if within == 1:
                clip_dir_gt.mkdir(parents=True, exist_ok=True)
                clip_dir_lq.mkdir(parents=True, exist_ok=True)

            frame_gt = cv2.imread(str(p_gt), cv2.IMREAD_COLOR)
            frame_lq = cv2.imread(str(p_lq), cv2.IMREAD_COLOR)
            if frame_gt is None:
                raise RuntimeError(f"Okuma hatasi (GT): {p_gt}")
            if frame_lq is None:
                raise RuntimeError(f"Okuma hatasi (LQ): {p_lq}")

            out_name = f"im{within}.png"
            out_gt = clip_dir_gt / out_name
            out_lq = clip_dir_lq / out_name
            if not cv2.imwrite(str(out_gt), frame_gt):
                raise RuntimeError(f"Yazma hatasi: {out_gt}")
            if not cv2.imwrite(str(out_lq), frame_lq):
                raise RuntimeError(f"Yazma hatasi: {out_lq}")

        total_clips = total // clip_len
        for i in range(1, total_clips + 1):
            f_ann.write(f"{i:05d} {clip_len}\n")

    # Validate each clip has exact frame count on both sides
    for i in range(1, (total // clip_len) + 1):
        clip_name = f"{i:05d}"
        gt_dir = original_root / clip_name
        lq_dir = corrupted_root / clip_name
        gt_frames = sorted(gt_dir.glob("im*.png"))
        lq_frames = sorted(lq_dir.glob("im*.png"))
        if len(gt_frames) != clip_len or len(lq_frames) != clip_len:
            raise RuntimeError(
                f"Klip dogrulama hatasi: {clip_name} icin "
                f"GT={len(gt_frames)}, LQ={len(lq_frames)} (beklenen {clip_len})."
            )

    return total, total // clip_len


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Vimeo-90K style dataset (original/corrupted) from two aligned videos "
            "for MMagic BasicFramesDataset."
        )
    )
    src = parser.add_argument_group("source (choose one)")
    src.add_argument("--gt-video", help="Ground truth video path (orijinal.mp4)")
    src.add_argument("--lq-video", help="Low-quality/corrupted video path (bozuk.mp4)")
    src.add_argument("--gt-frames-dir", help="Folder containing GT frames (e.g. dataset/original)")
    src.add_argument("--lq-frames-dir", help="Folder containing LQ frames (e.g. dataset/corrupted)")
    src.add_argument("--frames-pattern", default="frame_*.jpg", help="Glob pattern for frames (default: frame_*.jpg)")
    src.add_argument(
        "--drop-last",
        action="store_true",
        help="If total frames is not divisible by clip length, drop remainder frames at the end.",
    )

    parser.add_argument("--out-root", default="dataset_vimeo", help="Output root folder (default: dataset_vimeo)")
    parser.add_argument("--clip-len", type=int, default=15, help="Frames per clip/folder (default: 15)")
    parser.add_argument(
        "--ann-path",
        default=str(Path("content") / "train_list.txt"),
        help="Annotation output path (default: content/train_list.txt)",
    )
    args = parser.parse_args()

    out_root = Path(args.out_root)
    ann_path = Path(args.ann_path)

    use_videos = bool(args.gt_video or args.lq_video)
    use_frames = bool(args.gt_frames_dir or args.lq_frames_dir)
    if use_videos == use_frames:
        raise SystemExit(
            "Kaynak secimi hatali. Ya --gt-video/--lq-video verin ya da --gt-frames-dir/--lq-frames-dir verin."
        )

    if use_videos:
        if not (args.gt_video and args.lq_video):
            raise SystemExit("--gt-video ve --lq-video birlikte verilmeli.")
        total_frames, total_clips = build_basicframes_dataset(
            gt_video=Path(args.gt_video),
            lq_video=Path(args.lq_video),
            out_root=out_root,
            clip_len=args.clip_len,
            ann_path=ann_path,
        )
    else:
        if not (args.gt_frames_dir and args.lq_frames_dir):
            raise SystemExit("--gt-frames-dir ve --lq-frames-dir birlikte verilmeli.")
        total_frames, total_clips = build_basicframes_dataset_from_frames(
            gt_frames_dir=Path(args.gt_frames_dir),
            lq_frames_dir=Path(args.lq_frames_dir),
            out_root=out_root,
            clip_len=args.clip_len,
            ann_path=ann_path,
            pattern=args.frames_pattern,
            drop_last=bool(args.drop_last),
        )

    print(f"Tamamlandi. Toplam kare cifti: {total_frames}, toplam klip: {total_clips}")
    print(f"Dataset root: {out_root.resolve()}")
    print(f"Annotation:   {ann_path.resolve()}")

    # MMagic config snippet (BasicFramesDataset):
    # data_root = r"/content/dataset"  # or the absolute path you used in --out-root
    # data_prefix = dict(gt="original", lq="corrupted")
    # ann_file = r"/content/train_list.txt"


if __name__ == "__main__":
    main()
