from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def _sorted_images(img_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    images.sort(key=lambda p: p.name)
    return images


def images_to_video(img_dir: Path, out_path: Path, fps: float, codec: str = "mp4v") -> None:
    images = _sorted_images(img_dir)
    if not images:
        raise ValueError(f"No images found in: {img_dir}")

    first = cv2.imread(str(images[0]))
    if first is None:
        raise ValueError(f"Failed to read first image: {images[0]}")

    h, w = first.shape[:2]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {out_path}")

    try:
        for p in images:
            frame = cv2.imread(str(p))
            if frame is None:
                raise ValueError(f"Failed to read image: {p}")
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert MOT20 img1 frames into mp4 videos")
    parser.add_argument(
        "--root",
        type=str,
        default=str(Path("data/MOT20/train")),
        help="MOT20 train root folder (contains MOT20-01, MOT20-02, ...)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path("data/MOT20/videos")),
        help="Output directory for generated videos",
    )
    parser.add_argument("--fps", type=float, default=25.0, help="FPS for output videos")
    parser.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        help="FourCC codec (e.g. mp4v, avc1). mp4v is most portable.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.output_dir)

    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    seq_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("MOT20-")])
    if not seq_dirs:
        raise ValueError(f"No MOT20-* dirs found under: {root}")

    for seq_dir in seq_dirs:
        img_dir = seq_dir / "img1"
        if not img_dir.exists():
            print(f"[SKIP] No img1 folder: {img_dir}")
            continue

        out_path = out_dir / f"{seq_dir.name}.mp4"
        print(f"[INFO] Writing: {out_path} (from {img_dir})")
        images_to_video(img_dir, out_path, fps=float(args.fps), codec=str(args.codec))

    print("[DONE]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
