"""
Extracts frames from a video file and saves them as JPEGs to data/raw/images/.

Run from the project root:
    python -m src.extract_frames --video data/raw/videos/recording.mp4
    python -m src.extract_frames --video data/raw/videos/recording.mp4 --every 15
"""

import argparse
import cv2
from pathlib import Path

RAW_IMAGES_DIR = Path("data/raw/images")


def extract_frames(video_path: Path, every_n_frames: int = 30, prefix: str = "") -> int:
    """
    Saves one frame every `every_n_frames` from `video_path` to RAW_IMAGES_DIR.
    Skips frames that already exist on disk.
    Returns the number of newly saved frames.
    """
    RAW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    stem = prefix or video_path.stem

    print(f"Video : {video_path.name}  ({total} frames @ {fps:.1f} fps)")
    print(f"Saving every {every_n_frames} frame(s) -> {RAW_IMAGES_DIR}")

    saved = 0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % every_n_frames == 0:
            out_path = RAW_IMAGES_DIR / f"{stem}_frame{frame_idx:06d}.jpg"
            if not out_path.exists():
                cv2.imwrite(str(out_path), frame)
                saved += 1
        frame_idx += 1

    cap.release()
    print(f"Saved {saved} new frame(s).")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames from a video file.")
    parser.add_argument("--video", type=Path, required=True,
                        help="Path to the input video file")
    parser.add_argument("--every", type=int, default=30,
                        help="Extract one frame every N frames (default: 30)")
    parser.add_argument("--prefix", type=str, default="",
                        help="Filename prefix for saved frames (default: video stem)")
    args = parser.parse_args()

    if not args.video.exists():
        print(f"File not found: {args.video}")
        return

    extract_frames(args.video, every_n_frames=args.every, prefix=args.prefix)


if __name__ == "__main__":
    main()
