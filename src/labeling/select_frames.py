"""
Copies a selection of images from data/raw/images/ into data/to_annotate/.

Only copies images that are not already staged. Does not modify raw data.

Run from the project root:
    python -m src.labeling.select_frames               # copies 20 random images
    python -m src.labeling.select_frames --count 50
    python -m src.labeling.select_frames --all
"""

import argparse
import random
import shutil
from pathlib import Path

RAW_IMAGES_DIR = Path("data/raw/images")
TO_ANNOTATE_DIR = Path("data/to_annotate")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def select_frames(count: int | None = 20, seed: int | None = None) -> list[Path]:
    """
    Copies up to `count` unstaged images to TO_ANNOTATE_DIR.
    Pass count=None to copy all available images.
    Returns list of copied file paths.
    """
    if not RAW_IMAGES_DIR.exists():
        raise FileNotFoundError(f"Raw images directory not found: {RAW_IMAGES_DIR}")

    TO_ANNOTATE_DIR.mkdir(parents=True, exist_ok=True)

    candidates = [
        p for p in sorted(RAW_IMAGES_DIR.iterdir())
        if p.suffix.lower() in IMAGE_EXTENSIONS
        and not (TO_ANNOTATE_DIR / p.name).exists()
    ]

    if not candidates:
        print("No new images to stage — all raw images are already in to_annotate/.")
        return []

    if count is None:
        selected = candidates
    else:
        if seed is not None:
            random.seed(seed)
        selected = random.sample(candidates, min(count, len(candidates)))

    copied = []
    for src in sorted(selected):
        dst = TO_ANNOTATE_DIR / src.name
        shutil.copy2(src, dst)
        copied.append(dst)
        print(f"  Staged: {src.name}")

    print(f"\nStaged {len(copied)} image(s) -> {TO_ANNOTATE_DIR}")
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage images for annotation.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--count", type=int, default=20,
                       help="Number of images to copy (default: 20)")
    group.add_argument("--all", action="store_true",
                       help="Copy all available images")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible selection")
    args = parser.parse_args()

    select_frames(
        count=None if args.all else args.count,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
