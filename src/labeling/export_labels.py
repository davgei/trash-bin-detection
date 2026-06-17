"""
Exports verified image-label pairs from data/to_annotate/ into the final
dataset structure under data/annotated/, split into train / val / test.

Only images that have a matching .txt label file are exported.
Existing files in data/annotated/ are NOT overwritten.

Hard examples (images where YOLO was wrong during assisted annotation) are
duplicated N extra times in the train split so YOLO sees them more often.

Run from the project root:
    python -m src.labeling.export_labels
    python -m src.labeling.export_labels --split 0.8 0.1 0.1
    python -m src.labeling.export_labels --hard-repeat 4
"""

import argparse
import json
import random
import shutil
from pathlib import Path

TO_ANNOTATE_DIR = Path("data/to_annotate")
ANNOTATED_DIR   = Path("data/annotated")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _find_labeled_pairs(directory: Path) -> list[tuple[Path, Path]]:
    """Returns (image, label) pairs where both files exist."""
    pairs = []
    for img in sorted(directory.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = img.with_suffix(".txt")
        if label.exists():
            pairs.append((img, label))
    return pairs


def _is_hard_example(img: Path) -> bool:
    """Returns True if annotate.py flagged this image as a hard example."""
    sidecar = img.with_suffix(".json")
    if not sidecar.exists():
        return False
    try:
        return json.loads(sidecar.read_text()).get("hard_example", False)
    except (json.JSONDecodeError, OSError):
        return False


def _copy_hard_duplicates(img: Path, label: Path, split: str, repeat: int) -> int:
    """
    Copies the image and label `repeat` extra times into the train split,
    with _h2, _h3 … suffixes. Returns the number of copies actually written.
    """
    written = 0
    for n in range(2, repeat + 2):
        dst_img   = ANNOTATED_DIR / "images" / split / f"{img.stem}_h{n}{img.suffix}"
        dst_label = ANNOTATED_DIR / "labels" / split / f"{label.stem}_h{n}.txt"
        if dst_img.exists():
            continue
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, dst_img)
        shutil.copy2(label, dst_label)
        written += 1
    return written


def _copy_to_split(img: Path, label: Path, split: str) -> bool:
    """
    Copies image and label into data/annotated/{images,labels}/{split}/.
    Returns False if the destination already exists (skips without overwriting).
    """
    dst_img   = ANNOTATED_DIR / "images" / split / img.name
    dst_label = ANNOTATED_DIR / "labels" / split / label.name

    if dst_img.exists():
        return False

    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img, dst_img)
    shutil.copy2(label, dst_label)
    return True


def export_labels(train_ratio: float = 0.7,
                  val_ratio: float   = 0.2,
                  test_ratio: float  = 0.1,
                  seed: int = 42,
                  hard_repeat: int = 3) -> dict[str, int]:
    """
    Splits labeled pairs and copies them to data/annotated/.
    Hard examples (flagged by annotate.py) are duplicated `hard_repeat` extra
    times in the train split so YOLO trains on them more often.
    Returns a dict with image counts per split.
    """
    pairs = _find_labeled_pairs(TO_ANNOTATE_DIR)
    if not pairs:
        print(f"No labeled images found in {TO_ANNOTATE_DIR}")
        return {"train": 0, "val": 0, "test": 0}

    random.seed(seed)
    random.shuffle(pairs)

    n       = len(pairs)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    splits = [
        ("train", pairs[:n_train]),
        ("val",   pairs[n_train:n_train + n_val]),
        ("test",  pairs[n_train + n_val:]),
    ]

    counts: dict[str, int] = {}
    print(f"Exporting {n} labeled pair(s) from {TO_ANNOTATE_DIR} ...")
    for split_name, split_pairs in splits:
        copied = 0
        hard_copies = 0
        for img, lbl in split_pairs:
            if _copy_to_split(img, lbl, split_name):
                copied += 1
                if split_name == "train" and _is_hard_example(img):
                    hard_copies += _copy_hard_duplicates(img, lbl, split_name, hard_repeat)
        skipped = len(split_pairs) - copied
        counts[split_name] = copied
        msg = f"  {split_name}: {copied} copied"
        if hard_copies:
            msg += f" (+{hard_copies} hard example duplicate(s))"
        if skipped:
            msg += f", {skipped} already existed (skipped)"
        print(msg)

    print(f"\nDone -> {ANNOTATED_DIR}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Export labeled images to annotated dataset.")
    parser.add_argument("--split", type=float, nargs=3,
                        default=[0.7, 0.2, 0.1],
                        metavar=("TRAIN", "VAL", "TEST"),
                        help="Split ratios, must sum to 1.0 (default: 0.7 0.2 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible split (default: 42)")
    parser.add_argument("--hard-repeat", type=int, default=3,
                        help="Extra copies of hard examples in train split (default: 3)")
    args = parser.parse_args()

    train_r, val_r, test_r = args.split
    if abs(train_r + val_r + test_r - 1.0) > 0.01:
        parser.error(f"Split ratios must sum to 1.0 (got {train_r + val_r + test_r:.2f})")

    export_labels(train_ratio=train_r, val_ratio=val_r,
                  test_ratio=test_r, seed=args.seed,
                  hard_repeat=args.hard_repeat)


if __name__ == "__main__":
    main()
