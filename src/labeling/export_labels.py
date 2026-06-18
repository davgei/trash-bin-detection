"""
Exports verified image-label pairs from data/to_annotate/ into the master pool
(data/annotated_backup/images/train and labels/train).

Hard examples (images where YOLO was wrong during assisted annotation) are
duplicated N extra times in the pool so YOLO sees them more often in training.

After export, prepare_dataset.py rebuilds the active train/val/test split from
the pool using content-hash deduplication and hard-example-aware splitting.

Run from the project root:
    python -m src.labeling.export_labels
    python -m src.labeling.export_labels --hard-repeat 4
"""

import argparse
import json
import shutil
from pathlib import Path

TO_ANNOTATE_DIR = Path("data/to_annotate")
POOL_DIR        = Path("data/annotated_backup")   # master pool; prepare_dataset.py splits from here
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


def _copy_to_pool(img: Path, label: Path, repeat: int) -> tuple[int, int]:
    """
    Copies image+label into POOL_DIR/images/train and POOL_DIR/labels/train.
    Hard examples (repeat > 0) are duplicated with _h2, _h3 … suffixes.
    Returns (base_copied, hard_copies_written).
    """
    dst_img   = POOL_DIR / "images" / "train" / img.name
    dst_label = POOL_DIR / "labels" / "train" / label.name

    if dst_img.exists():
        return 0, 0

    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img, dst_img)
    shutil.copy2(label, dst_label)

    hard_copies = 0
    for n in range(2, repeat + 2):
        dup_img   = POOL_DIR / "images" / "train" / f"{img.stem}_h{n}{img.suffix}"
        dup_label = POOL_DIR / "labels" / "train" / f"{label.stem}_h{n}.txt"
        if not dup_img.exists():
            shutil.copy2(img, dup_img)
            shutil.copy2(label, dup_label)
            hard_copies += 1

    return 1, hard_copies


def export_labels(hard_repeat: int = 3) -> dict[str, int]:
    """
    Copies newly labeled pairs from data/to_annotate/ into the master pool
    (data/annotated_backup/images/train and labels/train).

    Hard examples (flagged by annotate.py) are duplicated `hard_repeat` extra
    times so YOLO sees them more often in the next training run.

    After export, run prepare_dataset.py to rebuild the active train/val/test split.
    """
    pairs = _find_labeled_pairs(TO_ANNOTATE_DIR)
    if not pairs:
        print(f"No labeled images found in {TO_ANNOTATE_DIR}")
        return {"added": 0, "hard_copies": 0}

    print(f"Exporting {len(pairs)} labeled pair(s) to pool ({POOL_DIR}) ...")
    added = 0
    hard_total = 0
    skipped = 0
    for img, lbl in pairs:
        repeat = hard_repeat if _is_hard_example(img) else 0
        base, hard = _copy_to_pool(img, lbl, repeat)
        if base:
            added += 1
            hard_total += hard
        else:
            skipped += 1

    msg = f"  {added} new image(s) added to pool"
    if hard_total:
        msg += f" (+{hard_total} hard example duplicate(s))"
    if skipped:
        msg += f", {skipped} already in pool (skipped)"
    print(msg)

    for img, lbl in pairs:
        img.unlink(missing_ok=True)
        img.with_suffix(".txt").unlink(missing_ok=True)
        img.with_suffix(".json").unlink(missing_ok=True)
    print(f"  Cleaned up {len(pairs)} annotated file(s) from {TO_ANNOTATE_DIR}")

    print(f"\nPool updated -> run 'python -m src.prepare_dataset' to rebuild train/val/test splits")
    return {"added": added, "hard_copies": hard_total}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export labeled images to the dataset pool.")
    parser.add_argument("--hard-repeat", type=int, default=3,
                        help="Extra copies of hard examples in the pool (default: 3)")
    args = parser.parse_args()
    export_labels(hard_repeat=args.hard_repeat)


if __name__ == "__main__":
    main()
