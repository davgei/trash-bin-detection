"""
Rebuilds disjoint train/val/test splits for the annotated dataset.

The dataset uses oversampling for hard-example mining: images the model
previously misclassified are duplicated (e.g. the '_h2'/'_h3'/'_h4' copies) so
the model trains on them several times. Those duplicates must stay entirely in
the training split. The val and test splits must contain only distinct images
that appear nowhere else, so the evaluation numbers stay honest.

Policy:
  - Any image whose content appears more than once  -> all copies go to train.
  - Images whose content appears exactly once        -> split into train/val/test.

See docs/dataset.md for the full rationale.

Run from the project root:
    python -m src.prepare_dataset
    python -m src.prepare_dataset --val-frac 0.15 --test-frac 0.15 --seed 1
"""

import argparse
import hashlib
import random
import shutil
from collections import defaultdict
from pathlib import Path

ANNOTATED_DIR = Path("data/annotated")
BACKUP_DIR = Path("data/annotated_backup")
SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def _content_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def group_pairs_by_content(source: Path) -> dict[str, list[tuple[Path, Path]]]:
    """
    Walks every image under source/images/* and groups image+label pairs by image
    content hash. Identical images saved under different names land in one group.
    Raises if an image has no matching label file.
    """
    groups: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for split in SPLITS:
        for image_path in sorted((source / "images" / split).iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label_path = source / "labels" / split / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"No label file for image: {image_path}")
            groups[_content_hash(image_path)].append((image_path, label_path))
    return groups


def partition(
    groups: dict[str, list[tuple[Path, Path]]],
    val_frac: float,
    test_frac: float,
    seed: int,
) -> dict[str, list[tuple[Path, Path]]]:
    """
    Locks every duplicated-content group into train, then splits the
    single-copy images into train/val/test by the given fractions.
    """
    train_locked: list[tuple[Path, Path]] = []
    singles: list[tuple[Path, Path]] = []
    for pairs in groups.values():
        if len(pairs) > 1:
            train_locked.extend(pairs)
        else:
            singles.append(pairs[0])

    random.Random(seed).shuffle(singles)
    n_val = round(len(singles) * val_frac)
    n_test = round(len(singles) * test_frac)

    val = singles[:n_val]
    test = singles[n_val:n_val + n_test]
    train_singles = singles[n_val + n_test:]

    return {
        "train": train_locked + train_singles,
        "val": val,
        "test": test,
    }


def clear_split_dirs(root: Path) -> None:
    """Removes all files (images, labels, caches, .gitkeep) from the split dirs."""
    for kind in ("images", "labels"):
        for split in SPLITS:
            split_dir = root / kind / split
            split_dir.mkdir(parents=True, exist_ok=True)
            for path in split_dir.iterdir():
                if path.is_file():
                    path.unlink()
    for cache in root.glob("labels/*.cache"):
        cache.unlink()


def write_splits(root: Path, splits: dict[str, list[tuple[Path, Path]]]) -> None:
    """Copies each image+label pair into its assigned split folder."""
    for split, pairs in splits.items():
        for image_path, label_path in pairs:
            shutil.copy2(image_path, root / "images" / split / image_path.name)
            shutil.copy2(label_path, root / "labels" / split / label_path.name)


def verify_no_content_leak(root: Path) -> None:
    """Asserts no image content in val or test also appears in train."""
    content_by_split: dict[str, set[str]] = {}
    for split in SPLITS:
        content_by_split[split] = {
            _content_hash(p)
            for p in (root / "images" / split).iterdir()
            if p.suffix.lower() in IMAGE_SUFFIXES
        }
    for split in ("val", "test"):
        leak = content_by_split[split] & content_by_split["train"]
        if leak:
            raise AssertionError(f"{len(leak)} image(s) leak between train and {split}")
    if content_by_split["val"] & content_by_split["test"]:
        raise AssertionError("Image content shared between val and test")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild disjoint dataset splits.")
    parser.add_argument("--val-frac", type=float, default=0.15,
                        help="Fraction of single-copy images for validation (default: 0.15)")
    parser.add_argument("--test-frac", type=float, default=0.15,
                        help="Fraction of single-copy images for test (default: 0.15)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for the split (default: 0)")
    args = parser.parse_args()

    if not ANNOTATED_DIR.exists():
        raise FileNotFoundError(f"Dataset not found: {ANNOTATED_DIR}")

    if BACKUP_DIR.exists():
        print(f"Using existing backup as source of truth: {BACKUP_DIR}")
    else:
        print(f"Backing up {ANNOTATED_DIR} -> {BACKUP_DIR} ...")
        shutil.copytree(ANNOTATED_DIR, BACKUP_DIR)

    groups = group_pairs_by_content(BACKUP_DIR)
    n_dup = sum(1 for pairs in groups.values() if len(pairs) > 1)
    print(f"Found {len(groups)} unique images "
          f"({n_dup} duplicated -> locked to train, {len(groups) - n_dup} single-copy).")

    splits = partition(groups, args.val_frac, args.test_frac, args.seed)
    clear_split_dirs(ANNOTATED_DIR)
    write_splits(ANNOTATED_DIR, splits)
    verify_no_content_leak(ANNOTATED_DIR)

    print("\nNew splits (image files written to disk):")
    for split in SPLITS:
        on_disk = sum(
            1 for p in (ANNOTATED_DIR / "images" / split).iterdir()
            if p.suffix.lower() in IMAGE_SUFFIXES
        )
        print(f"  {split:5s}: {on_disk} files")
    print(f"\nSource pool kept at: {BACKUP_DIR}")


if __name__ == "__main__":
    main()
