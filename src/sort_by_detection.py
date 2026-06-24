"""
Sorts images into "has bins" and "no bins" by running YOLO bin detection.

Runs the trained model over every image in --images and places each into
<output>/has_bins or <output>/no_bins depending on whether the model detects any
trash_bin (class 0) above --conf. Useful for triaging freshly fetched Street View
images before annotation, since many panoramas contain no bin at all.

Files are copied by default (the source is left intact). Pass --move to relocate
them instead — that turns the queue into a true split with no duplicates.

Run from the project root:
    py -3.14 -m src.sort_by_detection
    py -3.14 -m src.sort_by_detection --images data/to_annotate --move
    py -3.14 -m src.sort_by_detection --weights models/trained/trash_bin_yolo11n_best.pt
"""

import argparse
import shutil
from pathlib import Path

CLASS_TRASH_BIN = 0
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_IMAGES  = Path("data/to_annotate")
DEFAULT_WEIGHTS = Path("models/trained/colab_seg/weights/best.pt")
DEFAULT_OUTPUT  = Path("data/sorted")
DEFAULT_CONF    = 0.25


def collect_images(images_path: Path) -> list[Path]:
    """Returns image files directly under images_path (non-recursive)."""
    if not images_path.is_dir():
        raise FileNotFoundError(f"Fant ikke bildemappe: {images_path}")
    return sorted(
        p for p in images_path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def detects_bin(seg_model: object, image_path: Path, conf: float) -> bool:
    """True if the model detects at least one trash_bin (class 0) above conf.

    predict already filters by conf, so any returned class-0 box counts. The seg
    model also predicts class 1 (ground), which is ignored here. An unreadable
    image counts as no detection rather than crashing.
    """
    try:
        result = seg_model.predict(str(image_path), conf=conf, verbose=False)[0]
    except Exception:
        return False
    if result.boxes is None or len(result.boxes) == 0:
        return False
    classes = result.boxes.cls.cpu().numpy().astype(int)
    return bool((classes == CLASS_TRASH_BIN).any())


def place(src: Path, dest_dir: Path, move: bool) -> bool:
    """Copies or moves src into dest_dir. Returns False if a file already exists."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists():
        return False
    if move:
        shutil.move(str(src), str(dest))
    else:
        shutil.copy2(str(src), str(dest))
    return True


def sort_images(images_path: Path, weights: Path, output_dir: Path,
                conf: float, move: bool, limit: int | None) -> None:
    from ultralytics import YOLO

    if not weights.exists():
        raise FileNotFoundError(f"Fant ikke vekter: {weights}")

    images = collect_images(images_path)
    if limit is not None:
        images = images[:limit]
    print(f"Bilder å sortere: {len(images)}")
    if not images:
        return

    print(f"Laster modell: {weights}")
    seg_model = YOLO(str(weights))

    has_dir = output_dir / "has_bins"
    no_dir = output_dir / "no_bins"
    action = "Flytter" if move else "Kopierer"

    n_has = n_no = n_skip = 0
    for idx, image_path in enumerate(images):
        found = detects_bin(seg_model, image_path, conf)
        dest_dir = has_dir if found else no_dir
        if place(image_path, dest_dir, move):
            if found:
                n_has += 1
            else:
                n_no += 1
        else:
            n_skip += 1
            print(f"  Finnes allerede, hopper over: {image_path.name}")
        print(f"[{idx + 1}/{len(images)}] {image_path.name}: "
              f"{'kasse' if found else 'ingen kasse'}")

    print(f"\nFerdig ({action.lower()}):")
    print(f"  med kasse:   {n_has}  -> {has_dir}")
    print(f"  uten kasse:  {n_no}  -> {no_dir}")
    if n_skip:
        print(f"  hoppet over (fantes fra før): {n_skip}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sorter bilder i has_bins/no_bins ved YOLO-deteksjon av søppelkasser."
    )
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES,
                        help=f"Mappe med bilder å sortere (default: {DEFAULT_IMAGES})")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                        help=f"YOLO-vekter (default: {DEFAULT_WEIGHTS})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Mappe for has_bins/ og no_bins/ (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF,
                        help=f"Konfidensterskel for deteksjon (default: {DEFAULT_CONF})")
    parser.add_argument("--move", action="store_true",
                        help="Flytt filene i stedet for å kopiere (ekte splitt, ingen duplikater)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stopp etter N bilder")
    args = parser.parse_args()

    sort_images(
        images_path=args.images,
        weights=args.weights,
        output_dir=args.output,
        conf=args.conf,
        move=args.move,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
