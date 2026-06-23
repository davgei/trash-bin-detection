"""
Lokal gjennomgang av forberegnede segmenteringsmasker.

Laster pkl + preview-bilde fra data/annotated_seg/.precompute/ — ingen ML-modeller
lastes. Forutsetter at notebooks/seg_precompute_review.ipynb er kjørt i Colab først.

Tastatur:
    a / Enter / Space   godkjenn -> skriver label + kopierer bilde til annotated_seg
    s                   hopp over permanent (.skip-markør)
    f                   flagg til manuell gjennomgang (.skip-markør + CSV-logg)
    o                   toggle overlay på/av (preview-bilde vs råbilde)
    q / Esc             avslutt; fremgang er lagret

Kjør fra prosjektroten:
    py -3.14 -m src.labeling.seg_cache_review
    py -3.14 -m src.labeling.seg_cache_review --splits train
"""

import argparse
import csv
import shutil
from pathlib import Path

import cv2

SOURCE_DIR = Path("data/annotated")
OUTPUT_DIR = Path("data/annotated_seg")
SPLITS     = ("train", "val", "test")

REVIEW_COLUMNS = ["filename", "split", "bin_index", "sam_conf", "reason"]
WINDOW         = "seg-review"
ACCEPT_KEYS    = (ord("a"), 13, 32)
QUIT_KEYS      = (ord("q"), 27)


def _label_path(output: Path, split: str, stem: str) -> Path:
    return output / "labels" / split / f"{stem}.txt"


def _skip_path(output: Path, split: str, stem: str) -> Path:
    return output / "labels" / split / f"{stem}.skip"


def _write_seg_label(path: Path, bin_polygons: list[list[float]],
                     ground_polygons: list[list[float]]) -> None:
    lines: list[str] = []
    for poly in bin_polygons:
        lines.append("0 " + " ".join(f"{v:.6f}" for v in poly))
    for poly in ground_polygons:
        lines.append("1 " + " ".join(f"{v:.6f}" for v in poly))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _append_review_rows(output: Path, rows: list[dict]) -> None:
    if not rows:
        return
    review_path = output / "review_flags.csv"
    write_header = not review_path.exists()
    with open(review_path, "a", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=REVIEW_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _draw_hud(img, lines: list[str]) -> None:
    y = img.shape[0] - 10
    for line in reversed(lines):
        cv2.putText(img, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y -= 22


def build_worklist(cache_dir: Path, output: Path,
                   splits: tuple[str, ...]) -> list[tuple[str, Path, Path]]:
    """Returns (split, pkl_path, preview_path) for images still needing review."""
    worklist: list[tuple[str, Path, Path]] = []
    for split in splits:
        cache_split = cache_dir / split
        if not cache_split.exists():
            continue
        for pkl in sorted(cache_split.glob("*.pkl")):
            stem = pkl.stem
            preview = cache_split / f"{stem}.jpg"
            if not preview.exists():
                print(f"  ADVARSEL: mangler preview for {stem} — hopper over (kjør forberegning på nytt)")
                continue
            if _label_path(output, split, stem).exists():
                continue
            if _skip_path(output, split, stem).exists():
                continue
            worklist.append((split, pkl, preview))
    return worklist


def main() -> None:
    import pickle

    parser = argparse.ArgumentParser(
        description="Gjennomgå forberegnede SAM2-masker lokalt (ingen ML nødvendig)."
    )
    parser.add_argument("--source", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS))
    parser.add_argument("--limit", type=int, default=None,
                        help="Stopp etter N bilder denne økten")
    args = parser.parse_args()

    cache_dir = args.output / ".precompute"
    if not cache_dir.exists():
        print("Cache-mappen finnes ikke. Kjør Colab-notebooken (forberegning) først.")
        return

    for kind in ("images", "labels", "previews"):
        for split in args.splits:
            (args.output / kind / split).mkdir(parents=True, exist_ok=True)

    worklist = build_worklist(cache_dir, args.output, tuple(args.splits))
    if not worklist:
        print("Ingen bilder å gjennomgå — alt er allerede godkjent eller hoppet over.")
        return

    limit = args.limit if args.limit is not None else len(worklist)
    worklist = worklist[:limit]
    print(f"{len(worklist)} bilde(r) i køen.")
    print("a/Enter/Space=godkjenn  s=hopp over  f=flagg  o=overlay  q=avslutt")

    accepted = skipped = flagged = 0
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    for index, (split, pkl_path, preview_path) in enumerate(worklist):
        with open(pkl_path, "rb") as fh:
            entry = pickle.load(fh)

        image_path = Path(entry["image_path"])
        stem = image_path.stem
        bin_polygons: list[list[float]] = entry["bin_polygons"]
        ground_polygons: list[list[float]] = entry["ground_polygons"]
        is_flagged: bool = entry["flagged"]
        flags: list[dict] = entry["flags"]

        # image_path from pkl may be an absolute Colab/Drive path — fall back to local source
        if not image_path.exists():
            image_path = args.source / "images" / split / image_path.name

        raw_image = cv2.imread(str(image_path))
        preview_image = cv2.imread(str(preview_path))

        if raw_image is None or preview_image is None:
            print(f"  Kan ikke lese bilde eller preview: {image_path.name} — hopper over")
            continue

        show_overlay = True
        decision = None

        while decision is None:
            display = (preview_image if show_overlay else raw_image).copy()
            _draw_hud(display, [
                f"{index + 1}/{len(worklist)}  [{split}]  {image_path.name}",
                f"kasser: {len(bin_polygons)}  bakke: {len(ground_polygons)}"
                + ("  FLAGGET" if is_flagged else ""),
            ])
            cv2.imshow(WINDOW, display)
            key = cv2.waitKey(0) & 0xFF
            if key in ACCEPT_KEYS:
                decision = "accept"
            elif key == ord("s"):
                decision = "skip"
            elif key == ord("f"):
                decision = "flag"
            elif key == ord("o"):
                show_overlay = not show_overlay
            elif key in QUIT_KEYS:
                decision = "quit"

        if decision == "quit":
            break

        if decision == "accept":
            out_label   = args.output / "labels"   / split / f"{stem}.txt"
            out_image   = args.output / "images"   / split / image_path.name
            out_preview = args.output / "previews" / split / f"{stem}.jpg"
            shutil.copy2(image_path, out_image)
            _write_seg_label(out_label, bin_polygons, ground_polygons)
            shutil.copy2(preview_path, out_preview)
            _append_review_rows(args.output, [
                {"filename": image_path.name, "split": split, **flag}
                for flag in flags
            ])
            accepted += 1
            print(f"  godkjent: {image_path.name}" + ("  [FLAGGET]" if is_flagged else ""))

        elif decision == "flag":
            rows = [{"filename": image_path.name, "split": split,
                     "bin_index": -1, "sam_conf": "", "reason": "manual_flag"}]
            rows += [{"filename": image_path.name, "split": split, **flag} for flag in flags]
            _append_review_rows(args.output, rows)
            _skip_path(args.output, split, stem).touch()
            flagged += 1
            print(f"  flagget: {image_path.name}")

        elif decision == "skip":
            _skip_path(args.output, split, stem).touch()
            skipped += 1
            print(f"  hoppet over: {image_path.name}")

    cv2.destroyAllWindows()
    print(f"\nFerdig: {accepted} godkjent, {skipped} hoppet over, {flagged} flagget.")
    if accepted:
        print(f"Datasett: {args.output}")


if __name__ == "__main__":
    main()
