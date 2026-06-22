"""
Interactive review tool for the YOLO-seg dataset (classes: 0 trash_bin, 1 ground).

Unlike the batch labeller (src.labeling.sam2_seg_autolabel), this runs SAM2 and
the ground model on ONE image at a time and shows the proposed masks in an OpenCV
window so you approve/skip each image yourself. It reuses the exact same pipeline,
so an accepted label is byte-identical to what the batch would have written.

While you look at the current image, the next few images are segmented ahead in a
background thread (a look-ahead buffer, default 5). After the first image,
approving feels near-instant — and the buffer absorbs bursts where you blast
through several quick decisions in a row.

Controls (shown in the window):
    a / Enter / Space   approve  -> writes image + label + preview to the output set
    s                   skip     -> nothing saved; the image reappears next run
    f                   flag     -> log to review_flags.csv and skip (manual-fix pile)
    o                   toggle the mask overlay on/off (compare against the raw image)
    q / Esc             quit     -> progress is saved; resumes where you left off

It only reads from --source; never writes there and never touches
data/annotated_backup. Images that already have a seg label are skipped (resume),
unless you pass --overwrite.

The OpenCV window needs a local display, so run this on your own machine (not on
a headless Colab). On this CPU-only box the first image takes a few seconds; the
prefetch hides the wait on the rest.

Run from the project root:
    py -3.14 -m src.labeling.sam2_seg_review                 # full review
    py -3.14 -m src.labeling.sam2_seg_review --splits val    # one split only
"""

import argparse
import csv
import shutil
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import torch

from src.labeling.sam2_seg_autolabel import (
    DEFAULT_SAM_MODEL,
    DEFAULT_SEMANTIC_MODEL,
    OUTPUT_DIR,
    REVIEW_COLUMNS,
    SOURCE_DIR,
    SPLITS,
    BIN_CLIP_MARGIN,
    SegComputation,
    compute_seg_for_image,
    list_split_images,
    load_sam,
    load_semantic,
    make_preview,
    prepare_output_dirs,
    read_bin_boxes,
    write_seg_label,
)

WINDOW = "seg-review"
ACCEPT_KEYS = (ord("a"), 13, 32)   # a, Enter, Space
QUIT_KEYS   = (ord("q"), 27)       # q, Esc
PREFETCH_DEFAULT = 5


def build_worklist(source: Path, output: Path, splits: tuple[str, ...],
                   overwrite: bool) -> list[tuple[str, Path]]:
    """Returns (split, image_path) pairs still needing a label, in split order."""
    worklist: list[tuple[str, Path]] = []
    for split in splits:
        for image_path in list_split_images(source, split):
            out_label = output / "labels" / split / f"{image_path.stem}.txt"
            if out_label.exists() and not overwrite:
                continue
            worklist.append((split, image_path))
    return worklist


def process_one(item: tuple[str, Path], source: Path, sam: object, processor: object,
                semantic: object, ground_ids: list[int], device: str,
                clip_margin: float):
    """Loads one image and computes its segmentation. Returns None if unreadable."""
    split, image_path = item
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        height, width = image.shape[:2]
        label_path = source / "labels" / split / f"{image_path.stem}.txt"
        boxes = read_bin_boxes(label_path, width, height)
        comp = compute_seg_for_image(image, boxes, sam, image_path, processor,
                                     semantic, ground_ids, device, clip_margin)
        return image, boxes, comp
    except Exception as error:  # keep the loop alive on a single bad image
        print(f"FEIL ved {image_path.name}: {error}")
        return None


def draw_hud(image, lines: list[str]):
    """Draws status lines bottom-left with a dark outline for readability."""
    y = image.shape[0] - 10
    for line in reversed(lines):
        cv2.putText(image, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y -= 22


def append_review_rows(review_path: Path, rows: list[dict]) -> None:
    """Appends flag rows, writing the header only if the file does not exist yet."""
    if not rows:
        return
    write_header = not review_path.exists()
    with open(review_path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def save_accepted(output: Path, split: str, image_path: Path, image,
                  boxes, comp: SegComputation) -> None:
    out_label   = output / "labels"   / split / f"{image_path.stem}.txt"
    out_image   = output / "images"   / split / image_path.name
    out_preview = output / "previews" / split / f"{image_path.stem}.jpg"
    shutil.copy2(image_path, out_image)
    write_seg_label(out_label, comp.bin_polygons, comp.ground_polygons)
    preview = make_preview(image, comp.clipped_masks, comp.ground_mask, boxes,
                           comp.flagged)
    cv2.imwrite(str(out_preview), preview)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactively review SAM2 + ground masks and approve YOLO-seg labels."
    )
    parser.add_argument("--source", type=Path, default=SOURCE_DIR,
                        help=f"Detection dataset to read images and boxes from (default: {SOURCE_DIR})")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR,
                        help=f"Where to write the seg dataset (default: {OUTPUT_DIR})")
    parser.add_argument("--sam-model", type=str, default=DEFAULT_SAM_MODEL,
                        help=f"SAM2 weights for bin masks (default: {DEFAULT_SAM_MODEL})")
    parser.add_argument("--semantic-model", type=str, default=DEFAULT_SEMANTIC_MODEL,
                        help=f"HF ADE20K semantic model for ground (default: {DEFAULT_SEMANTIC_MODEL})")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS),
                        help="Which splits to review (default: train val test)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after reviewing N images this session")
    parser.add_argument("--overwrite", action="store_true",
                        help="Also review images whose seg label already exists")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device, e.g. cpu or cuda (default: auto-detect)")
    parser.add_argument("--clip-margin", type=float, default=BIN_CLIP_MARGIN,
                        help=f"Expand each box by this fraction before clipping the bin mask (default: {BIN_CLIP_MARGIN})")
    parser.add_argument("--prefetch", type=int, default=PREFETCH_DEFAULT,
                        help=f"How many images to segment ahead in the background (default: {PREFETCH_DEFAULT})")
    args = parser.parse_args()

    if not args.source.exists():
        print(f"Kildemappe ikke funnet: {args.source}")
        return

    worklist = build_worklist(args.source, args.output, tuple(args.splits), args.overwrite)
    if not worklist:
        print("Ingenting å gjennomgå — alle bilder har allerede en seg-label "
              "(bruk --overwrite for å se dem på nytt).")
        return

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Enhet: {device}")
    print(f"Laster SAM2 ({args.sam_model}) ...")
    sam = load_sam(args.sam_model)
    print(f"Laster semantisk modell ({args.semantic_model}) ...")
    processor, semantic, ground_ids = load_semantic(args.semantic_model, device)
    if not ground_ids:
        print("ADVARSEL: fant ingen ground-klasser i modellen — sjekk --semantic-model.")
    else:
        names = ", ".join(semantic.config.id2label[i] for i in ground_ids)
        print(f"Ground-klasser: {names}")

    prepare_output_dirs(args.output, tuple(args.splits))
    review_path = args.output / "review_flags.csv"
    print(f"\n{len(worklist)} bilde(r) å gjennomgå. "
          f"Taster: a=godkjenn  s=hopp over  f=flagg  o=overlay  q=avslutt")

    accepted = skipped = flagged = 0
    limit = args.limit if args.limit is not None else len(worklist)
    last_index = min(limit, len(worklist))

    prefetch = max(1, args.prefetch)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    # One worker only: the SAM2 and ground model objects are shared and not safe
    # to call from several threads at once. The look-ahead depth comes from the
    # buffer below, not from extra workers — the single worker keeps grinding
    # ahead so up to `prefetch` images are ready by the time you reach them.
    with ThreadPoolExecutor(max_workers=1) as pool:
        compute = lambda item: process_one(item, args.source, sam, processor,
                                           semantic, ground_ids, device, args.clip_margin)
        pending: deque = deque()
        submitted = 0

        def top_up() -> None:
            nonlocal submitted
            while len(pending) < prefetch and submitted < last_index:
                pending.append(pool.submit(compute, worklist[submitted]))
                submitted += 1

        top_up()
        quit_requested = False
        for index in range(last_index):
            split, image_path = worklist[index]
            future = pending.popleft()
            top_up()
            result = future.result()

            if result is None:
                print(f"  hopper over (kan ikke behandle): {image_path.name}")
                continue
            image, boxes, comp = result

            show_overlay = True
            decision = None
            while decision is None:
                if show_overlay:
                    display = make_preview(image, comp.clipped_masks, comp.ground_mask,
                                           boxes, comp.flagged)
                else:
                    display = image.copy()
                draw_hud(display, [
                    f"{index + 1}/{last_index}  [{split}]  {image_path.name}",
                    f"kasser: {len(comp.bin_polygons)}   bakke: {len(comp.ground_polygons)}"
                    f"{'   FLAGGET' if comp.flagged else ''}",
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
                quit_requested = True
                for queued in pending:
                    queued.cancel()
                break
            if decision == "accept":
                save_accepted(args.output, split, image_path, image, boxes, comp)
                append_review_rows(review_path, [
                    {"filename": image_path.name, "split": split, **flag}
                    for flag in comp.flags
                ])
                accepted += 1
                print(f"  godkjent: {image_path.name}"
                      f"{'  [FLAGGET]' if comp.flagged else ''}")
            elif decision == "flag":
                rows = [{"filename": image_path.name, "split": split,
                         "bin_index": -1, "sam_conf": "", "reason": "manual_flag"}]
                rows += [{"filename": image_path.name, "split": split, **flag}
                         for flag in comp.flags]
                append_review_rows(review_path, rows)
                flagged += 1
                print(f"  flagget: {image_path.name}")
            else:
                skipped += 1

    cv2.destroyAllWindows()
    print(f"\nFerdig denne økten: {accepted} godkjent, {skipped} hoppet over, "
          f"{flagged} flagget."
          f"{'  (avsluttet før slutten)' if quit_requested else ''}")
    if accepted:
        print(f"Datasett: {args.output}  (forhåndsvisninger i {args.output}/previews)")


if __name__ == "__main__":
    main()
