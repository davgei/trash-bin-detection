"""
Builds a YOLO segmentation dataset (classes: 0 trash_bin, 1 ground) from the
existing detection dataset, without any manual mask drawing.

For each image in data/annotated/ it:
    1. Reads the existing YOLO bounding-box label (the human-verified bins).
    2. Uses each box as a prompt to SAM2 (local, free) -> an instance mask that
       covers the visible bin. The mask is clipped to a slightly expanded box so
       SAM2 cannot bleed onto a fence or object beside the bin.
    3. Runs a pretrained ADE20K semantic model and merges its road / sidewalk /
       earth / grass classes into one ground mask (bin pixels removed so the two
       classes never overlap).
    4. Flags uncertain bin masks (low SAM2 confidence, empty or tiny mask) into a
       review CSV for manual checking.
    5. Converts both classes to YOLO-seg polygons and writes a preview image with
       the masks drawn on top.

It only reads from --source (default data/annotated); it never writes there and
never touches data/annotated_backup. The new dataset mirrors the source's
train/val/test split exactly, so the content-hash disjointness and the
hard-example-in-train rule are preserved automatically.

The model weights download once and run locally on GPU if available, else CPU.

Run from the project root:
    py -3.14 -m src.labeling.sam2_seg_autolabel --limit 3      # quick smoke test
    py -3.14 -m src.labeling.sam2_seg_autolabel                # full dataset
"""

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

SOURCE_DIR     = Path("data/annotated")
OUTPUT_DIR     = Path("data/annotated_seg")
SPLITS         = ("train", "val", "test")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")

CLASS_TRASH_BIN = 0
CLASS_GROUND    = 1

DEFAULT_SAM_MODEL      = "sam2.1_b.pt"
DEFAULT_SEMANTIC_MODEL = "nvidia/segformer-b5-finetuned-ade-640-640"
GROUND_KEYWORDS        = ("road", "sidewalk", "pavement", "earth", "grass",
                          "path", "field", "land", "sand", "dirt track")

MIN_SAM_CONF      = 0.70   # below this the bin mask is flagged for review
MIN_BIN_AREA_FRAC = 0.10   # mask must cover at least this fraction of its box
BIN_CLIP_MARGIN   = 0.15   # expand the box by this fraction before clipping

BIN_MIN_POLY_AREA    = 30
GROUND_MIN_POLY_AREA = 200
POLY_EPS_FRAC        = 0.003

REVIEW_COLUMNS = ["filename", "split", "bin_index", "sam_conf", "reason"]


def list_split_images(source: Path, split: str) -> list[Path]:
    image_dir = source / "images" / split
    if not image_dir.exists():
        return []
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def read_bin_boxes(label_path: Path, width: int, height: int
                   ) -> list[tuple[float, float, float, float]]:
    """Reads the trash_bin boxes from a YOLO detection label as pixel xyxy."""
    boxes: list[tuple[float, float, float, float]] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        if int(float(parts[0])) != CLASS_TRASH_BIN:
            continue
        cx, cy, bw, bh = (float(v) for v in parts[1:])
        boxes.append((
            (cx - bw / 2) * width,
            (cy - bh / 2) * height,
            (cx + bw / 2) * width,
            (cy + bh / 2) * height,
        ))
    return boxes


def expand_box(box: tuple[float, float, float, float], width: int, height: int,
               margin: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    return (
        max(0, int(x1 - bw * margin)),
        max(0, int(y1 - bh * margin)),
        min(width, int(x2 + bw * margin)),
        min(height, int(y2 + bh * margin)),
    )


def clip_mask_to_box(mask: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    clipped = np.zeros_like(mask)
    clipped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return clipped


def mask_to_polygons(mask: np.ndarray, width: int, height: int, min_area: float,
                     eps_frac: float = POLY_EPS_FRAC,
                     largest_only: bool = False) -> list[list[float]]:
    """Converts a binary mask into normalised YOLO-seg polygons."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if largest_only and contours:
        contours = [max(contours, key=cv2.contourArea)]
    polygons: list[list[float]] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        eps = eps_frac * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2).astype(float)
        if len(approx) < 3:
            continue
        approx[:, 0] = np.clip(approx[:, 0] / width, 0.0, 1.0)
        approx[:, 1] = np.clip(approx[:, 1] / height, 0.0, 1.0)
        polygons.append(approx.flatten().tolist())
    return polygons


def assess_bin_mask(mask: np.ndarray, box: tuple[float, float, float, float],
                    conf: float) -> tuple[bool, str]:
    """Decides whether a bin mask needs manual review and why."""
    area = int(mask.sum())
    if area == 0:
        return True, "empty_mask"
    x1, y1, x2, y2 = box
    box_area = max((x2 - x1) * (y2 - y1), 1.0)
    reasons: list[str] = []
    if conf < MIN_SAM_CONF:
        reasons.append(f"low_conf({conf:.2f})")
    if area / box_area < MIN_BIN_AREA_FRAC:
        reasons.append("tiny_mask")
    return bool(reasons), ";".join(reasons)


def load_sam(model_path: str) -> object:
    from ultralytics import SAM
    return SAM(model_path)


def sam_bin_masks(sam: object, image_path: Path,
                  boxes: list[tuple[float, float, float, float]],
                  device: str) -> tuple[list[np.ndarray], list[float]]:
    """Runs SAM2 with one box prompt per bin, returns aligned masks and confidences."""
    if not boxes:
        return [], []
    results = sam(str(image_path), bboxes=[list(b) for b in boxes],
                  device=device, verbose=False)
    result = results[0]
    if result.masks is None:
        return [], []
    masks = [m.cpu().numpy().astype(np.uint8) for m in result.masks.data]
    if result.boxes is not None and result.boxes.conf is not None:
        confs = [float(c) for c in result.boxes.conf.cpu().tolist()]
    else:
        confs = [1.0] * len(masks)
    return masks, confs


def name_matches_ground(name: str) -> bool:
    """True if any ground keyword equals a whole synonym in an ADE20K label.

    Matches per comma-separated synonym (e.g. "land, ground, soil") rather than by
    substring, so "land" does not falsely match "kitchen island".
    """
    tokens = [token.strip().lower() for token in name.split(",")]
    return any(keyword in tokens for keyword in GROUND_KEYWORDS)


def load_semantic(model_name: str, device: str) -> tuple[object, object, list[int]]:
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_name).to(device).eval()
    ground_ids = [
        int(i) for i, name in model.config.id2label.items()
        if name_matches_ground(name)
    ]
    return processor, model, ground_ids


def semantic_ground_mask(processor: object, model: object, ground_ids: list[int],
                         rgb: np.ndarray, device: str) -> np.ndarray:
    height, width = rgb.shape[:2]
    inputs = processor(images=rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    upsampled = F.interpolate(logits, size=(height, width), mode="bilinear",
                              align_corners=False)
    prediction = upsampled.argmax(dim=1)[0].cpu().numpy()
    return np.isin(prediction, ground_ids).astype(np.uint8)


@dataclass
class SegComputation:
    """The full segmentation result for one image, ready to preview or save."""
    clipped_masks: list[np.ndarray]
    bin_polygons: list[list[float]]
    ground_mask: np.ndarray
    ground_polygons: list[list[float]]
    flags: list[dict]
    flagged: bool


def compute_seg_for_image(image: np.ndarray,
                          boxes: list[tuple[float, float, float, float]],
                          sam: object, image_path: Path, processor: object,
                          semantic: object, ground_ids: list[int], device: str,
                          clip_margin: float) -> SegComputation:
    """Runs SAM2 + the ground model on one image and returns masks, polygons and flags.

    Shared by the batch labeller and the interactive review tool so both produce
    byte-identical labels. Flags carry only bin_index / sam_conf / reason; the
    caller adds filename and split before writing the review CSV.
    """
    height, width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    masks, confs = sam_bin_masks(sam, image_path, boxes, device)

    flags: list[dict] = []
    flagged = False
    if len(masks) != len(boxes):
        flags.append({"bin_index": -1, "sam_conf": "",
                      "reason": f"mask_count_mismatch({len(masks)}/{len(boxes)})"})
        flagged = True

    clipped_masks: list[np.ndarray] = []
    bin_polygons: list[list[float]] = []
    for index in range(min(len(masks), len(boxes))):
        expanded = expand_box(boxes[index], width, height, clip_margin)
        clipped = clip_mask_to_box(masks[index], expanded)
        clipped_masks.append(clipped)
        is_flagged, reason = assess_bin_mask(clipped, boxes[index], confs[index])
        if is_flagged:
            flagged = True
            flags.append({"bin_index": index, "sam_conf": f"{confs[index]:.3f}",
                          "reason": reason})
        bin_polygons.extend(
            mask_to_polygons(clipped, width, height, BIN_MIN_POLY_AREA, largest_only=True)
        )

    ground_mask = semantic_ground_mask(processor, semantic, ground_ids, rgb, device)
    if clipped_masks:
        union = np.zeros((height, width), dtype=np.uint8)
        for mask in clipped_masks:
            union |= mask
        ground_mask[union.astype(bool)] = 0
    ground_polygons = mask_to_polygons(ground_mask, width, height, GROUND_MIN_POLY_AREA)

    return SegComputation(clipped_masks, bin_polygons, ground_mask, ground_polygons,
                          flags, flagged)


def make_preview(image: np.ndarray, bin_masks: list[np.ndarray],
                 ground_mask: np.ndarray,
                 boxes: list[tuple[float, float, float, float]],
                 flagged: bool) -> np.ndarray:
    overlay = image.copy()
    overlay[ground_mask.astype(bool)] = (0, 180, 0)
    for mask in bin_masks:
        overlay[mask.astype(bool)] = (0, 0, 220)
    blended = cv2.addWeighted(overlay, 0.45, image, 0.55, 0)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(blended, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 1)
    if flagged:
        cv2.putText(blended, "FLAGGED", (5, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)
    return blended


def write_seg_label(path: Path, bin_polygons: list[list[float]],
                    ground_polygons: list[list[float]]) -> None:
    lines: list[str] = []
    for polygon in bin_polygons:
        lines.append(f"{CLASS_TRASH_BIN} " + " ".join(f"{v:.6f}" for v in polygon))
    for polygon in ground_polygons:
        lines.append(f"{CLASS_GROUND} " + " ".join(f"{v:.6f}" for v in polygon))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def prepare_output_dirs(output: Path, splits: tuple[str, ...]) -> None:
    for kind in ("images", "labels", "previews"):
        for split in splits:
            (output / kind / split).mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-label a YOLO-seg dataset (trash_bin + ground) with SAM2 and ADE20K."
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
                        help="Which splits to process (default: train val test)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after N newly processed images (already-done are skipped)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-process images whose seg label already exists")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device, e.g. cpu or cuda (default: auto-detect)")
    parser.add_argument("--clip-margin", type=float, default=BIN_CLIP_MARGIN,
                        help=f"Expand each box by this fraction before clipping the bin mask (default: {BIN_CLIP_MARGIN})")
    args = parser.parse_args()

    if not args.source.exists():
        print(f"Kildemappe ikke funnet: {args.source}")
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
    review_rows: list[dict] = []
    produced = skipped = flagged_images = 0

    for split in args.splits:
        images = list_split_images(args.source, split)
        print(f"\n[{split}] {len(images)} bilde(r)")
        for image_path in images:
            if args.limit is not None and produced >= args.limit:
                break

            out_label   = args.output / "labels"   / split / f"{image_path.stem}.txt"
            out_image   = args.output / "images"    / split / image_path.name
            out_preview = args.output / "previews"  / split / f"{image_path.stem}.jpg"
            if out_label.exists() and not args.overwrite:
                skipped += 1
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                print(f"  KAN IKKE LESE: {image_path.name}")
                continue
            height, width = image.shape[:2]

            label_path = args.source / "labels" / split / f"{image_path.stem}.txt"
            boxes = read_bin_boxes(label_path, width, height)

            comp = compute_seg_for_image(image, boxes, sam, image_path, processor,
                                         semantic, ground_ids, device, args.clip_margin)
            for flag in comp.flags:
                review_rows.append({"filename": image_path.name, "split": split, **flag})

            shutil.copy2(image_path, out_image)
            write_seg_label(out_label, comp.bin_polygons, comp.ground_polygons)
            preview = make_preview(image, comp.clipped_masks, comp.ground_mask, boxes,
                                   comp.flagged)
            cv2.imwrite(str(out_preview), preview)

            produced += 1
            if comp.flagged:
                flagged_images += 1
            print(f"  {image_path.name}: {len(comp.bin_polygons)} kasse-, "
                  f"{len(comp.ground_polygons)} ground-polygon(er)"
                  f"{'  [FLAGGET]' if comp.flagged else ''}")
        if args.limit is not None and produced >= args.limit:
            break

    if review_rows:
        review_path = args.output / "review_flags.csv"
        with open(review_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
            writer.writeheader()
            writer.writerows(review_rows)
        print(f"\n{len(review_rows)} flagg skrevet til {review_path}")

    print(f"\nFerdig: {produced} behandlet, {skipped} hoppet over (fantes), "
          f"{flagged_images} bilde(r) flagget for manuell kontroll.")
    if produced:
        print(f"Datasett: {args.output}  (forhåndsvisninger i {args.output}/previews)")


if __name__ == "__main__":
    main()
