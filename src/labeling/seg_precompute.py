"""
Forbereg segmenteringsmasker lokalt (CPU eller GPU), uten Colab.

Gjør den tunge SAM2 + ADE20K-jobben på din egen maskin og skriver samme
.precompute/-cache (pkl + preview) som notebooks/seg_precompute_review.ipynb
lager i Colab. Etterpå går gjennomgangen helt uten ML:

    py -3.14 -m src.labeling.seg_cache_review

Den deler nøyaktig pipeline med src.labeling.sam2_seg_autolabel, så en godkjent
maske blir byte-identisk med det batch-verktøyet ville skrevet.

Gjenopptakbart og krasj-trygt: hvert bilde skrives til cachen med en gang, og
bilder som allerede er beregnet (eller godkjent/hoppet over) hoppes over. Stopp
når som helst med Ctrl+C og start igjen — den fortsetter der den slapp. Kjør i
batcher med --limit, eller la den gå gjennom alt i bakgrunnen.

Den semantiske bakke-modellen er stor (B5) og treg på CPU. Vil du ha det
raskere på bekostning av maskekvalitet:
    --semantic-model nvidia/segformer-b0-finetuned-ade-512-512

Leser kun fra --source (default data/annotated); rører aldri data/annotated_backup.

Kjør fra prosjektroten:
    py -3.14 -m src.labeling.seg_precompute                 # alt som gjenstår
    py -3.14 -m src.labeling.seg_precompute --limit 50      # én batch på 50
    py -3.14 -m src.labeling.seg_precompute --splits train  # bare train-splitten
"""

import argparse
import pickle
from pathlib import Path

import cv2

from src.labeling.sam2_seg_autolabel import (
    BIN_CLIP_MARGIN,
    DEFAULT_SAM_MODEL,
    DEFAULT_SEMANTIC_MODEL,
    compute_seg_for_image,
    list_split_images,
    load_sam,
    load_semantic,
    make_preview,
    prepare_output_dirs,
    read_bin_boxes,
)

SOURCE_DIR = Path("data/annotated")
OUTPUT_DIR = Path("data/annotated_seg")
SPLITS     = ("train", "val", "test")


def _label_path(output: Path, split: str, stem: str) -> Path:
    return output / "labels" / split / f"{stem}.txt"


def _skip_path(output: Path, split: str, stem: str) -> Path:
    return output / "labels" / split / f"{stem}.skip"


def _cache_pkl(cache_dir: Path, split: str, stem: str) -> Path:
    return cache_dir / split / f"{stem}.pkl"


def _cache_preview(cache_dir: Path, split: str, stem: str) -> Path:
    return cache_dir / split / f"{stem}.jpg"


def build_pending(source: Path, output: Path, cache_dir: Path,
                  splits: tuple[str, ...]) -> tuple[list[tuple[str, Path]], int, int]:
    """Returns (pending, n_done, n_cached) — images still needing a precompute pass."""
    pending: list[tuple[str, Path]] = []
    n_done = n_cached = 0
    for split in splits:
        for image_path in list_split_images(source, split):
            stem = image_path.stem
            if _label_path(output, split, stem).exists() or _skip_path(output, split, stem).exists():
                n_done += 1
                continue
            if _cache_pkl(cache_dir, split, stem).exists() and _cache_preview(cache_dir, split, stem).exists():
                n_cached += 1
                continue
            pending.append((split, image_path))
    return pending, n_done, n_cached


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(
        description="Forbereg SAM2 + bakke-masker lokalt til .precompute/-cachen."
    )
    parser.add_argument("--source", type=Path, default=SOURCE_DIR,
                        help=f"Deteksjonsdatasett å lese bilder + bokser fra (default: {SOURCE_DIR})")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR,
                        help=f"Seg-datasettmappe; cachen havner i <output>/.precompute (default: {OUTPUT_DIR})")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS),
                        help="Hvilke splitter som behandles (default: train val test)")
    parser.add_argument("--sam-model", type=str, default=DEFAULT_SAM_MODEL,
                        help=f"SAM2-vekter (default: {DEFAULT_SAM_MODEL})")
    parser.add_argument("--semantic-model", type=str, default=DEFAULT_SEMANTIC_MODEL,
                        help=f"ADE20K-modell for bakke (default: {DEFAULT_SEMANTIC_MODEL})")
    parser.add_argument("--clip-margin", type=float, default=BIN_CLIP_MARGIN,
                        help=f"Utvid boksen med denne andelen før klipping (default: {BIN_CLIP_MARGIN})")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda eller cpu (default: auto)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stopp etter N bilder denne kjøringen (batching)")
    args = parser.parse_args()

    source: Path = args.source
    output: Path = args.output
    cache_dir = output / ".precompute"
    splits = tuple(args.splits)

    if not source.exists():
        print(f"Kildemappe ikke funnet: {source}")
        return

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Enhet: {device}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    prepare_output_dirs(output, splits)

    pending, n_done, n_cached = build_pending(source, output, cache_dir, splits)
    print(f"Allerede godkjent/hoppet over : {n_done}")
    print(f"Allerede i cache              : {n_cached}")
    print(f"Skal beregnes nå              : {len(pending)}")

    if not pending:
        print("Ingenting å beregne — alt er allerede i cache eller behandlet.")
        return

    if args.limit is not None:
        pending = pending[:args.limit]
        print(f"Begrenset til {len(pending)} bilde(r) denne kjøringen.")

    print(f"Laster SAM2 ({args.sam_model}) ...", flush=True)
    sam = load_sam(args.sam_model)
    print(f"Laster semantisk modell ({args.semantic_model}) ...", flush=True)
    processor, semantic, ground_ids = load_semantic(args.semantic_model, device)
    if not ground_ids:
        print("ADVARSEL: ingen ground-klasser funnet — bakke-masken blir tom.")
    else:
        names = ", ".join(semantic.config.id2label[i] for i in ground_ids)
        print(f"Ground-klasser: {names}")

    done = flagged_count = failed = 0
    for i, (split, image_path) in enumerate(pending):
        stem = image_path.stem
        print(f"[{i + 1}/{len(pending)}] {split}/{image_path.name}", end=" ... ", flush=True)
        image = cv2.imread(str(image_path))
        if image is None:
            print("KAN IKKE LESES — hopper over")
            failed += 1
            continue
        height, width = image.shape[:2]
        boxes = read_bin_boxes(source / "labels" / split / f"{stem}.txt", width, height)
        try:
            comp = compute_seg_for_image(
                image, boxes, sam, image_path,
                processor, semantic, ground_ids, device, args.clip_margin,
            )
            entry = {
                "split": split,
                "image_filename": image_path.name,
                "boxes": [list(b) for b in boxes],
                "bin_polygons": comp.bin_polygons,
                "ground_polygons": comp.ground_polygons,
                "flagged": bool(comp.flagged),
                "flags": list(comp.flags),
            }
            (cache_dir / split).mkdir(parents=True, exist_ok=True)
            with open(_cache_pkl(cache_dir, split, stem), "wb") as fh:
                pickle.dump(entry, fh, protocol=5)
            preview = make_preview(image, comp.clipped_masks, comp.ground_mask, boxes, comp.flagged)
            cv2.imwrite(str(_cache_preview(cache_dir, split, stem)), preview)
            done += 1
            if comp.flagged:
                flagged_count += 1
            print("OK" + ("  [FLAGGET]" if comp.flagged else ""), flush=True)
        except Exception as exc:
            failed += 1
            print(f"FEIL: {exc}", flush=True)

    print(f"\nFerdig: {done} beregnet ({flagged_count} flagget), {failed} feilet.")
    print(f"Cache: {cache_dir}")
    print("Neste steg (lokalt, ingen ML):  py -3.14 -m src.labeling.seg_cache_review")


if __name__ == "__main__":
    main()
