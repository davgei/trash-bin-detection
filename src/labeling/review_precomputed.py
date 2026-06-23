"""Gjennomgang av SAM2-forhåndsberegnede segmenteringer (lokalt, ingen GPU).

Les overlay-bilder og polygon-JSON fra data/sam2_precomputed/, vis dem i et
OpenCV-vindu, og skriv YOLO-seg-etiketter til data/annotated_seg/ ved godkjenning.
Split (train/val/test) leses fra JSON-sidecar-filen.

Tastesnarveier:
  a / Enter   Godkjenn  — skriver SAM2-polygonen til annotated_seg/labels/{split}/
  f           Flagg     — oppretter .flag-sidecar, ingen etikett
  s / Space   Hopp      — går videre uten å lagre noe
  q / Esc     Avslutt   — stopper umiddelbart
"""

import argparse
import json
from pathlib import Path

import cv2


PRECOMP_DIR  = Path("data/sam2_precomputed")
SEG_LABEL_DIR = Path("data/annotated_seg/labels")
WINDOW       = "a=godkjenn  f=flagg  s/Space=hopp  q/Esc=avslutt"


def seg_label_path(precomp_dir: Path, overlay_path: Path) -> Path | None:
    """Finn destinasjonssti for seg-etiketten basert på split i JSON."""
    json_path = precomp_dir / (overlay_path.stem + ".json")
    if not json_path.exists():
        return None
    data  = json.loads(json_path.read_text(encoding="utf-8"))
    split = data.get("split", "train")
    return SEG_LABEL_DIR / split / overlay_path.with_suffix(".txt").name


def build_worklist(precomp_dir: Path) -> list[Path]:
    overlays = sorted(precomp_dir.glob("*.jpg")) + sorted(precomp_dir.glob("*.png"))
    result = []
    for ov in overlays:
        dst  = seg_label_path(precomp_dir, ov)
        flag = precomp_dir / ov.with_suffix(".flag").name
        if flag.exists():
            continue
        if dst is not None and dst.exists():
            continue
        result.append(ov)
    return result


def write_label(overlay_path: Path, precomp_dir: Path) -> None:
    json_path = precomp_dir / (overlay_path.stem + ".json")
    dst       = seg_label_path(precomp_dir, overlay_path)
    if dst is None:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    data     = json.loads(json_path.read_text(encoding="utf-8"))
    polygons = data.get("polygons", [])
    lines    = ["0 " + " ".join(f"{v:.6f}" for v in poly) for poly in polygons]
    dst.write_text("\n".join(lines), encoding="utf-8")


def write_flag(overlay_path: Path, precomp_dir: Path) -> None:
    (precomp_dir / overlay_path.with_suffix(".flag").name).touch()


def run_review(precomp_dir: Path) -> None:
    worklist = build_worklist(precomp_dir)
    if not worklist:
        print("Ingen bilder å gjennomgå.")
        return

    total = len(worklist)
    print(f"{total} bilder å gjennomgå")
    print("a/Enter=godkjenn  f=flagg  s/Space=hopp  q/Esc=avslutt\n")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    approved = flagged = skipped = 0

    for idx, overlay_path in enumerate(worklist):
        img = cv2.imread(str(overlay_path))
        if img is None:
            continue

        h   = img.shape[0]
        hud = f"{idx + 1}/{total}  {overlay_path.stem}  +{approved} ok  f{flagged} flagget"
        cv2.putText(img, hud, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, hud, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW, img)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("a"), 13):
                write_label(overlay_path, precomp_dir)
                approved += 1
                break
            elif key == ord("f"):
                write_flag(overlay_path, precomp_dir)
                flagged += 1
                break
            elif key in (ord("s"), 32):
                skipped += 1
                break
            elif key in (ord("q"), 27):
                cv2.destroyAllWindows()
                _print_summary(approved, flagged, skipped)
                return

    cv2.destroyAllWindows()
    _print_summary(approved, flagged, skipped)
    print("\nEtiketter lagret i data/annotated_seg/labels/")
    print("Kjør for å trene:  py -3.14 -m src.train_seg")


def _print_summary(approved: int, flagged: int, skipped: int) -> None:
    print(f"\nFerdig.  Godkjent: {approved}  Flagget: {flagged}  Hoppet: {skipped}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gjennomgå SAM2-forhåndsberegnede segmenteringer lokalt."
    )
    parser.add_argument("--precomp-dir", type=Path, default=PRECOMP_DIR)
    args = parser.parse_args()
    run_review(args.precomp_dir)


if __name__ == "__main__":
    main()
