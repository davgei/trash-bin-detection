"""
Trains a YOLO segmentation model on the seg dataset (classes: 0 trash_bin, 1 ground).

This is the segmentation counterpart to src/train.py. It reads the seg dataset
built by src.labeling.sam2_seg_autolabel / sam2_seg_review (data/annotated_seg)
via configs/data_seg.yaml, and fine-tunes a YOLO11 segmentation model.

Because the base model has a "-seg" suffix, Ultralytics runs the segment task and
saves under runs/segment/..., so this never overwrites the detection weights in
runs/detect/.... The detection workflow (src/train.py) is untouched.

Run from the project root:
    py -3.14 -m src.train_seg
    py -3.14 -m src.train_seg --epochs 100 --name seg2
    py -3.14 -m src.train_seg --model yolo11s-seg.pt
"""

import argparse
from pathlib import Path

DATA_YAML  = Path("configs/data_seg.yaml")
MODELS_DIR = Path("models/trained")


def train(
    base_model: str = "yolo11n-seg.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 8,
    run_name: str = "seg",
    patience: int = 20,
) -> Path:
    """
    Fine-tunes a pretrained YOLO segmentation model on data/annotated_seg/.
    Saves weights to runs/segment/models/trained/<run_name>/weights/.
    Returns the path to best.pt.
    """
    from ultralytics import YOLO  # imported here so the module loads without ultralytics installed

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Data config not found: {DATA_YAML}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(base_model)
    results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        project=str(MODELS_DIR),
        name=run_name,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nBest weights saved to: {best}")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO segmentation on the trash bin + ground dataset.")
    parser.add_argument("--model", type=str, default="yolo11n-seg.pt",
                        help="Base segmentation model to fine-tune (default: yolo11n-seg.pt)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs (default: 50)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size in pixels (default: 640)")
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size (default: 8)")
    parser.add_argument("--name", type=str, default="seg",
                        help="Name for this training run (default: seg)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping: epochs without improvement before stopping (default: 20)")
    args = parser.parse_args()

    train(
        base_model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        run_name=args.name,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
