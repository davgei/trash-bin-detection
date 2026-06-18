"""
Trains a YOLO model on the annotated trash bin dataset.

Run from the project root:
    python -m src.train
    python -m src.train --epochs 100
    python -m src.train --model yolov8s.pt --name run2
"""

import argparse
from pathlib import Path

DATA_YAML   = Path("configs/data.yaml")
MODELS_DIR  = Path("models/trained")


def train(
    base_model: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 8,
    run_name: str = "run",
    patience: int = 10,
) -> Path:
    """
    Fine-tunes a pretrained YOLO model on data/annotated/.
    Saves weights to models/trained/<run_name>/weights/.
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
    print(f"\nTo annotate with this model:")
    print(f"  python -m src.labeling.annotate --mode assisted --model {best}")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO on the trash bin dataset.")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Base model to fine-tune (default: yolov8n.pt — smallest/fastest)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs (default: 50)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size in pixels (default: 640)")
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size (default: 8)")
    parser.add_argument("--name", type=str, default="run",
                        help="Name for this training run (default: run)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping: epochs without improvement before stopping (default: 10)")
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
