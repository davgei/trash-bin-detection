"""
Evaluates a trained YOLO model on the held-out test split.

The test split is never seen during training or validation, so these numbers
are the honest estimate of real-world performance.

Run from the project root:
    python -m src.evaluate
    python -m src.evaluate --model runs/detect/models/trained/run3/weights/best.pt
"""

import argparse
from pathlib import Path

DATA_YAML = Path("configs/data.yaml")
DEFAULT_WEIGHTS = Path("runs/detect/models/trained/run3/weights/best.pt")
OUTPUT_DIR = Path("outputs/evaluation")


def evaluate(weights: Path, split: str = "test") -> None:
    """Runs validation on the given split and prints the key detection metrics."""
    from ultralytics import YOLO

    if not weights.exists():
        raise FileNotFoundError(f"Model weights not found: {weights}")
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Data config not found: {DATA_YAML}")

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(DATA_YAML),
        split=split,
        project=str(OUTPUT_DIR),
        name=weights.parent.parent.name,
    )

    box = metrics.box
    print(f"\nEvaluation on '{split}' split ({weights}):")
    print(f"  Precision   : {box.mp:.3f}")
    print(f"  Recall      : {box.mr:.3f}")
    print(f"  mAP50       : {box.map50:.3f}")
    print(f"  mAP50-95    : {box.map:.3f}")
    print(f"\nPlots and full results saved under: {OUTPUT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a YOLO model on the test split.")
    parser.add_argument("--model", type=Path, default=DEFAULT_WEIGHTS,
                        help=f"Path to model weights (default: {DEFAULT_WEIGHTS})")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Dataset split to evaluate on (default: test)")
    args = parser.parse_args()

    evaluate(weights=args.model, split=args.split)


if __name__ == "__main__":
    main()
