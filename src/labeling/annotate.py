"""
OpenCV annotation tool for creating YOLO training labels.

Phase A — manual mode (no model needed):
    Draw bounding boxes with the mouse.
    s / Enter  : save boxes and move to next image
                 (if no boxes drawn: asks for confirmation before saving as background)
    b          : mark image as background — no trash bin present (saves empty label)
    z          : undo the last box
    c          : clear all boxes for this image
    Escape     : skip image without saving any label
    q          : save current boxes and quit

Phase B — assisted mode (requires a trained YOLO model):
    YOLO proposes bounding boxes (shown in gold).
    a / Enter  : accept all proposals and move to next image
    b          : mark image as background — no trash bin present (saves empty label)
    r          : reject proposals and switch to manual drawing for this image
    s          : skip image without saving any label
    q          : quit

Run from the project root:
    python -m src.labeling.annotate --mode manual
    python -m src.labeling.annotate --mode assisted --model models/trained/best.pt
    python -m src.labeling.annotate --mode assisted --model models/trained/best.pt --conf 0.3
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from .export_labels import export_labels

TO_ANNOTATE_DIR = Path("data/to_annotate")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CLASS_ID = 0  # trash_bin

_COLOR_MANUAL   = (50, 205,  50)   # lime green  (BGR)
_COLOR_PROPOSAL = (0,  215, 255)   # gold        (BGR)
_COLOR_ACTIVE   = (255, 100,  50)  # blue while dragging (BGR)
_WINDOW = "Annotate - trash bin detection"
_MAX_DISPLAY = 1280  # max pixels in either dimension when displaying


class _SessionStats:
    """Tracks how often YOLO's proposals were accepted without correction."""

    def __init__(self) -> None:
        self.correct = 0
        self.total   = 0

    def record(self, was_correct: bool) -> None:
        self.total += 1
        if was_correct:
            self.correct += 1

    def _accuracy(self) -> float | None:
        return self.correct / self.total if self.total > 0 else None

    def render(self, frame: np.ndarray) -> None:
        """Draws a small accuracy badge in the top-right corner of frame."""
        if self.total == 0:
            text  = "YOLO: --"
            color = (160, 160, 160)
        else:
            pct   = int(self._accuracy() * 100)
            text  = f"YOLO: {self.correct}/{self.total} ({pct}%)"
            if pct >= 80:
                color = (50, 205, 50)    # green
            elif pct >= 50:
                color = (0, 165, 255)    # orange
            else:
                color = (0, 60, 255)     # red

        h, w = frame.shape[:2]
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        x = w - tw - 12
        y = th + 10
        cv2.rectangle(frame, (x - 6, y - th - 6), (x + tw + 6, y + 6), (30, 30, 30), -1)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


class _DrawState:
    """Tracks live mouse drawing and finished boxes for one image."""

    def __init__(self, image: np.ndarray) -> None:
        orig_h, orig_w = image.shape[:2]
        self.scale = min(_MAX_DISPLAY / max(orig_h, orig_w), 1.0)
        if self.scale < 1.0:
            dw = int(orig_w * self.scale)
            dh = int(orig_h * self.scale)
            self.image = cv2.resize(image, (dw, dh), interpolation=cv2.INTER_AREA)
        else:
            self.image = image.copy()
        self.boxes: list[tuple[int, int, int, int]] = []  # confirmed (x1,y1,x2,y2) in display coords
        self._drawing = False
        self._x0 = self._y0 = self._x1 = self._y1 = 0

    def mouse_callback(self, event: int, x: int, y: int,
                       flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drawing = True
            self._x0, self._y0, self._x1, self._y1 = x, y, x, y
        elif event == cv2.EVENT_MOUSEMOVE and self._drawing:
            self._x1, self._y1 = x, y
        elif event == cv2.EVENT_LBUTTONUP and self._drawing:
            self._drawing = False
            x1, y1 = min(self._x0, x), min(self._y0, y)
            x2, y2 = max(self._x0, x), max(self._y0, y)
            if x2 - x1 > 5 and y2 - y1 > 5:
                self.boxes.append((x1, y1, x2, y2))

    def render(self, proposals: Optional[list[tuple[int, int, int, int]]] = None) -> np.ndarray:
        frame = self.image.copy()
        if proposals:
            for x1, y1, x2, y2 in proposals:
                cv2.rectangle(frame, (x1, y1), (x2, y2), _COLOR_PROPOSAL, 2)
                cv2.putText(frame, "YOLO", (x1, max(y1 - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, _COLOR_PROPOSAL, 1)
        for x1, y1, x2, y2 in self.boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.rectangle(frame, (x1, y1), (x2, y2), _COLOR_MANUAL, 2)
        if self._drawing:
            x1 = min(self._x0, self._x1)
            y1 = min(self._y0, self._y1)
            x2 = max(self._x0, self._x1)
            y2 = max(self._y0, self._y1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)        # svart kontur
            cv2.rectangle(frame, (x1, y1), (x2, y2), _COLOR_ACTIVE, 2)    # farge oppå
        return frame


def _overlay_help(frame: np.ndarray, text: str) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 28), (w, h), (30, 30, 30), -1)
    cv2.putText(frame, text, (8, h - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (210, 210, 210), 1)


def _pixel_to_yolo(box: tuple[int, int, int, int], w: int, h: int) -> str:
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2) / w
    cy = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def _save_label(image_path: Path,
                boxes: list[tuple[int, int, int, int]],
                img_w: int, img_h: int,
                scale: float = 1.0) -> None:
    label_path = image_path.with_suffix(".txt")
    # Convert display-space coords back to original image coords before normalising
    orig_boxes = [
        (int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale))
        for x1, y1, x2, y2 in boxes
    ]
    lines = [_pixel_to_yolo(b, img_w, img_h) for b in orig_boxes]
    label_path.write_text("\n".join(lines))
    print(f"  Saved {len(lines)} box(es) -> {label_path.name}")


def _mark_hard_example(image_path: Path) -> None:
    """
    Writes a sidecar .json file flagging this image as a hard example.
    export_labels.py reads this flag and duplicates the image in the train split
    so YOLO sees it more often during the next training round.
    """
    image_path.with_suffix(".json").write_text('{"hard_example": true}')
    print(f"  Flagged as hard example: {image_path.name}")


def _run_manual(image_path: Path, image: np.ndarray) -> str:
    """
    Manual annotation for one image.
    Returns 'next' or 'quit'.
    """
    orig_h, orig_w = image.shape[:2]
    state = _DrawState(image)  # scales image to fit _MAX_DISPLAY
    cv2.namedWindow(_WINDOW, cv2.WINDOW_AUTOSIZE)  # AUTOSIZE: window = image = no coord mismatch
    cv2.setMouseCallback(_WINDOW, state.mouse_callback)

    _HELP_NORMAL  = "Draw box | s/Enter: save+next | b: no bin here | z: undo | c: clear | Esc: skip | q: quit"
    _HELP_CONFIRM = "No boxes — s/Enter: confirm background (no trash bin) | Esc: cancel"

    confirm_background = False

    while True:
        frame = state.render()

        if confirm_background:
            cv2.putText(frame, "No boxes — confirm as background?",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1)
            _overlay_help(frame, _HELP_CONFIRM)
        else:
            cv2.putText(frame, f"{len(state.boxes)} box(es)  —  {image_path.name}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _COLOR_MANUAL, 1)
            _overlay_help(frame, _HELP_NORMAL)

        cv2.imshow(_WINDOW, frame)

        key = cv2.waitKey(15) & 0xFF

        if state.boxes:
            confirm_background = False  # drawing a box cancels the confirmation

        if key in (ord("s"), 13):
            if state.boxes:
                _save_label(image_path, state.boxes, orig_w, orig_h, state.scale)
                return "next"
            elif confirm_background:
                _save_label(image_path, [], orig_w, orig_h)   # intentional background label
                return "next"
            else:
                confirm_background = True           # first press with no boxes -> ask
        elif key == ord("b"):                       # explicit "no bin in this image"
            _save_label(image_path, [], orig_w, orig_h)
            return "next"
        elif key == ord("z"):
            if state.boxes:
                state.boxes.pop()
            confirm_background = False
        elif key == ord("c"):
            state.boxes.clear()
            confirm_background = False
        elif key == 27:                             # Escape
            if confirm_background:
                confirm_background = False          # cancel the confirmation
            else:
                print(f"  Skipped {image_path.name}")
                return "next"
        elif key == ord("q"):
            if state.boxes:
                _save_label(image_path, state.boxes, orig_w, orig_h, state.scale)
            return "quit"


def _run_assisted(image_path: Path, image: np.ndarray,
                  proposals: list[tuple[int, int, int, int]],
                  stats: _SessionStats) -> str:
    """
    Shows YOLO proposals for one image.
    If user presses r, switches to manual drawing mode in the same window
    without leaving this function — this ensures the mouse callback is
    registered before the user starts drawing.
    Returns 'next' or 'quit'.
    """
    orig_h, orig_w = image.shape[:2]
    draw_state  = _DrawState(image)  # scales image to fit _MAX_DISPLAY
    manual_mode = False
    confirm_bg  = False
    # Scale YOLO proposals (original image coords) into display coords for rendering
    disp_proposals = [
        (int(x1 * draw_state.scale), int(y1 * draw_state.scale),
         int(x2 * draw_state.scale), int(y2 * draw_state.scale))
        for x1, y1, x2, y2 in proposals
    ]
    cv2.namedWindow(_WINDOW, cv2.WINDOW_AUTOSIZE)  # AUTOSIZE: window = image = no coord mismatch

    _HELP_WITH    = (f"{len(proposals)} proposal(s) | "
                     "a/Enter: accept | b: no bin here | r: redraw | s: skip | q: quit")
    _HELP_NONE    = "No YOLO proposals | b: background | r: draw manually | s: skip | q: quit"
    _HELP_MANUAL  = "Draw box | s/Enter: save | b: no bin here | z: undo | c: clear | Esc: skip | q: quit"
    _HELP_CONFIRM = "No boxes -- s/Enter: confirm background | Esc: cancel"

    while True:
        if manual_mode:
            frame = draw_state.render()
            if confirm_bg:
                cv2.putText(frame, "No boxes -- confirm as background?",
                            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1)
                _overlay_help(frame, _HELP_CONFIRM)
            else:
                cv2.putText(frame, f"{len(draw_state.boxes)} box(es)  --  {image_path.name}",
                            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _COLOR_MANUAL, 1)
                _overlay_help(frame, _HELP_MANUAL)
        else:
            frame = draw_state.render(proposals=disp_proposals)
            if disp_proposals:
                cv2.putText(frame, image_path.name,
                            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _COLOR_PROPOSAL, 1)
                _overlay_help(frame, _HELP_WITH)
            else:
                cv2.putText(frame, f"No proposals -- {image_path.name}",
                            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1)
                _overlay_help(frame, _HELP_NONE)

        stats.render(frame)
        cv2.imshow(_WINDOW, frame)
        key = cv2.waitKey(15) & 0xFF

        if draw_state.boxes:
            confirm_bg = False

        if not manual_mode:
            if key in (ord("a"), 13) and disp_proposals:
                stats.record(True)
                # proposals are already in original image coords (not display coords)
                _save_label(image_path, proposals, orig_w, orig_h)
                return "next"
            elif key == ord("b"):
                stats.record(not disp_proposals)
                if disp_proposals:
                    _mark_hard_example(image_path)
                _save_label(image_path, [], orig_w, orig_h)
                return "next"
            elif key == ord("r"):
                stats.record(False)
                _mark_hard_example(image_path)
                manual_mode = True
                cv2.setMouseCallback(_WINDOW, draw_state.mouse_callback)
            elif key == ord("s"):
                print(f"  Skipped {image_path.name}")
                return "next"
            elif key == ord("q"):
                return "quit"
        else:
            if key in (ord("s"), 13):
                if draw_state.boxes:
                    _save_label(image_path, draw_state.boxes, orig_w, orig_h, draw_state.scale)
                    return "next"
                elif confirm_bg:
                    _save_label(image_path, [], orig_w, orig_h)
                    return "next"
                else:
                    confirm_bg = True
            elif key == ord("b"):
                _save_label(image_path, [], orig_w, orig_h)
                return "next"
            elif key == ord("z"):
                if draw_state.boxes:
                    draw_state.boxes.pop()
                confirm_bg = False
            elif key == ord("c"):
                draw_state.boxes.clear()
                confirm_bg = False
            elif key == 27:
                if confirm_bg:
                    confirm_bg = False
                else:
                    print(f"  Skipped {image_path.name}")
                    return "next"
            elif key == ord("q"):
                if draw_state.boxes:
                    _save_label(image_path, draw_state.boxes, orig_w, orig_h, draw_state.scale)
                return "quit"


def _unlabeled_images(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
        and not p.with_suffix(".txt").exists()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate images for YOLO training.")
    parser.add_argument("--mode", choices=["manual", "assisted"], default="manual",
                        help="manual: draw from scratch  |  assisted: review YOLO proposals")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to YOLO weights — required for assisted mode")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for YOLO proposals (default: 0.25)")
    parser.add_argument("--dir", type=Path, default=TO_ANNOTATE_DIR,
                        help=f"Image directory (default: {TO_ANNOTATE_DIR})")
    args = parser.parse_args()

    if args.mode == "assisted" and args.model is None:
        parser.error("--model is required for assisted mode")

    if not args.dir.exists():
        print(f"Directory not found: {args.dir}")
        return

    images = _unlabeled_images(args.dir)
    if not images:
        print("No unlabeled images found.")
        return

    print(f"Found {len(images)} unlabeled image(s) in {args.dir}")

    model = None
    stats = _SessionStats()
    if args.mode == "assisted":
        from ultralytics import YOLO
        model = YOLO(args.model)
        print(f"Model: {args.model}  (conf >= {args.conf})")

    for i, image_path in enumerate(images):
        print(f"\n[{i + 1}/{len(images)}] {image_path.name}")
        image = cv2.imread(str(image_path))
        if image is None:
            print("  Could not read image, skipping.")
            continue

        if args.mode == "manual" or model is None:
            result = _run_manual(image_path, image)
        else:
            yolo_out = model.predict(image, conf=args.conf, verbose=False)
            proposals: list[tuple[int, int, int, int]] = []
            if yolo_out and yolo_out[0].boxes is not None:
                for xyxy in yolo_out[0].boxes.xyxy.tolist():
                    proposals.append((int(xyxy[0]), int(xyxy[1]),
                                      int(xyxy[2]), int(xyxy[3])))
            result = _run_assisted(image_path, image, proposals, stats)

        if result == "quit":
            print("\nQuitting.")
            break

    cv2.destroyAllWindows()
    if stats.total > 0:
        pct = int(stats.correct / stats.total * 100)
        print(f"\nYOLO accuracy this session: {stats.correct}/{stats.total} ({pct}%)")

    print("\nExporting labels to data/annotated/ ...")
    export_labels()
    print("Done.")


if __name__ == "__main__":
    main()
