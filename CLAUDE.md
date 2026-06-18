# CLAUDE.md — Working Rules for This Project

## Scope and Pacing

- Work in small, verifiable steps. Do not implement multiple project phases in one go.
- Implement only what the current task requires. Do not add features, abstractions, or
  scaffolding for future phases unless explicitly asked.
- Before starting a non-trivial implementation, confirm the approach with the user.

## Environment and Tooling

**Python interpreter — read this first.** All project dependencies (`ultralytics`, `torch`,
`opencv-python`, `numpy`) are installed in the **global Python 3.14**, *not* in `.venv`.
The local `.venv/` exists but is empty (only `pip`), so `.venv/Scripts/python.exe` will
fail with `ModuleNotFoundError`. Run every command with the `py` launcher:

```
py -3.14 -m src.train
py -3.14 -m src.prepare_dataset
py -3.14 -m src.evaluate
```

To locate the interpreter that actually has the packages (do not assume `.venv`):
```
py -0p                                              # list installed Pythons
py -3.14 -c "import ultralytics; print(ultralytics.__file__)"
```

**Where trained weights live.** `src/train.py` passes `project="models/trained"`, but
Ultralytics prepends `runs/detect/`, so weights are actually saved to:

```
runs/detect/models/trained/<run-name>/weights/best.pt
```

**Dataset layout and annotation flow.**

```
data/to_annotate/       ← drop new images here before annotating
data/annotated_backup/  ← master pool (source of truth); never delete
data/annotated/         ← active train/val/test split read by training/eval
data/raw/               ← original source material (videos, GPS) — NOT annotation queue
```

Full flow for adding new images:
1. Copy new images into `data/to_annotate/`
2. `py -3.14 -m src.labeling.annotate [--mode assisted --model <weights>]`
   - draws boxes, saves `.txt` labels alongside images in `to_annotate/`
   - at session end: exports to pool (`annotated_backup/`) and rebuilds `data/annotated/`
3. Training reads from `data/annotated/` via `configs/data.yaml`

To rebuild `data/annotated/` manually (e.g. after changing split fractions):
`py -3.14 -m src.prepare_dataset`

## Code Quality

- Use type hints in all Python functions and method signatures.
- Use clear, descriptive names for variables, functions, and classes.
- Write no comments unless the reason behind the code is genuinely non-obvious.
- Prioritize simple and readable code over clever or compact solutions.

## Data Handling

- Never overwrite or modify files in `data/raw/`. Raw data is read-only.
- Do not assume that a file or dataset exists — always check before using it.
- Do not create example or synthetic data without being asked to.

## Git and Repository Hygiene

- Do not commit large data files, images, or videos to Git.
- Do not commit model weights (`.pt`, `.pth`, `.onnx`, `.bin`, etc.) to Git.
- Keep `data/`, `models/pretrained/`, `models/trained/`, and `outputs/` out of version
  control (already handled by `.gitignore`).
- Use `.gitkeep` to preserve empty tracked directories.

## Dataset Annotation and Active Learning

The labeling workflow follows two phases:

### Phase A — Manual bootstrapping (first 10–20 images)
The user manually draws bounding boxes from scratch. Output must be saved in YOLO label
format: one `.txt` file per image, one row per object, normalized coordinates.

```
<class_id> <x_center> <y_center> <width> <height>
```

### Phase B — Model-assisted labeling
Once a small initial dataset exists, train a preliminary YOLO model and use it to propose
bounding boxes on new images. The user reviews each proposal:

- **Accept**: save the proposed box as-is (one keypress)
- **Reject and redraw**: discard the proposal, let the user draw the correct box manually
- **Skip**: do not save a label for this image

This workflow is called *active learning* or *model-assisted labeling* and is fully
supported by YOLO (Ultralytics). The key tool to build for this is an annotation helper
(`src/annotate.py` or similar) with a simple GUI — OpenCV or a lightweight web interface
(e.g. Gradio or Streamlit) are both viable options.

### Rules for the annotation tool
- It must never overwrite an existing verified label without confirmation.
- Proposed boxes from YOLO and manually drawn boxes must be saved in the same format.
- The tool must make it easy to resume a session — skip already-labelled images.
- Confidence threshold for YOLO proposals should be configurable, not hardcoded.
- Use OpenCV for the GUI — no web framework, no server, draw directly in an image window.

### Iteration loop
```
annotate 10–20 images manually
→ train initial YOLO model
→ run model on unlabelled images → get proposed boxes
→ user accepts / corrects proposals
→ add to training set
→ retrain → repeat
```

Each iteration should improve proposal quality. Stop when the model proposes boxes that
rarely need correction.

### Hard Example Mining (oversampling)

Images the model previously misclassified are intentionally **duplicated** in the training
set so the model trains on them several times. These hard examples are stored as exact,
byte-identical copies with a `_h2`, `_h3`, `_h4` … suffix on the filename, e.g.
`Skjermbilde 2026-06-17 145926.png`, `..._h2.png`, `..._h3.png`, `..._h4.png`.

This imposes two hard rules on dataset splitting:
- **All copies of a duplicated image must stay in the training split.** Duplicates must
  never appear in val or test.
- **val and test must contain only single-copy images** — distinct images that appear
  nowhere else — so evaluation stays honest (no train/val/test leakage).

`src/prepare_dataset.py` enforces this. It groups images by **content hash** (md5 of the
file, not filename), locks any content that appears more than once into `train`, and splits
the single-copy images into train/val/test. It then verifies content-level disjointness and
preserves the duplicate count (the oversampling signal stays intact). **Always re-split with
this script — never split by filename**, because the duplicate copies have different names
but identical content, so a filename-based split silently leaks them across splits.

## Explaining Changes

- When making structural changes (new modules, refactoring, dependency additions),
  briefly explain the reason before proceeding.
- Do not rename, move, or delete existing files without confirming with the user.
