# CLAUDE.md — Working Rules for This Project

## Scope and Pacing

- Work in small, verifiable steps. Do not implement multiple project phases in one go.
- Implement only what the current task requires. Do not add features, abstractions, or
  scaffolding for future phases unless explicitly asked.
- Before starting a non-trivial implementation, confirm the approach with the user.

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

## Explaining Changes

- When making structural changes (new modules, refactoring, dependency additions),
  briefly explain the reason before proceeding.
- Do not rename, move, or delete existing files without confirming with the user.
