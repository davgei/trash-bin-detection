# Project: Trash Bin Detection and Spatial Assessment

## Background

Renovasjons- og gjenvinningsetaten (REN) in Oslo manages waste collection across the city.
REN has GPS positions for existing trash bins, as well as images and video sequences from
cameras mounted on waste collection vehicles.

The goal is to use this data to automate spatial assessment at existing collection points —
reducing the need for manual field surveys.

## Long-term Goal

Develop a system that can automatically assess whether there is physical space for additional
waste containers at existing collection points.

## Planned Pipeline

```text
images from waste vehicle
→ identify trash bins and obstacles in 2D
→ create or use a point cloud of the area
→ link 2D detections to 3D points
→ create 3D bounding boxes around objects
→ measure available area
→ assess whether an additional trash bin fits
```

## Available Data (Assumed)

### Images (`data/raw/images/`)
Images captured by cameras mounted on waste collection vehicles.

### GPS and location data (`data/raw/gps/`)
GPS coordinates and/or house numbers that correspond to the images — used to identify
which collection point (søppelpunkt) each image belongs to. The pairing between an image
and its GPS location is the key link that makes spatial assessment possible.

### Videos (`data/raw/videos/`)
Optionally: video sequences from the same cameras, used to extract additional frames.

### Point clouds (future, `data/pointclouds/`)
Pre-existing or generated point clouds (e.g. from LiDAR or photogrammetry) covering the
same collection points.

## Project Phases

### Phase 1: 2D Object Detection

Identify trash bins in standard 2D images using a YOLO-based model.

The model returns:
- class: `trash_bin`
- confidence score
- 2D bounding box

### Phase 2: 3D Label Generation

Use 2D detections together with camera calibration and a corresponding point cloud to
generate 3D bounding box proposals:

1. Project 3D points into the image plane
2. Find points that fall inside the 2D bounding box
3. Filter out ground, background, and noise
4. Cluster points likely belonging to the trash bin
5. Fit a 3D bounding box proposal
6. Review, correct, and save as training label

### Phase 3: 3D Model Training

Train a point cloud-based 3D object detection model using the labels from Phase 2.

Candidate architectures:
- PointPillars
- VoteNet
- CenterPoint
- PointNet++

### Phase 4: Spatial Assessment

Use 3D detections to measure available floor area around existing bins and determine
whether an additional container can be placed at the location.

## Technical Challenges

- Camera calibration and accurate 2D-to-3D projection
- Point cloud filtering (separating ground, walls, and background from objects)
- Generating high-quality 3D training labels from noisy point data
- Handling varying lighting conditions, occlusions, and weather
- Aligning GPS coordinates with point cloud and image data
- Defining a reliable spatial model for "enough room for one more bin"
