"""
Estimates GPS coordinates of detected trash bins from Street View images.

YOLO-seg predicts trash_bin instances (class 0) and their masks. Each bin is
then back-projected to a GPS coordinate by one of two methods (--method):

    ground  Ray through the bin's lowest mask pixel (where it meets the ground)
            is intersected with a flat ground plane at the known camera height.
            No depth model. Accurate for bins standing on roughly flat ground.
    depth   Depth Anything V2 (metric, outdoor) gives a metric depth map; the
            median depth over the mask is the distance along the ray through the
            mask centroid. Handles bins not on flat ground, but monocular metric
            depth is poorly scaled for Street View's wide FOV.
    both    Try ground first; fall back to depth when the ground ray is too
            shallow to intersect reliably (distance exceeds --max-ground-dist).

Both methods share one pinhole camera model rotated by the panorama heading and
pitch, projecting into a local East-North frame added to the camera (panorama)
GPS. On validation against manifest ground truth, ground gave ~2.3 m median error
at FOV 80 (near-unbiased) versus ~22 m for depth, so ground is the default.

Camera metadata (panorama lat/lng, heading, pitch) is read from the manifest
data/streetview_log.csv, keyed by image filename. The field-of-view is NOT
recorded there, so it is supplied via --fov (the fetcher's CLI default is 80).
Images fetched by src/fetch_streetview.py (address/coordinate mode) have no
manifest row and are skipped.

When the manifest has the bin's own coordinate (bin_lat/bin_lng) and the
camera-to-bin distance (distance_m), they are written out as ground truth so the
estimate can be validated.

Run from the project root:
    py -3.14 -m src.estimate_bin_positions
    py -3.14 -m src.estimate_bin_positions --images data/annotated_seg/images/test
    py -3.14 -m src.estimate_bin_positions --method both
    py -3.14 -m src.estimate_bin_positions --method depth --fov 80 --conf 0.3
"""

import argparse
import csv
from dataclasses import dataclass
from math import atan2, cos, degrees, pi, radians, sin, sqrt, tan
from pathlib import Path

import cv2
import numpy as np

from src.fetch_streetview_from_csv import CAMERA_HEIGHT_M, haversine_m, offset_coord

LOG_FILE            = Path("data/streetview_log.csv")
DEFAULT_IMAGES      = Path("data/annotated_seg/images")
DEFAULT_SEG_WEIGHTS = Path("models/trained/colab_seg/weights/best.pt")
DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
OUTPUT_DIR          = Path("outputs/bin_positions")

DEFAULT_FOV             = 80.0
DEFAULT_CONF            = 0.25
DEFAULT_METHOD          = "ground"
DEFAULT_MAX_GROUND_DIST = 60.0
METHODS = ("ground", "depth", "both")
CLASS_TRASH_BIN = 0

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

OUTPUT_COLUMNS = [
    "filename", "bin_index", "conf", "method", "depth_m",
    "bearing_deg", "horizontal_dist_m", "est_lat", "est_lng", "up_m",
    "gt_bin_lat", "gt_bin_lng", "error_m", "gt_distance_m", "dist_err_m",
]


@dataclass
class CameraMeta:
    pano_lat: float
    pano_lng: float
    heading: float
    pitch: float
    bin_lat: float | None
    bin_lng: float | None
    distance_m: float | None


@dataclass
class BinPosition:
    filename: str
    bin_index: int
    conf: float
    method: str
    depth_m: float | None
    bearing_deg: float
    horizontal_dist_m: float
    est_lat: float
    est_lng: float
    up_m: float
    gt_bin_lat: float | None
    gt_bin_lng: float | None
    error_m: float | None
    gt_distance_m: float | None
    dist_err_m: float | None


def _parse_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_metadata(log_file: Path) -> dict[str, CameraMeta]:
    """Reads the Street View manifest into a {filename: CameraMeta} map."""
    if not log_file.exists():
        raise FileNotFoundError(f"Fant ikke manifest: {log_file}")

    metadata: dict[str, CameraMeta] = {}
    with open(log_file, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            pano_lat = _parse_float(row.get("pano_lat"))
            pano_lng = _parse_float(row.get("pano_lng"))
            heading  = _parse_float(row.get("heading"))
            pitch    = _parse_float(row.get("pitch"))
            if None in (pano_lat, pano_lng, heading, pitch):
                continue
            metadata[row["filename"]] = CameraMeta(
                pano_lat=pano_lat,
                pano_lng=pano_lng,
                heading=heading,
                pitch=pitch,
                bin_lat=_parse_float(row.get("bin_lat")),
                bin_lng=_parse_float(row.get("bin_lng")),
                distance_m=_parse_float(row.get("distance_m")),
            )
    return metadata


def load_depth_model(model_name: str, device: str) -> tuple[object, object]:
    """Loads a HuggingFace metric depth-estimation model and its processor."""
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device).eval()
    return processor, model


def estimate_depth_map(image_bgr: np.ndarray, processor: object, model: object,
                       device: str) -> np.ndarray:
    """Returns a metric depth map (metres) at the image's native resolution."""
    import torch
    import torch.nn.functional as F

    height, width = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=image_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        predicted = model(**inputs).predicted_depth
    depth = F.interpolate(
        predicted.unsqueeze(1),
        size=(height, width),
        mode="bicubic",
        align_corners=False,
    )[0, 0]
    return depth.cpu().numpy()


def _ray_components(px: float, py: float, meta: CameraMeta, fov_h_deg: float,
                    width: int, height: int) -> tuple[float, float, float]:
    """Direction of the camera ray through pixel (px, py) in East-North-Up.

    Pinhole camera (square pixels, horizontal FOV fov_h_deg) rotated by the
    panorama heading and pitch. The returned vector has an implicit optical-axis
    component of 1 (it is not unit length); scaling it by a depth gives a 3D
    offset, and solving for a target height gives a ground intersection.
    """
    focal = (width / 2.0) / tan(radians(fov_h_deg) / 2.0)
    x_n = (px - width / 2.0) / focal
    y_n = (py - height / 2.0) / focal

    h = radians(meta.heading)
    p = radians(meta.pitch)
    sin_h, cos_h = sin(h), cos(h)
    sin_p, cos_p = sin(p), cos(p)

    east  = x_n * cos_h + y_n * sin_h * sin_p + sin_h * cos_p
    north = -x_n * sin_h + y_n * cos_h * sin_p + cos_h * cos_p
    up    = -y_n * cos_p + sin_p
    return east, north, up


def _to_coordinate(meta: CameraMeta, east: float, north: float, up: float,
                   ) -> tuple[float, float, float, float, float]:
    horizontal_dist = sqrt(east * east + north * north)
    bearing_deg = (degrees(atan2(east, north)) + 360.0) % 360.0
    est_lat, est_lng = offset_coord(meta.pano_lat, meta.pano_lng,
                                    bearing_deg, horizontal_dist)
    return est_lat, est_lng, bearing_deg, horizontal_dist, up


def backproject(px: float, py: float, depth_m: float, meta: CameraMeta,
                fov_h_deg: float, width: int, height: int,
                ) -> tuple[float, float, float, float, float]:
    """Back-projects a pixel at metric depth (along the optical axis) to GPS.

    Returns (est_lat, est_lng, bearing_deg, horizontal_dist_m, up_m), where up_m
    is the height of the point relative to the camera (negative = below).
    """
    east, north, up = _ray_components(px, py, meta, fov_h_deg, width, height)
    return _to_coordinate(meta, east * depth_m, north * depth_m, up * depth_m)


def backproject_ground(px: float, py: float, meta: CameraMeta, fov_h_deg: float,
                       width: int, height: int, camera_height_m: float,
                       max_dist_m: float,
                       ) -> tuple[float, float, float, float, float] | None:
    """Intersects the ray through a ground-contact pixel with the ground plane.

    Assumes flat ground camera_height_m below the camera. Returns the same
    5-tuple as backproject (up_m is -camera_height_m by construction), or None
    when the ray points at/above the horizon or the intersection lies beyond
    max_dist_m, where the flat-ground assumption is unreliable.
    """
    east, north, up = _ray_components(px, py, meta, fov_h_deg, width, height)
    if up >= -1e-6:
        return None
    t = camera_height_m / -up
    east, north = east * t, north * t
    if sqrt(east * east + north * north) > max_dist_m:
        return None
    return _to_coordinate(meta, east, north, -camera_height_m)


MASK_ALPHA   = 0.45
COLOR_GROUND = (0, 200, 0)      # BGR grønn — bakke-geometri
COLOR_DEPTH  = (0, 165, 255)    # BGR oransje — dybde
COLOR_RING   = (255, 255, 0)    # BGR cyan — 1 m bakkeradius rundt kassen


def _project_enu_to_pixel(east: float, north: float, up: float, meta: CameraMeta,
                          fov_h_deg: float, width: int, height: int,
                          ) -> tuple[float, float] | None:
    """Projects a camera-relative ENU offset to a pixel — inverse of _ray_components.

    The rotation in _ray_components is proper (orthonormal), so its inverse is the
    transpose. Returns None when the point is at or behind the image plane.
    """
    focal = (width / 2.0) / tan(radians(fov_h_deg) / 2.0)
    h = radians(meta.heading)
    p = radians(meta.pitch)
    sin_h, cos_h = sin(h), cos(h)
    sin_p, cos_p = sin(p), cos(p)

    cam_x = east * cos_h - north * sin_h
    cam_y = east * sin_h * sin_p + north * cos_h * sin_p - up * cos_p
    cam_z = east * sin_h * cos_p + north * cos_h * cos_p + up * sin_p
    if cam_z <= 1e-6:
        return None
    px = width / 2.0 + (cam_x / cam_z) * focal
    py = height / 2.0 + (cam_y / cam_z) * focal
    return px, py


def _ground_ring_pixels(east_c: float, north_c: float, meta: CameraMeta,
                        fov_h_deg: float, width: int, height: int,
                        camera_height_m: float, radius_m: float,
                        n_points: int = 48) -> np.ndarray | None:
    """Pixels of a horizontal circle of radius_m on the ground around (east_c, north_c).

    The circle lies on the flat ground plane (camera_height_m below the camera).
    Returns an (N, 1, 2) int32 array for cv2.polylines, or None if fewer than three
    points fall in front of the camera.
    """
    pts: list[list[int]] = []
    for k in range(n_points):
        angle = 2.0 * pi * k / n_points
        proj = _project_enu_to_pixel(
            east_c + radius_m * cos(angle), north_c + radius_m * sin(angle),
            -camera_height_m, meta, fov_h_deg, width, height,
        )
        if proj is not None:
            pts.append([int(round(proj[0])), int(round(proj[1]))])
    if len(pts) < 3:
        return None
    return np.array(pts, dtype=np.int32).reshape(-1, 1, 2)


def _draw_overlay(image: np.ndarray,
                  items: list[tuple[np.ndarray, float, float, "BinPosition"]],
                  out_path: Path, meta: CameraMeta, fov_h_deg: float,
                  camera_height_m: float, ground_radius_m: float) -> None:
    """Tints each bin mask, marks the back-projected pixel, and rings the ground.

    When ground_radius_m > 0 a horizontal circle of that radius is drawn on the
    ground plane around each bin's estimated base. If the geometry (FOV, pitch,
    camera height) is right, the ring reads as a believable real-world circle —
    a direct visual calibration check.
    """
    height, width = image.shape[:2]
    overlay = image.copy()
    for mask, _, _, pos in items:
        overlay[mask] = COLOR_GROUND if pos.method == "ground" else COLOR_DEPTH
    canvas = cv2.addWeighted(overlay, MASK_ALPHA, image, 1.0 - MASK_ALPHA, 0.0)

    if ground_radius_m > 0.0:
        for _, _, _, pos in items:
            bearing = radians(pos.bearing_deg)
            east_c = pos.horizontal_dist_m * sin(bearing)
            north_c = pos.horizontal_dist_m * cos(bearing)
            ring = _ground_ring_pixels(east_c, north_c, meta, fov_h_deg,
                                       width, height, camera_height_m, ground_radius_m)
            if ring is not None:
                cv2.polylines(canvas, [ring], True, COLOR_RING, 2, cv2.LINE_AA)

    for _, ux, uy, pos in items:
        color = COLOR_GROUND if pos.method == "ground" else COLOR_DEPTH
        point = (int(round(ux)), int(round(uy)))
        cv2.circle(canvas, point, 6, (255, 255, 255), -1)
        cv2.circle(canvas, point, 6, color, 2)
        label = f"{pos.method} {pos.horizontal_dist_m:.1f}m"
        if pos.error_m is not None:
            label += f" e{pos.error_m:.0f}m"
        org = (point[0] + 8, point[1] - 8)
        cv2.putText(canvas, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)


def _save_depth_vis(depth_map: np.ndarray, out_path: Path) -> None:
    """Writes a colour-mapped metric depth map (near=blue, far=red)."""
    finite = np.isfinite(depth_map)
    if not finite.any():
        return
    lo, hi = np.percentile(depth_map[finite], [2, 98])
    norm = np.clip((depth_map - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.imwrite(str(out_path), colored)


def process_image(image_path: Path, seg_model: object, depth_ctx: tuple | None,
                  device: str, meta: CameraMeta, fov_h_deg: float, conf: float,
                  method: str, camera_height_m: float, max_ground_dist_m: float,
                  vis_dir: Path | None = None,
                  ground_radius_m: float = 1.0) -> list[BinPosition]:
    """Detects bins and back-projects each to GPS via the chosen method.

    depth_ctx is (processor, depth_model) when depth may be used, else None. The
    depth map is computed lazily and at most once per image, only when a bin
    actually needs it.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  Kan ikke lese bilde: {image_path.name} — hopper over")
        return []
    height, width = image.shape[:2]

    result = seg_model.predict(str(image_path), conf=conf,
                               retina_masks=True, verbose=False)[0]
    if result.masks is None or result.boxes is None:
        return []

    masks = result.masks.data.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()

    bin_indices = [i for i, c in enumerate(classes) if c == CLASS_TRASH_BIN]
    if not bin_indices:
        return []

    depth_map: np.ndarray | None = None
    draw_items: list[tuple[np.ndarray, float, float, BinPosition]] = []

    positions: list[BinPosition] = []
    for bin_index, i in enumerate(bin_indices):
        mask = masks[i] > 0.5
        ys, xs = np.where(mask)
        if xs.size == 0:
            continue

        used_method: str | None = None
        depth_m: float | None = None
        used_px = used_py = 0.0
        result5: tuple[float, float, float, float, float] | None = None

        if method in ("ground", "both"):
            bottom_y = int(ys.max())
            bottom_x = float(xs[ys == bottom_y].mean())
            result5 = backproject_ground(bottom_x, float(bottom_y), meta,
                                         fov_h_deg, width, height,
                                         camera_height_m, max_ground_dist_m)
            if result5 is not None:
                used_method = "ground"
                used_px, used_py = bottom_x, float(bottom_y)

        if result5 is None and method in ("depth", "both"):
            if depth_map is None:
                processor, depth_model = depth_ctx
                depth_map = estimate_depth_map(image, processor, depth_model, device)
            depth_m = float(np.median(depth_map[mask]))
            if not np.isfinite(depth_m) or depth_m <= 0.5:
                print(f"  Ugyldig dybde ({depth_m:.2f} m) for kasse {bin_index} "
                      f"i {image_path.name} — hopper over")
                continue
            px, py = float(xs.mean()), float(ys.mean())
            result5 = backproject(px, py, depth_m, meta, fov_h_deg, width, height)
            used_method = "depth"
            used_px, used_py = px, py

        if result5 is None:
            print(f"  Bakke-stråle for grunn for kasse {bin_index} "
                  f"i {image_path.name} — hopper over")
            continue

        est_lat, est_lng, bearing_deg, horizontal_dist, up = result5

        error_m = None
        if meta.bin_lat is not None and meta.bin_lng is not None:
            error_m = haversine_m(est_lat, est_lng, meta.bin_lat, meta.bin_lng)
        dist_err_m = None
        if meta.distance_m is not None:
            dist_err_m = horizontal_dist - meta.distance_m

        position = BinPosition(
            filename=image_path.name,
            bin_index=bin_index,
            conf=float(confidences[i]),
            method=used_method,
            depth_m=depth_m,
            bearing_deg=bearing_deg,
            horizontal_dist_m=horizontal_dist,
            est_lat=est_lat,
            est_lng=est_lng,
            up_m=up,
            gt_bin_lat=meta.bin_lat,
            gt_bin_lng=meta.bin_lng,
            error_m=error_m,
            gt_distance_m=meta.distance_m,
            dist_err_m=dist_err_m,
        )
        positions.append(position)
        if vis_dir is not None:
            draw_items.append((mask, used_px, used_py, position))

    if vis_dir is not None:
        if draw_items:
            _draw_overlay(image, draw_items, vis_dir / image_path.name, meta,
                          fov_h_deg, camera_height_m, ground_radius_m)
        if depth_map is not None:
            _save_depth_vis(depth_map, vis_dir / f"{image_path.stem}_depth.jpg")

    return positions


def collect_images(images_path: Path,
                   manifest_names: set[str] | None = None) -> list[Path]:
    """Returns image files at a single path or recursively under a directory.

    When manifest_names is given (a set of bare filenames from the Street View
    manifest), only files whose name is in that set are returned — avoiding the
    80 % skip-rate that occurs when the annotated directory contains non-Street-View
    images.
    """
    if images_path.is_file():
        return [images_path]
    if not images_path.is_dir():
        raise FileNotFoundError(f"Fant ikke bilder: {images_path}")
    candidates = (
        p for p in images_path.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if manifest_names is not None:
        return sorted(p for p in candidates if p.name in manifest_names)
    return sorted(candidates)


def _write_csv(output_dir: Path, positions: list[BinPosition]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "bin_positions.csv"
    with open(out_path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for pos in positions:
            writer.writerow({
                "filename": pos.filename,
                "bin_index": pos.bin_index,
                "conf": f"{pos.conf:.3f}",
                "method": pos.method,
                "depth_m": "" if pos.depth_m is None else f"{pos.depth_m:.2f}",
                "bearing_deg": f"{pos.bearing_deg:.2f}",
                "horizontal_dist_m": f"{pos.horizontal_dist_m:.2f}",
                "est_lat": f"{pos.est_lat:.7f}",
                "est_lng": f"{pos.est_lng:.7f}",
                "up_m": f"{pos.up_m:.2f}",
                "gt_bin_lat": "" if pos.gt_bin_lat is None else f"{pos.gt_bin_lat:.7f}",
                "gt_bin_lng": "" if pos.gt_bin_lng is None else f"{pos.gt_bin_lng:.7f}",
                "error_m": "" if pos.error_m is None else f"{pos.error_m:.2f}",
                "gt_distance_m": "" if pos.gt_distance_m is None else f"{pos.gt_distance_m:.1f}",
                "dist_err_m": "" if pos.dist_err_m is None else f"{pos.dist_err_m:.2f}",
            })
    return out_path


def estimate(images_path: Path, seg_weights: Path, depth_model_name: str,
             log_file: Path, output_dir: Path, fov_h_deg: float, conf: float,
             device: str | None, limit: int | None, method: str,
             camera_height_m: float, max_ground_dist_m: float,
             save_vis: bool, ground_radius_m: float) -> None:
    """Runs the full pipeline over the images and writes a CSV of positions."""
    from ultralytics import YOLO

    if method not in METHODS:
        raise ValueError(f"Ukjent metode {method!r}, velg blant {METHODS}")
    if not 0.0 < fov_h_deg < 180.0:
        raise ValueError(f"FOV må være mellom 0 og 180 grader, fikk {fov_h_deg}")
    if not seg_weights.exists():
        raise FileNotFoundError(f"Fant ikke seg-vekter: {seg_weights}")

    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Enhet: {device}  Metode: {method}")

    metadata = load_metadata(log_file)
    print(f"Manifest: {len(metadata)} bilder med kameradata")

    manifest_names: set[str] | None = (
        set(metadata.keys()) if images_path.is_dir() else None
    )
    images = collect_images(images_path, manifest_names)
    if limit is not None:
        images = images[:limit]
    print(f"Bilder å behandle: {len(images)}")

    print(f"Laster seg-modell: {seg_weights}")
    seg_model = YOLO(str(seg_weights))

    depth_ctx: tuple | None = None
    if method in ("depth", "both"):
        print(f"Laster dybdemodell: {depth_model_name}")
        depth_ctx = load_depth_model(depth_model_name, device)

    vis_dir: Path | None = None
    if save_vis:
        vis_dir = output_dir / "previews"
        vis_dir.mkdir(parents=True, exist_ok=True)

    all_positions: list[BinPosition] = []
    n_no_meta = 0
    for idx, image_path in enumerate(images):
        meta = metadata.get(image_path.name)
        if meta is None:
            n_no_meta += 1
            continue
        positions = process_image(image_path, seg_model, depth_ctx, device, meta,
                                  fov_h_deg, conf, method, camera_height_m,
                                  max_ground_dist_m, vis_dir, ground_radius_m)
        all_positions.extend(positions)
        print(f"[{idx + 1}/{len(images)}] {image_path.name}: {len(positions)} kasse(r)")

    if n_no_meta:
        print(f"\n{n_no_meta} bilde(r) uten kameradata i manifestet — hoppet over.")

    if not all_positions:
        print("Ingen kasser å rapportere.")
        return

    out_path = _write_csv(output_dir, all_positions)

    by_method: dict[str, int] = {}
    for pos in all_positions:
        by_method[pos.method] = by_method.get(pos.method, 0) + 1
    method_summary = ", ".join(f"{k}: {v}" for k, v in sorted(by_method.items()))

    errors = [p.error_m for p in all_positions if p.error_m is not None]
    print(f"\nFerdig: {len(all_positions)} kasse(r) i {out_path}  ({method_summary})")
    if vis_dir is not None:
        print(f"Visualiseringer: {vis_dir}")
    if errors:
        errors_arr = np.array(errors)
        print(f"Avvik fra fasit (bin_lat/lng): "
              f"median {np.median(errors_arr):.1f} m, "
              f"snitt {errors_arr.mean():.1f} m, "
              f"maks {errors_arr.max():.1f} m  (n={len(errors)})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimer GPS-posisjon til detekterte søppelkasser via YOLO-seg + bakke-geometri/dybde."
    )
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES,
                        help=f"Bilde eller mappe med bilder (default: {DEFAULT_IMAGES})")
    parser.add_argument("--method", type=str, default=DEFAULT_METHOD, choices=METHODS,
                        help=f"ground=bakke-geometri, depth=DAV2-dybde, both=bakke med dybde-fallback (default: {DEFAULT_METHOD})")
    parser.add_argument("--seg-weights", type=Path, default=DEFAULT_SEG_WEIGHTS,
                        help=f"YOLO-seg vekter (default: {DEFAULT_SEG_WEIGHTS})")
    parser.add_argument("--depth-model", type=str, default=DEFAULT_DEPTH_MODEL,
                        help=f"HuggingFace dybdemodell (default: {DEFAULT_DEPTH_MODEL})")
    parser.add_argument("--log", type=Path, default=LOG_FILE,
                        help=f"Street View-manifest (default: {LOG_FILE})")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR,
                        help=f"Mappe for resultat-CSV (default: {OUTPUT_DIR})")
    parser.add_argument("--fov", type=float, default=DEFAULT_FOV,
                        help=f"Horisontal FOV i grader, må matche hentingen (default: {DEFAULT_FOV})")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF,
                        help=f"Konfidensterskel for seg-deteksjon (default: {DEFAULT_CONF})")
    parser.add_argument("--camera-height", type=float, default=CAMERA_HEIGHT_M,
                        help=f"Kamerahøyde over bakken i meter, for bakke-geometri (default: {CAMERA_HEIGHT_M})")
    parser.add_argument("--max-ground-dist", type=float, default=DEFAULT_MAX_GROUND_DIST,
                        help=f"Forkast bakke-estimat over denne avstanden i meter (default: {DEFAULT_MAX_GROUND_DIST})")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda eller cpu (default: auto)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stopp etter N bilder")
    parser.add_argument("--save-vis", action="store_true",
                        help="Lagre maske-overlegg (og dybdekart) til <output>/previews/")
    parser.add_argument("--ground-radius", type=float, default=1.0,
                        help="Radius i meter for bakke-ringen rundt hver kasse i overlegget; 0 skrur den av (default: 1.0)")
    args = parser.parse_args()

    estimate(
        images_path=args.images,
        seg_weights=args.seg_weights,
        depth_model_name=args.depth_model,
        log_file=args.log,
        output_dir=args.output,
        fov_h_deg=args.fov,
        conf=args.conf,
        device=args.device,
        limit=args.limit,
        method=args.method,
        camera_height_m=args.camera_height,
        max_ground_dist_m=args.max_ground_dist,
        save_vis=args.save_vis,
        ground_radius_m=args.ground_radius,
    )


if __name__ == "__main__":
    main()
