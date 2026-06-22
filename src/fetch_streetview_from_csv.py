"""
Fetches Street View images aimed directly at trash bins, using exact bin
coordinates from a CSV export (hentesteder, or the older Uttrekk_products).

The problem with address-based fetching is that the camera points in an
arbitrary default direction — often at a hedge or out into the road. Here we
already know the exact coordinate of each bin, so we can aim the camera at it:

    1. Read the bin coordinate from the CSV.
    2. Ask the Street View Metadata API (free) for the nearest panorama and its
       actual camera position.
    3. Compute the compass heading from the camera toward the bin.
    4. Compute a downward pitch from the camera-to-bin distance (the camera sits
       ~2.5 m up, the bin is on the ground).
    5. Fetch the static image for that panorama with the computed heading/pitch.

Because a bin coordinate often marks the property/area rather than the bin
itself, the nearest panorama does not always show the bin. So after fetching the
nearest panorama the trained YOLO model is run on it: if it detects a bin we move
on; if not, a second-nearest panorama is found (via free offset metadata queries,
since the REST API returns no neighbour list) and fetched too. During annotation
BOTH images are kept (YOLO has false negatives — the human reviewer decides).
Use --no-detect to reproduce the old single-fetch behaviour.

Images are saved to data/to_annotate/ so they flow into the existing annotation
pipeline. A log of every fetch is written to data/streetview_log.csv.

Requires the GOOGLE_MAPS_API_KEY environment variable to be set.

Run from the project root:
    py -3.14 -m src.fetch_streetview_from_csv
    py -3.14 -m src.fetch_streetview_from_csv --dry-run           # geometry only, no images, no detection
    py -3.14 -m src.fetch_streetview_from_csv --limit 5           # stop after ~5 new images
    py -3.14 -m src.fetch_streetview_from_csv --no-detect         # fetch nearest only, no YOLO, no retry
"""

import argparse
import csv
import numpy as np
import requests
from dataclasses import dataclass, field
from math import atan2, cos, degrees, radians, sin, sqrt
from pathlib import Path

from src.fetch_streetview import _api_key

CSV_FILE        = Path("data/hentesteder.csv")
TO_ANNOTATE_DIR = Path("data/to_annotate")
POOL_DIR        = Path("data/annotated_backup")
LOG_FILE        = Path("data/streetview_log.csv")
SPLITS          = ("train", "val", "test")

CAMERA_HEIGHT_M = 2.5   # approximate height of the Street View car camera
TARGET_HEIGHT_M = 0.5   # approximate height we aim at on the bin
MIN_PITCH_DEG   = -45.0

DEFAULT_MODEL = Path("models/trained/trash_bin_yolo11n_best.pt")
DEFAULT_CONF  = 0.25

SECOND_PANO_RADIUS_M  = 30
DIRECTIONAL_OFFSETS_M = (15.0, 25.0)
RING_BEARINGS_DEG     = (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)
DEFAULT_RING_RADIUS_M = 18.0
TEMPORAL_DUP_EPS_M    = 5.0

LOG_COLUMNS = [
    "filename", "product_numbers", "bin_lat", "bin_lng",
    "pano_id", "pano_lat", "pano_lng", "distance_m",
    "heading", "pitch", "capture_date", "status", "address",
    "attempts", "detected", "detection_conf", "pano_rank", "pair",
]


@dataclass
class Bin:
    product_number: str
    lat: float
    lng: float
    bin_type: str
    waste: str


@dataclass
class Location:
    lat: float
    lng: float
    bins: list[Bin] = field(default_factory=list)

    @property
    def product_numbers(self) -> list[str]:
        return sorted(b.product_number for b in self.bins)


def _parse_coord(value: str | None) -> float | None:
    """Parses a Norwegian-formatted coordinate ('59,9474') into a float."""
    if value is None:
        return None
    value = value.strip()
    if not value or value.lower() == "null":
        return None
    try:
        return float(value.replace(",", "."))
    except ValueError:
        return None


def _csv_delimiter(csv_path: Path) -> str:
    """Detects ';' (hentesteder export) vs ',' (Uttrekk export) from the header."""
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        header = f.readline()
    return ";" if header.count(";") >= header.count(",") else ","


def read_bins(csv_path: Path, active_only: bool = True) -> list[Bin]:
    bins: list[Bin] = []
    delimiter = _csv_delimiter(csv_path)
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter=delimiter):
            if active_only and "Active" in row and (row.get("Active") or "").strip() != "SANN":
                continue
            lat = _parse_coord(row.get("Breddegrad") or row.get("Latitude"))
            lng = _parse_coord(row.get("Lengdegrad") or row.get("Longitude"))
            if lat is None or lng is None:
                continue
            bins.append(Bin(
                product_number=(row.get("Beholderid") or row.get("ProductNumber") or "").strip(),
                lat=lat,
                lng=lng,
                bin_type=(row.get("Beholdertype") or row.get("BinType") or "").strip(),
                waste=(row.get("Fraksjon") or row.get("Info1") or "").strip(),
            ))
    return bins


def dedupe_by_location(bins: list[Bin]) -> list[Location]:
    """Groups bins that share the same coordinate into one location."""
    groups: dict[tuple[float, float], Location] = {}
    for b in bins:
        key = (round(b.lat, 6), round(b.lng, 6))
        if key not in groups:
            groups[key] = Location(lat=b.lat, lng=b.lng)
        groups[key].bins.append(b)
    return list(groups.values())


class BinIndex:
    """In-memory index for nearest-bin lookups, vectorised over all locations."""

    def __init__(self, locations: list[Location]) -> None:
        self.locations = locations
        self._lats = np.array([loc.lat for loc in locations], dtype=float)
        self._lngs = np.array([loc.lng for loc in locations], dtype=float)

    def nearest(self, lat: float, lng: float) -> tuple[Location | None, float]:
        """Returns the closest location and its distance in metres."""
        if not self.locations:
            return None, float("inf")
        r = 6371000.0
        phi1 = radians(lat)
        phi2 = np.radians(self._lats)
        dphi = np.radians(self._lats - lat)
        dlambda = np.radians(self._lngs - lng)
        a = np.sin(dphi / 2) ** 2 + cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
        dist = 2 * r * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        i = int(np.argmin(dist))
        return self.locations[i], float(dist[i])


def load_bin_index(csv_path: Path = CSV_FILE, active_only: bool = True) -> BinIndex:
    return BinIndex(dedupe_by_location(read_bins(csv_path, active_only)))


def bearing(from_lat: float, from_lng: float,
            to_lat: float, to_lng: float) -> float:
    """Initial compass bearing (0-360, 0=N, 90=E) from one point to another."""
    phi1, phi2 = radians(from_lat), radians(to_lat)
    dlambda = radians(to_lng - from_lng)
    x = sin(dlambda) * cos(phi2)
    y = cos(phi1) * sin(phi2) - sin(phi1) * cos(phi2) * cos(dlambda)
    return (degrees(atan2(x, y)) + 360.0) % 360.0


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance in metres."""
    r = 6371000.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lng2 - lng1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    return 2 * r * atan2(sqrt(a), sqrt(1 - a))


def auto_pitch(distance_m: float) -> float:
    """Downward angle so the camera looks at the bin on the ground."""
    drop = CAMERA_HEIGHT_M - TARGET_HEIGHT_M
    pitch = -degrees(atan2(drop, max(distance_m, 1.0)))
    return max(pitch, MIN_PITCH_DEG)


def location_to_filename(location: Location) -> str:
    return f"streetview_{location.product_numbers[0]}.jpg"


def already_fetched(filename: str) -> bool:
    if (TO_ANNOTATE_DIR / filename).exists():
        return True
    for split in SPLITS:
        if (POOL_DIR / "images" / split / filename).exists():
            return True
    return False


def streetview_metadata(lat: float, lng: float, api_key: str,
                        radius: int = 50) -> dict:
    url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {
        "location": f"{lat},{lng}",
        "radius": radius,
        "source": "outdoor",
        "key": api_key,
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


def fetch_streetview_by_pano(pano_id: str, heading: float, pitch: float,
                             output_path: Path, api_key: str,
                             size: str = "640x480", fov: int = 60, scale: int = 2) -> None:
    url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "size": size,
        "pano": pano_id,
        "heading": round(heading, 2),
        "pitch": round(pitch, 2),
        "fov": fov,
        "source": "outdoor",
        "key": api_key,
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    output_path.write_bytes(response.content)


def offset_coord(lat: float, lng: float, bearing_deg: float,
                 dist_m: float) -> tuple[float, float]:
    """Moves a coordinate dist_m metres along a compass bearing.

    Longitude degrees are scaled by cos(latitude), which matters at Oslo's
    ~60°N where a degree of longitude is only ~55 km.
    """
    d_lat = (dist_m * cos(radians(bearing_deg))) / 111320.0
    d_lng = (dist_m * sin(radians(bearing_deg))) / (111320.0 * cos(radians(lat)))
    return lat + d_lat, lng + d_lng


def second_nearest_pano(bin_lat: float, bin_lng: float, pano0_id: str,
                        pano0_lat: float, pano0_lng: float, api_key: str,
                        radius: int = SECOND_PANO_RADIUS_M,
                        ring_radius: float = DEFAULT_RING_RADIUS_M,
                        ) -> tuple[str, float, float, str] | None:
    """Finds a distinct panorama near the bin, other than pano0.

    The REST metadata API only ever returns the single nearest panorama, so to
    reach a different one we re-query from points offset away from pano0: two
    along the pano0->bin axis (the road usually continues that way) and a ring
    around the bin. Candidates are deduped by pano_id, pano0 and its temporal
    twins are dropped, and the one closest to the bin is returned. Metadata
    queries are free. Returns (pano_id, lat, lng, capture_date) or None.
    """
    axis = bearing(pano0_lat, pano0_lng, bin_lat, bin_lng)
    query_points = [offset_coord(bin_lat, bin_lng, axis, d) for d in DIRECTIONAL_OFFSETS_M]
    query_points += [offset_coord(bin_lat, bin_lng, b, ring_radius) for b in RING_BEARINGS_DEG]

    candidates: dict[str, tuple[float, float, str]] = {}
    for qlat, qlng in query_points:
        try:
            meta = streetview_metadata(qlat, qlng, api_key, radius)
        except Exception:
            continue
        if meta.get("status") != "OK":
            continue
        candidates[meta["pano_id"]] = (
            meta["location"]["lat"], meta["location"]["lng"], meta.get("date", ""),
        )
    candidates.pop(pano0_id, None)

    best: tuple[str, float, float, str] | None = None
    best_dist = float("inf")
    for pano_id, (plat, plng, pdate) in candidates.items():
        if haversine_m(plat, plng, pano0_lat, pano0_lng) < TEMPORAL_DUP_EPS_M:
            continue
        dist = haversine_m(plat, plng, bin_lat, bin_lng)
        if dist < best_dist:
            best_dist = dist
            best = (pano_id, plat, plng, pdate)
    return best


def detect_bin(model: object, image_path: Path, conf: float) -> tuple[bool, float]:
    """Runs YOLO on one image and returns (bin_detected, best_confidence).

    Single-class detector (class 0 = trash_bin), so any returned box is a bin.
    A corrupt or unreadable image counts as no detection rather than crashing.
    """
    try:
        results = model.predict(str(image_path), conf=conf, verbose=False)
    except Exception:
        return False, 0.0
    if not results:
        return False, 0.0
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return False, 0.0
    confidences = boxes.conf.tolist()
    if not confidences:
        return True, 0.0
    return True, float(max(confidences))


def second_location_filename(location: Location) -> str:
    return f"streetview_{location.product_numbers[0]}_p2.jpg"


def _log_row(filename: str, location: Location, pano_id: str, pano_lat: float,
             pano_lng: float, dist: float, heading: float, pitch: float,
             capture_date: str, status: str, address: str, attempts: int,
             detected: bool | None, conf: float | None, pano_rank: int,
             pair: str) -> dict:
    return {
        "filename": filename,
        "product_numbers": ";".join(location.product_numbers),
        "bin_lat": f"{location.lat:.7f}",
        "bin_lng": f"{location.lng:.7f}",
        "pano_id": pano_id,
        "pano_lat": f"{pano_lat:.7f}",
        "pano_lng": f"{pano_lng:.7f}",
        "distance_m": f"{dist:.1f}",
        "heading": f"{heading:.2f}",
        "pitch": f"{pitch:.2f}",
        "capture_date": capture_date,
        "status": status,
        "address": address,
        "attempts": str(attempts),
        "detected": "" if detected is None else str(detected),
        "detection_conf": "" if conf is None else f"{conf:.3f}",
        "pano_rank": str(pano_rank),
        "pair": pair,
    }


def reverse_geocode(lat: float, lng: float, api_key: str) -> str:
    """Looks up the nearest street address for a coordinate (optional, costs an API call)."""
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"latlng": f"{lat},{lng}", "key": api_key}
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    if data.get("status") != "OK" or not data.get("results"):
        return ""
    return data["results"][0]["formatted_address"]


def load_log() -> dict[str, dict]:
    if not LOG_FILE.exists():
        return {}
    with open(LOG_FILE, encoding="utf-8-sig", newline="") as f:
        return {row["filename"]: row for row in csv.DictReader(f)}


def write_log(rows: dict[str, dict]) -> None:
    with open(LOG_FILE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for filename in sorted(rows):
            writer.writerow(rows[filename])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Street View images aimed at bin coordinates from a CSV."
    )
    parser.add_argument("--csv", type=Path, default=CSV_FILE,
                        help=f"CSV with bin coordinates (default: {CSV_FILE})")
    parser.add_argument("--size", type=str, default="640x480",
                        help="Image size as WxH, max 640x640 for standard API key (default: 640x480)")
    parser.add_argument("--fov", type=int, default=120,
                        help="Field of view in degrees, 10-120 (default: 120)")
    parser.add_argument("--pitch", type=float, default=None,
                        help="Fixed pitch in degrees; if omitted, computed from distance")
    parser.add_argument("--radius", type=int, default=50,
                        help="Search radius in metres for the nearest panorama (default: 50)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after ~N new images (a location may yield 2 when it retries); "
                             "already-fetched are skipped and do not count")
    parser.add_argument("--include-inactive", action="store_true",
                        help="Also fetch bins marked Active=USANN")
    parser.add_argument("--reverse-geocode", action="store_true",
                        help="Also look up the street address for each location (extra API call)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute headings/pitch and query metadata, but do not download images (no detection)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help=f"YOLO weights for the detect-then-retry decision (default: {DEFAULT_MODEL})")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF,
                        help=f"Confidence threshold for the bin-detection decision (default: {DEFAULT_CONF})")
    parser.add_argument("--max-attempts", type=int, default=2,
                        help="Panoramas tried per location: 1 = nearest only, "
                             "2 = also second-nearest on a miss (default: 2)")
    parser.add_argument("--ring-radius", type=float, default=DEFAULT_RING_RADIUS_M,
                        help=f"Ring radius in metres when searching for the second panorama "
                             f"(default: {DEFAULT_RING_RADIUS_M})")
    parser.add_argument("--no-detect", action="store_true",
                        help="Disable YOLO detection and the second-panorama retry (old single-fetch behaviour)")
    args = parser.parse_args()

    api_key = _api_key()

    if not args.csv.exists():
        print(f"CSV ikke funnet: {args.csv}")
        return

    bins = read_bins(args.csv, active_only=not args.include_inactive)
    locations = dedupe_by_location(bins)

    print(f"Leste {len(bins)} kasse(r) -> {len(locations)} unike steder")
    if args.limit is not None:
        print(f"Stopper etter ~{args.limit} nye bilde(r).")
    if args.dry_run:
        print("DRY RUN: henter ikke bilder, bare geometri og metadata.\n")

    TO_ANNOTATE_DIR.mkdir(parents=True, exist_ok=True)
    log = load_log()

    detect_enabled = not args.no_detect and not args.dry_run
    model = None
    if detect_enabled:
        if not args.model.exists():
            print(f"YOLO-vekter ikke funnet: {args.model}")
            print("Angi --model, eller bruk --no-detect for å hente uten deteksjon.")
            return
        print(f"Laster YOLO (CPU) fra {args.model} ...")
        from ultralytics import YOLO
        model = YOLO(str(args.model))
        print(f"Deteksjon på: conf >= {args.conf}, opptil {args.max_attempts} panorama per sted.\n")

    fetched = skipped = no_imagery = failed = produced = 0
    logged_pano_ids: set[str] = {row["pano_id"] for row in log.values() if row.get("pano_id")}

    for i, loc in enumerate(locations, start=1):
        if args.limit is not None and produced >= args.limit:
            break

        filename = location_to_filename(loc)
        if already_fetched(filename):
            print(f"[{i}/{len(locations)}] Hopper over (finnes): {filename}")
            skipped += 1
            continue

        print(f"[{i}/{len(locations)}] {filename}  ({loc.lat:.6f}, {loc.lng:.6f})", end=" ")
        try:
            meta = streetview_metadata(loc.lat, loc.lng, api_key, args.radius)
        except Exception as e:
            print(f"-> METADATA-FEIL: {e}")
            failed += 1
            continue

        status = meta.get("status", "UNKNOWN")
        if status != "OK":
            print(f"-> ingen panorama ({status})")
            no_imagery += 1
            continue

        pano0_id  = meta["pano_id"]
        if pano0_id in logged_pano_ids:
            print(f"-> hopper over: panorama {pano0_id[:12]}… allerede hentet fra et annet sted")
            skipped += 1
            continue
        pano0_lat = meta["location"]["lat"]
        pano0_lng = meta["location"]["lng"]
        date0     = meta.get("date", "")
        dist0     = haversine_m(pano0_lat, pano0_lng, loc.lat, loc.lng)
        head0     = bearing(pano0_lat, pano0_lng, loc.lat, loc.lng)
        pitch0    = args.pitch if args.pitch is not None else auto_pitch(dist0)

        address = ""
        if args.reverse_geocode:
            try:
                address = reverse_geocode(loc.lat, loc.lng, api_key)
            except Exception:
                address = ""

        print(f"-> pano {dist0:.0f}m unna, heading {head0:.0f}°, pitch {pitch0:.0f}°")

        if model is None:
            if not args.dry_run:
                try:
                    fetch_streetview_by_pano(
                        pano0_id, head0, pitch0, TO_ANNOTATE_DIR / filename,
                        api_key, args.size, args.fov,
                    )
                    fetched += 1
                    print("   [lagret]")
                except Exception as e:
                    print(f"   BILDE-FEIL: {e}")
                    failed += 1
                    continue
            log[filename] = _log_row(
                filename, loc, pano0_id, pano0_lat, pano0_lng, dist0, head0,
                pitch0, date0, status, address,
                attempts=1, detected=None, conf=None, pano_rank=1, pair="",
            )
            logged_pano_ids.add(pano0_id)
            produced += 1
            continue

        try:
            fetch_streetview_by_pano(
                pano0_id, head0, pitch0, TO_ANNOTATE_DIR / filename,
                api_key, args.size, args.fov,
            )
        except Exception as e:
            print(f"   BILDE-FEIL (nærmeste): {e}")
            failed += 1
            continue
        fetched += 1
        detected0, conf0 = detect_bin(model, TO_ANNOTATE_DIR / filename, args.conf)
        print(f"   nærmeste: {'KASSE funnet' if detected0 else 'ingen kasse'} (conf {conf0:.2f})")

        if detected0 or args.max_attempts < 2:
            log[filename] = _log_row(
                filename, loc, pano0_id, pano0_lat, pano0_lng, dist0, head0,
                pitch0, date0, status, address,
                attempts=1, detected=detected0, conf=conf0, pano_rank=1, pair="",
            )
            logged_pano_ids.add(pano0_id)
            produced += 1
            continue

        second = second_nearest_pano(
            loc.lat, loc.lng, pano0_id, pano0_lat, pano0_lng,
            api_key, SECOND_PANO_RADIUS_M, args.ring_radius,
        )
        if second is None:
            print("   fant ikke et nest nærmeste panorama — beholder nærmeste")
            log[filename] = _log_row(
                filename, loc, pano0_id, pano0_lat, pano0_lng, dist0, head0,
                pitch0, date0, status, address,
                attempts=1, detected=detected0, conf=conf0, pano_rank=1, pair="",
            )
            logged_pano_ids.add(pano0_id)
            produced += 1
            continue

        pano1_id, pano1_lat, pano1_lng, date1 = second
        dist1     = haversine_m(pano1_lat, pano1_lng, loc.lat, loc.lng)
        head1     = bearing(pano1_lat, pano1_lng, loc.lat, loc.lng)
        pitch1    = args.pitch if args.pitch is not None else auto_pitch(dist1)
        filename2 = second_location_filename(loc)

        try:
            fetch_streetview_by_pano(
                pano1_id, head1, pitch1, TO_ANNOTATE_DIR / filename2,
                api_key, args.size, args.fov,
            )
        except Exception as e:
            print(f"   BILDE-FEIL (nest nærmeste): {e} — beholder nærmeste")
            log[filename] = _log_row(
                filename, loc, pano0_id, pano0_lat, pano0_lng, dist0, head0,
                pitch0, date0, status, address,
                attempts=1, detected=detected0, conf=conf0, pano_rank=1, pair="",
            )
            logged_pano_ids.add(pano0_id)
            produced += 1
            continue
        fetched += 1
        detected1, conf1 = detect_bin(model, TO_ANNOTATE_DIR / filename2, args.conf)
        print(f"   nest nærmeste ({dist1:.0f}m): "
              f"{'KASSE funnet' if detected1 else 'ingen kasse'} (conf {conf1:.2f}) "
              f"— beholder begge bilder")

        log[filename] = _log_row(
            filename, loc, pano0_id, pano0_lat, pano0_lng, dist0, head0,
            pitch0, date0, status, address,
            attempts=2, detected=detected0, conf=conf0, pano_rank=1, pair=filename2,
        )
        log[filename2] = _log_row(
            filename2, loc, pano1_id, pano1_lat, pano1_lng, dist1, head1,
            pitch1, date1, status, address,
            attempts=2, detected=detected1, conf=conf1, pano_rank=2, pair=filename,
        )
        logged_pano_ids.add(pano0_id)
        logged_pano_ids.add(pano1_id)
        produced += 2

    if log:
        write_log(log)

    print(f"\nFerdig: {fetched} bilde(r) hentet (betalte API-kall), "
          f"{skipped} hoppet over, {no_imagery} uten panorama, {failed} feilet.")
    if not args.dry_run and fetched:
        print(f"Bilder lagret i {TO_ANNOTATE_DIR} — klar for annotering.")
    print(f"Logg: {LOG_FILE}")


if __name__ == "__main__":
    main()
