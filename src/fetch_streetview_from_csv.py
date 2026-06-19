"""
Fetches Street View images aimed directly at trash bins, using exact bin
coordinates from a CSV export (Uttrekk_products).

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

Images are saved to data/to_annotate/ so they flow into the existing annotation
pipeline. A log of every fetch is written to data/streetview_log.csv.

Requires the GOOGLE_MAPS_API_KEY environment variable to be set.

Run from the project root:
    py -3.14 -m src.fetch_streetview_from_csv
    py -3.14 -m src.fetch_streetview_from_csv --dry-run           # geometry only, no images
    py -3.14 -m src.fetch_streetview_from_csv --limit 5           # test on the first 5 locations
    py -3.14 -m src.fetch_streetview_from_csv --reverse-geocode   # also look up street addresses
"""

import argparse
import csv
import numpy as np
import requests
from dataclasses import dataclass, field
from math import atan2, cos, degrees, radians, sin, sqrt
from pathlib import Path

from src.fetch_streetview import _api_key

CSV_FILE        = Path("data/Uttrekk_products(result).csv")
TO_ANNOTATE_DIR = Path("data/to_annotate")
POOL_DIR        = Path("data/annotated_backup")
LOG_FILE        = Path("data/streetview_log.csv")
SPLITS          = ("train", "val", "test")

CAMERA_HEIGHT_M = 2.5   # approximate height of the Street View car camera
TARGET_HEIGHT_M = 0.5   # approximate height we aim at on the bin
MIN_PITCH_DEG   = -45.0

LOG_COLUMNS = [
    "filename", "product_numbers", "bin_lat", "bin_lng",
    "pano_id", "pano_lat", "pano_lng", "distance_m",
    "heading", "pitch", "capture_date", "status", "address",
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


def read_bins(csv_path: Path, active_only: bool = True) -> list[Bin]:
    bins: list[Bin] = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if active_only and (row.get("Active") or "").strip() != "SANN":
                continue
            lat = _parse_coord(row.get("Latitude"))
            lng = _parse_coord(row.get("Longitude"))
            if lat is None or lng is None:
                continue
            bins.append(Bin(
                product_number=(row.get("ProductNumber") or "").strip(),
                lat=lat,
                lng=lng,
                bin_type=(row.get("BinType") or "").strip(),
                waste=(row.get("Info1") or "").strip(),
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
                             size: str = "640x480", fov: int = 120) -> None:
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
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
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
                        help="Stop after N newly fetched images; already-fetched are skipped "
                             "and do not count (for incremental testing)")
    parser.add_argument("--include-inactive", action="store_true",
                        help="Also fetch bins marked Active=USANN")
    parser.add_argument("--reverse-geocode", action="store_true",
                        help="Also look up the street address for each location (extra API call)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute headings/pitch and query metadata, but do not download images")
    args = parser.parse_args()

    api_key = _api_key()

    if not args.csv.exists():
        print(f"CSV ikke funnet: {args.csv}")
        return

    bins = read_bins(args.csv, active_only=not args.include_inactive)
    locations = dedupe_by_location(bins)

    print(f"Leste {len(bins)} kasse(r) -> {len(locations)} unike steder")
    if args.limit is not None:
        print(f"Stopper etter {args.limit} nye bilde(r).")
    if args.dry_run:
        print("DRY RUN: henter ikke bilder, bare geometri og metadata.\n")

    TO_ANNOTATE_DIR.mkdir(parents=True, exist_ok=True)
    log = load_log()

    fetched = skipped = no_imagery = failed = produced = 0

    for i, loc in enumerate(locations, start=1):
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

        pano_id  = meta["pano_id"]
        pano_lat = meta["location"]["lat"]
        pano_lng = meta["location"]["lng"]
        dist     = haversine_m(pano_lat, pano_lng, loc.lat, loc.lng)
        head     = bearing(pano_lat, pano_lng, loc.lat, loc.lng)
        pitch    = args.pitch if args.pitch is not None else auto_pitch(dist)

        address = ""
        if args.reverse_geocode:
            try:
                address = reverse_geocode(loc.lat, loc.lng, api_key)
            except Exception:
                address = ""

        print(f"-> pano {dist:.0f}m unna, heading {head:.0f}°, pitch {pitch:.0f}°", end="")

        if not args.dry_run:
            try:
                fetch_streetview_by_pano(
                    pano_id, head, pitch, TO_ANNOTATE_DIR / filename,
                    api_key, args.size, args.fov,
                )
                fetched += 1
                print("  [lagret]")
            except Exception as e:
                print(f"  BILDE-FEIL: {e}")
                failed += 1
                continue
        else:
            print()

        log[filename] = {
            "filename": filename,
            "product_numbers": ";".join(loc.product_numbers),
            "bin_lat": f"{loc.lat:.7f}",
            "bin_lng": f"{loc.lng:.7f}",
            "pano_id": pano_id,
            "pano_lat": f"{pano_lat:.7f}",
            "pano_lng": f"{pano_lng:.7f}",
            "distance_m": f"{dist:.1f}",
            "heading": f"{head:.2f}",
            "pitch": f"{pitch:.2f}",
            "capture_date": meta.get("date", ""),
            "status": status,
            "address": address,
        }

        produced += 1
        if args.limit is not None and produced >= args.limit:
            break

    if log:
        write_log(log)

    print(f"\nFerdig: {fetched} hentet, {skipped} hoppet over, "
          f"{no_imagery} uten panorama, {failed} feilet.")
    if not args.dry_run and fetched:
        print(f"Bilder lagret i {TO_ANNOTATE_DIR} — klar for annotering.")
    print(f"Logg: {LOG_FILE}")


if __name__ == "__main__":
    main()
