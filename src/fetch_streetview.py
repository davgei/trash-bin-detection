"""
Fetches Street View images for a list of street addresses or coordinates and
saves them to data/to_annotate/ so they can be annotated in the next session.

Each line in the input file is either:
    - a street address, e.g.  Karl Johans gate 1, Oslo
    - a coordinate (dot decimals), e.g.  59.947495, 10.669440

Coordinates are treated as exact bin locations: the script finds the nearest
panorama and aims the camera straight at the coordinate, which avoids the camera
pointing at a hedge or out into the road.

Addresses are geocoded, then snapped to the nearest bin coordinate in the bin
CSV (--csv). The camera is aimed at that bin. If no bin is within
--max-snap-distance metres, the address is skipped with a warning (it likely has
no bin in the CSV, or the geocoding was off).

Each entry is fetched exactly once. Before fetching, the script checks whether
an image already exists in data/to_annotate/ or data/annotated_backup/. After a
successful fetch the entry is removed from the input file so it is never fetched
again, even if the script is interrupted and re-run.

Requires the GOOGLE_MAPS_API_KEY environment variable to be set.

Run from the project root:
    py -3.14 -m src.fetch_streetview
    py -3.14 -m src.fetch_streetview --addresses data/streetview_addresses.txt
"""

import argparse
import os
import re
import requests
from pathlib import Path

ADDRESSES_FILE  = Path("data/streetview_addresses.txt")
TO_ANNOTATE_DIR = Path("data/to_annotate")
POOL_DIR        = Path("data/annotated_backup")
SPLITS          = ("train", "val", "test")


def _api_key() -> str:
    key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not key:
        raise EnvironmentError(
            "GOOGLE_MAPS_API_KEY is not set. "
            "Set it with: $env:GOOGLE_MAPS_API_KEY='your-key'  (PowerShell) "
            "or: set GOOGLE_MAPS_API_KEY=your-key  (cmd)"
        )
    return key


def address_to_filename(address: str) -> str:
    slug = address.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"\s+", "_", slug)
    return f"streetview_{slug}.jpg"


def parse_coordinate(line: str) -> tuple[float, float] | None:
    """Parses 'lat, lng' or 'lat lng' (dot decimals) into a coordinate, else None."""
    parts = re.split(r"[,\s]+", line.strip())
    if len(parts) != 2:
        return None
    try:
        lat, lng = float(parts[0]), float(parts[1])
    except ValueError:
        return None
    if -90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0:
        return (lat, lng)
    return None


def coord_to_filename(lat: float, lng: float) -> str:
    slug = f"{lat:.6f}_{lng:.6f}".replace(".", "_").replace("-", "m")
    return f"streetview_coord_{slug}.jpg"


def already_fetched(filename: str) -> bool:
    """Returns True if this image already exists in to_annotate or the pool."""
    if (TO_ANNOTATE_DIR / filename).exists():
        return True
    for split in SPLITS:
        if (POOL_DIR / "images" / split / filename).exists():
            return True
    return False


def read_addresses(path: Path) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def remove_address(path: Path, address: str) -> None:
    """Removes one address from the file, leaving all others intact."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    updated = [l for l in lines if l.strip() != address and not (l.strip() == "" and address == "")]
    path.write_text("".join(updated), encoding="utf-8")


def geocode_address(address: str, api_key: str) -> tuple[float, float]:
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": address, "key": api_key}
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    if data["status"] != "OK":
        raise ValueError(f"Geocoding feilet for '{address}': {data['status']}")
    location = data["results"][0]["geometry"]["location"]
    return location["lat"], location["lng"]


def fetch_streetview_image(
    lat: float,
    lng: float,
    output_path: Path,
    api_key: str,
    size: str = "640x480",
    fov: int = 120,
) -> None:
    url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "size": size,
        "location": f"{lat},{lng}",
        "fov": fov,
        "key": api_key,
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    output_path.write_bytes(response.content)


def _fetch_aimed(bin_lat: float, bin_lng: float, output_path: Path,
                 api_key: str, size: str, fov: int, radius: int = 50) -> None:
    """Finds the nearest panorama and aims the camera at the exact coordinate."""
    from src.fetch_streetview_from_csv import (
        auto_pitch, bearing, fetch_streetview_by_pano,
        haversine_m, streetview_metadata,
    )
    meta = streetview_metadata(bin_lat, bin_lng, api_key, radius)
    if meta.get("status") != "OK":
        raise ValueError(f"ingen panorama ({meta.get('status')})")
    pano_id  = meta["pano_id"]
    pano_lat = meta["location"]["lat"]
    pano_lng = meta["location"]["lng"]
    dist  = haversine_m(pano_lat, pano_lng, bin_lat, bin_lng)
    head  = bearing(pano_lat, pano_lng, bin_lat, bin_lng)
    pitch = auto_pitch(dist)
    fetch_streetview_by_pano(pano_id, head, pitch, output_path, api_key, size, fov)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Street View images for a list of addresses."
    )
    parser.add_argument(
        "--addresses", type=Path, default=ADDRESSES_FILE,
        help=f"Text file with one address per line (default: {ADDRESSES_FILE})",
    )
    parser.add_argument(
        "--size", type=str, default="640x480",
        help="Image size as WxH, max 640x640 for standard API key (default: 640x480)",
    )
    parser.add_argument(
        "--fov", type=int, default=120,
        help="Field of view in degrees (default: 120)",
    )
    parser.add_argument(
        "--radius", type=int, default=50,
        help="Search radius in metres for the nearest panorama (default: 50)",
    )
    parser.add_argument(
        "--csv", type=Path, default=Path("data/hentesteder.csv"),
        help="Bin coordinate CSV used to snap addresses to the nearest bin",
    )
    parser.add_argument(
        "--max-snap-distance", type=float, default=30.0,
        help="Max metres an address may be from a bin before it is skipped (default: 30)",
    )
    args = parser.parse_args()

    api_key = _api_key()

    if not args.addresses.exists():
        print(f"Adressefil ikke funnet: {args.addresses}")
        print(f"Opprett filen og legg til adresser, én per linje.")
        return

    entries = read_addresses(args.addresses)
    if not entries:
        print("Ingen adresser eller koordinater å hente.")
        return

    TO_ANNOTATE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fant {len(entries)} oppføring(er) i {args.addresses}")

    bin_index = None
    location_to_filename = None
    if any(parse_coordinate(e) is None for e in entries):
        from src.fetch_streetview_from_csv import load_bin_index
        from src.fetch_streetview_from_csv import location_to_filename as location_to_filename
        if not args.csv.exists():
            print(f"Adresse-oppslag krever kasse-CSV: {args.csv} (bruk --csv)")
            return
        bin_index = load_bin_index(args.csv)
        print(f"Lastet {len(bin_index.locations)} kassesteder fra {args.csv}")

    fetched = 0
    skipped = 0
    failed  = 0
    no_bin  = 0

    for entry in entries:
        coord = parse_coordinate(entry)

        if coord:
            filename = coord_to_filename(*coord)
            if already_fetched(filename):
                print(f"  Hopper over (finnes allerede): {entry}")
                remove_address(args.addresses, entry)
                skipped += 1
                continue
            print(f"  Koordinat: {entry} ...", end=" ", flush=True)
            try:
                _fetch_aimed(coord[0], coord[1], TO_ANNOTATE_DIR / filename,
                             api_key, args.size, args.fov, args.radius)
                remove_address(args.addresses, entry)
                print(f"lagret som {filename}")
                fetched += 1
            except Exception as e:
                print(f"FEIL: {e}")
                failed += 1
            continue

        print(f"  Adresse: {entry} ...", end=" ", flush=True)
        try:
            glat, glng = geocode_address(entry, api_key)
        except Exception as e:
            print(f"GEOKODING FEILET: {e}")
            failed += 1
            continue

        loc, dist = bin_index.nearest(glat, glng)
        if loc is None or dist > args.max_snap_distance:
            nearest = f"nærmeste {dist:.0f} m" if loc is not None else "ingen kasser i CSV"
            print(f"ingen kasse innen {args.max_snap_distance:.0f} m ({nearest}) — hoppet over")
            no_bin += 1
            continue

        filename = location_to_filename(loc)
        if already_fetched(filename):
            print(f"kasse finnes allerede ({filename}) — hoppet over")
            remove_address(args.addresses, entry)
            skipped += 1
            continue

        try:
            _fetch_aimed(loc.lat, loc.lng, TO_ANNOTATE_DIR / filename,
                         api_key, args.size, args.fov, args.radius)
            remove_address(args.addresses, entry)
            print(f"-> kasse {dist:.0f} m unna, lagret som {filename}")
            fetched += 1
        except Exception as e:
            print(f"FEIL: {e}")
            failed += 1

    print(f"\nFerdig: {fetched} hentet, {skipped} hoppet over, "
          f"{no_bin} uten kasse i nærheten, {failed} feilet.")
    if fetched:
        print(f"Bilder lagret i {TO_ANNOTATE_DIR} — klar for annotering.")


if __name__ == "__main__":
    main()
