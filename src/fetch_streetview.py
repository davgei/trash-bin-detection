"""
Fetches Street View images for a list of street addresses and saves them to
data/to_annotate/ so they can be annotated in the next session.

Each address is fetched exactly once. Before fetching, the script checks
whether an image for that address already exists in data/to_annotate/ or
data/annotated_backup/. After a successful fetch the address is removed from
the input file so it is never fetched again, even if the script is interrupted
and re-run.

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
        help="Image size as WxH (default: 640x480)",
    )
    parser.add_argument(
        "--fov", type=int, default=120,
        help="Field of view in degrees (default: 120)",
    )
    args = parser.parse_args()

    api_key = _api_key()

    if not args.addresses.exists():
        print(f"Adressefil ikke funnet: {args.addresses}")
        print(f"Opprett filen og legg til adresser, én per linje.")
        return

    addresses = read_addresses(args.addresses)
    if not addresses:
        print("Ingen adresser å hente.")
        return

    TO_ANNOTATE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fant {len(addresses)} adresse(r) i {args.addresses}")

    fetched = 0
    skipped = 0
    failed  = 0

    for address in addresses:
        filename = address_to_filename(address)

        if already_fetched(filename):
            print(f"  Hopper over (finnes allerede): {address}")
            remove_address(args.addresses, address)
            skipped += 1
            continue

        print(f"  Henter: {address} ...", end=" ", flush=True)
        try:
            lat, lng = geocode_address(address, api_key)
            output_path = TO_ANNOTATE_DIR / filename
            fetch_streetview_image(lat, lng, output_path, api_key, args.size, args.fov)
            remove_address(args.addresses, address)
            print(f"lagret som {filename}")
            fetched += 1
        except Exception as e:
            print(f"FEIL: {e}")
            failed += 1

    print(f"\nFerdig: {fetched} hentet, {skipped} hoppet over, {failed} feilet.")
    if fetched:
        print(f"Bilder lagret i {TO_ANNOTATE_DIR} — klar for annotering.")


if __name__ == "__main__":
    main()
