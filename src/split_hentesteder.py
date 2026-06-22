"""
Splits the large hentesteder.csv into chunk files of N unique collection points,
so a single fetch run touches at most N coordinates — a safety cap on Street View
API spend.

The source has one row per container, so each collection point (coordinate)
repeats many times (different fractions/routes). This keeps one row per unique
coordinate (the first row seen, with all original columns) and writes chunks of
--chunk-size points to data/hentesteder_chunks/.

Deduplication uses the same 6-decimal coordinate rounding as the fetch script, so
each chunk yields about --chunk-size actual Street View fetches.

Run from the project root:
    py -3.14 -m src.split_hentesteder
    py -3.14 -m src.split_hentesteder --chunk-size 1000
"""

import argparse
import csv
from pathlib import Path

from src.fetch_streetview_from_csv import _parse_coord

SOURCE     = Path("data/hentesteder.csv")
OUTPUT_DIR = Path("data/hentesteder_chunks")
DELIMITER  = ";"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split hentesteder.csv into chunks of unique collection points."
    )
    parser.add_argument("--input", type=Path, default=SOURCE,
                        help=f"Source CSV (default: {SOURCE})")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help=f"Where to write chunk files (default: {OUTPUT_DIR})")
    parser.add_argument("--chunk-size", type=int, default=1000,
                        help="Unique collection points per file (default: 1000)")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Fil ikke funnet: {args.input}")
        return

    with open(args.input, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=DELIMITER)
        fieldnames = reader.fieldnames or []
        seen: set[tuple[float, float]] = set()
        unique_rows: list[dict] = []
        total = 0
        no_coord = 0
        for row in reader:
            total += 1
            lat = _parse_coord(row.get("Breddegrad"))
            lng = _parse_coord(row.get("Lengdegrad"))
            if lat is None or lng is None:
                no_coord += 1
                continue
            key = (round(lat, 6), round(lng, 6))
            if key in seen:
                continue
            seen.add(key)
            unique_rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for old in args.output_dir.glob("hentesteder_*.csv"):
        old.unlink()

    count = len(unique_rows)
    num_files = -(-count // args.chunk_size) if count else 0
    for i in range(num_files):
        chunk = unique_rows[i * args.chunk_size:(i + 1) * args.chunk_size]
        out_path = args.output_dir / f"hentesteder_{i + 1:03d}.csv"
        with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=DELIMITER,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(chunk)

    print(f"Leste {total} rader fra {args.input}")
    print(f"Unike hentesteder: {count}  (hoppet over {no_coord} uten koordinat)")
    print(f"Skrev {num_files} fil(er) à opptil {args.chunk_size} punkter til {args.output_dir}")


if __name__ == "__main__":
    main()
