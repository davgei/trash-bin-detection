"""
Reports how effective the second-panorama retry is, from data/streetview_log.csv.

Among locations where YOLO missed the nearest panorama and a second-nearest
panorama was fetched (a "second attempt", logged as pano_rank=2), this prints the
percentage where YOLO then found a bin on that second attempt.

Run from the project root:
    py -3.14 -m src.streetview_stats
    py -3.14 -m src.streetview_stats --log data/streetview_log.csv
"""

import argparse
import csv
from pathlib import Path

LOG_FILE = Path("data/streetview_log.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report the second-attempt detection rate from the Street View log."
    )
    parser.add_argument("--log", type=Path, default=LOG_FILE,
                        help=f"Street View log CSV (default: {LOG_FILE})")
    args = parser.parse_args()

    if not args.log.exists():
        print(f"Logg ikke funnet: {args.log}")
        return

    with open(args.log, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    second_attempts = [r for r in rows if r.get("pano_rank") == "2"]
    if not second_attempts:
        print("Ingen andreforsøk i loggen ennå (ingen rad med pano_rank=2).")
        return

    found = [r for r in second_attempts if r.get("detected") == "True"]
    pct = 100.0 * len(found) / len(second_attempts)

    print(f"Andreforsøk (nærmeste bommet, hentet nest nærmeste): {len(second_attempts)}")
    print(f"Fant kasse på andre forsøk:                          {len(found)}")
    print(f"Prosent:                                             {pct:.1f}%")


if __name__ == "__main__":
    main()
