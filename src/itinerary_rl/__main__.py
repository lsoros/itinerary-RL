"""Print a short read-back of a flight extract. Does not write the file anywhere."""

from __future__ import annotations

import argparse
import sys

from itinerary_rl.loader import LoadError, iter_flights


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read a BTS on-time extract and summarize it.")
    parser.add_argument("path", help="Path to the CSV. Not stored in the repo.")
    args = parser.parse_args(argv)

    print(f"reading {args.path}", flush=True)
    try:
        first = None
        rows = cancelled = diverted = 0
        for record in iter_flights(args.path):
            rows += 1
            cancelled += int(record.cancelled)
            diverted += int(record.diverted)
            if first is None:
                first = record
    except LoadError as exc:
        print(exc, file=sys.stderr)
        return 1

    if first is None:
        print("flight extract has a header and no rows", file=sys.stderr)
        return 1

    print(f"rows: {rows}")
    print(f"cancelled: {cancelled}")
    print(f"diverted: {diverted}")
    print(
        "first: "
        f"{first.fl_date} {first.weekday_name} "
        f"{first.marketing_carrier} {first.marketing_flight_number} "
        f"{first.origin} {first.scheduled_departure} -> "
        f"{first.dest} {first.scheduled_arrival} "
        f"cancelled={first.cancelled} diverted={first.diverted} "
        f"delay={first.arrival_delay_minutes} miles={first.distance_miles}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
