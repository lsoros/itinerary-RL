"""Problem YAML: airports, carriers, connection bound, and departure allow-list.

This file does not carry reward weights or a calendar date. Those stay in
their own configs so a problem can be reused under another score.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from itinerary_rl.clock import normalize_hhmm, parse_hhmm
from itinerary_rl.loader import WEEKDAYS

MAX_CONNECTIONS = 5
_WEEKDAY_INDEX = {name.lower(): index for index, name in enumerate(WEEKDAYS, start=1)}
_FIELDS = {
    "origin_airport",
    "destination_airport",
    "carriers",
    "max_connections",
    "departures",
}
_WINDOW_FIELDS = {"day", "start", "end"}


class ProblemError(ValueError):
    """The problem file does not match the contract."""


@dataclass(frozen=True)
class DepartureWindow:
    day_of_week: int
    start: str
    end: str

    @property
    def day_name(self) -> str:
        return WEEKDAYS[self.day_of_week - 1]

    def contains(self, scheduled_departure: str) -> bool:
        minute = parse_hhmm(scheduled_departure)
        return parse_hhmm(self.start) <= minute <= parse_hhmm(self.end)


@dataclass(frozen=True)
class Problem:
    origin_airport: str
    destination_airport: str
    carriers: tuple[str, ...]
    max_connections: int
    departures: tuple[DepartureWindow, ...]

    @property
    def max_flights(self) -> int:
        return self.max_connections + 1

    def allows(self, day_of_week: int, scheduled_departure: str) -> bool:
        return any(
            window.day_of_week == day_of_week and window.contains(scheduled_departure)
            for window in self.departures
        )

    def allows_carrier(self, marketing_carrier: str) -> bool:
        return marketing_carrier in self.carriers


def load_problem(path: str | Path, *, max_connections_ceiling: int = MAX_CONNECTIONS) -> Problem:
    source = Path(path)
    if not source.is_file():
        raise ProblemError(f"problem file not found: {source}")
    try:
        loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ProblemError(f"{source}: {exc}") from exc
    return parse_problem(loaded, max_connections_ceiling=max_connections_ceiling)


def parse_problem(loaded: object, *, max_connections_ceiling: int = MAX_CONNECTIONS) -> Problem:
    if not isinstance(loaded, dict):
        raise ProblemError("problem file must be a mapping")
    unknown = sorted(set(loaded) - _FIELDS)
    if unknown:
        raise ProblemError(f"unknown problem fields: {', '.join(unknown)}")
    missing = sorted(_FIELDS - set(loaded))
    if missing:
        raise ProblemError(f"missing problem fields: {', '.join(missing)}")

    origin = _airport(loaded["origin_airport"], "origin_airport")
    dest = _airport(loaded["destination_airport"], "destination_airport")
    if origin == dest:
        raise ProblemError("origin_airport and destination_airport must differ")

    return Problem(
        origin_airport=origin,
        destination_airport=dest,
        carriers=_carriers(loaded["carriers"]),
        max_connections=_connections(loaded["max_connections"], max_connections_ceiling),
        departures=_departures(loaded["departures"]),
    )


def _airport(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ProblemError(f"{field} must be an airport code")
    code = value.strip().upper()
    if len(code) != 3 or not code.isalpha():
        raise ProblemError(f"{field} must be a three-letter airport code, got {value!r}")
    return code


def _carriers(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ProblemError("carriers must be a non-empty list")
    codes: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ProblemError(f"carrier must be a string, got {item!r}")
        code = item.strip().upper()
        if len(code) != 2 or not code.isalnum():
            raise ProblemError(f"carrier must be a two-character marketing code, got {item!r}")
        if code not in codes:
            codes.append(code)
    return tuple(codes)


def _connections(value: object, ceiling: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProblemError(f"max_connections must be an integer, got {value!r}")
    if value < 0 or value > ceiling:
        raise ProblemError(f"max_connections must be from 0 to {ceiling}, got {value}")
    return value


def _departures(value: object) -> tuple[DepartureWindow, ...]:
    if not isinstance(value, list) or not value:
        raise ProblemError("departures must be a non-empty list")
    windows: list[DepartureWindow] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise ProblemError(f"departures[{index}] must be a mapping")
        unknown = sorted(set(item) - _WINDOW_FIELDS)
        if unknown:
            raise ProblemError(f"departures[{index}] has unknown fields: {', '.join(unknown)}")
        missing = sorted(_WINDOW_FIELDS - set(item))
        if missing:
            raise ProblemError(f"departures[{index}] is missing {', '.join(missing)}")
        start = _clock(item["start"], f"departures[{index}].start")
        end = _clock(item["end"], f"departures[{index}].end")
        if parse_hhmm(start) > parse_hhmm(end):
            raise ProblemError(
                f"departures[{index}] start is after end; split an overnight window across two days"
            )
        windows.append(
            DepartureWindow(
                day_of_week=_day(item["day"], f"departures[{index}].day"),
                start=start,
                end=end,
            )
        )
    return tuple(windows)


def _day(value: object, field: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value in range(1, 8):
        return value
    if isinstance(value, str) and value.strip().lower() in _WEEKDAY_INDEX:
        return _WEEKDAY_INDEX[value.strip().lower()]
    raise ProblemError(f"{field} must be a weekday name or 1..7, got {value!r}")


def _clock(value: object, field: str) -> str:
    try:
        return normalize_hhmm(value)
    except ValueError as exc:
        raise ProblemError(f"{field}: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read a problem YAML and print the allow-list checks.")
    parser.add_argument("path", help="Path to the problem file.")
    args = parser.parse_args(argv)
    try:
        problem = load_problem(args.path)
    except ProblemError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(
        f"{problem.origin_airport} -> {problem.destination_airport} "
        f"carriers={','.join(problem.carriers)} "
        f"max_connections={problem.max_connections} max_flights={problem.max_flights}"
    )
    for window in problem.departures:
        print(f"window: {window.day_name} {window.start}-{window.end}")
    probes = (
        (1, "1059"),
        (1, "1100"),
        (1, "1400"),
        (1, "1401"),
        (2, "0900"),
        (2, "0901"),
        (3, "1200"),
    )
    for day, clock in probes:
        name = WEEKDAYS[day - 1]
        print(f"allows {name} {clock}: {problem.allows(day, clock)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
