"""Streaming reader for a Marketing Carrier On-Time Performance extract.

The file stays outside the repo. Callers pass a path at runtime. Extra columns
are kept so a later index can use fields this slice does not interpret.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

# BTS DAY_OF_WEEK is 1 = Monday through 7 = Sunday.
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

REQUIRED_COLUMNS = (
    "FL_DATE",
    "DAY_OF_WEEK",
    "MKT_UNIQUE_CARRIER",
    "MKT_CARRIER_FL_NUM",
    "OP_UNIQUE_CARRIER",
    "OP_CARRIER_FL_NUM",
    "ORIGIN",
    "ORIGIN_CITY_NAME",
    "DEST",
    "DEST_CITY_NAME",
    "CRS_DEP_TIME",
    "CRS_ARR_TIME",
    "ARR_DELAY_NEW",
    "CANCELLED",
    "DIVERTED",
    "DIV1_AIRPORT",
    "DISTANCE",
)


class LoadError(ValueError):
    """The extract cannot be read as the contract requires."""


@dataclass(frozen=True)
class FlightRecord:
    """One historical flight. Times are zero-padded HHMM strings."""

    fl_date: str
    day_of_week: int
    marketing_carrier: str
    marketing_flight_number: str
    operating_carrier: str
    operating_flight_number: str
    origin: str
    origin_city_name: str
    dest: str
    dest_city_name: str
    scheduled_departure: str
    scheduled_arrival: str
    arrival_delay_minutes: float | None
    cancelled: bool
    diverted: bool
    diversion_airport: str | None
    distance_miles: float | None
    extras: Mapping[str, str]

    @property
    def weekday_name(self) -> str:
        return WEEKDAYS[self.day_of_week - 1]


def iter_flights(path: str | Path, *, encoding: str = "utf-8-sig") -> Iterator[FlightRecord]:
    """Yield flights from a BTS CSV. Does not load the file into memory."""
    source = Path(path)
    if not source.is_file():
        raise LoadError(f"flight extract not found: {source}")

    with source.open(newline="", encoding=encoding) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise LoadError(f"flight extract has no header: {source}")
        missing = [name for name in REQUIRED_COLUMNS if name not in reader.fieldnames]
        if missing:
            raise LoadError(f"flight extract is missing columns: {', '.join(missing)}")

        for line_number, row in enumerate(reader, start=2):
            try:
                yield _record(row)
            except LoadError as exc:
                raise LoadError(f"{source}:{line_number}: {exc}") from exc


def _record(row: Mapping[str, str | None]) -> FlightRecord:
    extras = {
        key: value
        for key, value in row.items()
        if key not in REQUIRED_COLUMNS and value not in (None, "")
    }
    day = _required_int(row, "DAY_OF_WEEK")
    if day not in range(1, 8):
        raise LoadError(f"DAY_OF_WEEK must be 1..7, got {day}")
    return FlightRecord(
        fl_date=_required(row, "FL_DATE"),
        day_of_week=day,
        marketing_carrier=_required(row, "MKT_UNIQUE_CARRIER"),
        marketing_flight_number=_required(row, "MKT_CARRIER_FL_NUM"),
        operating_carrier=_required(row, "OP_UNIQUE_CARRIER"),
        operating_flight_number=_required(row, "OP_CARRIER_FL_NUM"),
        origin=_required(row, "ORIGIN"),
        origin_city_name=_required(row, "ORIGIN_CITY_NAME"),
        dest=_required(row, "DEST"),
        dest_city_name=_required(row, "DEST_CITY_NAME"),
        scheduled_departure=_hhmm(_required(row, "CRS_DEP_TIME")),
        scheduled_arrival=_hhmm(_required(row, "CRS_ARR_TIME")),
        arrival_delay_minutes=_optional_float(row, "ARR_DELAY_NEW"),
        cancelled=_flag(row, "CANCELLED"),
        diverted=_flag(row, "DIVERTED"),
        diversion_airport=_optional(row, "DIV1_AIRPORT"),
        distance_miles=_optional_float(row, "DISTANCE"),
        extras=extras,
    )


def _required(row: Mapping[str, str | None], name: str) -> str:
    value = (row.get(name) or "").strip()
    if not value:
        raise LoadError(f"missing {name}")
    return value


def _optional(row: Mapping[str, str | None], name: str) -> str | None:
    value = (row.get(name) or "").strip()
    return value or None


def _required_int(row: Mapping[str, str | None], name: str) -> int:
    raw = _required(row, name)
    try:
        return int(float(raw))
    except ValueError as exc:
        raise LoadError(f"{name} is not an integer: {raw}") from exc


def _optional_float(row: Mapping[str, str | None], name: str) -> float | None:
    raw = _optional(row, name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise LoadError(f"{name} is not a number: {raw}") from exc


def _flag(row: Mapping[str, str | None], name: str) -> bool:
    raw = _optional(row, name)
    if raw is None:
        return False
    try:
        return float(raw) != 0.0
    except ValueError as exc:
        raise LoadError(f"{name} is not a flag: {raw}") from exc


def _hhmm(raw: str) -> str:
    digits = raw.strip()
    if digits.endswith(".0"):
        digits = digits[:-2]
    if not digits.isdigit() or not 1 <= len(digits) <= 4:
        raise LoadError(f"scheduled time is not HHMM: {raw}")
    return digits.zfill(4)
