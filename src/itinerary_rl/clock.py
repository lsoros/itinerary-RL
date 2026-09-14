"""Clocks for scheduled and delay-adjusted arrivals.

Minutes are measured from midnight of the departure weekday. A value at or
above 1440 is the next day, which is how an overnight connection stays positive.
"""

from __future__ import annotations

MINUTES_PER_DAY = 24 * 60


def normalize_hhmm(value: object) -> str:
    """Accept 1100, "1100", and "11:00". Returns zero-padded HHMM."""
    if isinstance(value, bool) or value is None:
        raise ValueError(f"scheduled time is not HHMM: {value}")
    if isinstance(value, int):
        raw = str(value)
    else:
        raw = str(value).strip().replace(":", "")
    if not raw.isdigit() or not 1 <= len(raw) <= 4:
        raise ValueError(f"scheduled time is not HHMM: {value}")
    hhmm = raw.zfill(4)
    parse_hhmm(hhmm)
    return hhmm


def parse_hhmm(value: str) -> int:
    if len(value) != 4 or not value.isdigit():
        raise ValueError(f"scheduled time is not HHMM: {value}")
    hours, minutes = int(value[:2]), int(value[2:])
    if hours > 23 or minutes > 59:
        raise ValueError(f"scheduled time is not a valid clock: {value}")
    return hours * 60 + minutes


def format_hhmm(minute_of_day: int) -> str:
    minute_of_day %= MINUTES_PER_DAY
    return f"{minute_of_day // 60:02d}{minute_of_day % 60:02d}"


def scheduled_arrival_offset(departure_hhmm: str, arrival_hhmm: str) -> int:
    """Minutes from departure-day midnight to scheduled arrival, including red-eyes."""
    departure = parse_hhmm(departure_hhmm)
    arrival = parse_hhmm(arrival_hhmm)
    if arrival < departure:
        arrival += MINUTES_PER_DAY
    return arrival


def offset_clock(offset_minutes: float) -> tuple[str, int]:
    """HHMM and whole days after the departure weekday."""
    if offset_minutes < 0:
        raise ValueError("arrival offset cannot be earlier than departure-day midnight")
    day = int(offset_minutes // MINUTES_PER_DAY)
    minute = int(offset_minutes % MINUTES_PER_DAY)
    return format_hhmm(minute), day
