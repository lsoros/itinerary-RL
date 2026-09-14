"""Generic-week index: one prototype per matched service, plus a cancellation rate.

Rows are streamed and aggregated. The finished index does not keep the extract.
Two different flights that share an action stay separate and are marked ambiguous
rather than merged.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Literal

from itinerary_rl.clock import scheduled_arrival_offset
from itinerary_rl.loader import WEEKDAYS, FlightRecord, iter_flights

Status = Literal["cancelled", "diverted", "operated"]
# Ties prefer the earlier name: a cancelled/diverted tie is cancelled.
_STATUS_TIE = ("cancelled", "diverted", "operated")

ServiceIdentity = tuple[str, str, str]
MatchKey = tuple[str, str, str, int, str]
ReliabilityKey = tuple[str, str, int, str, str]


@dataclass(frozen=True)
class ServicePrototype:
    """Fixed outcome of one action on the weekday template."""

    origin: str
    dest: str
    marketing_carrier: str
    day_of_week: int
    scheduled_departure: str
    ambiguous: bool
    operating_carrier: str | None
    operating_flight_number: str | None
    marketing_flight_number: str | None
    status: Status | None
    diversion_airport: str | None
    scheduled_arrival: str | None
    scheduled_arrival_offset_minutes: int | None
    arrival_delay_minutes: float | None
    tracked_arrival_offset_minutes: float | None
    cancellation_rate: float | None
    distance_miles: float | None
    sample_count: int

    @property
    def weekday_name(self) -> str:
        return WEEKDAYS[self.day_of_week - 1]


@dataclass
class WeekIndex:
    """In-memory index built once from an extract. Lookup is by action fields."""

    prototypes: dict[MatchKey, ServicePrototype]

    def get(
        self,
        *,
        origin: str,
        dest: str,
        marketing_carrier: str,
        day_of_week: int,
        scheduled_departure: str,
    ) -> ServicePrototype | None:
        return self.prototypes.get(
            (origin, dest, marketing_carrier, day_of_week, scheduled_departure)
        )

    def __len__(self) -> int:
        return len(self.prototypes)

    def __iter__(self) -> Iterator[ServicePrototype]:
        return iter(self.prototypes.values())


@dataclass
class _Reliability:
    cancelled: int = 0
    total: int = 0

    @property
    def rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.cancelled / self.total


@dataclass
class _Service:
    identities: set[ServiceIdentity] = field(default_factory=set)
    cancelled: int = 0
    diverted: int = 0
    operated: int = 0
    diversion_airports: dict[str, int] = field(default_factory=dict)
    scheduled_arrivals: dict[str, int] = field(default_factory=dict)
    marketing_flight_numbers: dict[str, int] = field(default_factory=dict)
    delays: list[float] = field(default_factory=list)
    distances: list[float] = field(default_factory=list)

    def add(self, record: FlightRecord) -> None:
        self.identities.add(
            (
                record.marketing_flight_number,
                record.operating_carrier,
                record.operating_flight_number,
            )
        )
        self.marketing_flight_numbers[record.marketing_flight_number] = (
            self.marketing_flight_numbers.get(record.marketing_flight_number, 0) + 1
        )
        if record.distance_miles is not None:
            self.distances.append(record.distance_miles)
        if record.scheduled_arrival:
            self.scheduled_arrivals[record.scheduled_arrival] = (
                self.scheduled_arrivals.get(record.scheduled_arrival, 0) + 1
            )
        status = _row_status(record)
        if status == "cancelled":
            self.cancelled += 1
        elif status == "diverted":
            self.diverted += 1
            if record.diversion_airport:
                self.diversion_airports[record.diversion_airport] = (
                    self.diversion_airports.get(record.diversion_airport, 0) + 1
                )
        else:
            self.operated += 1
            if record.arrival_delay_minutes is not None:
                self.delays.append(record.arrival_delay_minutes)

    @property
    def sample_count(self) -> int:
        return self.cancelled + self.diverted + self.operated


def build_index(path: str | Path, *, progress_every: int = 100_000) -> WeekIndex:
    """Stream an extract into prototypes. Prints progress so a long file is not silent."""
    services: dict[MatchKey, _Service] = {}
    reliability: dict[ReliabilityKey, _Reliability] = {}
    rows = 0
    for record in iter_flights(path):
        rows += 1
        if progress_every and rows % progress_every == 0:
            print(f"indexed {rows} rows", flush=True)
        key = _match_key(record)
        services.setdefault(key, _Service()).add(record)
        rate_key = _reliability_key(record)
        bucket = reliability.setdefault(rate_key, _Reliability())
        bucket.total += 1
        bucket.cancelled += int(record.cancelled)
    print(f"indexed {rows} rows", flush=True)
    return WeekIndex(
        prototypes={
            key: _freeze(key, acc, reliability) for key, acc in services.items()
        }
    )


def _match_key(record: FlightRecord) -> MatchKey:
    return (
        record.origin,
        record.dest,
        record.marketing_carrier,
        record.day_of_week,
        record.scheduled_departure,
    )


def _reliability_key(record: FlightRecord) -> ReliabilityKey:
    return (
        record.operating_carrier,
        record.operating_flight_number,
        record.day_of_week,
        record.origin,
        record.dest,
    )


def _row_status(record: FlightRecord) -> Status:
    if record.cancelled:
        return "cancelled"
    if record.diverted:
        return "diverted"
    return "operated"


def _freeze(
    key: MatchKey,
    acc: _Service,
    reliability: dict[ReliabilityKey, _Reliability],
) -> ServicePrototype:
    origin, dest, carrier, day, departure = key
    scheduled_arrival = _mode(acc.scheduled_arrivals)
    arrival_offset = (
        scheduled_arrival_offset(departure, scheduled_arrival)
        if scheduled_arrival is not None
        else None
    )
    common = dict(
        origin=origin,
        dest=dest,
        marketing_carrier=carrier,
        day_of_week=day,
        scheduled_departure=departure,
        scheduled_arrival=scheduled_arrival,
        scheduled_arrival_offset_minutes=arrival_offset,
        distance_miles=_median(acc.distances),
        sample_count=acc.sample_count,
    )
    if len(acc.identities) != 1:
        return ServicePrototype(
            ambiguous=True,
            operating_carrier=None,
            operating_flight_number=None,
            marketing_flight_number=None,
            status=None,
            diversion_airport=None,
            arrival_delay_minutes=None,
            tracked_arrival_offset_minutes=None,
            cancellation_rate=None,
            **common,
        )

    marketing_flight, operating_carrier, operating_flight = next(iter(acc.identities))
    status = _majority(acc)
    delay = _median(acc.delays) if status == "operated" else None
    tracked = None
    if status == "operated" and arrival_offset is not None:
        tracked = arrival_offset + (delay or 0.0)
    rate_key = (operating_carrier, operating_flight, day, origin, dest)
    return ServicePrototype(
        ambiguous=False,
        operating_carrier=operating_carrier,
        operating_flight_number=operating_flight,
        marketing_flight_number=marketing_flight,
        status=status,
        diversion_airport=_mode(acc.diversion_airports) if status == "diverted" else None,
        arrival_delay_minutes=delay,
        tracked_arrival_offset_minutes=tracked,
        cancellation_rate=reliability[rate_key].rate,
        **common,
    )


def _majority(acc: _Service) -> Status:
    counts = {
        "cancelled": acc.cancelled,
        "diverted": acc.diverted,
        "operated": acc.operated,
    }
    return max(_STATUS_TIE, key=lambda status: (counts[status], -_STATUS_TIE.index(status)))


def _mode(counts: dict[str, int]) -> str | None:
    if not counts:
        return None
    return min(counts, key=lambda item: (-counts[item], item))


def _median(values: Iterable[float]) -> float | None:
    sample = list(values)
    if not sample:
        return None
    return float(statistics.median(sample))


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    from itinerary_rl.loader import LoadError

    parser = argparse.ArgumentParser(description="Build a generic-week index from a BTS extract.")
    parser.add_argument("path", help="Path to the CSV. Not stored in the repo.")
    args = parser.parse_args(argv)

    print(f"reading {args.path}", flush=True)
    try:
        index = build_index(args.path)
    except LoadError as exc:
        print(exc, file=sys.stderr)
        return 1

    counts = {"cancelled": 0, "diverted": 0, "operated": 0}
    ambiguous = 0
    for item in index:
        ambiguous += int(item.ambiguous)
        if item.status is not None:
            counts[item.status] += 1
    print(f"services: {len(index)}")
    print(f"ambiguous: {ambiguous}")
    print(
        "prototypes: "
        f"cancelled={counts['cancelled']} diverted={counts['diverted']} operated={counts['operated']}"
    )
    example = index.get(
        origin="JFK",
        dest="LAX",
        marketing_carrier="AA",
        day_of_week=4,
        scheduled_departure="0700",
    )
    print(f"example: {_format_example(example)}")
    return 0


def _format_example(item: ServicePrototype | None) -> str:
    if item is None:
        return "AA JFK 0700 Thursday -> LAX not found"
    if item.ambiguous or item.status is None:
        return "AA JFK 0700 Thursday -> LAX is ambiguous"
    from itinerary_rl.clock import offset_clock

    tracked = "none"
    if item.tracked_arrival_offset_minutes is not None:
        hhmm, day = offset_clock(item.tracked_arrival_offset_minutes)
        tracked = f"{hhmm}+{day}d"
    rate = "none" if item.cancellation_rate is None else f"{item.cancellation_rate:.4f}"
    delay = "none" if item.arrival_delay_minutes is None else f"{item.arrival_delay_minutes:.1f}"
    return (
        f"{item.marketing_carrier} {item.marketing_flight_number} "
        f"{item.weekday_name} {item.origin} {item.scheduled_departure} -> {item.dest} "
        f"status={item.status} samples={item.sample_count} "
        f"cancel_rate={rate} delay={delay} tracked={tracked} "
        f"op={item.operating_carrier}{item.operating_flight_number}"
    )


if __name__ == "__main__":
    raise SystemExit(main())

