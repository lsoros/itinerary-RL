"""v1 term catalog and the two weight lists.

The environment list is a step reward. The verifier list scores a finished
trajectory and is the only list allowed to weight ``invalid``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from itinerary_rl.clock import MINUTES_PER_DAY, parse_hhmm, scheduled_arrival_offset
from itinerary_rl.engine import Action, Episode
from itinerary_rl.index import WeekIndex, build_index
from itinerary_rl.loader import LoadError
from itinerary_rl.problem import Problem, ProblemError, load_problem

CATALOG = (
    "cancellation_rate",
    "distance",
    "trip_time_scheduled",
    "trip_time_tracked",
    "distance_to_go",
    "invalid",
)
AGGREGATES = ("any_leg", "mean", "max")
Aggregate = Literal["any_leg", "mean", "max"]


class RewardError(ValueError):
    """A weight list does not match the catalog."""


@dataclass(frozen=True)
class WeightedTerm:
    name: str
    weight: float
    aggregate: Aggregate | None = None


@dataclass(frozen=True)
class Scheme:
    terms: tuple[WeightedTerm, ...]
    allows_invalid: bool


@dataclass(frozen=True)
class Leg:
    origin: str
    dest: str
    day_of_week: int
    scheduled_departure: str
    scheduled_arrival: str | None
    scheduled_arrival_offset: float | None
    tracked_arrival_offset: float | None
    distance_miles: float | None
    cancellation_rate: float | None
    status: str | None


class Scorer:
    """One scheme, one index. Caches direct-airport miles for ``distance_to_go``."""

    def __init__(self, scheme: Scheme, index: WeekIndex, problem: Problem) -> None:
        self.scheme = scheme
        self.index = index
        self.problem = problem
        self._direct: dict[tuple[str, str], float] | None = None

    def step_reward(self, before: list[Leg], after: list[Leg]) -> float:
        total = 0.0
        for term in self.scheme.terms:
            if term.name == "distance_to_go":
                total += term.weight * _distance_to_go_delta(self, before, after)
            elif term.name == "invalid":
                continue
            elif term.name == "trip_time_tracked" and after and after[-1].status == "cancelled":
                total += term.weight * (0.0 - _value(term, before, self))
            else:
                total += term.weight * (_value(term, after, self) - _value(term, before, self))
        return total

    def score(self, legs: list[Leg], invalid_count: int, ended: str | None) -> float:
        total = 0.0
        for term in self.scheme.terms:
            if term.name == "invalid":
                total += term.weight * invalid_count
            elif term.name == "distance_to_go":
                total += term.weight * _distance_to_go_net(self, legs)
            elif term.name == "trip_time_tracked" and ended == "cancelled":
                continue
            else:
                total += term.weight * _value(term, legs, self)
        return total

    def direct_miles(self, origin: str, dest: str) -> float | None:
        if origin == dest:
            return 0.0
        if self._direct is None:
            self._direct = _direct_table(self.index)
        return self._direct.get((origin, dest))


def load_scheme(path: str | Path, *, allows_invalid: bool) -> Scheme:
    source = Path(path)
    if not source.is_file():
        raise RewardError(f"weight list not found: {source}")
    try:
        loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RewardError(f"{source}: {exc}") from exc
    return parse_scheme(loaded, allows_invalid=allows_invalid)


def parse_scheme(loaded: object, *, allows_invalid: bool) -> Scheme:
    if not isinstance(loaded, dict) or set(loaded) != {"terms"}:
        raise RewardError("weight list must contain only a terms field")
    raw_terms = loaded["terms"]
    if not isinstance(raw_terms, list):
        raise RewardError("terms must be a list")
    terms: list[WeightedTerm] = []
    for index, item in enumerate(raw_terms, start=1):
        terms.append(_term(item, index, allows_invalid=allows_invalid))
    return Scheme(terms=tuple(terms), allows_invalid=allows_invalid)


def step_legs(origin: str, action, service, moved: bool) -> Leg:
    return _leg(origin, action.day_of_week, action.scheduled_departure, service)


def legs_from_episode(episode: Episode) -> list[Leg]:
    origin = episode.problem.origin_airport
    legs: list[Leg] = []
    for record in episode.records:
        if record.service is None or record.action is None:
            continue
        legs.append(_leg(origin, record.action.day_of_week, record.action.scheduled_departure, record.service))
        if record.moved:
            origin = record.observation.airport
    return legs


def _term(item: object, index: int, *, allows_invalid: bool) -> WeightedTerm:
    if not isinstance(item, dict):
        raise RewardError(f"terms[{index}] must be a mapping")
    unknown = sorted(set(item) - {"name", "weight", "aggregate"})
    if unknown:
        raise RewardError(f"terms[{index}] has unknown fields: {', '.join(unknown)}")
    name = item.get("name")
    if name not in CATALOG:
        raise RewardError(f"terms[{index}] is not in the catalog: {name!r}")
    if name == "invalid" and not allows_invalid:
        raise RewardError("invalid is a verifier term and cannot be in the environment list")
    weight = item.get("weight")
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise RewardError(f"terms[{index}].weight must be a number")
    aggregate = item.get("aggregate")
    if name == "cancellation_rate":
        if aggregate is None:
            aggregate = "any_leg"
        if aggregate not in AGGREGATES:
            raise RewardError(f"terms[{index}].aggregate must be any_leg, mean, or max")
    elif aggregate is not None:
        raise RewardError(f"terms[{index}] does not take aggregate")
    return WeightedTerm(name=name, weight=float(weight), aggregate=aggregate)


def _value(term: WeightedTerm, legs: list[Leg], scorer: Scorer) -> float:
    if term.name == "cancellation_rate":
        return _cancellation_rate(legs, term.aggregate or "any_leg")
    if term.name == "distance":
        return sum(leg.distance_miles or 0.0 for leg in legs)
    if term.name == "trip_time_scheduled":
        return _span(legs, tracked=False)
    if term.name == "trip_time_tracked":
        return _span(legs, tracked=True)
    if term.name == "distance_to_go":
        return _distance_to_go_net(scorer, legs)
    if term.name == "invalid":
        return 0.0
    raise RewardError(f"unknown term: {term.name}")


def _cancellation_rate(legs: list[Leg], aggregate: str) -> float:
    rates = [leg.cancellation_rate for leg in legs if leg.cancellation_rate is not None]
    if not rates:
        return 0.0
    if aggregate == "mean":
        return sum(rates) / len(rates)
    if aggregate == "max":
        return max(rates)
    survival = 1.0
    for rate in rates:
        survival *= 1.0 - rate
    return 1.0 - survival


def _span(legs: list[Leg], *, tracked: bool) -> float:
    if not legs:
        return 0.0
    start = parse_hhmm(legs[0].scheduled_departure)
    end = start
    midnight = 0
    cursor_day = legs[0].day_of_week
    for index, leg in enumerate(legs):
        if index > 0:
            days = (leg.day_of_week - cursor_day) % 7
            candidate = midnight + days * MINUTES_PER_DAY + parse_hhmm(leg.scheduled_departure)
            if candidate < end:
                days += 7
            midnight += days * MINUTES_PER_DAY
            cursor_day = leg.day_of_week
        departure = midnight + parse_hhmm(leg.scheduled_departure)
        if index == 0:
            start = departure
        end = departure
        if leg.status == "cancelled":
            break
        arrival = leg.tracked_arrival_offset if tracked else leg.scheduled_arrival_offset
        if arrival is None:
            arrival = leg.scheduled_arrival_offset
        if arrival is None:
            continue
        end = midnight + arrival
    return float(end - start)


def _distance_to_go_delta(scorer: Scorer, before: list[Leg], after: list[Leg]) -> float:
    if len(after) <= len(before):
        return 0.0
    origin = before[-1].dest if before else scorer.problem.origin_airport
    landed = after[-1].dest if after[-1].status != "cancelled" else after[-1].origin
    before_miles = scorer.direct_miles(origin, scorer.problem.destination_airport)
    after_miles = 0.0 if landed == scorer.problem.destination_airport else scorer.direct_miles(
        landed, scorer.problem.destination_airport
    )
    if before_miles is None or after_miles is None:
        return 0.0
    return before_miles - after_miles


def _distance_to_go_net(scorer: Scorer, legs: list[Leg]) -> float:
    return _distance_to_go_delta(scorer, [], legs) if legs else 0.0


def _direct_table(index: WeekIndex) -> dict[tuple[str, str], float]:
    best: dict[tuple[str, str], float] = {}
    for service in index:
        if service.ambiguous or service.distance_miles is None:
            continue
        key = (service.origin, service.dest)
        miles = service.distance_miles
        if key not in best or miles < best[key]:
            best[key] = miles
    return best


def _leg(origin: str, day_of_week: int, scheduled_departure: str, service) -> Leg:
    arrival_offset = service.scheduled_arrival_offset_minutes
    if arrival_offset is None and service.scheduled_arrival:
        arrival_offset = scheduled_arrival_offset(scheduled_departure, service.scheduled_arrival)
    tracked = service.tracked_arrival_offset_minutes
    landed = origin if service.status == "cancelled" else (
        service.diversion_airport if service.status == "diverted" and service.diversion_airport else service.dest
    )
    return Leg(
        origin=origin,
        dest=landed,
        day_of_week=day_of_week,
        scheduled_departure=scheduled_departure,
        scheduled_arrival=service.scheduled_arrival,
        scheduled_arrival_offset=arrival_offset,
        tracked_arrival_offset=tracked,
        distance_miles=service.distance_miles,
        cancellation_rate=service.cancellation_rate,
        status=service.status,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score one reached trip and one illegal request.")
    parser.add_argument("problem")
    parser.add_argument("environment_weights")
    parser.add_argument("verifier_weights")
    parser.add_argument("extract")
    args = parser.parse_args(argv)
    try:
        problem = load_problem(args.problem)
        environment = load_scheme(args.environment_weights, allows_invalid=False)
        verifier = load_scheme(args.verifier_weights, allows_invalid=True)
    except (ProblemError, RewardError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"reading {args.extract}", flush=True)
    try:
        index = build_index(args.extract)
    except LoadError as exc:
        print(exc, file=sys.stderr)
        return 1

    env_scorer = Scorer(environment, index, problem)
    ver_scorer = Scorer(verifier, index, problem)
    from itinerary_rl.engine import _find_nonstop

    nonstop = _find_nonstop(problem, index)
    if nonstop is None:
        print("no in-window nonstop", file=sys.stderr)
        return 1
    reached = Episode(problem, index, scorer=env_scorer)
    reached.reset()
    observation = reached.step(nonstop)
    reached_legs = legs_from_episode(reached)
    rejected = Episode(problem, index)
    rejected.reset()
    rejected.step(
        Action(dest="ZZZ", marketing_carrier="AA", scheduled_departure="1100", day_of_week=1)
    )
    print(f"env_reward: {observation.reward}")
    print(f"verifier_reached: {ver_scorer.score(reached_legs, reached.invalid_count, reached.ended)}")
    print(
        "verifier_illegal: "
        f"{ver_scorer.score(legs_from_episode(rejected), rejected.invalid_count, rejected.ended)}"
    )
    print(f"cancellation_rate: {_cancellation_rate(reached_legs, 'any_leg')}")
    print(f"distance: {sum(leg.distance_miles or 0.0 for leg in reached_legs)}")
    print(f"trip_time_scheduled: {_span(reached_legs, tracked=False)}")
    print(f"trip_time_tracked: {_span(reached_legs, tracked=True)}")
    print(f"distance_to_go: {_distance_to_go_net(env_scorer, reached_legs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
