"""In-process episode. Resolves an action, or rejects it without crashing.

Attach a scorer to fill ``observation.reward``. Without one, the step reward
stays zero and the transition does not change. The in-memory records are the
trajectory; this module does not write a file.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Literal

from itinerary_rl.clock import MINUTES_PER_DAY, offset_clock, parse_hhmm, shift_weekday
from itinerary_rl.index import ServicePrototype, WeekIndex, build_index
from itinerary_rl.loader import WEEKDAYS, LoadError
from itinerary_rl.problem import Problem, ProblemError, load_problem

DEFAULT_STEP_CAP = 64
DEFAULT_MIN_CONNECTION_MINUTES = 45
EndedReason = Literal["reached", "cancelled", "max_connections", "no_connection", "step_cap"]


class EpisodeError(RuntimeError):
    """The caller asked the episode to continue after it ended."""


@dataclass(frozen=True)
class Action:
    dest: str
    marketing_carrier: str
    scheduled_departure: str
    day_of_week: int

    @property
    def weekday_name(self) -> str:
        return WEEKDAYS[self.day_of_week - 1]


@dataclass(frozen=True)
class Observation:
    """What the agent sees. No verifier score and no illegal-request flag."""

    airport: str
    weekday: int | None
    clock: str | None
    resulting_airport: str | None
    reward: float
    done: bool


@dataclass(frozen=True)
class StepRecord:
    """One request. ``invalid`` is the single bucket the verifier will penalize."""

    invalid: bool
    moved: bool
    ended: EndedReason | None
    observation: Observation
    action: Action | None = None
    service: ServicePrototype | None = None


@dataclass
class Episode:
    problem: Problem
    index: WeekIndex
    min_connection_minutes: int = DEFAULT_MIN_CONNECTION_MINUTES
    step_cap: int = DEFAULT_STEP_CAP
    airport: str = ""
    weekday: int | None = None
    clock: str | None = None
    resulting_airport: str | None = None
    done: bool = False
    ended: EndedReason | None = None
    flights_taken: int = 0
    steps_taken: int = 0
    _anchor_weekday: int | None = None
    _ready_offset: float | None = None
    records: list[StepRecord] = field(default_factory=list)
    scorer: object | None = None
    _reward: float = 0.0

    def reset(self) -> Observation:
        self.airport = self.problem.origin_airport
        self.weekday = None
        self.clock = None
        self.resulting_airport = None
        self.done = False
        self.ended = None
        self.flights_taken = 0
        self.steps_taken = 0
        self._anchor_weekday = None
        self._ready_offset = None
        self.records = []
        self._reward = 0.0
        return self._observation()

    def copy(self) -> Episode:
        """Branch for search. The scorer is dropped; transitions stay the same."""
        clone = Episode(
            problem=self.problem,
            index=self.index,
            min_connection_minutes=self.min_connection_minutes,
            step_cap=self.step_cap,
            airport=self.airport,
            weekday=self.weekday,
            clock=self.clock,
            resulting_airport=self.resulting_airport,
            done=self.done,
            ended=self.ended,
            flights_taken=self.flights_taken,
            steps_taken=self.steps_taken,
            scorer=None,
            _reward=0.0,
        )
        clone._anchor_weekday = self._anchor_weekday
        clone._ready_offset = self._ready_offset
        clone.records = list(self.records)
        return clone

    def legal_actions(self) -> list[Action]:
        """Actions the engine would accept from this position. Sorted for determinism."""
        if self.done:
            return []
        actions: list[Action] = []
        for service in self.index:
            if service.origin != self.airport:
                continue
            action = _action(service)
            if self._is_legal(action):
                actions.append(action)
        actions.sort(
            key=lambda action: (
                action.dest,
                action.marketing_carrier,
                action.day_of_week,
                action.scheduled_departure,
            )
        )
        return actions

    def step(self, action: Action) -> Observation:
        if self.done:
            raise EpisodeError("episode has ended")
        self.steps_taken += 1
        if not self._is_legal(action):
            return self._reject(action)
        return self._fly(action)

    @property
    def invalid_count(self) -> int:
        return sum(record.invalid for record in self.records)

    def _reject(self, action: Action | None = None) -> Observation:
        ended: EndedReason | None = None
        if self.steps_taken >= self.step_cap:
            self.done = True
            self.ended = ended = "step_cap"
        self._reward = 0.0
        observation = self._observation()
        self.records.append(
            StepRecord(
                invalid=True,
                moved=False,
                ended=ended,
                observation=observation,
                action=action,
            )
        )
        return observation

    def _fly(self, action: Action) -> Observation:
        service = self.index.get(
            origin=self.airport,
            dest=action.dest,
            marketing_carrier=action.marketing_carrier,
            day_of_week=action.day_of_week,
            scheduled_departure=action.scheduled_departure,
        )
        if service is None or service.ambiguous or service.status is None:
            return self._reject(action)
        if service.status == "cancelled":
            return self._cancel(action, service)
        if service.status == "diverted":
            if not service.diversion_airport:
                return self._reject(action)
            return self._land(
                action,
                service,
                airport=service.diversion_airport,
                arrival_offset=service.scheduled_arrival_offset_minutes,
            )
        return self._land(
            action,
            service,
            airport=service.dest,
            arrival_offset=service.tracked_arrival_offset_minutes,
        )

    def _cancel(self, action: Action, service: ServicePrototype) -> Observation:
        self.done = True
        self.ended = "cancelled"
        self.resulting_airport = self.airport
        self._reward = self._step_reward(action, service, moved=False)
        observation = self._observation()
        self.records.append(
            StepRecord(
                invalid=False,
                moved=False,
                ended="cancelled",
                observation=observation,
                action=action,
                service=service,
            )
        )
        return observation

    def _land(
        self,
        action: Action,
        service: ServicePrototype,
        *,
        airport: str,
        arrival_offset: float | None,
    ) -> Observation:
        offset = self._arrival_offset(action, arrival_offset)
        self.airport = airport
        self.flights_taken += 1
        self._anchor_weekday = action.day_of_week
        self._ready_offset = offset + self.min_connection_minutes
        clock, day_shift = offset_clock(offset)
        self.weekday = shift_weekday(action.day_of_week, day_shift)
        self.clock = clock
        self.resulting_airport = airport
        ended = self._ending_after_landing(airport)
        if ended is not None:
            self.done = True
            self.ended = ended
        self._reward = self._step_reward(action, service, moved=True)
        observation = self._observation()
        self.records.append(
            StepRecord(
                invalid=False,
                moved=True,
                ended=ended,
                observation=observation,
                action=action,
                service=service,
            )
        )
        return observation

    def _ending_after_landing(self, airport: str) -> EndedReason | None:
        if airport == self.problem.destination_airport:
            return "reached"
        if self.flights_taken >= self.problem.max_flights:
            return "max_connections"
        if not self._has_continuation():
            return "no_connection"
        return None

    def _arrival_offset(self, action: Action, arrival_offset: float | None) -> float:
        if arrival_offset is not None:
            return arrival_offset
        return float(parse_hhmm(action.scheduled_departure))

    def _is_legal(self, action: Action) -> bool:
        if action.dest == self.airport:
            return False
        if not self.problem.allows_carrier(action.marketing_carrier):
            return False
        if not self.problem.allows(action.day_of_week, action.scheduled_departure):
            return False
        if self._anchor_weekday is not None and self._ready_offset is not None:
            if self._departure_offset(action) < self._ready_offset:
                return False
        if self.flights_taken >= self.problem.max_flights:
            return False
        service = self.index.get(
            origin=self.airport,
            dest=action.dest,
            marketing_carrier=action.marketing_carrier,
            day_of_week=action.day_of_week,
            scheduled_departure=action.scheduled_departure,
        )
        return service is not None and not service.ambiguous and service.status is not None

    def _step_reward(self, action: Action, service: ServicePrototype, *, moved: bool) -> float:
        if self.scorer is None:
            return 0.0
        from itinerary_rl.terms import legs_from_episode, step_legs

        before = legs_from_episode(self)
        after = before + [step_legs(self.problem.origin_airport if not before else before[-1].dest, action, service, moved)]
        return self.scorer.step_reward(before, after)

    def _departure_offset(self, action: Action) -> float:
        assert self._anchor_weekday is not None
        days = (action.day_of_week - self._anchor_weekday) % 7
        offset = days * MINUTES_PER_DAY + parse_hhmm(action.scheduled_departure)
        if self._ready_offset is not None and offset < self._ready_offset:
            offset += 7 * MINUTES_PER_DAY
        return float(offset)

    def _has_continuation(self) -> bool:
        for service in self.index:
            if service.ambiguous or service.status is None:
                continue
            if service.origin != self.airport:
                continue
            if not self.problem.allows_carrier(service.marketing_carrier):
                continue
            if not self.problem.allows(service.day_of_week, service.scheduled_departure):
                continue
            action = Action(
                dest=service.dest,
                marketing_carrier=service.marketing_carrier,
                scheduled_departure=service.scheduled_departure,
                day_of_week=service.day_of_week,
            )
            if self._departure_offset(action) >= (self._ready_offset or 0):
                return True
        return False

    def _observation(self) -> Observation:
        return Observation(
            airport=self.airport,
            weekday=self.weekday,
            clock=self.clock,
            resulting_airport=self.resulting_airport,
            reward=self._reward,
            done=self.done,
        )


def play(problem: Problem, index: WeekIndex) -> tuple[str, str]:
    """Play one in-window nonstop, if one exists, and one illegal request."""
    episode = Episode(problem, index)
    episode.reset()
    nonstop = _find_nonstop(problem, index)
    if nonstop is None:
        reached = "no in-window nonstop from the origin to the destination"
    else:
        observation = episode.step(nonstop)
        reached = (
            f"{nonstop.marketing_carrier} {nonstop.weekday_name} {problem.origin_airport} "
            f"{nonstop.scheduled_departure} -> {nonstop.dest} "
            f"airport={observation.airport} done={observation.done} "
            f"ended={episode.ended} reward={observation.reward} "
            f"clock={observation.weekday and WEEKDAYS[observation.weekday - 1]} {observation.clock}"
        )

    episode.reset()
    illegal = episode.step(
        Action(dest="ZZZ", marketing_carrier="AA", scheduled_departure="1100", day_of_week=1)
    )
    rejected = (
        f"airport={illegal.airport} done={illegal.done} "
        f"invalid={episode.invalid_count} reward={illegal.reward}"
    )
    return reached, rejected


def _find_nonstop(problem: Problem, index: WeekIndex) -> Action | None:
    for service in index:
        if not _usable(problem, service):
            continue
        if service.origin != problem.origin_airport or service.dest != problem.destination_airport:
            continue
        if service.status != "operated":
            continue
        return _action(service)
    return None


def _usable(problem: Problem, service: ServicePrototype) -> bool:
    return (
        not service.ambiguous
        and service.status is not None
        and problem.allows_carrier(service.marketing_carrier)
        and problem.allows(service.day_of_week, service.scheduled_departure)
    )


def _action(service: ServicePrototype) -> Action:
    return Action(
        dest=service.dest,
        marketing_carrier=service.marketing_carrier,
        scheduled_departure=service.scheduled_departure,
        day_of_week=service.day_of_week,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Play one nonstop and one illegal request.")
    parser.add_argument("problem", help="Problem YAML.")
    parser.add_argument("extract", help="BTS CSV path. Not stored in the repo.")
    args = parser.parse_args(argv)
    try:
        problem = load_problem(args.problem)
    except ProblemError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"reading {args.extract}", flush=True)
    try:
        index = build_index(args.extract)
    except LoadError as exc:
        print(exc, file=sys.stderr)
        return 1
    reached, rejected = play(problem, index)
    print(f"nonstop: {reached}")
    print(f"illegal: {rejected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
