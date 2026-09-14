"""Verifier search. Scores finished walks under the verifier list only.

Ground truth prefers a path that reaches the destination. If none do, it falls
back to cancelled or stuck legal walks. Feasibility is the episode engine.
Cancellation rate only ranks; it does not drop a path that still reaches.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from itinerary_rl.engine import Action, Episode, _action
from itinerary_rl.index import ServicePrototype, WeekIndex, build_index
from itinerary_rl.loader import LoadError, WEEKDAYS
from itinerary_rl.problem import Problem, ProblemError, load_problem
from itinerary_rl.terms import Leg, RewardError, Scorer, legs_from_episode, load_scheme

FALLBACK_ENDINGS = frozenset({"cancelled", "max_connections", "no_connection"})


@dataclass(frozen=True)
class GroundTruth:
    """Best legal walk under the verifier weights."""

    pass_used: str
    score: float
    ended: str | None
    actions: tuple[Action, ...]
    legs: tuple[Leg, ...]


def find_ground_truth(
    problem: Problem,
    index: WeekIndex,
    scorer: Scorer,
    *,
    min_connection_minutes: int = 45,
) -> GroundTruth | None:
    """Enumerate legal walks. Prefer reaching; else cancelled or stuck."""
    best_reached: GroundTruth | None = None
    best_fallback: GroundTruth | None = None
    by_origin = _by_origin(index)
    root = Episode(problem, index, min_connection_minutes=min_connection_minutes, scorer=None)
    root.reset()
    best_reached, best_fallback = _search(
        root, by_origin, scorer, best_reached, best_fallback
    )
    return best_reached if best_reached is not None else best_fallback


def _by_origin(index: WeekIndex) -> dict[str, list[ServicePrototype]]:
    groups: dict[str, list[ServicePrototype]] = {}
    for service in index:
        if service.ambiguous or service.status is None:
            continue
        groups.setdefault(service.origin, []).append(service)
    for services in groups.values():
        services.sort(
            key=lambda service: (
                service.dest,
                service.marketing_carrier,
                service.day_of_week,
                service.scheduled_departure,
            )
        )
    return groups


def _legal_actions(
    episode: Episode, by_origin: dict[str, list[ServicePrototype]]
) -> list[Action]:
    actions: list[Action] = []
    for service in by_origin.get(episode.airport, ()):
        action = _action(service)
        if episode._is_legal(action):
            actions.append(action)
    return actions


def _search(
    episode: Episode,
    by_origin: dict[str, list[ServicePrototype]],
    scorer: Scorer,
    best_reached: GroundTruth | None,
    best_fallback: GroundTruth | None,
) -> tuple[GroundTruth | None, GroundTruth | None]:
    if episode.done:
        candidate = _candidate(episode, scorer)
        if candidate is None:
            return best_reached, best_fallback
        if candidate.ended == "reached":
            return _better(best_reached, candidate), best_fallback
        if candidate.ended in FALLBACK_ENDINGS:
            return best_reached, _better(best_fallback, candidate)
        return best_reached, best_fallback
    for action in _legal_actions(episode, by_origin):
        branch = episode.copy()
        branch.step(action)
        best_reached, best_fallback = _search(
            branch, by_origin, scorer, best_reached, best_fallback
        )
    return best_reached, best_fallback


def _better(current: GroundTruth | None, candidate: GroundTruth) -> GroundTruth:
    if current is None or _rank(candidate) > _rank(current):
        return candidate
    return current


def _candidate(episode: Episode, scorer: Scorer) -> GroundTruth | None:
    if episode.ended is None:
        return None
    if episode.ended == "step_cap":
        return None
    legs = tuple(legs_from_episode(episode))
    actions = tuple(record.action for record in episode.records if record.action is not None)
    score = scorer.score(list(legs), invalid_count=0, ended=episode.ended)
    return GroundTruth(
        pass_used="reached" if episode.ended == "reached" else "fallback",
        score=score,
        ended=episode.ended,
        actions=actions,
        legs=legs,
    )


def _rank(candidate: GroundTruth) -> tuple:
    """Higher score wins. Ties prefer fewer flights, then a stable action order."""
    action_key = tuple(
        (
            action.dest,
            action.marketing_carrier,
            action.day_of_week,
            action.scheduled_departure,
        )
        for action in candidate.actions
    )
    return (candidate.score, -len(candidate.actions), action_key)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Find the verifier ground-truth path.")
    parser.add_argument("problem")
    parser.add_argument("verifier_weights")
    parser.add_argument("extract")
    args = parser.parse_args(argv)
    try:
        problem = load_problem(args.problem)
        scheme = load_scheme(args.verifier_weights, allows_invalid=True)
    except (ProblemError, RewardError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"reading {args.extract}", flush=True)
    try:
        index = build_index(args.extract)
    except LoadError as exc:
        print(exc, file=sys.stderr)
        return 1
    scorer = Scorer(scheme, index, problem)
    truth = find_ground_truth(problem, index, scorer)
    if truth is None:
        print("ground_truth: none")
        return 0
    print(f"pass: {truth.pass_used}")
    print(f"ended: {truth.ended}")
    print(f"score: {truth.score}")
    print(f"flights: {len(truth.actions)}")
    for index_no, (action, leg) in enumerate(zip(truth.actions, truth.legs), start=1):
        day = WEEKDAYS[action.day_of_week - 1]
        print(
            f"leg{index_no}: {action.marketing_carrier} {day} "
            f"{leg.origin}->{action.dest} dep={action.scheduled_departure} "
            f"status={leg.status} cancel_rate={leg.cancellation_rate} "
            f"miles={leg.distance_miles}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
