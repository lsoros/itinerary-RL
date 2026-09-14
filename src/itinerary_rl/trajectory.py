"""Trajectory log for the verifier.

The environment reward is not stored. A later change to the reward list must
not change a log that has already been written.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from itinerary_rl.engine import Action, Episode, _find_nonstop
from itinerary_rl.index import build_index
from itinerary_rl.loader import LoadError
from itinerary_rl.problem import Problem, ProblemError, load_problem


def trajectory(episode: Episode) -> dict[str, Any]:
    """Finished action sequence, including the single invalid bucket."""
    problem = episode.problem
    return {
        "origin_airport": problem.origin_airport,
        "destination_airport": problem.destination_airport,
        "carriers": list(problem.carriers),
        "max_connections": problem.max_connections,
        "ended": episode.ended,
        "invalid_count": episode.invalid_count,
        "steps": [_step(record) for record in episode.records],
    }


def _step(record) -> dict[str, Any]:
    action = record.action
    service = record.service
    payload: dict[str, Any] = {
        "invalid": record.invalid,
        "moved": record.moved,
        "ended": record.ended,
        "action": None
        if action is None
        else {
            "dest": action.dest,
            "marketing_carrier": action.marketing_carrier,
            "scheduled_departure": action.scheduled_departure,
            "day_of_week": action.day_of_week,
        },
        "resolved": None,
        "position": {
            "airport": record.observation.airport,
            "weekday": record.observation.weekday,
            "clock": record.observation.clock,
        },
    }
    if service is not None:
        payload["resolved"] = {
            "operating_carrier": service.operating_carrier,
            "operating_flight_number": service.operating_flight_number,
            "marketing_flight_number": service.marketing_flight_number,
            "status": service.status,
            "origin": service.origin,
            "dest": service.dest,
            "scheduled_departure": service.scheduled_departure,
            "scheduled_arrival": service.scheduled_arrival,
            "distance_miles": service.distance_miles,
            "cancellation_rate": service.cancellation_rate,
        }
    return payload


def demonstration(problem: Problem, index) -> tuple[Episode, Episode]:
    reached = Episode(problem, index)
    reached.reset()
    nonstop = _find_nonstop(problem, index)
    if nonstop is not None:
        reached.step(nonstop)

    rejected = Episode(problem, index)
    rejected.reset()
    rejected.step(
        Action(dest="ZZZ", marketing_carrier="AA", scheduled_departure="1100", day_of_week=1)
    )
    return reached, rejected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write two sample trajectories as JSON.")
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

    reached, rejected = demonstration(problem, index)
    reached_log = trajectory(reached)
    rejected_log = trajectory(rejected)
    print(
        "reached_log: "
        f"steps={len(reached_log['steps'])} "
        f"invalid={reached_log['invalid_count']} "
        f"ended={reached_log['ended']} "
        f"resolved={_resolved_label(reached_log)}"
    )
    print(
        "illegal_log: "
        f"steps={len(rejected_log['steps'])} "
        f"invalid={rejected_log['invalid_count']} "
        f"ended={rejected_log['ended']} "
        f"action={_action_label(rejected_log)}"
    )
    print(json.dumps({"reached": reached_log, "illegal": rejected_log}, indent=2))
    return 0


def _resolved_label(log: dict[str, Any]) -> str:
    if not log["steps"] or log["steps"][0]["resolved"] is None:
        return "none"
    resolved = log["steps"][0]["resolved"]
    return f"{resolved['operating_carrier']}{resolved['operating_flight_number']}"


def _action_label(log: dict[str, Any]) -> str:
    if not log["steps"] or log["steps"][0]["action"] is None:
        return "none"
    return log["steps"][0]["action"]["dest"]


if __name__ == "__main__":
    raise SystemExit(main())
