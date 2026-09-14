"""Clients of reset and step. Neither policy lives in the environment.

Random is a sanity check. It is seeded so the same problem and extract produce
the same trajectory. Greedy is the interface a later policy fills in.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass

from itinerary_rl.clock import format_hhmm
from itinerary_rl.engine import Action, Episode, Observation
from itinerary_rl.index import WeekIndex, build_index
from itinerary_rl.loader import LoadError
from itinerary_rl.problem import Problem, ProblemError, load_problem
from itinerary_rl.terms import RewardError, Scorer, legs_from_episode, load_scheme

# Share of steps that name a flight which is not in the index. The rest draw a
# known service, which can still be illegal for this position.
UNMATCHED_RATE = 0.25


class Policy:
    """Chooses the next request from the observation the agent is allowed to see."""

    def act(self, observation: Observation) -> Action:
        raise NotImplementedError


class GreedyPolicy(Policy):
    def act(self, observation: Observation) -> Action:
        raise NotImplementedError("greedy policy is not part of this build")


class RandomPolicy(Policy):
    """Seeded mixture of a known service and a request that cannot match."""

    def __init__(
        self,
        index: WeekIndex,
        seed: int,
        *,
        unmatched_rate: float = UNMATCHED_RATE,
    ) -> None:
        if not 0.0 <= unmatched_rate <= 1.0:
            raise ValueError("unmatched_rate must be between 0 and 1")
        self._rng = random.Random(seed)
        self._services = tuple(index)
        self._unmatched_rate = unmatched_rate

    def act(self, observation: Observation) -> Action:
        del observation
        if not self._services or self._rng.random() < self._unmatched_rate:
            return Action(
                dest="ZZZ",
                marketing_carrier="ZZ",
                scheduled_departure=format_hhmm(self._rng.randrange(24 * 60)),
                day_of_week=self._rng.randint(1, 7),
            )
        service = self._rng.choice(self._services)
        return Action(
            dest=service.dest,
            marketing_carrier=service.marketing_carrier,
            scheduled_departure=service.scheduled_departure,
            day_of_week=service.day_of_week,
        )


@dataclass(frozen=True)
class Rollout:
    ended: str | None
    invalid_count: int
    steps: int
    env_return: float
    verifier_score: float


def rollout(policy: Policy, episode: Episode, verifier: Scorer) -> Rollout:
    observation = episode.reset()
    env_return = 0.0
    while not observation.done:
        observation = episode.step(policy.act(observation))
        env_return += observation.reward
    return Rollout(
        ended=episode.ended,
        invalid_count=episode.invalid_count,
        steps=episode.steps_taken,
        env_return=env_return,
        verifier_score=verifier.score(
            legs_from_episode(episode), episode.invalid_count, episode.ended
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the seeded random baseline.")
    parser.add_argument("problem")
    parser.add_argument("environment_weights")
    parser.add_argument("verifier_weights")
    parser.add_argument("extract")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", choices=("random", "greedy"), default="random")
    parser.add_argument("--unmatched-rate", type=float, default=UNMATCHED_RATE)
    args = parser.parse_args(argv)
    if args.policy == "greedy":
        print("greedy policy is not part of this build", file=sys.stderr)
        return 2
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
    try:
        policy = RandomPolicy(index, args.seed, unmatched_rate=args.unmatched_rate)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    result = rollout(policy, _episode(problem, index, environment), Scorer(verifier, index, problem))
    print(f"seed: {args.seed}")
    print(f"ended: {result.ended}")
    print(f"steps: {result.steps}")
    print(f"invalid_count: {result.invalid_count}")
    print(f"env_return: {result.env_return}")
    print(f"verifier_score: {result.verifier_score}")
    return 0


def _episode(problem: Problem, index: WeekIndex, environment) -> Episode:
    return Episode(
        problem,
        index,
        scorer=Scorer(environment, index, problem),
    )


if __name__ == "__main__":
    raise SystemExit(main())
