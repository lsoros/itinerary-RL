"""OpenEnv adapter. One session owns an episode. Every session shares the index."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml
from openenv.core.env_server.interfaces import Environment

from itinerary_rl.api import FlightAction, FlightObservation, FlightState
from itinerary_rl.engine import Action, Episode
from itinerary_rl.index import WeekIndex, build_index
from itinerary_rl.problem import Problem, ProblemError, load_problem, parse_problem
from itinerary_rl.terms import RewardError, Scheme, Scorer, load_scheme

_REPO_REWARD = Path(__file__).resolve().parents[2] / "examples" / "env_reward.yaml"


class ConfigError(RuntimeError):
    """The server was started without a dataset or an environment weight list."""


@dataclass(frozen=True)
class Runtime:
    """Process-wide data. Built once. Not part of a problem file."""

    index: WeekIndex
    scheme: Scheme
    min_connection_minutes: int
    step_cap: int


def load_runtime() -> Runtime:
    dataset = os.environ.get("ITINERARY_DATASET")
    if not dataset:
        raise ConfigError("ITINERARY_DATASET is required and must be a mounted extract path")
    print(f"reading {dataset}", flush=True)
    index = build_index(dataset)
    print("index ready", flush=True)
    return Runtime(
        index=index,
        scheme=load_scheme(_reward_path(), allows_invalid=False),
        min_connection_minutes=_positive_int("ITINERARY_MIN_CONNECTION", 45),
        step_cap=_positive_int("ITINERARY_STEP_CAP", 64),
    )


class ItineraryEnvironment(Environment[FlightAction, FlightObservation, FlightState]):
    """Calls the existing episode. A seed does not change the transition."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, runtime: Runtime) -> None:
        super().__init__()
        self._runtime = runtime
        self._episode: Episode | None = None
        self._state = FlightState()

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        problem: str | None = None,
        problem_yaml: str | None = None,
        **kwargs: Any,
    ) -> FlightObservation:
        del seed, kwargs
        loaded = _load_problem(problem, problem_yaml)
        episode = Episode(
            loaded,
            self._runtime.index,
            min_connection_minutes=self._runtime.min_connection_minutes,
            step_cap=self._runtime.step_cap,
            scorer=Scorer(self._runtime.scheme, self._runtime.index, loaded),
        )
        observation = episode.reset()
        self._episode = episode
        self._state = FlightState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            airport=observation.airport,
            ended=None,
        )
        return _observation(observation)

    def step(
        self,
        action: FlightAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> FlightObservation:
        del timeout_s, kwargs
        if self._episode is None:
            raise RuntimeError("reset before step")
        observation = self._episode.step(
            Action(
                dest=action.dest,
                marketing_carrier=action.marketing_carrier,
                scheduled_departure=action.scheduled_departure,
                day_of_week=action.day_of_week,
            )
        )
        self._state.step_count = self._episode.steps_taken
        self._state.airport = observation.airport
        self._state.ended = self._episode.ended
        return _observation(observation)

    @property
    def state(self) -> FlightState:
        return self._state


def _load_problem(path: str | None, text: str | None) -> Problem:
    if (path is None) == (text is None):
        raise ProblemError("reset requires exactly one of problem or problem_yaml")
    if path is not None:
        return load_problem(path)
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ProblemError(str(exc)) from exc
    return parse_problem(loaded)


def _observation(observation) -> FlightObservation:
    return FlightObservation(
        airport=observation.airport,
        weekday=observation.weekday,
        clock=observation.clock,
        resulting_airport=observation.resulting_airport,
        reward=observation.reward,
        done=observation.done,
    )


def _reward_path() -> Path:
    raw = os.environ.get("ITINERARY_REWARD")
    if raw:
        path = Path(raw)
        if not path.is_file():
            raise RewardError(f"environment weight list not found: {path}")
        return path
    if _REPO_REWARD.is_file():
        return _REPO_REWARD
    raise ConfigError("set ITINERARY_REWARD to an environment weight list")


def _positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a positive integer") from exc
    if value < 1:
        raise ConfigError(f"{name} must be a positive integer")
    return value
