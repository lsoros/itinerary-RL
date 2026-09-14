"""OpenEnv action, observation, and state for one itinerary step.

These are the wire types. They do not add a verifier score or an illegal-request
flag. ``ended`` is operator state, not part of the observation.
"""

from __future__ import annotations

from openenv.core.env_server.types import Action, Observation, State


class FlightAction(Action):
    dest: str
    marketing_carrier: str
    scheduled_departure: str
    day_of_week: int


class FlightObservation(Observation):
    airport: str
    weekday: int | None = None
    clock: str | None = None
    resulting_airport: str | None = None


class FlightState(State):
    airport: str = ""
    ended: str | None = None
