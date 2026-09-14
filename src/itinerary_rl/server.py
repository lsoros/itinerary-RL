"""Local OpenEnv server. The extract is a mount, not part of the image."""

from __future__ import annotations

import sys

import uvicorn
from openenv.core.env_server.http_server import create_app

from itinerary_rl.api import FlightAction, FlightObservation
from itinerary_rl.environment import ConfigError, ItineraryEnvironment, load_runtime
from itinerary_rl.loader import LoadError
from itinerary_rl.terms import RewardError


def main(argv: list[str] | None = None) -> int:
    del argv
    try:
        runtime = load_runtime()
    except (ConfigError, LoadError, RewardError) as exc:
        print(exc, file=sys.stderr)
        return 1
    app = create_app(
        lambda: ItineraryEnvironment(runtime),
        FlightAction,
        FlightObservation,
        env_name="itinerary",
    )
    uvicorn.run(app, host="0.0.0.0", port=8000)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
