"""OpenEnv client. Talks to a running server; it does not load the extract."""

from __future__ import annotations

import argparse
from typing import Any

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from itinerary_rl.api import FlightAction, FlightObservation, FlightState

# The Thursday JFK 1930 -> LAX service used to check the weight lists.
_CHECK_ACTION = FlightAction(
    dest="LAX",
    marketing_carrier="AA",
    scheduled_departure="1930",
    day_of_week=4,
)


class ItineraryClient(EnvClient[FlightAction, FlightObservation, FlightState]):
    def _step_payload(self, action: FlightAction) -> dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[FlightObservation]:
        body = dict(payload["observation"])
        body["reward"] = payload.get("reward")
        body["done"] = payload.get("done", False)
        observation = FlightObservation.model_validate(body)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> FlightState:
        return FlightState.model_validate(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reset and take one known legal step.")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--problem", default="/app/examples/jfk_lax.yaml")
    args = parser.parse_args(argv)
    with ItineraryClient(base_url=args.url, connect_timeout_s=30.0).sync() as client:
        reset = client.reset(problem=args.problem)
        step = client.step(_CHECK_ACTION)
    print(
        "reset "
        f"airport={reset.observation.airport} "
        f"done={reset.observation.done} "
        f"reward={reset.observation.reward}"
    )
    print(
        "step "
        f"airport={step.observation.airport} "
        f"weekday={step.observation.weekday} "
        f"clock={step.observation.clock} "
        f"done={step.observation.done} "
        f"reward={step.observation.reward}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
