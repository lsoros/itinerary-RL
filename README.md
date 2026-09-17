# intinerary-RL
This is a simple [OpenEnv](https://github.com/huggingface/openenv) and [Docker](https://www.docker.com/) environment for evaluating RL strategies. There are no trained policies to be found here, only a random agent with support for a heuristic-based greedy algorithm. 

I made this environment via conversation with a mix of Cursor Grok 4.6, Claude Opus 5, GPT 5.6-Sol, Codex 5.3, and Claude Opus 4.5. Please note that Sol has very rarely been known to permanently erase data, so its use is not broadly advised.

The specifications and contract for v1 can be found in [docs/spec.md](docs/spec.md). The base environment for a flight itinerary recommendation system has been implemented and minimally validated with just one test case, which is a flight from JFK -> LAX. Most active time was spent on design, with every (incremental) build test passing as codebase components were added. No design decisions were revisited during the build phase. See also [planned extensions for v2](docs/spec.md).

## Usage
Problem instances are specified via YAML files, as in the [following example](https://github.com/lsoros/itinerary-RL/blob/main/examples/jfk_lax.yaml):

```yaml
origin_airport: JFK
destination_airport: LAX
carriers: ["AA"]
max_connections: 1
departures:
  - day: Monday
    start: "1100"
    end: "1400"
  - day: Tuesday
    start: "0900"
    end: "0900"
  - day: Thursday
    start: "1200"
    end: "2000"
```

Prerequisites for running the environment:

- Python 3.12+ (3.12 recommended)
- A [BTS](https://transtats.bts.gov/) Marketing Carrier On-Time Performance CSV on disk (not in this repo)
- Docker, only if you want the OpenEnv server in a container

Paths below assume a Linux or WSL shell from the repo root.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Please note that `openenv` pulls a large dependency set, and installation may take awhile. 

## Two ways to drive the environment

### 1. Python script

No Docker. You load the extract, build the index, and call `reset` / `step` on an `Episode`.

```python
from itinerary_rl.engine import Action, Episode
from itinerary_rl.index import build_index
from itinerary_rl.problem import load_problem
from itinerary_rl.terms import Scorer, load_scheme

problem = load_problem("examples/jfk_lax.yaml")
index = build_index("/path/to/T_ONTIME_MARKETING.csv")
scheme = load_scheme("examples/env_reward.yaml", allows_invalid=False)

episode = Episode(problem, index, scorer=Scorer(scheme, index, problem))
obs = episode.reset()
obs = episode.step(
    Action(
        dest="LAX",
        marketing_carrier="AA",
        scheduled_departure="1930",
        day_of_week=4,  # Thursday; BTS uses 1 = Monday
    )
)
print(obs.airport, obs.clock, obs.reward, obs.done)
```

**Action fields:** destination airport, marketing carrier, exact scheduled departure (`HHMM`), weekday (`1`–`7`).

**Observation fields:** `airport`, `weekday`, `clock`, `resulting_airport`, `reward` (environment step reward), `done`. 

### 2. CLI

Build the image (does not copy the flight data CSV):

```bash
docker build -t itinerary-rl .
```

Start the server. Mount the extract directory and set `ITINERARY_DATASET` to the file inside the container:

```bash
docker run --rm -p 8000:8000 \
  -v /path/to/extract_dir:/data:ro \
  -e ITINERARY_DATASET=/data/T_ONTIME_MARKETING.csv \
  itinerary-rl
```

Wait until the log shows `index ready` and Uvicorn is listening. Optional environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `ITINERARY_DATASET` | required | Path to the CSV inside the container |
| `ITINERARY_REWARD` | `/app/examples/env_reward.yaml` | Environment weight list |
| `ITINERARY_MIN_CONNECTION` | `45` | Minutes between arrival and next departure |
| `ITINERARY_STEP_CAP` | `64` | Max steps so illegal requests cannot spin |

From another terminal, with the package installed locally:

```python
from itinerary_rl.api import FlightAction
from itinerary_rl.client import ItineraryClient

with ItineraryClient(base_url="http://127.0.0.1:8000").sync() as client:
    result = client.reset(problem="/app/examples/jfk_lax.yaml")
    result = client.step(
        FlightAction(
            dest="LAX",
            marketing_carrier="AA",
            scheduled_departure="1930",
            day_of_week=4,
        )
    )
    print(result.observation.airport, result.reward, result.done)
```

`reset` needs exactly one of:

- `problem` — path to a problem YAML **visible inside the container** (the image already has `examples/`)
- `problem_yaml` — the YAML text as a string

Smoke-check script (same known legal step):

```bash
python -m itinerary_rl.client
```

## Problem and reward files

The environment looks at historical data and returns recommendations given generic day of week and time of day preferences. The reward signal is used during training, and is configurable via YAML ([example file](https://github.com/lsoros/itinerary-RL/blob/main/examples/env_reward.yaml)) to allow custom weighting. A reward verifier ([example file](https://github.com/lsoros/itinerary-RL/blob/main/examples/verifier_reward.yaml)) is used as an orthogonal signal to distinguish valid from invalid trajectories separately assign partial rewards. Depending on the training paradigm, fields in these separate signal configurations may be chosen such that they overlap, or not.

**Problem** (`examples/jfk_lax.yaml`): origin and destination airports, marketing carriers, `max_connections`, and departure allow-list (weekday + military time). 

**Environment weights** (`examples/env_reward.yaml`): fill `observation.reward`. May not include `invalid`.

**Verifier weights** (`examples/verifier_reward.yaml`): score finished trajectories and search for ground truth. May include `invalid`. Changing the environment list does not change what the verifier treats as optimal.

## Examples (JFK -> LAX)

Seeded random baseline:

```bash
python -m itinerary_rl.baselines \
  examples/jfk_lax.yaml \
  examples/env_reward.yaml \
  examples/verifier_reward.yaml \
  /path/to/T_ONTIME_MARKETING.csv \
  --seed 0
```

Ground-truth optimal path (calculated as part of the verifier):

```bash
python -m itinerary_rl.verifier \
  examples/jfk_lax.yaml \
  examples/verifier_reward.yaml \
  /path/to/T_ONTIME_MARKETING.csv
```

Score a known nonstop and an illegal/invalid request under both the reward and verifier weight lists:

```bash
python -m itinerary_rl.terms \
  examples/jfk_lax.yaml \
  examples/env_reward.yaml \
  examples/verifier_reward.yaml \
  /path/to/T_ONTIME_MARKETING.csv
```

The greedy baseline is an interface only (`GreedyPolicy`); the policy is not yet implemented.

## Other module entry points

Useful while debugging the stack without OpenEnv:

| Command | What it does |
|---|---|
| `python -m itinerary_rl PATH.csv` | Stream the extract; print row counts |
| `python -m itinerary_rl.index PATH.csv` | Build the week index; print prototype summary |
| `python -m itinerary_rl.problem examples/jfk_lax.yaml` | Load a problem; probe allow/deny times |
| `python -m itinerary_rl.engine PROBLEM.csv` | One legal nonstop (if any) and one illegal step |
| `python -m itinerary_rl.trajectory ...` | Same demos as JSON trajectories |

Each of these commands rebuilds the flight catalog index from the CSV when it needs one. That can take a minute on a large extract.

## Dataset note

Use the [BTS](https://transtats.bts.gov/) Marketing Carrier On-Time Performance field list. Keep the file outside git. January is enough to exercise the loader and engine; a multi-year extract is the same schema and the same commands.


