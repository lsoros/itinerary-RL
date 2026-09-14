FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
COPY examples /app/examples

RUN pip install --no-cache-dir .

ENV ITINERARY_REWARD=/app/examples/env_reward.yaml
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "-m", "itinerary_rl.server"]
