# Just the static `docker` CLI binary, no daemon -- sandbox.py uses it to
# talk to the HOST's Docker daemon via the mounted socket (Docker-out-of-
# Docker), not to run a nested daemon. Only the sandbox-runner service
# actually mounts the socket; see PLAN.md Phase 7 for why that access is
# deliberately never granted to worker/api containers.
FROM docker:27-cli AS docker-cli

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates git patch \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://raw.githubusercontent.com/gastownhall/beads/main/scripts/install.sh | bash \
    && git config --system user.email "worker@custos.local" \
    && git config --system user.name "custos-worker"

COPY --from=docker-cli /usr/local/bin/docker /usr/local/bin/docker

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY tests ./tests
COPY scripts ./scripts
COPY public ./public

ENV PYTHONPATH=/app/src

CMD ["python", "-m", "harness.worker"]
