# Just the static `docker` CLI binary plus the compose plugin, no daemon --
# sandbox.py uses it to talk to the HOST's Docker daemon via the mounted
# socket (Docker-out-of-Docker), not to run a nested daemon. Only the
# sandbox-runner service actually mounts the socket; see PLAN.md Phase 7
# for why that access is deliberately never granted to worker/api
# containers. docker:27-cli DOES bundle the compose plugin, but under
# /usr/local/libexec/docker/cli-plugins/, not /usr/local/lib/ (checked
# live inside the image -- an easy wrong guess since cli-plugins usually
# lives under lib/). Found the gap live when run_self_mod_deploy.py's own
# final `docker compose ... up -d` fell through to plain `docker --help`
# inside sandbox-runner, compose not being a recognized subcommand at all
# there (only /usr/local/bin/docker itself was ever being copied below).
FROM docker:27-cli AS docker-cli

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates git patch \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://raw.githubusercontent.com/gastownhall/beads/main/scripts/install.sh | bash \
    && git config --system user.email "worker@custos.local" \
    && git config --system user.name "custos-worker"

COPY --from=docker-cli /usr/local/bin/docker /usr/local/bin/docker
COPY --from=docker-cli /usr/local/libexec/docker/cli-plugins/docker-compose /usr/local/libexec/docker/cli-plugins/docker-compose

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY tests ./tests
COPY scripts ./scripts
COPY public ./public

ENV PYTHONPATH=/app/src

CMD ["python", "-m", "harness.worker"]
