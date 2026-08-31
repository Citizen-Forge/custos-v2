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

# Node, for agents working TypeScript projects. Found missing live
# (2026-08-31) by reading a working agent's own transcript: it ran `node`
# and `npm` and got "not found". Silent Run is TypeScript and was chosen
# precisely so agents could build and test headlessly, so without this an
# agent could write files and verify nothing -- worse than being blocked,
# because the ticket still looked workable and burned real inference.
#
# Baked into the shared image rather than reached through a per-project
# toolchain container, deliberately: the container route would mean
# giving a ticket's shell_exec access to the Docker socket, and PLAN.md
# Phase 7 keeps that to sandbox-runner alone. Unblocking TypeScript is
# not worth widening that boundary. The cost is that this image accretes
# a toolchain per language over time -- harness/toolchain.py exists so
# that at least fails loudly (a project declares what it needs and
# dispatch refuses work the toolchain can't support) instead of silently
# producing unverified work the way this gap did.
ARG NODE_MAJOR=22
RUN curl -fsSL https://deb.nodesource.com/setup_${NODE_MAJOR}.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && node --version && npm --version

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
