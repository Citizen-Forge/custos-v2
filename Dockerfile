FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://raw.githubusercontent.com/gastownhall/beads/main/scripts/install.sh | bash \
    && git config --system user.email "worker@custos.local" \
    && git config --system user.name "custos-worker"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY tests ./tests
COPY scripts ./scripts

ENV PYTHONPATH=/app/src

CMD ["python", "-m", "harness.worker"]
