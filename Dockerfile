FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Install system deps for matrix-nio[e2e] (libolm)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libolm-dev gcc curl git && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency files first (cache layer)
COPY pyproject.toml uv.lock ./

# Install dependencies (no dev group)
RUN uv sync --no-dev --no-install-project

# Copy source, readme, alembic
COPY README.md ./
COPY src/ src/
COPY alembic/ alembic/
COPY alembic.ini ./

# Install the project itself
RUN uv sync --no-dev

# Create data directory, copy bundled defaults
RUN mkdir -p /data/skills /data/memory
COPY data/skills/ /data/skills/
COPY data/soul.md /data/soul.md
COPY data/memory/ /data/memory/

EXPOSE 8080

HEALTHCHECK --interval=30s --retries=3 \
    CMD curl -f http://localhost:8080/healthz || exit 1

# Run migrations then start the app
CMD ["sh", "-c", "uv run --no-dev alembic upgrade head && uv run --no-dev aineko"]
