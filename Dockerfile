FROM ubuntu:24.04

WORKDIR /app

# Install system deps (python3.12, libolm for matrix-nio[e2e], ROCm runtime libs)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv \
        libolm-dev gcc clang curl ca-certificates git ripgrep \
        libnuma1 libdrm2 poppler-utils imagemagick file jq && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Copy dependency files first (cache layer)
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --no-install-project

# Copy source, readme, alembic
COPY README.md ./
COPY src/ src/
COPY alembic/ alembic/
COPY alembic.ini ./

# Install the project itself
RUN uv sync --no-dev

EXPOSE 8080

HEALTHCHECK --interval=30s --retries=3 \
    CMD curl -f http://localhost:8080/healthz || exit 1

# Run migrations then exec the app (exec replaces sh so Python gets SIGTERM)
CMD ["sh", "-c", "uv run --no-dev alembic upgrade head && exec uv run --no-dev aineko"]
