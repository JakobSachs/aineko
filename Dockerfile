FROM ubuntu:24.04

WORKDIR /app

# Install system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv \
        libolm-dev gcc g++ clang curl ca-certificates git ripgrep \
        libnuma1 libdrm2 libgl1 ffmpeg colmap \
        poppler-utils imagemagick file jq tzdata \
        nvidia-cuda-toolkit && \
    rm -rf /var/lib/apt/lists/*

# Install runtime setup deps that used to be installed by /data/setup.sh.
# Keeping them in their own layer makes boot deterministic and avoids apt work
# every time the container starts.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc-11 g++-11 ninja-build && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}" \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1

# Copy dependency files first (cache layer)
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --no-dev --no-install-project

# Copy source, readme, alembic
COPY README.md ./
COPY src/ src/
COPY alembic/ alembic/
COPY alembic.ini ./

# Install the project itself, then remove bytecode caches so corrupted .pyc
# files cannot get baked into the image.
RUN uv sync --no-dev && \
    find /app -name '*.pyc' -delete && \
    find /app -type d -name __pycache__ -exec rm -rf {} +

EXPOSE 8080

HEALTHCHECK --interval=30s --retries=3 \
    CMD curl -f http://localhost:8080/healthz || exit 1

# Run migrations then exec the app (exec replaces sh so Python gets SIGTERM)
CMD ["sh", "-c", "uv run --no-dev alembic upgrade head && exec uv run --no-dev aineko"]
