# syntax=docker/dockerfile:1.9
# Multi-stage build: small Python 3.13 runtime image.
# For GPU deployments swap the base of both stages to an `nvidia/cuda:12.x-cudnn-runtime`
# variant and add `onnxruntime-gpu` via the `gpu` extras group.

FROM python:3.14-slim AS builder

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONDONTWRITEBYTECODE=1

# Build deps + OpenCV runtime libs (needed at install time for some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Copy lockfile first so layer cache survives source changes
COPY pyproject.toml README.md ./
COPY src/ ./src/

# `--no-editable` produces a wheel install (smaller, faster import). `--no-dev`
# trims the development tools out of the runtime image.
RUN uv sync --no-dev --no-editable


FROM python:3.14-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 anpr \
    && mkdir -p /data && chown anpr:anpr /data

WORKDIR /app

# Copy venv + src from the builder
COPY --from=builder --chown=anpr:anpr /app /app

USER anpr

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1 \
    ANPR_DATABASE_URL="sqlite+aiosqlite:////data/anpr.db" \
    ANPR_API_HOST=0.0.0.0 \
    ANPR_API_PORT=8000 \
    ANPR_LOG_JSON=true

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "anpr.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
