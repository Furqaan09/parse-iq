# ---------- Stage 1: build the React app ----------
FROM node:24-alpine AS frontend
WORKDIR /src/frontend

# Dependencies first for caching
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
# vite.config.ts writes the build to ../api/static = /src/api/static
RUN npm run build


# ---------- Stage 2: the Python app ----------
FROM python:3.13-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /bin/uv

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Use the image's Python, never download one
ENV UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Dependencies first for caching
COPY api/pyproject.toml api/uv.lock api/.python-version ./
RUN uv sync --locked --no-dev

# Bake the reranker model into the image so a cold start doesn't download it
ENV FASTEMBED_CACHE_PATH=/app/.fastembed_cache
RUN .venv/bin/python -c "from fastembed.rerank.cross_encoder import TextCrossEncoder; TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2')"

COPY api/ ./
COPY --from=frontend /src/api/static ./static

# Run the venv's executables (uvicorn, alembic) without "uv run"
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]