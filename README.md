# ParseIQ

ParseIQ is an AI-powered document assistant. Upload documents such as PDFs,
emails or screenshots of text, then ask questions in plain English. Answers
come only from your documents, with citations back to the source page.

## How it works

1. **Upload** a PDF, image or text file. It's saved to disk and its metadata
   is recorded in Postgres.
2. **Chunk:** the document is split into section-aware text chunks. Images are
   OCR'd with Tesseract first.
3. **Embed:** each chunk is embedded locally with `all-MiniLM-L6-v2` and stored
   in a FAISS vector index.
4. **Ask:** the question is embedded, the closest chunks are retrieved from
   FAISS and reranked with a cross-encoder. An LLM served through Hugging Face
   Inference Providers then answers from those chunks, citing `[n]` sources.

## Stack

- **Backend:** FastAPI, PostgreSQL with SQLModel and Alembic, FAISS,
  sentence-transformers, Hugging Face Inference Providers. Dependencies are
  managed with [uv](https://docs.astral.sh/uv/).
- **Frontend:** React, TypeScript and Tailwind (Vite). FastAPI serves the built
  app, so production is one container on one port.

## Project structure

```
Dockerfile           # one image: builds React, then the Python app
docker-compose.yml   # db + app
api/
  app/
    main.py          # FastAPI app: /api routes, /files, the React build at /
    settings.py      # all configuration (env vars and api/.env)
    db.py            # database engine and session
    models.py        # SQLModel tables: Document, Chunk
    cli.py           # admin commands (reindex)
    routes/          # HTTP only: documents, ask
    ingestion/       # file -> text -> chunks -> vectors
    services/        # embeddings, FAISS index, reranker, RAG pipeline, LLM client
  migration/         # Alembic migrations
  rag_eval/          # retrieval and answer evaluation harness
  tests/             # unit, integration and eval tests
  storage/           # uploaded files and the FAISS index (gitignored)
frontend/
  src/
    components/      # UploadBox, DocumentList, ChatPanel
    lib/api.ts       # typed fetch wrappers for the backend API
```

## Getting started

You need Docker for Postgres, and a Hugging Face token for the chat LLM.

Create the two env files first:

```bash
cp .env.example .env            # POSTGRES_* values used by Docker Compose
cp api/.env.example api/.env    # DATABASE_URL and HF_TOKEN
```

The database password in `api/.env` must match `POSTGRES_PASSWORD` in `.env`.

### Run everything in Docker

```bash
docker compose up --build       # http://localhost:8000
```

The container applies database migrations when it starts. API docs are at
`http://localhost:8000/docs`.

### Develop locally

This also needs [uv](https://docs.astral.sh/uv/), Node.js 22 or newer, and
[Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
(`brew install tesseract` on macOS).

```bash
docker compose up -d db

cd api
uv sync                               # creates .venv from uv.lock
uv run alembic upgrade head
uv run uvicorn app.main:app --reload  # http://localhost:8000
```

In a second terminal:

```bash
cd frontend
npm install
npm run dev                           # http://localhost:5173
```

The Vite dev server forwards `/api` and `/files` to the backend.

### Tests and linting

Tests need a Postgres database whose name ends in `_test`. By default they use
`parseiq:parseiq_pw@localhost:5433/parseiq_test`. Set `POSTGRES_PASSWORD` to
`parseiq_pw`, or export your own `DATABASE_URL`.

```bash
docker compose exec db createdb -U parseiq parseiq_test   # once

cd api
uv run pytest                                     # calls to the paid LLM are skipped
uv run pytest -m "not integration and not eval"   # fast tests only
uv run ruff check . && uv run black --check .
```

### Rebuild the search index

```bash
cd api
uv run python -m app.cli reindex
```

This re-chunks every document and rebuilds the FAISS index from the database.
