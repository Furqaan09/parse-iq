# ParseIQ

ParseIQ is an AI-powered document assistant. Upload documents — class syllabi,
emails, screenshots of text or conversations — and ask questions in plain
English instead of reading through pages manually. Answers are grounded in
the source material and returned with citations back to the exact page/snippet
they came from.

## How it works

1. **Upload** a PDF, image, or text file. It's saved to disk and its metadata
   is recorded in Postgres.
2. **Chunk** — the document is split into overlapping, paragraph-aware text
   chunks (images are also OCR'd for their text content).
3. **Embed** — each chunk is embedded locally (`all-MiniLM-L6-v2` for text,
   CLIP for images) and stored in a FAISS vector index.
4. **Ask** — a question is embedded, the most relevant chunks are retrieved
   from FAISS, and an LLM (served via Hugging Face Inference Providers)
   answers strictly from that retrieved context, citing `[n]` sources back to
   the original document/page.

## Stack

- **Backend:** FastAPI (Python), PostgreSQL via SQLModel + Alembic, FAISS for
  vector search, Hugging Face Inference Providers for the chat LLM.
- **Frontend:** React + TypeScript + Tailwind (Vite).

## Project structure

```
api/
  app/
    core/       # DB engine/session, device selection
    models.py   # SQLModel tables (Document, Chunk, KVExtraction, Task)
    routes/      # FastAPI routers: documents, search, chat
    services/    # chunking, embeddings, FAISS index, RAG pipeline, LLM client
  migration/     # Alembic env + versioned migrations
  storage/       # uploaded files + FAISS indexes (gitignored)
frontend/
  src/
    components/  # UploadBox, DocumentList, ChatPanel
    lib/api.ts   # typed fetch wrappers for the backend API
```

## Getting started

### Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL running locally (or reachable via `DB_HOST`)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed
  system-wide (required by `pytesseract` for image text extraction) — e.g.
  `brew install tesseract` on macOS
- A Hugging Face account + API token (for the chat LLM)

### Backend

```bash
cd api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # add requirements-dev.txt for ruff/black

cp .env.example .env                   # fill in DB credentials + HF_TOKEN

alembic upgrade head                   # create/upgrade the schema
uvicorn app.main:app --reload          # http://localhost:8000
```

### Frontend

```bash
cd frontend
npm install
cp .env.example .env.local             # defaults to http://localhost:8000
npm run dev                            # http://localhost:5173
```

The dev server proxies `/api/*` to the backend (see `vite.config.ts`), so the
frontend and backend can be run independently on their default ports.

### Linting (backend)

```bash
cd api
pip install -r requirements-dev.txt
ruff check app
black app
```
