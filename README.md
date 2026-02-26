# healthrag — Healthcare Knowledge Assistant

A production-ready bilingual (EN/JA) RAG backend for clinicians to retrieve medical
guidelines and research summaries. Built with FastAPI, FAISS, and multilingual
sentence-transformers.

```
POST /ingest    ← upload .txt (auto-detects EN or JA)
POST /retrieve  ← semantic search, top-k with similarity scores
POST /generate  ← structured response, bilingual output toggle
GET  /health    ← liveness probe (no auth)
```

---

## Quick Start

```bash
# 1. Clone and enter the project
cd healthrag

# 2. Create virtual environment and install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env — set API_KEYS, and optionally TRANSLATION_BACKEND=mock for offline use

# 4. Start the server
uvicorn app.main:app --reload
# → http://localhost:8000/docs (Swagger UI)

# 5. Seed sample medical documents (EN + JA)
python scripts/seed_documents.py

# 6. Try it out
curl -X POST http://localhost:8000/retrieve \
  -H "X-API-Key: dev-key-healthrag" \
  -H "Content-Type: application/json" \
  -d '{"query": "Type 2 diabetes treatment recommendations", "top_k": 3}'
```

---

## API Reference

All endpoints except `/health` require the `X-API-Key` header.

### `POST /ingest`
Upload a `.txt` document (UTF-8 or Shift-JIS). Language is auto-detected.

```bash
curl -X POST http://localhost:8000/ingest \
  -H "X-API-Key: dev-key-healthrag" \
  -F "file=@my_guideline.txt"
```

**Response**:
```json
{
  "doc_id": "a1b2c3d4",
  "filename": "my_guideline.txt",
  "language": "en",
  "chunks_indexed": 4,
  "message": "Successfully ingested 4 chunk(s) from 'my_guideline.txt' (detected language: en)"
}
```

---

### `POST /retrieve`
Semantic search. Cross-lingual by default (EN query → JA results).

```json
{
  "query": "糖尿病の治療推奨事項",
  "top_k": 3,
  "language_filter": "ja"
}
```

**Response**:
```json
{
  "query": "糖尿病の治療推奨事項",
  "query_language": "ja",
  "results": [
    {
      "doc_id": "a1b2c3d4",
      "content": "メトホルミンは第一選択薬として推奨...",
      "language": "ja",
      "similarity_score": 0.9823,
      "chunk_index": 0
    }
  ],
  "total_found": 1
}
```

---

### `POST /generate`
Generate a structured answer. Use `output_language` to force translation.

```json
{
  "query": "What are the latest recommendations for Type 2 diabetes management?",
  "top_k": 3,
  "output_language": "ja"
}
```

**Response**:
```json
{
  "query": "What are the latest recommendations...",
  "query_language": "en",
  "output_language": "ja",
  "answer": "取得した医療ガイドラインに基づく回答：...",
  "sources": [...],
  "translated": true
}
```

---

## Authentication

All endpoints (except `/health`) require an `X-API-Key` header.

```bash
# Set in .env
API_KEYS=your-secret-key,another-key

# Pass in requests
-H "X-API-Key: your-secret-key"
```

Multiple keys are supported (comma-separated). Rotate keys without downtime by adding
the new key before removing the old one.

---

## Architecture

```
Client
  │
  ▼
FastAPI (app/main.py)
  │   X-API-Key auth (app/core/security.py)
  ├── POST /ingest
  │     └─► ingestion.py
  │           ├─► language.py      detect_language()  [langdetect]
  │           ├─► text splitter    Japanese-aware chunking
  │           └─► vector_store.py  FAISS.from_documents() / add_documents()
  │                 └─► embeddings.py  SentenceTransformer (384-dim multilingual)
  │
  ├── POST /retrieve
  │     └─► retrieval.py
  │           ├─► language.py      detect query language
  │           └─► vector_store.py  similarity_search_with_score()
  │                                L2 → cosine similarity conversion
  │
  └── POST /generate
        └─► generation.py
              ├─► retrieval.py     retrieve top-k docs
              ├─► EN/JA template   mock LLM (swappable for real LLM)
              └─► translation.py   translate() if output_language ≠ query_language
                    └─► deep-translator (Google) / googletrans / argostranslate (offline) / mock
```

**Embedding model**: `paraphrase-multilingual-MiniLM-L12-v2`
- 50+ languages in a shared 384-dimensional space
- English and Japanese queries retrieve semantically relevant documents in either language
- CPU inference ~20ms/document; no GPU required

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEYS` | `dev-key-healthrag` | Comma-separated valid API keys |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | HuggingFace model name |
| `FAISS_INDEX_PATH` | `data/faiss_index` | Persistent index location |
| `TRANSLATION_BACKEND` | `google` | `google` (internet), `googletrans` (unofficial), `argostranslate` (offline ~100 MB), or `mock` |
| `LLM_BACKEND` | `template` | `template` (no key), `openai` (needs `OPENAI_API_KEY`), `claude` (needs `ANTHROPIC_API_KEY`) |
| `LLM_MODEL` | `` | Model override — defaults to `gpt-4o-mini` (openai) or `claude-sonnet-4-6` (claude) |
| `OPENAI_API_KEY` | — | Required when `LLM_BACKEND=openai`; also `pip install langchain-openai` |
| `ANTHROPIC_API_KEY` | — | Required when `LLM_BACKEND=claude`; also `pip install langchain-anthropic` |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between adjacent chunks |
| `MAX_CONTEXT_CHARS` | `2000` | Max context fed to the LLM |
| `LOG_LEVEL` | `INFO` | Logging level |
| `ENVIRONMENT` | `development` | `development` → console logs; `production` → JSON |

---

## Running Tests

```bash
# All tests (uses mock translation — no network needed)
TRANSLATION_BACKEND=mock pytest tests/ -v --cov=app

# Single test file
pytest tests/test_retrieval.py -v
```

---

## Docker

### Option A — Pull pre-built image (fastest, no build required)

Works on any machine with Docker installed (supports `linux/amd64` and `linux/arm64`).

```bash
# 1. Get the project files (only need .env.example and docker-compose.yml)
git clone https://github.com/nayemabs/healthrag.git
cd healthrag

# 2. Configure
cp .env.example .env
# Open .env and set API_KEYS to a secret of your choice (default: dev-key-healthrag)

# 3. Pull and run
docker pull ghcr.io/nayemabs/healthrag:latest
docker run -d \
  --name healthrag \
  -p 8000:8000 \
  --env-file .env \
  -v healthrag_data:/app/data \
  ghcr.io/nayemabs/healthrag:latest

# 4. Verify
curl http://localhost:8000/health
# → {"status":"ok","app":"Healthcare Knowledge Assistant"}

# 5. Ingest a document
curl -X POST http://localhost:8000/ingest \
  -H "X-API-Key: dev-key-healthrag" \
  -F "file=@yourfile.txt"

# 6. Retrieve
curl -X POST http://localhost:8000/retrieve \
  -H "X-API-Key: dev-key-healthrag" \
  -H "Content-Type: application/json" \
  -d '{"query": "your question here", "top_k": 3}'

# 7. Generate an answer
curl -X POST http://localhost:8000/generate \
  -H "X-API-Key: dev-key-healthrag" \
  -H "Content-Type: application/json" \
  -d '{"query": "your question here"}'
```

The FAISS index is stored in the `healthrag_data` Docker named volume — it survives container
restarts and `docker pull` upgrades. To wipe it: `docker volume rm healthrag_data`

### Option B — Docker Compose (recommended for local dev)

```bash
git clone https://github.com/nayemabs/healthrag.git
cd healthrag
cp .env.example .env          # edit API_KEYS and other settings
docker compose up --build
```

The `faiss_data` named volume is created automatically. To wipe the index: `docker compose down -v`

### Option C — Build from source

```bash
git clone https://github.com/nayemabs/healthrag.git
cd healthrag
docker build -t healthrag .
docker run -d -p 8000:8000 --env-file .env -v healthrag_data:/app/data healthrag
```

---

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`):

| Event | Jobs |
|-------|------|
| Pull Request → main | lint + test |
| Push → main | lint + test → build Docker → push to `ghcr.io` |

Image tags: `latest` and short commit SHA. Layer caching via GitHub Actions cache.

---

## Design Notes

### Scalability

Each FastAPI instance is stateless and shares nothing except the FAISS index on a mounted
volume, so you can add replicas behind a load balancer without any coordination. FAISS comfortably
handles ~1M vectors per node at sub-millisecond latency; beyond that, swapping to Pinecone or
Weaviate only requires implementing the same `get_store` / `add_documents` interface in
`app/db/vector_store.py`. The sentence-transformer runs on CPU in about 20ms per document —
fast enough for the current synchronous `/ingest`. For high-throughput ingestion you'd push
documents onto a queue (SQS, Kafka) and return a job ID immediately, with workers embedding
in the background.

### Modularity

The codebase is intentionally flat: each service is a small module with a single public
function (`detect_language`, `translate`, `ingest_text`, `retrieve`, `generate`). That makes
it easy to swap parts out — the translation backend is picked from an env var, and the mock
LLM in `generation.py` is just a `template.format()` call that you can replace with any LCEL
chain in one line. A few things I'd add next: a cross-encoder reranker after FAISS retrieval
(precision on medical abbreviations is still rough), metadata filtering by document date or
specialty, and a RAGAS eval loop to track retrieval quality over time.
