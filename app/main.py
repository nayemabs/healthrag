"""FastAPI app for the bilingual healthcare knowledge assistant."""
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.core.security import require_api_key
from app.db.vector_store import load_store
from app.models.requests import GenerateRequest, RetrieveRequest
from app.models.responses import GenerateResponse, IngestResponse, RetrieveResponse
from app.services import generation, ingestion, retrieval

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings = get_settings()
    logger.info("startup", app=settings.app_name, environment=settings.environment)
    load_store()  # pre-load FAISS index if it exists on disk
    yield
    logger.info("shutdown")


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description=(
        "Bilingual (EN/JA) RAG-powered healthcare knowledge assistant. "
        "Ingest clinical guidelines, retrieve semantically relevant passages, "
        "and generate structured answers in English or Japanese."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "app": settings.app_name}


@app.post("/ingest", response_model=IngestResponse, tags=["Documents"])
async def ingest_document(
    file: UploadFile = File(..., description="Plain-text document (.txt) in English or Japanese"),
    _: str = Depends(require_api_key),
):
    if not (file.filename or "").lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported.")

    raw = await file.read()
    if not raw.strip():
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    # Try UTF-8 first, then Shift-JIS for legacy Japanese files
    text: str
    for encoding in ("utf-8", "shift-jis", "euc-jp"):
        try:
            text = raw.decode(encoding)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    else:
        raise HTTPException(status_code=400, detail="File encoding not supported. Use UTF-8 or Shift-JIS.")

    try:
        return ingestion.ingest_text(text, filename=file.filename or "upload.txt")
    except Exception as e:
        logger.error("ingest_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@app.post("/retrieve", response_model=RetrieveResponse, tags=["Search"])
async def retrieve_documents(
    request: RetrieveRequest,
    _: str = Depends(require_api_key),
):
    """Semantic search over ingested documents. Supports cross-lingual queries (EN ↔ JA)."""
    return retrieval.retrieve(request)


@app.post("/generate", response_model=GenerateResponse, tags=["Generation"])
async def generate_response(
    request: GenerateRequest,
    _: str = Depends(require_api_key),
):
    """Generate a response from retrieved passages. Set output_language to force translation."""
    return generation.generate(request)
