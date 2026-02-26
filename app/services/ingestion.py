import uuid

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.embeddings import get_embeddings
from app.db.vector_store import add_documents
from app.models.responses import IngestResponse
from app.services.language import detect_language

logger = get_logger(__name__)


def ingest_text(text: str, filename: str = "unknown.txt") -> IngestResponse:
    settings = get_settings()

    language = detect_language(text)
    doc_id = str(uuid.uuid4())[:8]

    logger.info("ingesting", doc_id=doc_id, language=language, filename=filename, chars=len(text))

    # include JA sentence terminators so chunks don't cut mid-sentence
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", "。", "、", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_text(text)
    if not chunks:
        chunks = [text]  # single-chunk fallback for very short docs

    documents = [
        Document(
            page_content=chunk,
            metadata={
                "doc_id": doc_id,
                "filename": filename,
                "language": language,
                "chunk_index": i,
                "total_chunks": len(chunks),
            },
        )
        for i, chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()
    add_documents(documents, embeddings)

    logger.info("ingestion_complete", doc_id=doc_id, chunks=len(chunks), language=language)

    return IngestResponse(
        doc_id=doc_id,
        filename=filename,
        language=language,
        chunks_indexed=len(chunks),
        message=(
            f"Successfully ingested {len(chunks)} chunk(s) from '{filename}' "
            f"(detected language: {language})"
        ),
    )
