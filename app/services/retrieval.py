"""Semantic retrieval over the FAISS index.

Similarity is converted from FAISS L2 distance to cosine similarity using
cos_sim = 1 - d^2/2 (valid for unit-normalized embeddings).
"""
from app.core.logging import get_logger
from app.db.vector_store import get_store
from app.models.requests import RetrieveRequest
from app.models.responses import DocumentResult, RetrieveResponse
from app.services.language import detect_language

logger = get_logger(__name__)


def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    query_language = detect_language(request.query)
    store = get_store()

    if store is None:
        logger.warning("retrieve_empty_store")
        return RetrieveResponse(
            query=request.query,
            query_language=query_language,
            results=[],
            total_found=0,
        )

    # Fetch extra candidates when language filtering so we still get top_k after filter
    fetch_k = request.top_k * 3 if request.language_filter else request.top_k

    try:
        raw = store.similarity_search_with_score(request.query, k=fetch_k)
    except Exception as e:
        logger.error("retrieval_failed", error=str(e))
        return RetrieveResponse(
            query=request.query,
            query_language=query_language,
            results=[],
            total_found=0,
        )

    results: list[DocumentResult] = []
    for doc, l2_dist in raw:
        meta = doc.metadata
        if request.language_filter and meta.get("language") != request.language_filter:
            continue

        # Convert L2 distance to cosine similarity (valid for unit-normalized embeddings)
        similarity = max(0.0, min(1.0, 1.0 - (l2_dist ** 2) / 2.0))

        results.append(
            DocumentResult(
                doc_id=meta.get("doc_id", "unknown"),
                content=doc.page_content,
                language=meta.get("language", "unknown"),
                similarity_score=round(similarity, 4),
                chunk_index=meta.get("chunk_index", 0),
            )
        )

        if len(results) >= request.top_k:
            break

    logger.info("retrieved", count=len(results), query_lang=query_language, filter=request.language_filter)

    return RetrieveResponse(
        query=request.query,
        query_language=query_language,
        results=results,
        total_found=len(results),
    )
