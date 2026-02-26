import os
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

os.environ.setdefault("TRANSLATION_BACKEND", "mock")
os.environ.setdefault("API_KEYS", "dev-key-healthrag")
os.environ.setdefault("FAISS_INDEX_PATH", "/tmp/healthrag_test_faiss")

from app.main import app  # noqa: E402
from app.models.responses import DocumentResult, RetrieveResponse

TEST_API_KEY = "dev-key-healthrag"


def make_document_result(
    content: str,
    doc_id: str = "t001",
    language: str = "en",
    similarity_score: float = 0.95,
    chunk_index: int = 0,
) -> DocumentResult:
    return DocumentResult(
        doc_id=doc_id,
        content=content,
        language=language,
        similarity_score=similarity_score,
        chunk_index=chunk_index,
    )


def make_retrieve_response(
    docs: list[DocumentResult],
    query: str = "test",
    query_language: str = "en",
) -> RetrieveResponse:
    return RetrieveResponse(
        query=query,
        query_language=query_language,
        results=docs,
        total_found=len(docs),
    )


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def auth_headers():
    return {"X-API-Key": TEST_API_KEY}


@pytest.fixture
def en_doc():
    return Document(
        page_content=(
            "Type 2 diabetes management includes lifestyle modifications, "
            "metformin as first-line pharmacotherapy, and HbA1c monitoring every 3 months. "
            "Blood pressure targets should be <130/80 mmHg."
        ),
        metadata={
            "doc_id": "test001",
            "filename": "diabetes_guideline_en.txt",
            "language": "en",
            "chunk_index": 0,
            "total_chunks": 1,
        },
    )


@pytest.fixture
def ja_doc():
    return Document(
        page_content=(
            "2型糖尿病の管理には、生活習慣の改善、メトホルミンの第一選択薬、"
            "および3ヶ月ごとのHbA1cモニタリングが含まれます。"
            "血圧目標値は130/80 mmHg未満とすべきです。"
        ),
        metadata={
            "doc_id": "test002",
            "filename": "diabetes_guideline_ja.txt",
            "language": "ja",
            "chunk_index": 0,
            "total_chunks": 1,
        },
    )


@pytest.fixture
def mock_faiss_store(en_doc, ja_doc):
    store = MagicMock()
    store.similarity_search_with_score.return_value = [
        (en_doc, 0.12),
        (ja_doc, 0.45),
    ]
    return store
