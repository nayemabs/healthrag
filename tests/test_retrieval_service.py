"""Unit tests for the retrieval service (not the HTTP endpoint)."""
import math
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.models.requests import RetrieveRequest
from app.services.retrieval import retrieve

def _make_doc(content: str, doc_id: str = "d001", language: str = "en", chunk_index: int = 0) -> Document:
    return Document(
        page_content=content,
        metadata={
            "doc_id": doc_id,
            "filename": f"{doc_id}.txt",
            "language": language,
            "chunk_index": chunk_index,
            "total_chunks": 1,
        },
    )

def _make_store(*pairs: tuple[Document, float]) -> MagicMock:
    """Build a mock FAISS store that returns the given (doc, l2_dist) pairs."""
    store = MagicMock()
    store.similarity_search_with_score.return_value = list(pairs)
    return store

class TestSimilarityFormula:
    """
    formula: similarity = max(0.0, min(1.0, 1.0 - (l2_dist ** 2) / 2.0))
    Unit-normalised embeddings: d=0 → perfect match (1.0); d=√2 → orthogonal (0.0).
    """

    def test_perfect_match_l2_zero(self):
        doc = _make_doc("Perfect match.")
        store = _make_store((doc, 0.0))
        with patch("app.services.retrieval.get_store", return_value=store):
            resp = retrieve(RetrieveRequest(query="Perfect match", top_k=1))
        assert resp.results[0].similarity_score == 1.0

    def test_orthogonal_vectors_l2_sqrt2(self):
        doc = _make_doc("Orthogonal content.")
        store = _make_store((doc, math.sqrt(2)))
        with patch("app.services.retrieval.get_store", return_value=store):
            resp = retrieve(RetrieveRequest(query="query", top_k=1))
        assert resp.results[0].similarity_score == 0.0

    def test_mid_similarity_l2_one(self):
        # d=1 → 1 - 0.5 = 0.5
        doc = _make_doc("Medium similarity.")
        store = _make_store((doc, 1.0))
        with patch("app.services.retrieval.get_store", return_value=store):
            resp = retrieve(RetrieveRequest(query="query", top_k=1))
        assert resp.results[0].similarity_score == pytest.approx(0.5, abs=0.001)

    def test_high_similarity_l2_half(self):
        # d=0.5 → 1 - 0.125 = 0.875
        doc = _make_doc("High similarity.")
        store = _make_store((doc, 0.5))
        with patch("app.services.retrieval.get_store", return_value=store):
            resp = retrieve(RetrieveRequest(query="query", top_k=1))
        assert resp.results[0].similarity_score == pytest.approx(0.875, abs=0.001)

    def test_clamped_to_zero_for_large_l2(self):
        # d=2.0 → 1 - 2.0 = -1.0 → clamped to 0.0
        doc = _make_doc("Very dissimilar.")
        store = _make_store((doc, 2.0))
        with patch("app.services.retrieval.get_store", return_value=store):
            resp = retrieve(RetrieveRequest(query="query", top_k=1))
        assert resp.results[0].similarity_score == 0.0

    def test_score_always_in_0_1_range(self):
        docs_and_dists = [
            (_make_doc(f"Doc {i}", doc_id=f"d{i:03d}"), dist)
            for i, dist in enumerate([0.0, 0.3, 0.7, 1.0, 1.2, 1.5, math.sqrt(2)])
        ]
        store = _make_store(*docs_and_dists)
        with patch("app.services.retrieval.get_store", return_value=store):
            resp = retrieve(RetrieveRequest(query="query", top_k=10))
        for result in resp.results:
            assert 0.0 <= result.similarity_score <= 1.0

    def test_score_rounded_to_4_decimal_places(self):
        doc = _make_doc("Precision check.")
        store = _make_store((doc, 0.333))
        with patch("app.services.retrieval.get_store", return_value=store):
            resp = retrieve(RetrieveRequest(query="query", top_k=1))
        score = resp.results[0].similarity_score
        # Rounded to 4 decimal places — str representation should have ≤4 decimal digits
        assert score == round(score, 4)

class TestTopK:
    def test_returns_at_most_top_k_results(self):
        docs = [(_make_doc(f"Doc {i}", doc_id=f"d{i:03d}"), 0.1 * i) for i in range(10)]
        store = _make_store(*docs)
        with patch("app.services.retrieval.get_store", return_value=store):
            resp = retrieve(RetrieveRequest(query="query", top_k=3))
        assert len(resp.results) <= 3

    def test_top_k_1_returns_single_result(self):
        docs = [(_make_doc(f"Doc {i}", doc_id=f"d{i:03d}"), 0.2 * i) for i in range(5)]
        store = _make_store(*docs)
        with patch("app.services.retrieval.get_store", return_value=store):
            resp = retrieve(RetrieveRequest(query="query", top_k=1))
        assert len(resp.results) == 1

    def test_fewer_docs_than_top_k_returns_all(self):
        docs = [(_make_doc("Only doc", doc_id="solo"), 0.2)]
        store = _make_store(*docs)
        with patch("app.services.retrieval.get_store", return_value=store):
            resp = retrieve(RetrieveRequest(query="query", top_k=5))
        assert len(resp.results) == 1

    def test_fetch_k_multiplied_by_3_when_language_filter(self):
        """With a language_filter, retrieval fetches top_k*3 candidates so filtering
        doesn't starve the final result set."""
        store = MagicMock()
        store.similarity_search_with_score.return_value = []
        with patch("app.services.retrieval.get_store", return_value=store):
            retrieve(RetrieveRequest(query="query", top_k=4, language_filter="en"))
        called_k = store.similarity_search_with_score.call_args[1]["k"]
        assert called_k == 12  # 4 * 3

    def test_fetch_k_equals_top_k_without_language_filter(self):
        store = MagicMock()
        store.similarity_search_with_score.return_value = []
        with patch("app.services.retrieval.get_store", return_value=store):
            retrieve(RetrieveRequest(query="query", top_k=5))
        called_k = store.similarity_search_with_score.call_args[1]["k"]
        assert called_k == 5

class TestLanguageFilter:
    def test_en_filter_excludes_ja_docs(self):
        en_doc = _make_doc("English content.", language="en")
        ja_doc = _make_doc("日本語コンテンツ", doc_id="d002", language="ja")
        store = _make_store((en_doc, 0.2), (ja_doc, 0.3))
        with patch("app.services.retrieval.get_store", return_value=store):
            resp = retrieve(RetrieveRequest(query="query", top_k=5, language_filter="en"))
        assert all(r.language == "en" for r in resp.results)
        assert len(resp.results) == 1

    def test_ja_filter_excludes_en_docs(self):
        en_doc = _make_doc("English content.", language="en")
        ja_doc = _make_doc("日本語コンテンツ", doc_id="d002", language="ja")
        store = _make_store((en_doc, 0.2), (ja_doc, 0.3))
        with patch("app.services.retrieval.get_store", return_value=store):
            resp = retrieve(RetrieveRequest(query="query", top_k=5, language_filter="ja"))
        assert all(r.language == "ja" for r in resp.results)
        assert len(resp.results) == 1

    def test_no_filter_returns_mixed_language_results(self):
        en_doc = _make_doc("English content.", language="en")
        ja_doc = _make_doc("日本語コンテンツ", doc_id="d002", language="ja")
        store = _make_store((en_doc, 0.2), (ja_doc, 0.3))
        with patch("app.services.retrieval.get_store", return_value=store):
            resp = retrieve(RetrieveRequest(query="query", top_k=5))
        languages = {r.language for r in resp.results}
        assert "en" in languages
        assert "ja" in languages

    def test_filter_with_no_matching_docs_returns_empty(self):
        en_doc = _make_doc("English only.", language="en")
        store = _make_store((en_doc, 0.2))
        with patch("app.services.retrieval.get_store", return_value=store):
            resp = retrieve(RetrieveRequest(query="query", top_k=3, language_filter="ja"))
        assert resp.results == []
        assert resp.total_found == 0

class TestFAISSErrorHandling:
    def test_none_store_returns_empty(self):
        with patch("app.services.retrieval.get_store", return_value=None):
            resp = retrieve(RetrieveRequest(query="test query", top_k=3))
        assert resp.results == []
        assert resp.total_found == 0
        assert resp.query == "test query"

    def test_faiss_exception_returns_empty(self):
        store = MagicMock()
        store.similarity_search_with_score.side_effect = RuntimeError("FAISS index corrupted")
        with patch("app.services.retrieval.get_store", return_value=store):
            resp = retrieve(RetrieveRequest(query="test query", top_k=3))
        assert resp.results == []
        assert resp.total_found == 0

    def test_faiss_exception_does_not_raise(self):
        store = MagicMock()
        store.similarity_search_with_score.side_effect = Exception("unexpected error")
        with patch("app.services.retrieval.get_store", return_value=store):
            # Should not propagate the exception
            resp = retrieve(RetrieveRequest(query="query", top_k=3))
        assert resp is not None

class TestResultMetadata:
    def test_result_fields_populated_from_document_metadata(self):
        doc = _make_doc("Clinical guideline content.", doc_id="abc12345", language="en", chunk_index=2)
        store = _make_store((doc, 0.4))
        with patch("app.services.retrieval.get_store", return_value=store):
            resp = retrieve(RetrieveRequest(query="query", top_k=1))
        result = resp.results[0]
        assert result.doc_id == "abc12345"
        assert result.language == "en"
        assert result.chunk_index == 2
        assert result.content == "Clinical guideline content."

    def test_missing_metadata_uses_defaults(self):
        # Document with no metadata at all
        doc = Document(page_content="No metadata doc.", metadata={})
        store = _make_store((doc, 0.5))
        with patch("app.services.retrieval.get_store", return_value=store):
            resp = retrieve(RetrieveRequest(query="query", top_k=1))
        result = resp.results[0]
        assert result.doc_id == "unknown"
        assert result.language == "unknown"
        assert result.chunk_index == 0

    def test_query_language_detected_correctly(self):
        store = MagicMock()
        store.similarity_search_with_score.return_value = []
        with patch("app.services.retrieval.get_store", return_value=store):
            resp = retrieve(RetrieveRequest(
                query="2型糖尿病の治療推奨事項は何ですか？", top_k=3
            ))
        assert resp.query_language == "ja"

    def test_total_found_matches_results_count(self):
        docs = [
            (_make_doc(f"Doc {i}", doc_id=f"d{i:03d}"), 0.1 + i * 0.1)
            for i in range(3)
        ]
        store = _make_store(*docs)
        with patch("app.services.retrieval.get_store", return_value=store):
            resp = retrieve(RetrieveRequest(query="query", top_k=3))
        assert resp.total_found == len(resp.results)
