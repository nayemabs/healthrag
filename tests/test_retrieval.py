from unittest.mock import MagicMock, patch


def test_retrieve_returns_top_k_results(client, auth_headers, mock_faiss_store):
    with patch("app.services.retrieval.get_store", return_value=mock_faiss_store):
        response = client.post(
            "/retrieve",
            json={"query": "What is the recommended treatment for type 2 diabetes?", "top_k": 2},
            headers=auth_headers,
        )

    assert response.status_code == 200
    data = response.json()
    assert data["query_language"] == "en"
    assert len(data["results"]) == 2
    assert data["total_found"] == 2


def test_retrieve_similarity_scores_in_range(client, auth_headers, mock_faiss_store):
    with patch("app.services.retrieval.get_store", return_value=mock_faiss_store):
        response = client.post(
            "/retrieve",
            json={"query": "diabetes treatment guidelines"},
            headers=auth_headers,
        )

    data = response.json()
    for result in data["results"]:
        assert 0.0 <= result["similarity_score"] <= 1.0


def test_retrieve_detects_japanese_query(client, auth_headers, mock_faiss_store):
    with patch("app.services.retrieval.get_store", return_value=mock_faiss_store):
        response = client.post(
            "/retrieve",
            json={"query": "2型糖尿病の治療推奨事項は何ですか？"},
            headers=auth_headers,
        )

    assert response.status_code == 200
    assert response.json()["query_language"] == "ja"


def test_retrieve_language_filter_en(client, auth_headers, mock_faiss_store):
    with patch("app.services.retrieval.get_store", return_value=mock_faiss_store):
        response = client.post(
            "/retrieve",
            json={"query": "diabetes management", "top_k": 3, "language_filter": "en"},
            headers=auth_headers,
        )

    assert response.status_code == 200
    for result in response.json()["results"]:
        assert result["language"] == "en"


def test_retrieve_language_filter_ja(client, auth_headers, mock_faiss_store):
    with patch("app.services.retrieval.get_store", return_value=mock_faiss_store):
        response = client.post(
            "/retrieve",
            json={"query": "diabetes", "top_k": 3, "language_filter": "ja"},
            headers=auth_headers,
        )

    assert response.status_code == 200
    for result in response.json()["results"]:
        assert result["language"] == "ja"


def test_retrieve_empty_store_returns_empty_list(client, auth_headers):
    with patch("app.services.retrieval.get_store", return_value=None):
        response = client.post(
            "/retrieve",
            json={"query": "test query"},
            headers=auth_headers,
        )

    assert response.status_code == 200
    data = response.json()
    assert data["total_found"] == 0
    assert data["results"] == []


def test_retrieve_result_contains_required_fields(client, auth_headers, mock_faiss_store):
    with patch("app.services.retrieval.get_store", return_value=mock_faiss_store):
        response = client.post(
            "/retrieve",
            json={"query": "diabetes"},
            headers=auth_headers,
        )

    result = response.json()["results"][0]
    assert "doc_id" in result
    assert "content" in result
    assert "language" in result
    assert "similarity_score" in result
    assert "chunk_index" in result


def test_retrieve_requires_api_key(client):
    response = client.post("/retrieve", json={"query": "test"})
    assert response.status_code == 401


def test_retrieve_faiss_exception_returns_empty_200(client, auth_headers):
    store = MagicMock()
    store.similarity_search_with_score.side_effect = RuntimeError("index error")
    with patch("app.services.retrieval.get_store", return_value=store):
        response = client.post(
            "/retrieve",
            json={"query": "diabetes"},
            headers=auth_headers,
        )
    assert response.status_code == 200
    data = response.json()
    assert data["results"] == []
    assert data["total_found"] == 0


def test_retrieve_default_top_k_is_three(client, auth_headers, mock_faiss_store):
    with patch("app.services.retrieval.get_store", return_value=mock_faiss_store):
        response = client.post(
            "/retrieve",
            json={"query": "diabetes"},
            headers=auth_headers,
        )
    assert response.status_code == 200
    assert len(response.json()["results"]) <= 3


def test_retrieve_no_language_filter_returns_all_languages(client, auth_headers, mock_faiss_store):
    with patch("app.services.retrieval.get_store", return_value=mock_faiss_store):
        response = client.post(
            "/retrieve",
            json={"query": "diabetes"},
            headers=auth_headers,
        )
    languages = {r["language"] for r in response.json()["results"]}
    assert "en" in languages
    assert "ja" in languages


def test_retrieve_query_echoed_in_response(client, auth_headers, mock_faiss_store):
    query_text = "What are the HbA1c targets for type 2 diabetes?"
    with patch("app.services.retrieval.get_store", return_value=mock_faiss_store):
        response = client.post(
            "/retrieve",
            json={"query": query_text},
            headers=auth_headers,
        )
    assert response.json()["query"] == query_text
