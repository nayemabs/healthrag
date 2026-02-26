import io
from unittest.mock import MagicMock, patch

_PATCH_ADD = "app.services.ingestion.add_documents"
_PATCH_EMB = "app.services.ingestion.get_embeddings"


def test_ingest_english_document(client, auth_headers):
    text = (
        "Type 2 diabetes management guidelines recommend metformin as first-line therapy. "
        "Regular HbA1c monitoring every three months is advised."
    )
    with patch(_PATCH_EMB) as mock_emb, patch(_PATCH_ADD) as mock_add:
        mock_emb.return_value = MagicMock()
        mock_add.return_value = MagicMock()

        response = client.post(
            "/ingest",
            files={"file": ("diabetes_en.txt", io.BytesIO(text.encode()), "text/plain")},
            headers=auth_headers,
        )

    assert response.status_code == 200
    data = response.json()
    assert data["language"] == "en"
    assert data["chunks_indexed"] >= 1
    assert data["filename"] == "diabetes_en.txt"
    assert "doc_id" in data
    assert len(data["doc_id"]) == 8


def test_ingest_japanese_document(client, auth_headers):
    text = (
        "2型糖尿病の管理ガイドライン：メトホルミンが第一選択薬として推奨されます。"
        "3ヶ月ごとのHbA1c測定が必要です。血糖コントロールの維持が重要です。"
    )
    with patch(_PATCH_EMB) as mock_emb, patch(_PATCH_ADD) as mock_add:
        mock_emb.return_value = MagicMock()
        mock_add.return_value = MagicMock()

        response = client.post(
            "/ingest",
            files={"file": ("diabetes_ja.txt", io.BytesIO(text.encode()), "text/plain")},
            headers=auth_headers,
        )

    assert response.status_code == 200
    data = response.json()
    assert data["language"] == "ja"
    assert data["chunks_indexed"] >= 1


def test_ingest_rejects_non_txt(client, auth_headers):
    response = client.post(
        "/ingest",
        files={"file": ("report.pdf", b"%PDF-1.4 content", "application/pdf")},
        headers=auth_headers,
    )
    assert response.status_code == 400
    assert "txt" in response.json()["detail"].lower()


def test_ingest_rejects_empty_file(client, auth_headers):
    response = client.post(
        "/ingest",
        files={"file": ("empty.txt", b"   \n  ", "text/plain")},
        headers=auth_headers,
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_ingest_requires_api_key(client):
    response = client.post(
        "/ingest",
        files={"file": ("test.txt", b"hello world", "text/plain")},
    )
    assert response.status_code == 401


def test_ingest_response_contains_message(client, auth_headers):
    text = "Hypertension management guidelines state target BP < 130/80."
    with patch(_PATCH_EMB) as mock_emb, patch(_PATCH_ADD) as mock_add:
        mock_emb.return_value = MagicMock()
        mock_add.return_value = MagicMock()

        response = client.post(
            "/ingest",
            files={"file": ("hypertension.txt", io.BytesIO(text.encode()), "text/plain")},
            headers=auth_headers,
        )

    assert response.status_code == 200
    assert "Successfully ingested" in response.json()["message"]


def test_ingest_doc_id_is_eight_hex_characters(client, auth_headers):
    text = "Blood pressure management guidelines recommend target below 130/80 mmHg."
    with patch(_PATCH_EMB) as mock_emb, patch(_PATCH_ADD) as mock_add:
        mock_emb.return_value = MagicMock()
        mock_add.return_value = MagicMock()

        response = client.post(
            "/ingest",
            files={"file": ("bp.txt", io.BytesIO(text.encode()), "text/plain")},
            headers=auth_headers,
        )

    doc_id = response.json()["doc_id"]
    assert len(doc_id) == 8
    assert all(c in "0123456789abcdef" for c in doc_id)


def test_ingest_doc_ids_are_unique_across_calls(client, auth_headers):
    text = "Some medical guideline text for uniqueness check."
    ids = set()
    with patch(_PATCH_EMB) as mock_emb, patch(_PATCH_ADD) as mock_add:
        mock_emb.return_value = MagicMock()
        mock_add.return_value = MagicMock()

        for i in range(5):
            response = client.post(
                "/ingest",
                files={"file": (f"doc{i}.txt", io.BytesIO(text.encode()), "text/plain")},
                headers=auth_headers,
            )
            assert response.status_code == 200
            ids.add(response.json()["doc_id"])
    assert len(ids) == 5


def test_ingest_message_contains_filename_and_language(client, auth_headers):
    text = "Metformin is first-line therapy for type 2 diabetes management guidelines."
    filename = "diabetes_guidelines.txt"
    with patch(_PATCH_EMB) as mock_emb, patch(_PATCH_ADD) as mock_add:
        mock_emb.return_value = MagicMock()
        mock_add.return_value = MagicMock()

        response = client.post(
            "/ingest",
            files={"file": (filename, io.BytesIO(text.encode()), "text/plain")},
            headers=auth_headers,
        )

    data = response.json()
    assert filename in data["message"]
    assert data["language"] in data["message"]


def test_ingest_long_text_produces_multiple_chunks(client, auth_headers):
    text = "A " * 300  # 600 chars > chunk_size=512, should split into 2+
    with patch(_PATCH_EMB) as mock_emb, patch(_PATCH_ADD) as mock_add:
        mock_emb.return_value = MagicMock()
        mock_add.return_value = MagicMock()

        response = client.post(
            "/ingest",
            files={"file": ("long.txt", io.BytesIO(text.encode()), "text/plain")},
            headers=auth_headers,
        )

    assert response.status_code == 200
    assert response.json()["chunks_indexed"] >= 2
