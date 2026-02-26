from unittest.mock import patch


def test_health_requires_no_auth(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_retrieve_missing_api_key_returns_401(client):
    response = client.post("/retrieve", json={"query": "diabetes"})
    assert response.status_code == 401
    assert "X-API-Key" in response.json()["detail"]


def test_retrieve_wrong_api_key_returns_401(client):
    response = client.post(
        "/retrieve",
        json={"query": "diabetes"},
        headers={"X-API-Key": "totally-wrong-key"},
    )
    assert response.status_code == 401


def test_ingest_missing_api_key_returns_401(client):
    response = client.post("/ingest", files={"file": ("test.txt", b"some text", "text/plain")})
    assert response.status_code == 401


def test_generate_missing_api_key_returns_401(client):
    response = client.post("/generate", json={"query": "test"})
    assert response.status_code == 401


def test_valid_api_key_passes(client, auth_headers):
    with patch("app.services.retrieval.get_store", return_value=None):
        response = client.post("/retrieve", json={"query": "test"}, headers=auth_headers)
    assert response.status_code == 200


def test_second_api_key_in_list_also_authenticates(client):
    import os
    from app.core.config import get_settings
    get_settings.cache_clear()
    original = os.environ.get("API_KEYS", "dev-key-healthrag")
    os.environ["API_KEYS"] = "primary-key,secondary-key"
    get_settings.cache_clear()
    try:
        with patch("app.services.retrieval.get_store", return_value=None):
            r1 = client.post(
                "/retrieve", json={"query": "test"}, headers={"X-API-Key": "primary-key"}
            )
            r2 = client.post(
                "/retrieve", json={"query": "test"}, headers={"X-API-Key": "secondary-key"}
            )
        assert r1.status_code == 200
        assert r2.status_code == 200
    finally:
        os.environ["API_KEYS"] = original
        get_settings.cache_clear()


def test_api_keys_parsed_with_whitespace_around_commas(client):
    import os
    from app.core.config import get_settings
    get_settings.cache_clear()
    original = os.environ.get("API_KEYS", "dev-key-healthrag")
    os.environ["API_KEYS"] = "  spaced-key  ,  other-key  "
    get_settings.cache_clear()
    try:
        with patch("app.services.retrieval.get_store", return_value=None):
            response = client.post(
                "/retrieve", json={"query": "test"}, headers={"X-API-Key": "spaced-key"}
            )
        assert response.status_code == 200
    finally:
        os.environ["API_KEYS"] = original
        get_settings.cache_clear()


def test_empty_string_api_key_rejected(client):
    response = client.post(
        "/retrieve", json={"query": "test"}, headers={"X-API-Key": ""}
    )
    assert response.status_code == 401


def test_error_detail_mentions_api_key(client):
    response = client.post("/retrieve", json={"query": "test"})
    detail = response.json()["detail"]
    assert "API" in detail or "key" in detail.lower()
