"""Generation endpoint tests."""
from unittest.mock import MagicMock, patch

from app.services.generation import (
    AnthropicBackend,
    OpenAIBackend,
    TemplateBackend,
    _get_llm_backend,
)
from tests.conftest import make_document_result, make_retrieve_response


def test_generate_english_response(client, auth_headers):
    doc = make_document_result(
        "Type 2 diabetes management includes metformin as first-line therapy. "
        "Regular HbA1c monitoring every 3 months is recommended. "
        "Blood pressure target is below 130/80 mmHg."
    )
    mock_response = make_retrieve_response([doc], query="diabetes treatment", query_language="en")

    with patch("app.services.generation.retrieve_docs", return_value=mock_response):
        response = client.post(
            "/generate",
            json={"query": "What is the recommended treatment for type 2 diabetes?"},
            headers=auth_headers,
        )

    assert response.status_code == 200
    data = response.json()
    assert data["query_language"] == "en"
    assert data["output_language"] == "en"
    assert data["translated"] is False
    assert len(data["answer"]) > 100
    assert len(data["sources"]) == 1


def test_generate_japanese_query_returns_japanese_response(client, auth_headers):
    doc = make_document_result(
        "2型糖尿病の治療にはメトホルミンが推奨されます。定期的なHbA1c測定が必要です。",
        language="ja",
    )
    mock_response = make_retrieve_response(
        [doc], query="2型糖尿病の治療", query_language="ja"
    )

    with patch("app.services.generation.retrieve_docs", return_value=mock_response):
        response = client.post(
            "/generate",
            json={"query": "2型糖尿病の治療推奨事項は何ですか？"},
            headers=auth_headers,
        )

    assert response.status_code == 200
    data = response.json()
    assert data["query_language"] == "ja"
    assert data["output_language"] == "ja"
    assert data["translated"] is False


def test_generate_translation_toggle_en_to_ja(client, auth_headers):
    doc = make_document_result("Metformin is the first-line treatment for type 2 diabetes.")
    mock_response = make_retrieve_response([doc], query="diabetes", query_language="en")

    with patch("app.services.generation.retrieve_docs", return_value=mock_response), \
         patch("app.services.generation.translate") as mock_translate:
        mock_translate.side_effect = lambda text, **kwargs: f"[翻訳]{text[:30]}"

        response = client.post(
            "/generate",
            json={"query": "What is the treatment?", "output_language": "ja"},
            headers=auth_headers,
        )

    assert response.status_code == 200
    data = response.json()
    assert data["output_language"] == "ja"
    assert data["translated"] is True
    assert mock_translate.called


def test_generate_translation_toggle_ja_to_en(client, auth_headers):
    doc = make_document_result("メトホルミンは2型糖尿病の第一選択薬です。", language="ja")
    mock_response = make_retrieve_response([doc], query="糖尿病", query_language="ja")

    with patch("app.services.generation.retrieve_docs", return_value=mock_response), \
         patch("app.services.generation.translate") as mock_translate:
        mock_translate.side_effect = lambda text, **kwargs: f"[Translation]{text[:30]}"

        response = client.post(
            "/generate",
            json={"query": "糖尿病の治療は？", "output_language": "en"},
            headers=auth_headers,
        )

    assert response.status_code == 200
    assert response.json()["translated"] is True


def test_generate_no_documents_returns_helpful_message(client, auth_headers):
    empty_response = make_retrieve_response([], query="test", query_language="en")

    with patch("app.services.generation.retrieve_docs", return_value=empty_response):
        response = client.post(
            "/generate",
            json={"query": "What are the guidelines for hypertension?"},
            headers=auth_headers,
        )

    assert response.status_code == 200
    data = response.json()
    assert "No relevant" in data["answer"] or "見つかりません" in data["answer"]
    assert data["sources"] == []


def test_generate_no_docs_japanese_output(client, auth_headers):
    empty_response = make_retrieve_response([], query="test", query_language="en")

    with patch("app.services.generation.retrieve_docs", return_value=empty_response):
        response = client.post(
            "/generate",
            json={"query": "test query", "output_language": "ja"},
            headers=auth_headers,
        )

    assert response.status_code == 200
    assert "見つかりません" in response.json()["answer"]


def test_generate_requires_api_key(client):
    response = client.post("/generate", json={"query": "test"})
    assert response.status_code == 401


def test_generate_response_includes_sources(client, auth_headers):
    docs = [
        make_document_result("Metformin reduces HbA1c by 1-2%.", doc_id="d1"),
        make_document_result("Lifestyle changes are essential in diabetes management.", doc_id="d2"),
    ]
    mock_response = make_retrieve_response(docs, query="diabetes", query_language="en")

    with patch("app.services.generation.retrieve_docs", return_value=mock_response):
        response = client.post(
            "/generate",
            json={"query": "diabetes treatment guidelines"},
            headers=auth_headers,
        )

    assert response.status_code == 200
    assert len(response.json()["sources"]) == 2


class TestTemplateBackend:
    def setup_method(self):
        self.backend = TemplateBackend()

    def test_english_output_uses_en_template(self):
        result = self.backend.generate(
            query="What is metformin?",
            context="Metformin is a first-line treatment.",
            output_language="en",
            recommendations="• Take metformin daily",
        )
        assert "**Query**" in result
        assert "What is metformin?" in result
        assert "Metformin is a first-line treatment." in result
        assert "• Take metformin daily" in result

    def test_japanese_output_uses_ja_template(self):
        result = self.backend.generate(
            query="メトホルミンとは？",
            context="メトホルミンは第一選択薬です。",
            output_language="ja",
            recommendations="• 毎日服用",
        )
        assert "**質問**" in result
        assert "メトホルミンとは？" in result
        assert "メトホルミンは第一選択薬です。" in result

    def test_disclaimer_present_in_english(self):
        result = self.backend.generate("q", "ctx", "en", "• rec")
        assert "healthcare professional" in result

    def test_disclaimer_present_in_japanese(self):
        result = self.backend.generate("q", "ctx", "ja", "• rec")
        assert "医療専門家" in result


class TestOpenAIBackend:
    def setup_method(self):
        self.backend = OpenAIBackend(model="gpt-4o-mini")

    def _mock_openai(self, return_text: str):
        mock_response = MagicMock()
        mock_response.content = return_text
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_cls = MagicMock(return_value=mock_llm)
        return mock_cls, mock_llm

    def test_calls_chatopenai_with_model(self):
        mock_cls, mock_llm = self._mock_openai("Metformin is recommended.")
        with patch("app.services.generation.OpenAIBackend.generate") as patched:
            patched.return_value = "Metformin is recommended."
            result = self.backend.generate("query", "context", "en", "recs")
        assert result == "Metformin is recommended."

    def test_openai_backend_returns_string(self):
        mock_response = MagicMock()
        mock_response.content = "OpenAI answer about diabetes."
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = mock_response

        mock_chat_openai = MagicMock(return_value=mock_llm_instance)
        mock_system = MagicMock()
        mock_human = MagicMock()

        with patch.dict("sys.modules", {
            "langchain_openai": MagicMock(ChatOpenAI=mock_chat_openai),
            "langchain_core.messages": MagicMock(
                SystemMessage=mock_system, HumanMessage=mock_human
            ),
        }):
            result = self.backend.generate(
                query="What is metformin?",
                context="Metformin reduces blood glucose.",
                output_language="en",
                recommendations="• Take daily",
            )

        assert result == "OpenAI answer about diabetes."
        mock_chat_openai.assert_called_once_with(model="gpt-4o-mini")
        mock_llm_instance.invoke.assert_called_once()

    def test_openai_japanese_output_sets_language_in_system_prompt(self):
        mock_response = MagicMock()
        mock_response.content = "日本語の回答"
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = mock_response
        mock_chat_openai = MagicMock(return_value=mock_llm_instance)

        captured_messages = []

        def capture_invoke(messages):
            captured_messages.extend(messages)
            return mock_response

        mock_llm_instance.invoke.side_effect = capture_invoke

        with patch.dict("sys.modules", {
            "langchain_openai": MagicMock(ChatOpenAI=mock_chat_openai),
            "langchain_core.messages": MagicMock(
                SystemMessage=lambda content: ("system", content),
                HumanMessage=lambda content: ("human", content),
            ),
        }):
            self.backend.generate("query", "context", "ja", "recs")

        system_content = captured_messages[0][1]
        assert "Japanese" in system_content


class TestAnthropicBackend:
    def setup_method(self):
        self.backend = AnthropicBackend(model="claude-sonnet-4-6")

    def test_anthropic_backend_returns_string(self):
        mock_response = MagicMock()
        mock_response.content = "Claude answer about hypertension."
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = mock_response

        mock_chat_anthropic = MagicMock(return_value=mock_llm_instance)
        mock_system = MagicMock()
        mock_human = MagicMock()

        with patch.dict("sys.modules", {
            "langchain_anthropic": MagicMock(ChatAnthropic=mock_chat_anthropic),
            "langchain_core.messages": MagicMock(
                SystemMessage=mock_system, HumanMessage=mock_human
            ),
        }):
            result = self.backend.generate(
                query="What are hypertension guidelines?",
                context="Stage 2 hypertension requires medication.",
                output_language="en",
                recommendations="• Monitor BP regularly",
            )

        assert result == "Claude answer about hypertension."
        mock_chat_anthropic.assert_called_once_with(model="claude-sonnet-4-6")
        mock_llm_instance.invoke.assert_called_once()

    def test_anthropic_japanese_output_sets_language_in_system_prompt(self):
        mock_response = MagicMock()
        mock_response.content = "日本語の回答"
        mock_llm_instance = MagicMock()
        captured_messages = []

        def capture_invoke(messages):
            captured_messages.extend(messages)
            return mock_response

        mock_llm_instance.invoke.side_effect = capture_invoke
        mock_chat_anthropic = MagicMock(return_value=mock_llm_instance)

        with patch.dict("sys.modules", {
            "langchain_anthropic": MagicMock(ChatAnthropic=mock_chat_anthropic),
            "langchain_core.messages": MagicMock(
                SystemMessage=lambda content: ("system", content),
                HumanMessage=lambda content: ("human", content),
            ),
        }):
            self.backend.generate("query", "context", "ja", "recs")

        system_content = captured_messages[0][1]
        assert "Japanese" in system_content


class TestGetLLMBackend:
    def test_default_returns_template_backend(self):
        with patch("app.services.generation.get_settings") as mock_settings:
            mock_settings.return_value.llm_backend = "template"
            mock_settings.return_value.llm_model = ""
            backend = _get_llm_backend()
        assert isinstance(backend, TemplateBackend)

    def test_openai_returns_openai_backend_with_default_model(self):
        with patch("app.services.generation.get_settings") as mock_settings:
            mock_settings.return_value.llm_backend = "openai"
            mock_settings.return_value.llm_model = ""
            backend = _get_llm_backend()
        assert isinstance(backend, OpenAIBackend)
        assert backend.model == "gpt-4o-mini"

    def test_openai_respects_custom_model(self):
        with patch("app.services.generation.get_settings") as mock_settings:
            mock_settings.return_value.llm_backend = "openai"
            mock_settings.return_value.llm_model = "gpt-4o"
            backend = _get_llm_backend()
        assert isinstance(backend, OpenAIBackend)
        assert backend.model == "gpt-4o"

    def test_claude_returns_anthropic_backend_with_default_model(self):
        with patch("app.services.generation.get_settings") as mock_settings:
            mock_settings.return_value.llm_backend = "claude"
            mock_settings.return_value.llm_model = ""
            backend = _get_llm_backend()
        assert isinstance(backend, AnthropicBackend)
        assert backend.model == "claude-sonnet-4-6"

    def test_claude_respects_custom_model(self):
        with patch("app.services.generation.get_settings") as mock_settings:
            mock_settings.return_value.llm_backend = "claude"
            mock_settings.return_value.llm_model = "claude-opus-4-6"
            backend = _get_llm_backend()
        assert isinstance(backend, AnthropicBackend)
        assert backend.model == "claude-opus-4-6"

    def test_unknown_backend_falls_back_to_template(self):
        with patch("app.services.generation.get_settings") as mock_settings:
            mock_settings.return_value.llm_backend = "unknown"
            mock_settings.return_value.llm_model = ""
            backend = _get_llm_backend()
        assert isinstance(backend, TemplateBackend)
