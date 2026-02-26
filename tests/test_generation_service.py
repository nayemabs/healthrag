from unittest.mock import patch

from app.models.requests import GenerateRequest
from app.services.generation import _build_context, _extract_recommendations, generate
from tests.conftest import make_document_result, make_retrieve_response


class TestBuildContext:
    def test_single_doc_formats_correctly(self):
        doc = make_document_result("Metformin reduces HbA1c.")
        ctx = _build_context([doc], max_chars=2000)
        assert "[Source 1]:" in ctx
        assert "Metformin reduces HbA1c." in ctx

    def test_multiple_docs_numbered(self):
        docs = [
            make_document_result("First guideline.", doc_id="d1"),
            make_document_result("Second guideline.", doc_id="d2"),
            make_document_result("Third guideline.", doc_id="d3"),
        ]
        ctx = _build_context(docs, max_chars=2000)
        assert "[Source 1]:" in ctx
        assert "[Source 2]:" in ctx
        assert "[Source 3]:" in ctx

    def test_truncates_at_max_chars(self):
        docs = [make_document_result(f"Content segment {i}." * 3, doc_id=f"d{i}") for i in range(5)]
        ctx = _build_context(docs, max_chars=50)
        assert len(ctx) <= 50 + len("[Source 1]: ")

    def test_empty_docs_returns_empty_string(self):
        assert _build_context([], max_chars=2000) == ""

    def test_docs_joined_with_double_newline(self):
        docs = [
            make_document_result("First.", doc_id="d1"),
            make_document_result("Second.", doc_id="d2"),
        ]
        ctx = _build_context(docs, max_chars=2000)
        assert "\n\n" in ctx

    def test_only_first_doc_when_max_chars_fits_one(self):
        doc1 = make_document_result("First short doc.")
        doc2 = make_document_result("Second doc content that would push total over limit.")
        ctx = _build_context([doc1, doc2], max_chars=30)
        assert "[Source 1]:" in ctx
        assert "[Source 2]:" not in ctx


class TestExtractRecommendations:
    def test_empty_docs_returns_no_recommendations_message(self):
        result = _extract_recommendations([])
        assert "No specific recommendations available" in result

    def test_bullets_start_with_dot(self):
        doc = make_document_result(
            "Metformin is first-line therapy. HbA1c should be monitored regularly. "
            "Lifestyle changes are essential. Blood pressure targets apply."
        )
        result = _extract_recommendations([doc])
        lines = [line for line in result.split("\n") if line.strip()]
        assert all(line.startswith("•") for line in lines)

    def test_at_most_three_bullets(self):
        doc = make_document_result(
            "First long recommendation sentence here. Second long recommendation here. "
            "Third important guideline here. Fourth recommendation also included. Fifth as well."
        )
        result = _extract_recommendations([doc])
        assert result.count("•") <= 3

    def test_short_sentences_excluded(self):
        doc = make_document_result("Short. " + "This is a longer recommendation sentence about diabetes. " * 3)
        result = _extract_recommendations([doc])
        assert "Short" not in result

    def test_uses_only_first_document(self):
        doc1 = make_document_result("First doc recommendation sentence that is long enough.")
        doc2 = make_document_result("Second doc content should NOT appear in bullets.")
        result = _extract_recommendations([doc1, doc2])
        assert "Second doc content" not in result

    def test_japanese_text_with_kuten_split(self):
        doc = make_document_result(
            "メトホルミンが第一選択薬として推奨されます。HbA1cを定期的に測定してください。血糖値の管理が重要です。",
            language="ja",
        )
        result = _extract_recommendations([doc])
        assert "•" in result


class TestGenerateTemplateSelection:
    def test_english_query_uses_en_template(self):
        doc = make_document_result("Metformin is first-line therapy for type 2 diabetes.")
        resp = make_retrieve_response([doc], query="diabetes treatment", query_language="en")
        with patch("app.services.generation.retrieve_docs", return_value=resp):
            result = generate(GenerateRequest(query="What is the treatment for diabetes?"))
        assert "Based on the retrieved medical guidelines" in result.answer
        assert result.output_language == "en"

    def test_japanese_query_uses_ja_template(self):
        doc = make_document_result("メトホルミンは第一選択薬として推奨されます。", language="ja")
        resp = make_retrieve_response([doc], query="糖尿病治療", query_language="ja")
        with patch("app.services.generation.retrieve_docs", return_value=resp):
            result = generate(GenerateRequest(query="2型糖尿病の治療法は何ですか？"))
        assert "取得した医療ガイドライン" in result.answer
        assert result.output_language == "ja"

    def test_en_query_with_output_ja_uses_ja_template(self):
        doc = make_document_result("Metformin is first-line therapy.")
        resp = make_retrieve_response([doc], query="diabetes", query_language="en")
        with patch("app.services.generation.retrieve_docs", return_value=resp), \
             patch("app.services.generation.translate", return_value="[翻訳]"):
            result = generate(GenerateRequest(query="What is the treatment?", output_language="ja"))
        assert "取得した医療ガイドライン" in result.answer

    def test_answer_contains_query_section(self):
        doc = make_document_result("Lifestyle changes help control blood sugar effectively.")
        resp = make_retrieve_response([doc], query="diabetes", query_language="en")
        with patch("app.services.generation.retrieve_docs", return_value=resp):
            result = generate(GenerateRequest(query="How to control blood sugar?"))
        assert "How to control blood sugar?" in result.answer


class TestGenerateTranslation:
    def test_translated_flag_true_when_output_differs(self):
        doc = make_document_result("Diabetes guideline content.")
        resp = make_retrieve_response([doc], query="diabetes", query_language="en")
        with patch("app.services.generation.retrieve_docs", return_value=resp), \
             patch("app.services.generation.translate", return_value="[翻訳]"):
            result = generate(GenerateRequest(query="Treatment?", output_language="ja"))
        assert result.translated is True

    def test_translated_flag_false_when_output_same(self):
        doc = make_document_result("Diabetes guideline.")
        resp = make_retrieve_response([doc], query="diabetes", query_language="en")
        with patch("app.services.generation.retrieve_docs", return_value=resp):
            result = generate(GenerateRequest(query="Treatment?"))
        assert result.translated is False

    def test_translate_called_for_context_recs_and_query(self):
        """When output_language differs, translate() is called exactly 3 times:
        context, recommendations, and display_query."""
        doc = make_document_result("Metformin reduces HbA1c. Lifestyle changes are important.")
        resp = make_retrieve_response([doc], query="diabetes", query_language="en")
        with patch("app.services.generation.retrieve_docs", return_value=resp), \
             patch("app.services.generation.translate", return_value="翻訳") as mock_t:
            generate(GenerateRequest(query="What is the treatment?", output_language="ja"))
        assert mock_t.call_count == 3

    def test_translate_not_called_when_same_language(self):
        doc = make_document_result("Content.")
        resp = make_retrieve_response([doc], query="diabetes", query_language="en")
        with patch("app.services.generation.retrieve_docs", return_value=resp), \
             patch("app.services.generation.translate") as mock_t:
            generate(GenerateRequest(query="Treatment?"))
        mock_t.assert_not_called()

    def test_translate_called_with_correct_languages(self):
        doc = make_document_result("English content about diabetes.")
        resp = make_retrieve_response([doc], query="diabetes", query_language="en")
        with patch("app.services.generation.retrieve_docs", return_value=resp), \
             patch("app.services.generation.translate", return_value="翻訳") as mock_t:
            generate(GenerateRequest(query="Treatment?", output_language="ja"))
        for c in mock_t.call_args_list:
            assert c.kwargs.get("source_lang") == "en" or c.args[1] == "en"
            assert c.kwargs.get("target_lang") == "ja" or c.args[2] == "ja"


class TestGenerateNoDocs:
    def test_en_query_no_docs_english_message(self):
        resp = make_retrieve_response([], query="test", query_language="en")
        with patch("app.services.generation.retrieve_docs", return_value=resp):
            result = generate(GenerateRequest(query="What is the treatment?"))
        assert "No relevant" in result.answer
        assert result.sources == []

    def test_ja_output_no_docs_japanese_message(self):
        resp = make_retrieve_response([], query="test", query_language="en")
        with patch("app.services.generation.retrieve_docs", return_value=resp):
            result = generate(GenerateRequest(query="test query", output_language="ja"))
        assert "見つかりません" in result.answer

    def test_no_docs_translated_flag_reflects_language_mismatch(self):
        resp = make_retrieve_response([], query="test", query_language="en")
        with patch("app.services.generation.retrieve_docs", return_value=resp):
            result = generate(GenerateRequest(query="test", output_language="ja"))
        assert result.translated is True

    def test_no_docs_sources_is_empty_list(self):
        resp = make_retrieve_response([], query="test", query_language="en")
        with patch("app.services.generation.retrieve_docs", return_value=resp):
            result = generate(GenerateRequest(query="test"))
        assert result.sources == []


class TestGenerateResponseShape:
    def test_response_contains_all_required_fields(self):
        doc = make_document_result("Guideline content.")
        resp = make_retrieve_response([doc], query="diabetes", query_language="en")
        with patch("app.services.generation.retrieve_docs", return_value=resp):
            result = generate(GenerateRequest(query="What is the treatment?"))
        assert result.query == "What is the treatment?"
        assert result.query_language == "en"
        assert result.output_language == "en"
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0
        assert isinstance(result.sources, list)
        assert isinstance(result.translated, bool)

    def test_sources_match_retrieved_docs(self):
        docs = [
            make_document_result("Guideline A.", doc_id="docA"),
            make_document_result("Guideline B.", doc_id="docB"),
        ]
        resp = make_retrieve_response(docs, query="diabetes", query_language="en")
        with patch("app.services.generation.retrieve_docs", return_value=resp):
            result = generate(GenerateRequest(query="Treatment?"))
        doc_ids = [s.doc_id for s in result.sources]
        assert "docA" in doc_ids
        assert "docB" in doc_ids

    def test_output_language_defaults_to_query_language(self):
        doc = make_document_result("Content.")
        resp = make_retrieve_response([doc], query="query", query_language="ja")
        with patch("app.services.generation.retrieve_docs", return_value=resp):
            result = generate(GenerateRequest(query="糖尿病の治療は？"))
        assert result.output_language == "ja"

    def test_answer_is_non_empty_string(self):
        doc = make_document_result("Medical guideline content about blood pressure management.")
        resp = make_retrieve_response([doc], query="blood pressure", query_language="en")
        with patch("app.services.generation.retrieve_docs", return_value=resp):
            result = generate(GenerateRequest(query="Blood pressure guidelines?"))
        assert isinstance(result.answer, str)
        assert len(result.answer) > 50
