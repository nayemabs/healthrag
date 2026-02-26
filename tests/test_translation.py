"""Unit tests for the translation service."""
import os
from unittest.mock import MagicMock, patch

import pytest

from app.services.translation import MockBackend, GoogletransBackend, _BACKENDS, _split_into_sentences, translate


@pytest.fixture(autouse=True)
def reset_settings_cache():
    # Clear the LRU cache so TRANSLATION_BACKEND env patches actually take effect
    from app.core.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class TestSplitIntoSentencesEnglish:
    def test_single_sentence_period(self):
        result = _split_into_sentences("Metformin is first-line therapy.", "en")
        assert result == ["Metformin is first-line therapy."]

    def test_multiple_sentences_period(self):
        text = "Metformin is recommended. HbA1c should be monitored. Lifestyle changes help."
        result = _split_into_sentences(text, "en")
        assert len(result) == 3
        assert result[0].startswith("Metformin")
        assert result[1].startswith("HbA1c")
        assert result[2].startswith("Lifestyle")

    def test_exclamation_and_question_marks(self):
        text = "Is this safe? Yes it is! Good to know."
        result = _split_into_sentences(text, "en")
        assert len(result) == 3

    def test_no_terminator_returns_original_as_list(self):
        result = _split_into_sentences("Hypertension management", "en")
        assert result == ["Hypertension management"]

    def test_leading_trailing_whitespace_stripped(self):
        text = "  Sentence one. Sentence two.  "
        result = _split_into_sentences(text, "en")
        for s in result:
            assert s == s.strip()

    def test_empty_parts_excluded(self):
        text = "First sentence.  Second sentence."
        result = _split_into_sentences(text, "en")
        assert all(s.strip() for s in result)

    def test_single_word_no_terminator(self):
        result = _split_into_sentences("Diabetes", "en")
        assert result == ["Diabetes"]


class TestSplitIntoSentencesJapanese:
    def test_single_sentence_kuten(self):
        result = _split_into_sentences("糖尿病の治療について。", "ja")
        assert result == ["糖尿病の治療について。"]

    def test_multiple_sentences_kuten(self):
        text = "メトホルミンが推奨されます。定期的な検査が必要です。血糖値を管理してください。"
        result = _split_into_sentences(text, "ja")
        assert len(result) == 3
        assert result[0] == "メトホルミンが推奨されます。"
        assert result[1] == "定期的な検査が必要です。"
        assert result[2] == "血糖値を管理してください。"

    def test_exclamation_and_question_marks(self):
        text = "これは正しいですか？はい！確認しました。"
        result = _split_into_sentences(text, "ja")
        assert len(result) == 3

    def test_terminators_remain_attached(self):
        text = "第一文。第二文。"
        result = _split_into_sentences(text, "ja")
        for s in result:
            assert s.endswith("。")

    def test_newline_separates_sentences(self):
        text = "第一段落。\n第二段落。"
        result = _split_into_sentences(text, "ja")
        assert "第一段落。" in result
        assert "第二段落。" in result

    def test_empty_string_returns_original(self):
        result = _split_into_sentences("", "ja")
        assert result == [""]

    def test_text_without_terminators_returns_original(self):
        result = _split_into_sentences("糖尿病の管理", "ja")
        assert result == ["糖尿病の管理"]


class TestMockBackend:
    def setup_method(self):
        self.backend = MockBackend()

    def test_en_to_ja_adds_japanese_tag(self):
        assert self.backend.translate("Hello world", "en", "ja") == "[翻訳] Hello world"

    def test_ja_to_en_adds_english_tag(self):
        assert self.backend.translate("こんにちは", "ja", "en") == "[Translation] こんにちは"

    def test_preserves_full_text(self):
        long_text = "Type 2 diabetes management includes metformin as first-line therapy."
        result = self.backend.translate(long_text, "en", "ja")
        assert long_text in result


class TestTranslateMockBackend:
    """These tests run under TRANSLATION_BACKEND=mock (set by conftest)."""

    def test_same_language_english_returns_unchanged(self):
        text = "Diabetes treatment guidelines"
        assert translate(text, "en", "en") is text

    def test_same_language_japanese_returns_unchanged(self):
        text = "糖尿病の治療について"
        assert translate(text, "ja", "ja") is text

    def test_mock_en_to_ja(self):
        result = translate("Hello world", "en", "ja")
        assert result == "[翻訳] Hello world"

    def test_mock_ja_to_en(self):
        result = translate("こんにちは", "ja", "en")
        assert result == "[Translation] こんにちは"

    def test_empty_string_same_language(self):
        assert translate("", "en", "en") == ""


class TestGoogletransBackend:
    """Unit tests for GoogletransBackend.

    googletrans is patched via sys.modules so these tests work regardless of
    whether the package is installed (it pins an old httpx that breaks on Py 3.13+).
    """

    def setup_method(self):
        self.backend = GoogletransBackend()

    def _mock_googletrans(self, return_value):
        """Return a sys.modules patch that makes Translator().translate() return *return_value*."""
        mock_instance = MagicMock()
        mock_instance.translate.return_value = return_value
        mock_module = MagicMock()
        mock_module.Translator.return_value = mock_instance
        return patch.dict("sys.modules", {"googletrans": mock_module})

    def test_en_to_ja_returns_translated_text(self):
        mock_result = MagicMock()
        mock_result.text = "テスト翻訳"
        with self._mock_googletrans(mock_result):
            result = self.backend.translate("Test translation", "en", "ja")
        assert result == "テスト翻訳"

    def test_ja_to_en_returns_translated_text(self):
        mock_result = MagicMock()
        mock_result.text = "Translated text"
        with self._mock_googletrans(mock_result):
            result = self.backend.translate("翻訳テキスト", "ja", "en")
        assert result == "Translated text"

    def test_batch_results_joined_with_japanese_separator(self):
        r1, r2 = MagicMock(), MagicMock()
        r1.text, r2.text = "テスト", "翻訳"
        with self._mock_googletrans([r1, r2]):
            result = self.backend.translate("Test. Translation.", "en", "ja")
        assert result == "テスト。翻訳"

    def test_batch_results_joined_with_space_for_english(self):
        r1, r2 = MagicMock(), MagicMock()
        r1.text, r2.text = "First.", "Second."
        with self._mock_googletrans([r1, r2]):
            result = self.backend.translate("最初。次。", "ja", "en")
        assert result == "First. Second."


class TestTranslateGoogletransBackend:
    def test_calls_googletrans_translate(self):
        with patch.dict(os.environ, {"TRANSLATION_BACKEND": "googletrans"}):
            with patch.object(_BACKENDS["googletrans"], "translate", return_value="グーグルトランス") as mock_gt:
                result = translate("Hello", "en", "ja")

        assert result == "グーグルトランス"
        mock_gt.assert_called_once_with("Hello", "en", "ja")

    def test_falls_back_to_mock_on_connection_error(self):
        with patch.dict(os.environ, {"TRANSLATION_BACKEND": "googletrans"}):
            with patch.object(_BACKENDS["googletrans"], "translate", side_effect=ConnectionError("network down")):
                result = translate("Hello", "en", "ja")

        assert result == "[翻訳] Hello"

    def test_falls_back_to_mock_on_exception(self):
        with patch.dict(os.environ, {"TRANSLATION_BACKEND": "googletrans"}):
            with patch.object(_BACKENDS["googletrans"], "translate", side_effect=Exception("API error")):
                result = translate("Some text", "ja", "en")

        assert result == "[Translation] Some text"

    def test_same_language_never_calls_googletrans(self):
        with patch.dict(os.environ, {"TRANSLATION_BACKEND": "googletrans"}):
            with patch.object(_BACKENDS["googletrans"], "translate") as mock_gt:
                result = translate("Same language", "en", "en")

        assert result == "Same language"
        mock_gt.assert_not_called()


class TestTranslateArgostranslateBackend:
    def test_calls_argostranslate_translate(self):
        with patch.dict(os.environ, {"TRANSLATION_BACKEND": "argostranslate"}):
            with patch.object(_BACKENDS["argostranslate"], "translate", return_value="テスト翻訳") as mock_argo:
                result = translate("Test text", "en", "ja")

        assert result == "テスト翻訳"
        mock_argo.assert_called_once_with("Test text", "en", "ja")

    def test_falls_back_to_mock_on_runtime_error(self):
        with patch.dict(os.environ, {"TRANSLATION_BACKEND": "argostranslate"}):
            with patch.object(_BACKENDS["argostranslate"], "translate", side_effect=RuntimeError("package not found")):
                result = translate("Test text", "en", "ja")

        assert result == "[翻訳] Test text"

    def test_falls_back_to_mock_on_value_error(self):
        with patch.dict(os.environ, {"TRANSLATION_BACKEND": "argostranslate"}):
            with patch.object(_BACKENDS["argostranslate"], "translate", side_effect=ValueError("no package for en → ja")):
                result = translate("Diabetes guidelines", "en", "ja")

        assert result == "[翻訳] Diabetes guidelines"

    def test_falls_back_to_mock_on_import_error(self):
        with patch.dict(os.environ, {"TRANSLATION_BACKEND": "argostranslate"}):
            with patch.object(_BACKENDS["argostranslate"], "translate", side_effect=ImportError("argostranslate not installed")):
                result = translate("Test", "en", "ja")

        assert result == "[翻訳] Test"

    def test_same_language_never_calls_argostranslate(self):
        with patch.dict(os.environ, {"TRANSLATION_BACKEND": "argostranslate"}):
            with patch.object(_BACKENDS["argostranslate"], "translate") as mock_argo:
                result = translate("Same language text", "en", "en")

        assert result == "Same language text"
        mock_argo.assert_not_called()

    def test_logs_success_metadata(self):
        with patch.dict(os.environ, {"TRANSLATION_BACKEND": "argostranslate"}):
            with patch.object(_BACKENDS["argostranslate"], "translate", return_value="翻訳された文章"):
                result = translate("Translated sentence", "en", "ja")

        assert len(result) > 0


class TestTranslateGoogleBackend:
    def test_calls_google_translate(self):
        with patch.dict(os.environ, {"TRANSLATION_BACKEND": "google"}):
            with patch.object(_BACKENDS["google"], "translate", return_value="グーグル翻訳") as mock_google:
                result = translate("Hello", "en", "ja")

        assert result == "グーグル翻訳"
        mock_google.assert_called_once_with("Hello", "en", "ja")

    def test_falls_back_to_mock_on_connection_error(self):
        with patch.dict(os.environ, {"TRANSLATION_BACKEND": "google"}):
            with patch.object(_BACKENDS["google"], "translate", side_effect=ConnectionError("network unavailable")):
                result = translate("Hello", "en", "ja")

        assert result == "[翻訳] Hello"

    def test_falls_back_to_mock_on_exception(self):
        with patch.dict(os.environ, {"TRANSLATION_BACKEND": "google"}):
            with patch.object(_BACKENDS["google"], "translate", side_effect=Exception("Google API error")):
                result = translate("Some text", "ja", "en")

        assert result == "[Translation] Some text"

    def test_same_language_never_calls_google(self):
        with patch.dict(os.environ, {"TRANSLATION_BACKEND": "google"}):
            with patch.object(_BACKENDS["google"], "translate") as mock_google:
                result = translate("Same language text", "en", "en")

        assert result == "Same language text"
        mock_google.assert_not_called()


class TestTranslateUnknownBackend:
    def test_unknown_backend_falls_back_to_mock_ja(self):
        from app.core.config import get_settings
        settings = get_settings()
        with patch.object(settings, "translation_backend", "nonexistent_backend"):
            with patch("app.core.config.get_settings", return_value=settings):
                result = translate("Fallback test", "en", "ja")
        assert result == "[翻訳] Fallback test"

    def test_unknown_backend_falls_back_to_mock_en(self):
        from app.core.config import get_settings
        settings = get_settings()
        with patch.object(settings, "translation_backend", "nonexistent_backend"):
            with patch("app.core.config.get_settings", return_value=settings):
                result = translate("フォールバックテスト", "ja", "en")
        assert result == "[Translation] フォールバックテスト"
