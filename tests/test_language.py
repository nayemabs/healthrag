"""Unit tests for language detection."""
from unittest.mock import patch

import pytest
from langdetect import LangDetectException

from app.services.language import detect_language


class TestDetectLanguageEnglish:
    def test_clear_english_medical_text(self):
        text = (
            "Type 2 diabetes management guidelines recommend metformin as "
            "first-line therapy. Regular HbA1c monitoring every three months is advised."
        )
        assert detect_language(text) == "en"

    def test_short_english_word(self):
        # Single word — langdetect may be uncertain, but common English words work
        with patch("app.services.language.detect", return_value="en"):
            assert detect_language("diabetes") == "en"

    def test_english_with_numbers(self):
        text = "Target HbA1c should be below 7.0% according to 2024 clinical guidelines."
        assert detect_language(text) == "en"

    def test_non_supported_language_falls_back_to_en(self):
        # French, German, etc. → 'en' fallback
        with patch("app.services.language.detect", return_value="fr"):
            assert detect_language("Bonjour") == "en"

    def test_spanish_falls_back_to_en(self):
        with patch("app.services.language.detect", return_value="es"):
            assert detect_language("Hola mundo") == "en"


class TestDetectLanguageJapanese:
    def test_clear_japanese_medical_text(self):
        text = (
            "2型糖尿病の管理ガイドライン：メトホルミンが第一選択薬として推奨されます。"
            "3ヶ月ごとのHbA1c測定が必要です。"
        )
        assert detect_language(text) == "ja"

    def test_hiragana_only(self):
        text = "にほんごのてすとぶんしょうです。ちょっとながいほうがいいですよね。"
        assert detect_language(text) == "ja"

    def test_katakana_medical_terms(self):
        with patch("app.services.language.detect", return_value="ja"):
            assert detect_language("メトホルミン インスリン") == "ja"

    def test_kanji_heavy_text(self):
        text = "糖尿病の治療において血糖値の管理が重要です。食事療法と運動療法を組み合わせる。"
        assert detect_language(text) == "ja"


class TestCJKCoercion:
    """langdetect sometimes classifies short Japanese text as zh-cn or ko.
    detect_language() should coerce these CJK variants back to 'ja'."""

    def test_zh_cn_coerced_to_ja(self):
        with patch("app.services.language.detect", return_value="zh-cn"):
            assert detect_language("some cjk text") == "ja"

    def test_zh_tw_coerced_to_ja(self):
        with patch("app.services.language.detect", return_value="zh-tw"):
            assert detect_language("some cjk text") == "ja"

    def test_ko_coerced_to_ja(self):
        with patch("app.services.language.detect", return_value="ko"):
            assert detect_language("some cjk text") == "ja"

    def test_zh_hk_coerced_to_ja(self):
        # Any lang starting with 'zh' should be coerced
        with patch("app.services.language.detect", return_value="zh-hk"):
            assert detect_language("some text") == "ja"


class TestLangDetectExceptionHandling:
    def test_lang_detect_exception_returns_en(self):
        with patch(
            "app.services.language.detect",
            side_effect=LangDetectException(0, "No features in profile"),
        ):
            assert detect_language("x") == "en"

    def test_lang_detect_exception_on_empty_string(self):
        # langdetect raises on empty/whitespace-only strings
        with patch(
            "app.services.language.detect",
            side_effect=LangDetectException(0, "No text"),
        ):
            assert detect_language("") == "en"

    def test_lang_detect_exception_on_numeric_only(self):
        with patch(
            "app.services.language.detect",
            side_effect=LangDetectException(0, "Insufficient features"),
        ):
            assert detect_language("12345") == "en"


class TestReturnedValues:
    """Ensure detect_language() only returns 'en' or 'ja', never anything else."""

    @pytest.mark.parametrize("mocked_lang", ["en", "ja", "zh-cn", "ko", "fr", "de", "ar", "ru"])
    def test_always_returns_en_or_ja(self, mocked_lang):
        with patch("app.services.language.detect", return_value=mocked_lang):
            result = detect_language("some text")
        assert result in ("en", "ja"), f"Expected en or ja, got {result!r} for mocked lang {mocked_lang!r}"
