import re
import threading
from abc import ABC, abstractmethod
from typing import Final

from app.core.config import get_settings
from app.core.constants import LanguageCode
from app.core.logging import get_logger

logger = get_logger(__name__)


def _split_into_sentences(text: str, lang: str) -> list[str]:
    if lang == "ja":
        parts = re.split(r"([。！？\n])", text)
        sentences: list[str] = []
        i = 0
        while i < len(parts):
            chunk = parts[i]
            if i + 1 < len(parts) and parts[i + 1] in "。！？":
                chunk += parts[i + 1]
                i += 2
            else:
                i += 1
            if chunk.strip():
                sentences.append(chunk.strip())
        return sentences or [text]
    else:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s for s in sentences if s.strip()] or [text]


class TranslationBackend(ABC):
    @abstractmethod
    def translate(self, text: str, source: str, target: str) -> str: ...


class MockBackend(TranslationBackend):
    def translate(self, text: str, source: str, target: str) -> str:
        return f"[翻訳] {text}" if target == LanguageCode.JA else f"[Translation] {text}"


class GoogletransBackend(TranslationBackend):
    """py-googletrans wrapper — no API key required, batch-capable."""

    def translate(self, text: str, source: str, target: str) -> str:
        from googletrans import Translator  # noqa: PLC0415

        sentences = _split_into_sentences(text, source)
        translator = Translator()
        results = translator.translate(sentences, src=source, dest=target)
        if isinstance(results, list):
            translated_parts = [r.text for r in results]
        else:
            translated_parts = [results.text]
        separator = "。" if target == LanguageCode.JA else " "
        return separator.join(translated_parts)


class GoogleBackend(TranslationBackend):
    def translate(self, text: str, source: str, target: str) -> str:
        from deep_translator import GoogleTranslator  # noqa: PLC0415

        return GoogleTranslator(source=source, target=target).translate(text)


class ArgostranslateBackend(TranslationBackend):
    def __init__(self) -> None:
        self._installed_pairs: set[tuple[str, str]] = set()
        self._lock = threading.Lock()

    def _ensure_package(self, source: str, target: str) -> None:
        pair = (source, target)
        if pair in self._installed_pairs:
            return
        with self._lock:
            if pair in self._installed_pairs:
                return
            from argostranslate import package as argpkg
            from argostranslate.translate import get_installed_languages

            installed = get_installed_languages()
            source_lang = next((l for l in installed if l.code == source), None)
            if source_lang:
                target_in_source = next(
                    (t for t in source_lang.translations_to if t.code == target), None
                )
                if target_in_source:
                    self._installed_pairs.add(pair)
                    logger.info("argostranslate_package_cached", source=source, target=target)
                    return

            logger.info("downloading_argostranslate_package", source=source, target=target)
            argpkg.update_package_index()
            available = argpkg.get_available_packages()
            pkg = next(
                (p for p in available if p.from_code == source and p.to_code == target), None
            )
            if not pkg:
                raise ValueError(f"No argostranslate package available for {source} → {target}")

            argpkg.install_from_path(pkg.download())
            self._installed_pairs.add(pair)
            logger.info("argostranslate_package_installed", source=source, target=target)

    def translate(self, text: str, source: str, target: str) -> str:
        from argostranslate.translate import translate

        self._ensure_package(source, target)
        sentences = _split_into_sentences(text, source)
        translated_parts = [translate(s, source, target) for s in sentences]
        separator = "。" if target == LanguageCode.JA else " "
        return separator.join(translated_parts)


_BACKENDS: Final[dict[str, TranslationBackend]] = {
    "mock": MockBackend(),
    "googletrans": GoogletransBackend(),
    "google": GoogleBackend(),
    "argostranslate": ArgostranslateBackend(),
}


def translate(text: str, source_lang: str, target_lang: str) -> str:
    """Translate between 'en' and 'ja'. Returns text unchanged if languages match.
    Falls back to mock on any error so the API keeps running."""
    if source_lang == target_lang:
        return text

    settings = get_settings()
    backend = _BACKENDS.get(settings.translation_backend, _BACKENDS["mock"])
    mock = _BACKENDS["mock"]

    try:
        result = backend.translate(text, source_lang, target_lang)
        if backend is not mock:
            logger.info(
                "translated",
                backend=settings.translation_backend,
                source=source_lang,
                target=target_lang,
                input_chars=len(text),
                output_chars=len(result),
            )
        return result
    except Exception as e:
        logger.warning("translation_failed", backend=settings.translation_backend, error=str(e))
        return mock.translate(text, source_lang, target_lang)
