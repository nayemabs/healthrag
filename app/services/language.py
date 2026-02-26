from langdetect import LangDetectException, detect

from app.core.constants import LanguageCode
from app.core.logging import get_logger

logger = get_logger(__name__)


def detect_language(text: str) -> LanguageCode:
    # langdetect can misclassify short Japanese as zh-cn/ko — coerce both to JA
    try:
        lang = detect(text)
        if lang in (LanguageCode.EN, LanguageCode.JA):
            return LanguageCode(lang)
        if lang.startswith("zh") or lang == "ko":
            return LanguageCode.JA
        return LanguageCode.EN
    except LangDetectException:
        logger.warning("lang_detect_failed", text_preview=text[:60])
        return LanguageCode.EN
