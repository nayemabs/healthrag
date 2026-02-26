from abc import ABC, abstractmethod
from typing import Final

from app.core.config import get_settings
from app.core.constants import LanguageCode
from app.core.logging import get_logger
from app.models.requests import GenerateRequest, RetrieveRequest
from app.models.responses import DocumentResult, GenerateResponse
from app.services.language import detect_language
from app.services.retrieval import retrieve as retrieve_docs
from app.services.translation import translate

logger = get_logger(__name__)

_EN_TEMPLATE: Final = """\
Based on the retrieved medical guidelines and research summaries:

**Query**: {query}

**Summary**:
{context}

**Key Recommendations**:
{recommendations}

*This response is generated from indexed clinical documents. Always verify with a \
qualified healthcare professional before making medical decisions.*"""

_JA_TEMPLATE: Final = """\
取得した医療ガイドラインおよび研究概要に基づく回答：

**質問**: {query}

**要約**:
{context}

**主な推奨事項**:
{recommendations}

*この回答は索引付けされた臨床文書から生成されています。医療上の判断を行う前に、\
必ず資格のある医療専門家にご確認ください。*"""

_RAG_SYSTEM_PROMPT: Final = """\
You are a clinical knowledge assistant. Answer the user's medical query using ONLY the \
retrieved documents provided. Do not add information beyond what appears in the context. \
Respond in {output_language}. End with a disclaimer that the user should verify with a \
qualified healthcare professional before making medical decisions."""

_RAG_USER_PROMPT: Final = """\
Retrieved medical documents:
{context}

Query: {query}"""

_NO_DOCS: Final[dict[LanguageCode, str]] = {
    LanguageCode.EN: "No relevant medical guidelines found in the knowledge base. Please ingest documents first.",
    LanguageCode.JA: "知識ベースに関連する医療ガイドラインが見つかりませんでした。先に文書を取り込んでください。",
}


class LLMBackend(ABC):
    @abstractmethod
    def generate(self, query: str, context: str, output_language: str, recommendations: str) -> str: ...


class TemplateBackend(LLMBackend):
    def generate(self, query: str, context: str, output_language: str, recommendations: str) -> str:
        template = _JA_TEMPLATE if output_language == LanguageCode.JA else _EN_TEMPLATE
        return template.format(query=query, context=context, recommendations=recommendations)


class OpenAIBackend(LLMBackend):
    def __init__(self, model: str) -> None:
        self.model = model

    def generate(self, query: str, context: str, output_language: str, recommendations: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415
        from langchain_openai import ChatOpenAI  # noqa: PLC0415

        lang_name = "Japanese" if output_language == LanguageCode.JA else "English"
        system = _RAG_SYSTEM_PROMPT.format(output_language=lang_name)
        user = _RAG_USER_PROMPT.format(context=context, query=query)
        llm = ChatOpenAI(model=self.model)
        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return str(response.content)


class AnthropicBackend(LLMBackend):
    def __init__(self, model: str) -> None:
        self.model = model

    def generate(self, query: str, context: str, output_language: str, recommendations: str) -> str:
        from langchain_anthropic import ChatAnthropic  # noqa: PLC0415
        from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415

        lang_name = "Japanese" if output_language == LanguageCode.JA else "English"
        system = _RAG_SYSTEM_PROMPT.format(output_language=lang_name)
        user = _RAG_USER_PROMPT.format(context=context, query=query)
        llm = ChatAnthropic(model=self.model)
        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return str(response.content)


def _get_llm_backend() -> LLMBackend:
    settings = get_settings()
    if settings.llm_backend == "openai":
        return OpenAIBackend(model=settings.llm_model or "gpt-4o-mini")
    if settings.llm_backend == "claude":
        return AnthropicBackend(model=settings.llm_model or "claude-sonnet-4-6")
    return TemplateBackend()


def _build_context(docs: list[DocumentResult], max_chars: int) -> str:
    parts: list[str] = []
    total = 0
    for i, doc in enumerate(docs, 1):
        segment = f"[Source {i}]: {doc.content}"
        if total + len(segment) > max_chars:
            break
        parts.append(segment)
        total += len(segment)
    return "\n\n".join(parts)


def _extract_recommendations(docs: list[DocumentResult]) -> str:
    if not docs:
        return "• No specific recommendations available."
    content = docs[0].content
    sentences = [
        s.strip()
        for s in content.replace("。", ". ").split(". ")
        if len(s.strip()) > 15
    ]
    bullets = sentences[:3] if sentences else [content[:200]]
    return "\n".join(f"• {b.rstrip('.')}" for b in bullets)


def generate(request: GenerateRequest) -> GenerateResponse:
    settings = get_settings()
    query_language = detect_language(request.query)
    output_language = request.output_language or query_language
    translated = output_language != query_language

    retrieve_response = retrieve_docs(RetrieveRequest(query=request.query, top_k=request.top_k))
    docs = retrieve_response.results

    logger.info(
        "generating",
        query_lang=query_language,
        output_lang=output_language,
        llm_backend=settings.llm_backend,
        sources=len(docs),
    )

    if not docs:
        return GenerateResponse(
            query=request.query,
            query_language=query_language,
            output_language=output_language,
            answer=_NO_DOCS.get(output_language, _NO_DOCS[LanguageCode.EN]),
            sources=[],
            translated=translated,
        )

    context = _build_context(docs, settings.max_context_chars)
    recommendations = _extract_recommendations(docs)
    display_query = request.query

    if translated:
        context, recommendations, display_query = [
            translate(x, source_lang=query_language, target_lang=output_language)
            for x in (context, recommendations, display_query)
        ]

    backend = _get_llm_backend()
    answer = backend.generate(
        query=display_query,
        context=context,
        output_language=output_language,
        recommendations=recommendations,
    )

    return GenerateResponse(
        query=request.query,
        query_language=query_language,
        output_language=output_language,
        answer=answer,
        sources=docs,
        translated=translated,
    )
