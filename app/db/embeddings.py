from functools import lru_cache

from langchain_core.embeddings import Embeddings

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class LocalEmbeddings(Embeddings):
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer
        logger.info("loading_embedding_model", model=model_name)
        self._model = SentenceTransformer(model_name)
        logger.info("embedding_model_ready", model=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode([text], normalize_embeddings=True, show_progress_bar=False)
        return vector[0].tolist()


@lru_cache(maxsize=1)
def get_embeddings() -> LocalEmbeddings:
    settings = get_settings()
    return LocalEmbeddings(settings.embedding_model)
