from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_store: Optional[FAISS] = None


def _index_path() -> Path:
    return Path(get_settings().faiss_index_path)


def load_store() -> Optional[FAISS]:
    global _store
    path = _index_path()
    if path.exists():
        try:
            from app.db.embeddings import get_embeddings
            _store = FAISS.load_local(
                str(path),
                get_embeddings(),
                allow_dangerous_deserialization=True,
            )
            logger.info("faiss_loaded", path=str(path))
        except Exception as e:
            logger.warning("faiss_load_failed", error=str(e))
            _store = None
    return _store


def save_store(store: FAISS) -> None:
    path = _index_path()
    path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(path))
    logger.info("faiss_saved", path=str(path))


def get_store() -> Optional[FAISS]:
    return _store


def add_documents(documents: list[Document], embeddings) -> FAISS:
    global _store
    if _store is None:
        _store = FAISS.from_documents(documents, embeddings)
    else:
        _store.add_documents(documents)
    save_store(_store)
    return _store


def reset_store() -> None:
    global _store
    _store = None
