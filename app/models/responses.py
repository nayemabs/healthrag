from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IngestResponse:
    doc_id: str
    filename: str
    language: str
    chunks_indexed: int
    message: str


@dataclass(frozen=True, slots=True)
class DocumentResult:
    doc_id: str
    content: str
    language: str
    similarity_score: float
    chunk_index: int


@dataclass(frozen=True, slots=True)
class RetrieveResponse:
    query: str
    query_language: str
    results: list[DocumentResult]
    total_found: int


@dataclass(frozen=True, slots=True)
class GenerateResponse:
    query: str
    query_language: str
    output_language: str
    answer: str
    sources: list[DocumentResult]
    translated: bool
