from pydantic import BaseModel, Field

from app.core.constants import LanguageCode


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)
    language_filter: LanguageCode | None = None


class GenerateRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)
    output_language: LanguageCode | None = None
