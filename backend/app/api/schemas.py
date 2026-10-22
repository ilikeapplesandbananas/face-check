from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl


class MatchResponse(BaseModel):
    indexed_face_id: UUID
    profile_id: UUID
    source_image_url: HttpUrl
    page_context_url: HttpUrl
    raw_cosine_similarity: float
    confidence_percent: float = Field(ge=0, le=100)


class SearchResponse(BaseModel):
    search_id: UUID
    matches: list[MatchResponse]


class FeedbackRequest(BaseModel):
    search_id: UUID
    indexed_face_id: UUID
    is_match: bool


class FeedbackResponse(BaseModel):
    feedback_id: UUID
    status: str
    processed_at: datetime | None = None
