from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class MatchResponse(BaseModel):
    indexed_face_id: UUID
    profile_id: UUID
    source_image_url: str
    page_context_url: str
    raw_cosine_similarity: float
    confidence_percent: float = Field(ge=0, le=100)
    match_tier: str = "Lookalike"


class SearchResponse(BaseModel):
    search_id: UUID
    matches: list[MatchResponse]
    detected_faces_count: int = 1


class FeedbackRequest(BaseModel):
    search_id: UUID
    indexed_face_id: UUID
    is_match: bool


class FeedbackResponse(BaseModel):
    feedback_id: UUID
    status: str
    processed_at: datetime | None = None


class CrawlRequest(BaseModel):
    page_url: str
    max_images: int = Field(default=30, ge=1, le=100)


class CrawlResponse(BaseModel):
    page_url: str
    indexed_faces_count: int
    status: str
