from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile, status
from qdrant_client import QdrantClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.dependencies import get_db, get_face_pipeline, get_qdrant
from app.api.schemas import FeedbackRequest, FeedbackResponse, MatchResponse, SearchResponse
from app.core.config import get_settings
from app.db.models import FeedbackEvent, IndexedFace, SearchSession
from app.services.face_pipeline import FacePipeline, FacePipelineError, ImageQualityError
from app.services.face_search import FaceSearchService
from app.services.feedback import process_confirmed_feedback


router = APIRouter(tags=["face-search"])


@router.post("/search", response_model=SearchResponse)
async def search(
    image: UploadFile = File(..., description="Image containing the face to search for."),
    top_k: int = Query(default=24, ge=1, le=100),
    db: Session = Depends(get_db),
    pipeline: FacePipeline = Depends(get_face_pipeline),
    qdrant: QdrantClient = Depends(get_qdrant),
) -> SearchResponse:
    settings = get_settings()
    image_bytes = await image.read(settings.max_upload_bytes + 1)
    await image.close()
    if len(image_bytes) > settings.max_upload_bytes:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Uploaded image is too large.")

    try:
        embedding = pipeline.extract(image_bytes)
        service = FaceSearchService(db, qdrant=qdrant, settings=settings)
        result = service.search(embedding.vector, top_k)
    except ImageQualityError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
    except FacePipelineError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error

    return SearchResponse(
        search_id=result.search_id,
        matches=[
            MatchResponse(
                indexed_face_id=match.indexed_face_id,
                profile_id=match.profile_id,
                source_image_url=match.source_image_url,
                page_context_url=match.page_context_url,
                raw_cosine_similarity=match.raw_cosine_similarity,
                confidence_percent=match.confidence_percent,
            )
            for match in result.matches
        ],
    )


@router.post("/feedback", response_model=FeedbackResponse)
def feedback(
    payload: FeedbackRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> FeedbackResponse:
    search_session = db.get(SearchSession, payload.search_id)
    if search_session is None or search_session.expires_at < datetime.now(UTC):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Search session is unavailable or expired.")

    target = db.get(IndexedFace, payload.indexed_face_id)
    if target is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Search result no longer exists.")
    if str(payload.indexed_face_id) not in search_session.result_indexed_face_ids:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Feedback target was not returned by this search.")

    existing = db.scalar(
        select(FeedbackEvent).where(
            FeedbackEvent.search_session_id == payload.search_id,
            FeedbackEvent.target_indexed_face_id == payload.indexed_face_id,
        )
    )
    if existing is not None:
        return FeedbackResponse(
            feedback_id=existing.id,
            status="already_recorded",
            processed_at=existing.processed_at,
        )

    event = FeedbackEvent(
        id=uuid4(),
        search_session_id=payload.search_id,
        target_indexed_face_id=payload.indexed_face_id,
        is_match=payload.is_match,
    )
    db.add(event)
    db.flush()
    if payload.is_match:
        background_tasks.add_task(process_confirmed_feedback, event.id)
        return FeedbackResponse(feedback_id=event.id, status="queued")
    return FeedbackResponse(feedback_id=event.id, status="recorded")
