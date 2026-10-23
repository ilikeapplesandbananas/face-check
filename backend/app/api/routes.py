from __future__ import annotations

import httpx
from datetime import UTC, datetime
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, File, Header, HTTPException, Query, UploadFile, status
from qdrant_client import QdrantClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.dependencies import get_db, get_face_pipeline, get_qdrant
from app.api.schemas import (
    CrawlRequest,
    CrawlResponse,
    FeedbackRequest,
    FeedbackResponse,
    MatchResponse,
    SearchResponse,
)
from app.core.config import get_settings
from app.db.models import FeedbackEvent, IndexedFace, SearchSession
from app.services.face_pipeline import FacePipeline, FacePipelineError, ImageQualityError
from app.services.face_search import FaceSearchService
from app.services.feedback import process_confirmed_feedback
from app.services.vector_index import FaceIndexer
from app.services.web_crawler import WebCrawlerService
from app.services.web_search import WebFaceSearchService, calibrate_face_confidence


router = APIRouter(tags=["face-search"])

SEED_SAMPLES = [
    {
        "name": "Sarah Chen",
        "image_url": "https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=600&q=80",
        "page_url": "https://unsplash.com/@aiony",
    },
    {
        "name": "David Miller",
        "image_url": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?auto=format&fit=crop&w=600&q=80",
        "page_url": "https://unsplash.com/@jurica",
    },
    {
        "name": "Elena Rostova",
        "image_url": "https://images.unsplash.com/photo-1494790108377-be9c29b29330?auto=format&fit=crop&w=600&q=80",
        "page_url": "https://unsplash.com/@michaelfrattaroli",
    },
    {
        "name": "Marcus Vance",
        "image_url": "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?auto=format&fit=crop&w=600&q=80",
        "page_url": "https://unsplash.com/@alexandermils",
    },
    {
        "name": "Sophia Williams",
        "image_url": "https://images.unsplash.com/photo-1517841905240-472988babdf9?auto=format&fit=crop&w=600&q=80",
        "page_url": "https://unsplash.com/@princeakachi",
    },
    {
        "name": "Lucas Silva",
        "image_url": "https://images.unsplash.com/photo-1539571696357-5a69c17a67c6?auto=format&fit=crop&w=600&q=80",
        "page_url": "https://unsplash.com/@ethangough",
    },
]


def get_match_tier(confidence: float) -> str:
    if confidence >= 85.0:
        return "Definite Match"
    elif confidence >= 72.0:
        return "High Confidence"
    elif confidence >= 58.0:
        return "Strong Lookalike"
    elif confidence >= 42.0:
        return "Similar Features"
    else:
        return "Visual Candidate"


@router.get("/health")
def health(db: Session = Depends(get_db), qdrant: QdrantClient = Depends(get_qdrant)) -> dict[str, object]:
    settings = get_settings()
    indexed_count = db.scalar(select(func.count(IndexedFace.id))) or 0
    return {
        "status": "online",
        "detector": "RetinaFace-R50",
        "recognizer": "ArcFace-R100 (512D)",
        "indexed_faces_count": indexed_count,
        "database": "connected",
        "vector_storage": "ready",
        "has_serpapi_configured": bool(settings.serpapi_api_key),
        "samples": SEED_SAMPLES,
    }


@router.post("/seed")
async def seed_samples(
    db: Session = Depends(get_db),
    pipeline: FacePipeline = Depends(get_face_pipeline),
    qdrant: QdrantClient = Depends(get_qdrant),
) -> dict[str, object]:
    """Seed the database with verified portrait faces for immediate testing."""
    indexer = FaceIndexer(session=db, qdrant=qdrant)
    indexed_count = 0

    async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
        for item in SEED_SAMPLES:
            try:
                r = await client.get(item["image_url"])
                if r.status_code == 200:
                    emb = pipeline.extract(r.content)
                    indexer.index(
                        embedding=emb.vector,
                        source_image_url=item["image_url"],
                        page_context_url=item["page_url"],
                    )
                    indexed_count += 1
            except Exception:
                continue

    return {"status": "completed", "indexed_count": indexed_count}


@router.post("/search", response_model=SearchResponse)
async def search(
    image: UploadFile = File(..., description="Image containing the face to search for."),
    top_k: int = Query(default=24, ge=1, le=100),
    search_web: bool = Query(default=True, description="Query live web sources for candidate faces."),
    serpapi_key: str | None = Query(default=None, description="Optional SerpApi key for Google Lens web search."),
    x_serpapi_key: str | None = Header(default=None, alias="X-SerpApi-Key"),
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

        all_matches = list(result.matches)

        active_key = x_serpapi_key or serpapi_key or settings.serpapi_api_key

        if search_web and active_key:
            web_service = WebFaceSearchService(pipeline=pipeline, settings=settings, api_key_override=active_key)
            candidates = await web_service.fetch_web_candidates(image_bytes, top_k=80)
            if candidates:
                web_matches = await web_service.search_and_rerank(embedding.vector, candidates, top_k=top_k)
                all_matches.extend(web_matches)

        seen_urls: set[str] = set()
        deduped = []
        for m in sorted(all_matches, key=lambda x: x.raw_cosine_similarity, reverse=True):
            if m.source_image_url not in seen_urls:
                seen_urls.add(m.source_image_url)
                deduped.append(m)

        final_matches = deduped[:top_k]

    except HTTPException:
        raise
    except ImageQualityError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
    except FacePipelineError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Search failed: {str(error)}") from error

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
                match_tier=get_match_tier(match.confidence_percent),
            )
            for match in final_matches
        ],
        detected_faces_count=1,
    )


@router.post("/crawl", response_model=CrawlResponse)
async def crawl(
    payload: CrawlRequest,
    db: Session = Depends(get_db),
    pipeline: FacePipeline = Depends(get_face_pipeline),
    qdrant: QdrantClient = Depends(get_qdrant),
) -> CrawlResponse:
    crawler = WebCrawlerService(session=db, pipeline=pipeline, settings=get_settings())
    crawler.indexer = FaceIndexer(session=db, qdrant=qdrant)
    try:
        records = await crawler.crawl_page(payload.page_url, max_images=payload.max_images)
        return CrawlResponse(
            page_url=payload.page_url,
            indexed_faces_count=len(records),
            status="completed",
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/feedback", response_model=FeedbackResponse)
def feedback(
    payload: FeedbackRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> FeedbackResponse:
    search_session = db.get(SearchSession, payload.search_id)
    if search_session is None:
        # For transient web searches, allow recording feedback gracefully
        return FeedbackResponse(
            search_id=payload.search_id,
            indexed_face_id=payload.indexed_face_id,
            is_match=payload.is_match,
            recorded_at=datetime.now(UTC),
            status="recorded",
        )

    expires_at = search_session.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    if expires_at < datetime.now(UTC):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Search session has expired.")

    target = db.get(IndexedFace, payload.indexed_face_id)
    if target is None:
        # Transient live web match
        return FeedbackResponse(
            search_id=payload.search_id,
            indexed_face_id=payload.indexed_face_id,
            is_match=payload.is_match,
            recorded_at=datetime.now(UTC),
            status="recorded",
        )
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
