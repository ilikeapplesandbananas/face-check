"""Cosine search and confidence translation for public result cards."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import numpy as np
from qdrant_client import QdrantClient
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.models import FaceProfile, IndexedFace, SearchSession
from app.services.vector_index import validate_embedding


CONFIDENCE_THRESHOLD = 60.0


def cosine_to_confidence_percent(cosine_similarity: float) -> float:
    """Translate raw cosine similarity to a bounded, displayable 0–100 score.

    This is a similarity percentage, not a calibrated probability of identity.
    A 60% threshold means a raw cosine similarity of at least 0.60.
    """
    return round(max(0.0, min(1.0, float(cosine_similarity))) * 100, 2)


@dataclass(frozen=True)
class SearchMatch:
    indexed_face_id: UUID
    profile_id: UUID
    source_image_url: str
    page_context_url: str
    raw_cosine_similarity: float
    confidence_percent: float


@dataclass(frozen=True)
class SearchResult:
    search_id: UUID
    matches: list[SearchMatch]


class FaceSearchService:
    def __init__(self, session: Session, qdrant: QdrantClient, settings: Settings | None = None) -> None:
        self.session = session
        self.qdrant = qdrant
        self.settings = settings or get_settings()

    def search(self, embedding: np.ndarray, top_k: int) -> SearchResult:
        vector = validate_embedding(embedding)
        # Stored only for the 24-hour feedback window. Expired sessions cascade to
        # their feedback audit records at the database level.
        self.session.execute(delete(SearchSession).where(SearchSession.expires_at < datetime.now(UTC)))
        search_session = SearchSession(
            id=uuid4(),
            query_embedding=vector,
            expires_at=datetime.now(UTC) + timedelta(hours=24),
        )
        self.session.add(search_session)
        self.session.flush()

        # Search beyond the requested card count because multiple points may map to
        # the same profile, and enrichment/master points need a public source card.
        response = self.qdrant.query_points(
            collection_name=self.settings.qdrant_collection,
            query=vector,
            limit=max(top_k * 8, top_k),
            with_payload=True,
            with_vectors=False,
        )
        hits = response.points
        profile_ids = self._profile_ids(hits)
        representatives = self._public_representatives(profile_ids)
        results: dict[UUID, SearchMatch] = {}

        for hit in hits:
            confidence = cosine_to_confidence_percent(hit.score)
            if confidence < CONFIDENCE_THRESHOLD:
                continue
            payload = hit.payload or {}
            profile_id = self._uuid_payload(payload, "profile_id")
            if profile_id is None:
                continue
            face = self._face_from_hit_or_profile(payload, profile_id, representatives)
            if face is None:
                continue
            candidate = SearchMatch(
                indexed_face_id=face.id,
                profile_id=profile_id,
                source_image_url=face.source_image_url,
                page_context_url=face.page_context_url,
                raw_cosine_similarity=round(float(hit.score), 6),
                confidence_percent=confidence,
            )
            if profile_id not in results:
                results[profile_id] = candidate
            if len(results) == top_k:
                break

        matches = list(results.values())
        search_session.result_indexed_face_ids = [str(match.indexed_face_id) for match in matches]
        return SearchResult(search_id=search_session.id, matches=matches)

    def _profile_ids(self, hits: list[object]) -> set[UUID]:
        ids: set[UUID] = set()
        for hit in hits:
            profile_id = self._uuid_payload(getattr(hit, "payload", None) or {}, "profile_id")
            if profile_id:
                ids.add(profile_id)
        return ids

    def _public_representatives(self, profile_ids: set[UUID]) -> dict[UUID, IndexedFace]:
        if not profile_ids:
            return {}
        rows = self.session.scalars(
            select(IndexedFace)
            .where(IndexedFace.profile_id.in_(profile_ids))
            .order_by(IndexedFace.created_at.desc())
        ).all()
        representatives: dict[UUID, IndexedFace] = {}
        for row in rows:
            representatives.setdefault(row.profile_id, row)
        return representatives

    def _face_from_hit_or_profile(
        self, payload: dict[str, object], profile_id: UUID, representatives: dict[UUID, IndexedFace]
    ) -> IndexedFace | None:
        face_id = self._uuid_payload(payload, "indexed_face_id")
        if face_id:
            face = self.session.get(IndexedFace, face_id)
            if face is not None and face.profile_id == profile_id:
                return face
        return representatives.get(profile_id)

    @staticmethod
    def _uuid_payload(payload: dict[str, object], key: str) -> UUID | None:
        value = payload.get(key)
        if not isinstance(value, str):
            return None
        try:
            return UUID(value)
        except ValueError:
            return None
