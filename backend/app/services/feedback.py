"""Background confirmation processing and master-vector enrichment."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from uuid import UUID, uuid4

import numpy as np
from qdrant_client.http import models as qmodels

from app.db.models import FaceProfile, FeedbackEmbedding, FeedbackEvent, IndexedFace, SearchSession
from app.db.session import create_session_factory
from app.services.vector_index import create_qdrant_client, ensure_face_collection, validate_embedding


logger = logging.getLogger(__name__)


def process_confirmed_feedback(feedback_event_id: UUID) -> None:
    """Run after a verified Yes response; it is idempotent per feedback event."""
    session_factory = create_session_factory()
    qdrant = create_qdrant_client()
    with session_factory() as session:
        event = session.get(FeedbackEvent, feedback_event_id)
        if event is None or not event.is_match or event.processed_at is not None:
            return
        search = session.get(SearchSession, event.search_session_id)
        target = session.get(IndexedFace, event.target_indexed_face_id)
        if search is None or target is None or search.expires_at < datetime.now(UTC):
            logger.warning("Feedback event %s has no usable search session or target.", feedback_event_id)
            return

        profile = session.get(FaceProfile, target.profile_id)
        if profile is None:
            logger.warning("Feedback event %s refers to a missing profile.", feedback_event_id)
            return

        # Settings are read inside the task because FastAPI BackgroundTasks runs after
        # the request lifecycle and must not reuse request-scoped state.
        from app.core.config import get_settings

        collection = get_settings().qdrant_collection
        ensure_face_collection(qdrant, collection)
        sample = np.asarray(validate_embedding(np.asarray(search.query_embedding)), dtype=np.float32)
        sample_vector_id = uuid4()
        master_vector_id = profile.master_vector_id or uuid4()

        try:
            previous_master = _load_master(qdrant, collection, profile.master_vector_id)
            count = profile.confirmed_sample_count
            if previous_master is None:
                count = 0
                master = sample
            else:
                master = _normalize((previous_master * count) + sample)

            qdrant.upsert(
                collection_name=collection,
                points=[
                    qmodels.PointStruct(
                        id=str(sample_vector_id),
                        vector=sample.tolist(),
                        payload={
                            "kind": "confirmed_feedback_sample",
                            "profile_id": str(profile.id),
                            "feedback_event_id": str(event.id),
                        },
                    ),
                    qmodels.PointStruct(
                        id=str(master_vector_id),
                        vector=master.tolist(),
                        payload={"kind": "master_vector", "profile_id": str(profile.id)},
                    ),
                ],
                wait=True,
            )
            session.add(
                FeedbackEmbedding(
                    id=uuid4(),
                    feedback_event_id=event.id,
                    profile_id=profile.id,
                    vector_id=sample_vector_id,
                )
            )
            profile.master_vector_id = master_vector_id
            profile.confirmed_sample_count = count + 1
            event.processed_at = datetime.now(UTC)
            session.commit()
        except Exception:
            session.rollback()
            logger.exception("Could not process confirmed feedback event %s", feedback_event_id)
            raise


def _load_master(qdrant: object, collection: str, vector_id: UUID | None) -> np.ndarray | None:
    if vector_id is None:
        return None
    points = qdrant.retrieve(collection_name=collection, ids=[str(vector_id)], with_vectors=True)
    if not points or points[0].vector is None:
        return None
    vector = points[0].vector
    if isinstance(vector, dict):
        return None
    return np.asarray(validate_embedding(np.asarray(vector)), dtype=np.float32)


def _normalize(vector: np.ndarray) -> np.ndarray:
    magnitude = float(np.linalg.norm(vector))
    if magnitude == 0:
        raise ValueError("Cannot create a zero master vector.")
    return vector / magnitude
