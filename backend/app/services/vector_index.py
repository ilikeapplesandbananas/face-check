"""Qdrant collection provisioning and consistent PostgreSQL/Qdrant indexing."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID, uuid4

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.models import FaceProfile, IndexedFace


EMBEDDING_DIMENSION = 512


@dataclass(frozen=True)
class IndexedFaceRecord:
    profile_id: UUID
    indexed_face_id: UUID
    vector_id: UUID


def create_qdrant_client(settings: Settings | None = None) -> QdrantClient:
    config = settings or get_settings()
    return QdrantClient(url=config.qdrant_url)


def ensure_face_collection(client: QdrantClient, collection_name: str) -> None:
    """Provision the fixed 512D cosine collection with an explicit HNSW index."""
    if client.collection_exists(collection_name):
        collection = client.get_collection(collection_name)
        vectors = collection.config.params.vectors
        if not isinstance(vectors, qmodels.VectorParams) or vectors.size != EMBEDDING_DIMENSION:
            raise RuntimeError(f"Qdrant collection '{collection_name}' does not have a 512D dense vector schema.")
        if vectors.distance != qmodels.Distance.COSINE:
            raise RuntimeError(f"Qdrant collection '{collection_name}' must use cosine distance.")
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(
            size=EMBEDDING_DIMENSION,
            distance=qmodels.Distance.COSINE,
            hnsw_config=qmodels.HnswConfigDiff(m=16, ef_construct=200, full_scan_threshold=10_000),
        ),
    )


def validate_embedding(vector: np.ndarray) -> list[float]:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    if values.shape != (EMBEDDING_DIMENSION,):
        raise ValueError(f"Face vector must contain {EMBEDDING_DIMENSION} dimensions; got {values.shape}.")
    if not np.isfinite(values).all():
        raise ValueError("Face vector contains non-finite values.")
    magnitude = float(np.linalg.norm(values))
    if magnitude == 0:
        raise ValueError("Face vector cannot be zero.")
    # Qdrant's COSINE distance normalizes vectors server-side. Normalize here too
    # so later averaging/master-vector operations always start with a valid sample.
    return (values / magnitude).tolist()


class FaceIndexer:
    """Atomically-as-practical writer across PostgreSQL and Qdrant.

    PostgreSQL is the metadata authority. A Qdrant write failure rolls back its
    pending metadata transaction; a rare metadata commit failure removes the
    newly inserted vector point as compensation.
    """

    def __init__(
        self,
        session: Session,
        qdrant: QdrantClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.session = session
        self.qdrant = qdrant or create_qdrant_client(self.settings)
        ensure_face_collection(self.qdrant, self.settings.qdrant_collection)

    def index(
        self,
        embedding: np.ndarray,
        source_image_url: str,
        page_context_url: str,
        profile_id: UUID | None = None,
    ) -> IndexedFaceRecord:
        if not source_image_url or not page_context_url:
            raise ValueError("Both source_image_url and page_context_url are required.")

        normalized_vector = validate_embedding(embedding)
        profile = self._get_or_create_profile(profile_id)
        vector_id = uuid4()
        indexed_face = IndexedFace(
            id=uuid4(),
            profile_id=profile.id,
            vector_id=vector_id,
            source_image_url=source_image_url,
            page_context_url=page_context_url,
        )
        self.session.add(indexed_face)
        self.session.flush()

        try:
            self.qdrant.upsert(
                collection_name=self.settings.qdrant_collection,
                points=[
                    qmodels.PointStruct(
                        id=str(vector_id),
                        vector=normalized_vector,
                        payload={
                            "profile_id": str(profile.id),
                            "indexed_face_id": str(indexed_face.id),
                        },
                    )
                ],
                wait=True,
            )
            self.session.commit()
        except Exception:
            self.session.rollback()
            try:
                self.qdrant.delete(
                    collection_name=self.settings.qdrant_collection,
                    points_selector=qmodels.PointIdsList(points=[str(vector_id)]),
                    wait=True,
                )
            except Exception:
                # A reconciliation job can safely delete a Qdrant point that has no
                # matching indexed_faces.vector_id row. Do not hide the original error.
                pass
            raise

        return IndexedFaceRecord(profile.id, indexed_face.id, vector_id)

    def _get_or_create_profile(self, profile_id: UUID | None) -> FaceProfile:
        if profile_id is None:
            profile = FaceProfile(id=uuid4())
            self.session.add(profile)
            self.session.flush()
            return profile

        profile = self.session.get(FaceProfile, profile_id)
        if profile is None:
            raise ValueError(f"Face profile '{profile_id}' does not exist.")
        return profile
