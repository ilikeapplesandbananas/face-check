from collections.abc import Generator

from fastapi import Request
from qdrant_client import QdrantClient
from sqlalchemy.orm import Session

from app.db.session import create_session_factory
from app.services.face_pipeline import FacePipeline
from app.services.vector_index import create_qdrant_client


def get_db(request: Request) -> Generator[Session, None, None]:
    session_factory = getattr(request.app.state, "session_factory", None)
    if session_factory is None:
        session_factory = create_session_factory()
        request.app.state.session_factory = session_factory

    with session_factory() as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise


def get_face_pipeline(request: Request) -> FacePipeline:
    pipeline = getattr(request.app.state, "face_pipeline", None)
    if pipeline is None:
        pipeline = FacePipeline()
        request.app.state.face_pipeline = pipeline
    return pipeline


def get_qdrant(request: Request) -> QdrantClient:
    qdrant = getattr(request.app.state, "qdrant", None)
    if qdrant is None:
        qdrant = create_qdrant_client()
        request.app.state.qdrant = qdrant
    return qdrant
