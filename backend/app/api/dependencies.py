from collections.abc import Generator

from fastapi import Request
from qdrant_client import QdrantClient
from sqlalchemy.orm import Session

from app.services.face_pipeline import FacePipeline


def get_db(request: Request) -> Generator[Session, None, None]:
    session_factory = request.app.state.session_factory
    with session_factory() as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise


def get_face_pipeline(request: Request) -> FacePipeline:
    return request.app.state.face_pipeline


def get_qdrant(request: Request) -> QdrantClient:
    return request.app.state.qdrant
