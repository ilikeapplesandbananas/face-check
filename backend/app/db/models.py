from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, Text, UniqueConstraint, Uuid, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class FaceProfile(Base):
    """The durable identity record shared by one or more indexed face samples."""

    __tablename__ = "face_profiles"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    master_vector_id: Mapped[UUID | None] = mapped_column(Uuid, nullable=True, unique=True)
    confirmed_sample_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    indexed_faces: Mapped[list[IndexedFace]] = relationship(back_populates="profile", cascade="all, delete-orphan")


class IndexedFace(Base):
    """PostgreSQL metadata for an individual Qdrant vector point."""

    __tablename__ = "indexed_faces"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    profile_id: Mapped[UUID] = mapped_column(ForeignKey("face_profiles.id", ondelete="CASCADE"), index=True)
    vector_id: Mapped[UUID] = mapped_column(Uuid, unique=True, index=True, nullable=False)
    source_image_url: Mapped[str] = mapped_column(Text, nullable=False)
    page_context_url: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    profile: Mapped[FaceProfile] = relationship(back_populates="indexed_faces")


class SearchSession(Base):
    """Short-lived, consent-bound search vector retained solely for feedback processing."""

    __tablename__ = "search_sessions"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    query_embedding: Mapped[list[float]] = mapped_column(JSON, nullable=False)
    result_indexed_face_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)


class FeedbackEvent(Base):
    """Auditable verification decision; positive events queue vector enrichment."""

    __tablename__ = "feedback_events"
    __table_args__ = (
        UniqueConstraint("search_session_id", "target_indexed_face_id", name="uq_feedback_per_result"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    search_session_id: Mapped[UUID] = mapped_column(ForeignKey("search_sessions.id", ondelete="CASCADE"), index=True)
    target_indexed_face_id: Mapped[UUID] = mapped_column(ForeignKey("indexed_faces.id", ondelete="CASCADE"), index=True)
    is_match: Mapped[bool] = mapped_column(nullable=False)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class FeedbackEmbedding(Base):
    """One confirmed query embedding, linked to the profile it enriches."""

    __tablename__ = "feedback_embeddings"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    feedback_event_id: Mapped[UUID] = mapped_column(ForeignKey("feedback_events.id", ondelete="CASCADE"), unique=True)
    profile_id: Mapped[UUID] = mapped_column(ForeignKey("face_profiles.id", ondelete="CASCADE"), index=True)
    vector_id: Mapped[UUID] = mapped_column(Uuid, unique=True, index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
