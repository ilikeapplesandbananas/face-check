from collections.abc import Generator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings
from app.db.models import Base


def create_database_engine(settings: Settings | None = None) -> Engine:
    config = settings or get_settings()
    return create_engine(config.database_url, pool_pre_ping=True)


def create_session_factory(settings: Settings | None = None) -> sessionmaker[Session]:
    return sessionmaker(bind=create_database_engine(settings), autoflush=False, expire_on_commit=False)


def initialize_schema(settings: Settings | None = None) -> None:
    """Create the metadata schema; migrations replace this in a managed deployment."""
    engine = create_database_engine(settings)
    Base.metadata.create_all(engine)


def session_scope(settings: Settings | None = None) -> Generator[Session, None, None]:
    factory = create_session_factory(settings)
    with factory() as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
