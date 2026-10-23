import os
from collections.abc import Generator
from pathlib import Path
from threading import Lock

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings
from app.db.models import Base

_engine: Engine | None = None
_engine_lock = Lock()


def create_database_engine(settings: Settings | None = None) -> Engine:
    global _engine
    if _engine is not None:
        return _engine

    with _engine_lock:
        if _engine is not None:
            return _engine

        config = settings or get_settings()
        try:
            eng = create_engine(config.database_url, pool_pre_ping=True)
            with eng.connect():
                pass
            _engine = eng
            return _engine
        except Exception:
            # Fallback to embedded SQLite with WAL mode & busy timeout to prevent "database is locked"
            os.makedirs("data", exist_ok=True)
            sqlite_args = {"timeout": 30.0, "check_same_thread": False}
            eng = create_engine(config.sqlite_url, connect_args=sqlite_args)

            @event.listens_for(eng, "connect")
            def _set_sqlite_pragma(dbapi_connection, _connection_record):
                try:
                    cursor = dbapi_connection.cursor()
                    cursor.execute("PRAGMA journal_mode=WAL")
                    cursor.execute("PRAGMA synchronous=NORMAL")
                    cursor.execute("PRAGMA busy_timeout=30000")
                    cursor.close()
                except Exception:
                    pass

            _engine = eng
            return _engine


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
