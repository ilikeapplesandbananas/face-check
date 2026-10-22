from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings
from app.db.session import create_session_factory, initialize_schema
from app.services.face_pipeline import FacePipeline
from app.services.vector_index import create_qdrant_client, ensure_face_collection


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    initialize_schema(settings)
    qdrant = create_qdrant_client(settings)
    ensure_face_collection(qdrant, settings.qdrant_collection)
    app.state.session_factory = create_session_factory(settings)
    app.state.qdrant = qdrant
    app.state.face_pipeline = FacePipeline(settings)
    yield


app = FastAPI(
    title="VeriFace API",
    version="0.1.0",
    description="Reverse face search with RetinaFace alignment, ArcFace embeddings, and verified vector enrichment.",
    lifespan=lifespan,
)

app.include_router(router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health", tags=["system"])
async def health_check() -> dict[str, str]:
    """Liveness endpoint used during local development and container orchestration."""
    return {"status": "ok", "service": "veriface-api"}
