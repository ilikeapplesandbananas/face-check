from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import get_settings
from app.db.session import create_session_factory, initialize_schema
from app.services.face_pipeline import FacePipeline
from app.services.vector_index import create_qdrant_client, ensure_face_collection

logger = logging.getLogger("veriface")


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s: %s", request.url, exc)
    origin = request.headers.get("origin") or "*"
    return JSONResponse(
        status_code=500,
        content={"detail": f"Biometric search processing error: {str(exc)}"},
        headers={
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        },
    )


@app.get("/health", tags=["system"])
async def health_check() -> dict[str, str]:
    """Liveness endpoint used during local development and container orchestration."""
    return {"status": "ok", "service": "veriface-api"}
