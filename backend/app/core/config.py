from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the fixed face-inference pipeline."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="VERIFACE_",
        extra="ignore",
    )

    retinaface_model_path: Path = Field(default=Path("models/retinaface_r50_v1.onnx"))
    arcface_model_path: Path = Field(default=Path("models/arcface_r100_v1.onnx"))
    onnx_providers: str = Field(default="CPUExecutionProvider")
    blur_threshold: float = Field(default=100.0, gt=0)
    min_face_size: int = Field(default=112, ge=112)
    database_url: str = Field(
        default="postgresql+psycopg://veriface:veriface-development-only@localhost:5432/veriface"
    )
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_collection: str = Field(default="face_embeddings", min_length=1)
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1)

    @property
    def provider_list(self) -> list[str]:
        providers = [provider.strip() for provider in self.onnx_providers.split(",")]
        return [provider for provider in providers if provider]


@lru_cache
def get_settings() -> Settings:
    return Settings()
