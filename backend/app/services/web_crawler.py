"""Web page face crawler and batch indexer.

Crawls a target webpage or profile, discovers all human images,
extracts and aligns every detected face, and indexes them into
PostgreSQL/SQLite metadata and Qdrant vector memory.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any
from urllib.parse import urljoin, urlparse

import cv2
import httpx
import numpy as np
from bs4 import BeautifulSoup
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.services.face_pipeline import FacePipeline, decode_image, align_face
from app.services.vector_index import FaceIndexer, IndexedFaceRecord

logger = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"


class WebCrawlerService:
    def __init__(self, session: Session, pipeline: FacePipeline, settings: Settings | None = None) -> None:
        self.session = session
        self.pipeline = pipeline
        self.settings = settings or get_settings()
        self.indexer = FaceIndexer(session=session, settings=self.settings)

    async def crawl_page(self, page_url: str, max_images: int = 50) -> list[IndexedFaceRecord]:
        """Fetch a page, extract all images, detect faces, and index them."""
        parsed = urlparse(page_url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid URL: {page_url}")

        image_urls = await self._discover_image_urls(page_url, max_images)
        logger.info("Found %d image candidates on %s", len(image_urls), page_url)

        indexed_records: list[IndexedFaceRecord] = []
        limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
        async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, limits=limits, follow_redirects=True) as client:
            for img_url in image_urls:
                try:
                    resp = await client.get(img_url, timeout=8.0)
                    if resp.status_code != 200 or len(resp.content) < 1000:
                        continue

                    # Check image and extract faces
                    image = decode_image(resp.content)
                    self.pipeline._load_models()
                    boxes, landmark_sets = self.pipeline._detector.detect(image, input_size=(640, 640), max_num=0)
                    if boxes is None or len(boxes) == 0:
                        continue

                    for box, landmarks in zip(boxes, landmark_sets, strict=True):
                        conf = float(box[4])
                        width = float(box[2] - box[0])
                        height = float(box[3] - box[1])
                        if conf < 0.65 or width < 80 or height < 80:
                            continue

                        aligned = align_face(image, np.asarray(landmarks, dtype=np.float32))
                        raw_emb = np.asarray(self.pipeline._recognizer.get_feat(aligned), dtype=np.float32).reshape(-1)
                        norm = float(np.linalg.norm(raw_emb))
                        if norm == 0 or not np.isfinite(norm):
                            continue

                        embedding = raw_emb / norm
                        record = self.indexer.index(
                            embedding=embedding,
                            source_image_url=img_url,
                            page_context_url=page_url,
                        )
                        indexed_records.append(record)
                except Exception as exc:
                    logger.debug("Failed processing image %s: %s", img_url, exc)
                    continue

        return indexed_records

    async def _discover_image_urls(self, page_url: str, max_images: int) -> list[str]:
        async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, follow_redirects=True) as client:
            resp = await client.get(page_url, timeout=12.0)
            if resp.status_code != 200:
                raise RuntimeError(f"Could not load webpage (HTTP {resp.status_code})")

            soup = BeautifulSoup(resp.text, "html.parser")
            found: set[str] = set()

            # 1. <img> tags
            for img in soup.find_all("img"):
                src = img.get("src") or img.get("data-src") or img.get("srcset")
                if src:
                    # If srcset, take first candidate
                    src_clean = src.split(",")[0].strip().split(" ")[0]
                    full_url = urljoin(page_url, src_clean)
                    if self._is_valid_image_url(full_url):
                        found.add(full_url)

            # 2. OpenGraph / Twitter meta image tags
            for meta in soup.find_all("meta"):
                prop = meta.get("property") or meta.get("name") or ""
                if prop in ("og:image", "twitter:image", "image"):
                    content = meta.get("content")
                    if content:
                        found.add(urljoin(page_url, content))

            return list(found)[:max_images]

    @staticmethod
    def _is_valid_image_url(url: str) -> bool:
        lower = url.lower()
        if any(skip in lower for skip in ("icon", "logo", "spinner", "badge", ".svg", "pixel.gif")):
            return False
        return any(ext in lower for ext in (".jpg", ".jpeg", ".png", ".webp")) or "image" in lower
