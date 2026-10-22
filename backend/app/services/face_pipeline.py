"""Strict RetinaFace -> five-point alignment -> ArcFace embedding pipeline.

This module deliberately does not use a detector's convenience embedding method.
Every face is aligned from the five RetinaFace keypoints before being sent to
ArcFace, keeping the image geometry deterministic and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from insightface.model_zoo import get_model

from app.core.config import Settings, get_settings


ARC_FACE_112_TEMPLATE = np.array(
    [
        [38.2946, 59.6963],  # left eye
        [73.5318, 59.5014],  # right eye
        [56.0252, 79.7366],  # nose
        [41.5493, 100.3655],  # left mouth corner
        [70.7299, 100.2041],  # right mouth corner
    ],
    dtype=np.float32,
)


class FacePipelineError(ValueError):
    """A safe, user-correctable failure in image intake or face inference."""


class ImageQualityError(FacePipelineError):
    """The supplied image cannot produce a reliable embedding."""


@dataclass(frozen=True)
class DetectedFace:
    bounding_box: tuple[float, float, float, float]
    detection_confidence: float
    landmarks: np.ndarray

    @property
    def width(self) -> float:
        return self.bounding_box[2] - self.bounding_box[0]

    @property
    def height(self) -> float:
        return self.bounding_box[3] - self.bounding_box[1]


@dataclass(frozen=True)
class FaceEmbedding:
    """Output retained for the later indexing/search and feedback steps."""

    vector: np.ndarray
    aligned_face: np.ndarray
    detected_face: DetectedFace
    laplacian_variance: float


def decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode an uploaded image to BGR, rejecting invalid and empty payloads."""
    if not image_bytes:
        raise FacePipelineError("The uploaded image is empty.")

    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise FacePipelineError("The uploaded file is not a supported image.")
    return image


def laplacian_variance(image: np.ndarray) -> float:
    """Return the standard blur metric used for the image-level quality gate."""
    if image is None or image.size == 0:
        raise ImageQualityError("The image cannot be evaluated for blur.")
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(grayscale, cv2.CV_64F).var())


def ensure_image_quality(image: np.ndarray, threshold: float) -> float:
    score = laplacian_variance(image)
    if score < threshold:
        raise ImageQualityError(
            f"The image is too blurry for a reliable face match (quality score {score:.1f}; minimum {threshold:.1f})."
        )
    return score


def ensure_face_size(face: DetectedFace, min_size: int = 112) -> None:
    if face.width < min_size or face.height < min_size:
        raise ImageQualityError(
            f"The detected face is too small ({face.width:.0f}x{face.height:.0f}px; minimum {min_size}x{min_size}px)."
        )


def align_face(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Align a face to the canonical 112x112 ArcFace crop using 5 landmarks."""
    source = np.asarray(landmarks, dtype=np.float32)
    if source.shape != (5, 2) or not np.isfinite(source).all():
        raise FacePipelineError("RetinaFace did not return five valid facial landmarks.")

    transform, _ = cv2.estimateAffinePartial2D(source, ARC_FACE_112_TEMPLATE, method=cv2.LMEDS)
    if transform is None:
        raise FacePipelineError("Face landmarks could not be aligned.")

    return cv2.warpAffine(
        image,
        transform,
        (112, 112),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


class FacePipeline:
    """Thread-safe lazy loader for configured RetinaFace and ArcFace ONNX models."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._detector = None
        self._recognizer = None
        self._model_lock = Lock()

    def _require_model_file(self, path: Path, label: str) -> str:
        if not path.is_file():
            raise RuntimeError(
                f"{label} model not found at '{path}'. Download the approved ONNX model and configure its path."
            )
        return str(path)

    def _load_models(self) -> None:
        if self._detector is not None and self._recognizer is not None:
            return

        with self._model_lock:
            if self._detector is None:
                detector_path = self._require_model_file(self.settings.retinaface_model_path, "RetinaFace")
                self._detector = get_model(detector_path, providers=self.settings.provider_list)
                self._detector.prepare(ctx_id=self._context_id(), nms_thresh=0.4)
            if self._recognizer is None:
                recognizer_path = self._require_model_file(self.settings.arcface_model_path, "ArcFace")
                self._recognizer = get_model(recognizer_path, providers=self.settings.provider_list)
                self._recognizer.prepare(ctx_id=self._context_id())

    def _context_id(self) -> int:
        """Use InsightFace's CPU context marker unless a CUDA provider is configured."""
        return 0 if any(provider.startswith("CUDA") for provider in self.settings.provider_list) else -1

    def detect_primary_face(self, image: np.ndarray) -> DetectedFace:
        self._load_models()
        boxes, landmark_sets = self._detector.detect(image, input_size=(640, 640), max_num=0)
        if boxes is None or len(boxes) == 0 or landmark_sets is None or len(landmark_sets) == 0:
            raise FacePipelineError("No face was detected in the image.")

        # A single embedding is required for search. Select the most confident detection,
        # then use face area as a deterministic tie-breaker.
        candidates: list[DetectedFace] = []
        for box, landmarks in zip(boxes, landmark_sets, strict=True):
            candidates.append(
                DetectedFace(
                    bounding_box=tuple(float(value) for value in box[:4]),
                    detection_confidence=float(box[4]),
                    landmarks=np.asarray(landmarks, dtype=np.float32),
                )
            )
        return max(candidates, key=lambda face: (face.detection_confidence, face.width * face.height))

    def extract(self, image_bytes: bytes) -> FaceEmbedding:
        """Apply quality gate, RetinaFace detection/alignment, then ArcFace embedding."""
        image = decode_image(image_bytes)
        quality_score = ensure_image_quality(image, self.settings.blur_threshold)
        face = self.detect_primary_face(image)
        ensure_face_size(face, self.settings.min_face_size)
        aligned = align_face(image, face.landmarks)

        raw_embedding = np.asarray(self._recognizer.get_feat(aligned), dtype=np.float32).reshape(-1)
        if raw_embedding.shape != (512,):
            raise RuntimeError(f"ArcFace must output a 512D embedding; got shape {raw_embedding.shape}.")
        magnitude = float(np.linalg.norm(raw_embedding))
        if not np.isfinite(magnitude) or magnitude == 0:
            raise RuntimeError("ArcFace returned an invalid embedding.")

        return FaceEmbedding(
            vector=raw_embedding / magnitude,
            aligned_face=aligned,
            detected_face=face,
            laplacian_variance=quality_score,
        )
