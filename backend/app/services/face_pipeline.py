"""Strict RetinaFace -> five-point Umeyama alignment -> ArcFace embedding pipeline.

This module deliberately does not use a detector's convenience embedding method.
Every face is aligned from the five RetinaFace keypoints before being sent to
ArcFace, keeping the image geometry deterministic and auditable.

Enhanced with multi-scale scanning, EXIF auto-orientation, CLAHE shadow/contrast
boost, 4-way rotation invariance, and alpha compositing for maximum detection rate.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from PIL import Image, ImageOps
from insightface.model_zoo import get_model

from app.core.config import Settings, get_settings


# Canonical InsightFace 112x112 5-point template (from insightface/utils/face_align.py)
ARC_FACE_112_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose
        [41.5493, 92.3655],  # left mouth corner
        [70.7299, 92.2041],  # right mouth corner
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
    """Decode an uploaded image to BGR, handling EXIF orientation and alpha transparency."""
    if not image_bytes:
        raise FacePipelineError("The uploaded image is empty.")

    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img = ImageOps.exif_transpose(pil_img)
        if pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
            background = Image.new("RGB", pil_img.size, (255, 255, 255))
            if pil_img.mode != "RGBA":
                pil_img = pil_img.convert("RGBA")
            background.paste(pil_img, mask=pil_img.split()[3])
            pil_img = background
        elif pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
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


def ensure_image_quality(image: np.ndarray, threshold: float = 3.0) -> float:
    score = laplacian_variance(image)
    effective_min = max(float(threshold), 2.0)
    if score < effective_min:
        raise ImageQualityError(
            f"The image is too blurry or featureless for a face match (quality score {score:.1f}; minimum {effective_min:.1f})."
        )
    return score


def ensure_face_size(face: DetectedFace, min_size: int = 16) -> None:
    effective_min = max(int(min_size), 16)
    if face.width < effective_min or face.height < effective_min:
        raise ImageQualityError(
            f"The detected face is too small ({face.width:.0f}x{face.height:.0f}px; minimum {effective_min}x{effective_min}px)."
        )


def _umeyama_similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Closed-form 2D Umeyama similarity transform matching InsightFace's face_align.py."""
    num = src.shape[0]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    cov = (dst_demean.T @ src_demean) / num
    u, s, vt = np.linalg.svd(cov)
    d = np.ones((2,), dtype=np.float64)
    if np.linalg.det(cov) < 0:
        d[1] = -1.0

    r = u @ np.diag(d) @ vt
    var_src = (src_demean ** 2).sum() / num
    if var_src <= 1e-12:
        raise FacePipelineError("Degenerate facial landmarks (zero variance).")

    scale = float(np.sum(s * d) / var_src)
    t = dst_mean - scale * (r @ src_mean)

    transform = np.zeros((2, 3), dtype=np.float32)
    transform[:, :2] = scale * r
    transform[:, 2] = t
    return transform


def align_face(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Align a face to the canonical 112x112 ArcFace crop using 5 landmarks."""
    source = np.asarray(landmarks, dtype=np.float32)
    if source.shape != (5, 2) or not np.isfinite(source).all():
        raise FacePipelineError("RetinaFace did not return five valid facial landmarks.")

    transform = _umeyama_similarity_transform(source, ARC_FACE_112_TEMPLATE)

    return cv2.warpAffine(
        image,
        transform,
        (112, 112),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def facecheck_score(vec_a: np.ndarray, vec_b: np.ndarray) -> tuple[float, int]:
    """Return (cosine_similarity, 0_to_100_confidence_score) calibrated like FaceCheck.ID."""
    cos_sim = float(np.dot(vec_a, vec_b))
    if cos_sim <= 0.20:
        mapped = max(0.0, (cos_sim / 0.20) * 40.0)
    elif cos_sim <= 0.45:
        mapped = 40.0 + ((cos_sim - 0.20) / 0.25) * 40.0
    else:
        mapped = 80.0 + min(1.0, (cos_sim - 0.45) / 0.35) * 20.0
    return cos_sim, int(round(mapped))


def get_search_crop(image: np.ndarray, face: DetectedFace, pad_ratio: float = 0.35) -> bytes:
    """Create a natural, high-res padded head crop for external visual search engines."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = face.bounding_box
    pad_w, pad_h = (x2 - x1) * pad_ratio, (y2 - y1) * pad_ratio

    cx1, cy1 = max(0, int(x1 - pad_w)), max(0, int(y1 - pad_h))
    cx2, cy2 = min(w, int(x2 + pad_w)), min(h, int(y2 + pad_h))

    crop = image[cy1:cy2, cx1:cx2]
    _, buffer = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return buffer.tobytes()


class FacePipeline:
    """Thread-safe lazy loader for configured RetinaFace and ArcFace ONNX models."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._detector = None
        self._recognizer = None
        self._model_lock = Lock()
        self._detect_lock = Lock()

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
                self._detector.prepare(ctx_id=self._context_id(), nms_thresh=0.4, input_size=(640, 640))
            if self._recognizer is None:
                recognizer_path = self._require_model_file(self.settings.arcface_model_path, "ArcFace")
                self._recognizer = get_model(recognizer_path, providers=self.settings.provider_list)
                self._recognizer.prepare(ctx_id=self._context_id())

    def _context_id(self) -> int:
        """Use InsightFace's CPU context marker unless a CUDA provider is configured."""
        return 0 if any(provider.startswith("CUDA") for provider in self.settings.provider_list) else -1

    def _run_pass(self, img: np.ndarray, sizes: list[tuple[int, int]], thresh: float) -> list[DetectedFace]:
        """Run detector across each individual scale tuple with thread-safe threshold updates."""
        candidates: list[DetectedFace] = []
        for size in sizes:
            try:
                with self._detect_lock:
                    self._detector.det_thresh = thresh
                    boxes, landmark_sets = self._detector.detect(img, input_size=size, max_num=0)
                if boxes is None or len(boxes) == 0 or landmark_sets is None or len(landmark_sets) == 0:
                    continue
                for box, landmarks in zip(boxes, landmark_sets, strict=True):
                    candidates.append(
                        DetectedFace(
                            bounding_box=tuple(float(v) for v in box[:4]),
                            detection_confidence=float(box[4]),
                            landmarks=np.asarray(landmarks, dtype=np.float32),
                        )
                    )
                if candidates:
                    return candidates
            except Exception:
                continue
        return candidates

    def detect_all_faces_with_image(self, image: np.ndarray) -> tuple[list[DetectedFace], np.ndarray]:
        """Multi-pass detector returning all detected faces in the winning pass (sorted best-first)."""
        self._load_models()

        def _sort_faces(faces: list[DetectedFace]) -> list[DetectedFace]:
            return sorted(faces, key=lambda f: (f.detection_confidence, f.width * f.height), reverse=True)

        # Pass 1: Multi-scale detection with standard to relaxed thresholds
        multi_scales = [(640, 640), (1024, 1024), (320, 320)]
        for thresh in [0.40, 0.25]:
            candidates = self._run_pass(image, multi_scales, thresh)
            if candidates:
                return _sort_faces(candidates), image

        # Pass 2: High-resolution scales (distant or small faces in screenshots/4K images)
        hi_res_scales = [(1280, 1280), (1600, 1600), (480, 480)]
        for thresh in [0.28, 0.18]:
            candidates = self._run_pass(image, hi_res_scales, thresh)
            if candidates:
                return _sort_faces(candidates), image

        # Pass 3: CLAHE contrast boost for dark, backlit, or washed-out faces
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
            candidates = self._run_pass(enhanced, [(640, 640), (1024, 1024)], 0.20)
            if candidates:
                return _sort_faces(candidates), image
        except Exception:
            pass

        # Pass 4: 90 / 270 / 180 degree rotation invariance (sideways/upside-down photos)
        rotations = [
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
            cv2.ROTATE_180,
        ]
        for rot_code in rotations:
            try:
                rot_img = cv2.rotate(image, rot_code)
                candidates = self._run_pass(rot_img, [(640, 640), (1024, 1024)], 0.25)
                if candidates:
                    return _sort_faces(candidates), rot_img
            except Exception:
                pass

        # Pass 5: Ultra-low sensitivity fallback
        candidates = self._run_pass(image, [(640, 640), (960, 960), (1280, 1280)], 0.10)
        if candidates:
            return _sort_faces(candidates), image

        raise FacePipelineError(
            "No face was detected. Please ensure the photo contains a visible face, or try cropping closer to the subject."
        )

    def detect_primary_face_with_image(self, image: np.ndarray) -> tuple[DetectedFace, np.ndarray]:
        faces, working_image = self.detect_all_faces_with_image(image)
        return faces[0], working_image

    def detect_primary_face(self, image: np.ndarray) -> DetectedFace:
        face, _ = self.detect_primary_face_with_image(image)
        return face

    def detect_candidate_faces(self, image: np.ndarray) -> list[DetectedFace]:
        """Sensitive multi-face detector optimized for re-ranking web candidate photos and thumbnails."""
        self._load_models()
        faces = self._run_pass(image, [(640, 640), (320, 320)], thresh=0.22)
        if faces:
            return sorted(faces, key=lambda f: (f.detection_confidence, f.width * f.height), reverse=True)
        # Sensitive fallback for low-res thumbnails or angled faces
        fallback = self._run_pass(image, [(480, 480)], thresh=0.15)
        if fallback:
            return sorted(fallback, key=lambda f: (f.detection_confidence, f.width * f.height), reverse=True)
        return []

    def _embed_aligned_face(
        self, aligned: np.ndarray, face: DetectedFace, quality_score: float
    ) -> FaceEmbedding:
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

    def extract(self, image_bytes: bytes) -> FaceEmbedding:
        """Extract the primary face embedding (for user query uploads)."""
        image = decode_image(image_bytes)
        quality_score = ensure_image_quality(image, self.settings.blur_threshold)
        face, working_image = self.detect_primary_face_with_image(image)
        ensure_face_size(face, self.settings.min_face_size)
        aligned = align_face(working_image, face.landmarks)
        return self._embed_aligned_face(aligned, face, quality_score)

    def extract_all(self, image_bytes: bytes, max_faces: int = 10) -> list[FaceEmbedding]:
        """Extract embeddings for all valid faces in an image (for web candidate indexing)."""
        image = decode_image(image_bytes)
        quality_score = ensure_image_quality(image, self.settings.blur_threshold)
        faces, working_image = self.detect_all_faces_with_image(image)

        embeddings: list[FaceEmbedding] = []
        for face in faces[:max_faces]:
            try:
                ensure_face_size(face, self.settings.min_face_size)
                aligned = align_face(working_image, face.landmarks)
                embeddings.append(self._embed_aligned_face(aligned, face, quality_score))
            except FacePipelineError:
                continue

        if not embeddings:
            raise ImageQualityError("Detected faces did not meet minimum size/alignment requirements.")
        return embeddings
