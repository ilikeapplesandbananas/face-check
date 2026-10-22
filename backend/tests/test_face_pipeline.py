import numpy as np
import pytest

from app.services.face_pipeline import (
    ARC_FACE_112_TEMPLATE,
    DetectedFace,
    ImageQualityError,
    align_face,
    ensure_face_size,
    laplacian_variance,
)


def test_laplacian_variance_distinguishes_detail_from_flat_image() -> None:
    flat_image = np.full((160, 160, 3), 127, dtype=np.uint8)
    detailed_image = flat_image.copy()
    detailed_image[::4, :, :] = 255

    assert laplacian_variance(flat_image) == 0
    assert laplacian_variance(detailed_image) > 100


def test_face_size_gate_requires_both_dimensions() -> None:
    too_small = DetectedFace((0, 0, 111, 200), 0.99, ARC_FACE_112_TEMPLATE)
    with pytest.raises(ImageQualityError, match="too small"):
        ensure_face_size(too_small)


def test_alignment_maps_canonical_landmarks_to_112_crop() -> None:
    image = np.zeros((160, 160, 3), dtype=np.uint8)
    for x, y in ARC_FACE_112_TEMPLATE.astype(int):
        image[y, x] = (255, 255, 255)

    aligned = align_face(image, ARC_FACE_112_TEMPLATE)
    assert aligned.shape == (112, 112, 3)
    for x, y in ARC_FACE_112_TEMPLATE.astype(int):
        assert aligned[y, x].max() > 0
