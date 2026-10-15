
import cv2
import numpy as np
import pytest

from forensic.compression.analyzer import analyze_compression


def test_compression_analysis(sample_image):
    result = analyze_compression(sample_image)

    assert result["technique"] == "compression"
    assert 0 <= result["score"] <= 1
    assert "mean_high_frequency" in result
    assert isinstance(result["evidence_found"], bool)


def test_compression_matches_block_dct_reference(sample_image):
    image = cv2.imread(str(sample_image), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    height, width = image.shape
    scores = []

    for y in range(0, height - height % 8, 8):
        for x in range(0, width - width % 8, 8):
            coefficients = cv2.dct(image[y:y + 8, x:x + 8])
            scores.append(float(np.abs(coefficients[4:, 4:]).mean()))

    result = analyze_compression(sample_image)

    assert result["mean_high_frequency"] == pytest.approx(
        round(float(np.mean(scores)), 4), abs=0.0001
    )
