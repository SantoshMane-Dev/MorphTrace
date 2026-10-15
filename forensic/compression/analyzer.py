from pathlib import Path

import cv2
import numpy as np


_DCT_FREQUENCIES = np.arange(8, dtype=np.float32)[:, None]
_DCT_POSITIONS = np.arange(8, dtype=np.float32)[None, :]
_DCT_BASIS = np.cos(
    np.pi * (2 * _DCT_POSITIONS + 1) * _DCT_FREQUENCIES / 16
).astype(np.float32)
_DCT_BASIS[0] /= np.sqrt(2)
_DCT_BASIS *= 0.5


def analyze_compression(image_path):
    image_path = Path(image_path)

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Unable to read image")

    height, width = image.shape

    usable_height = height - (height % 8)
    usable_width = width - (width % 8)

    if usable_height < 8 or usable_width < 8:
        return {
            "technique": "compression",
            "score": 0.0,
            "mean_high_frequency": 0.0,
            "suspicious_regions": [],
            "localized": False,
            "evidence_found": False,
        }

    cropped = image[:usable_height, :usable_width].astype(np.float32)

    blocks = cropped.reshape(
        usable_height // 8, 8, usable_width // 8, 8
    ).transpose(0, 2, 1, 3)
    dct_blocks = np.einsum(
        "ui,...ij,vj->...uv",
        _DCT_BASIS,
        blocks,
        _DCT_BASIS,
        optimize=True,
    )
    mean_high_frequency = float(np.abs(dct_blocks[..., 4:, 4:]).mean())

    # This is a heuristic forensic signal, not a probability.
    normalized_score = min(mean_high_frequency / 50.0, 1.0)

    return {
        "technique": "compression",
        "score": round(normalized_score, 4),
        "mean_high_frequency": round(mean_high_frequency, 4),
        "suspicious_regions": [],
        "localized": False,
        "evidence_found": normalized_score >= 0.30,
    }
