from pathlib import Path

import cv2
import numpy as np


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

    block_scores = []

    for y in range(0, usable_height, 8):
        for x in range(0, usable_width, 8):
            block = cropped[y:y + 8, x:x + 8]

            dct = cv2.dct(block)

            high_frequency = np.abs(dct[4:, 4:])
            score = float(np.mean(high_frequency))

            block_scores.append(score)

    mean_high_frequency = float(np.mean(block_scores))

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
