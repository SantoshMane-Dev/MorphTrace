from pathlib import Path

import cv2
import numpy as np


def analyze_noise(image_path):
    image_path = Path(image_path)

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Unable to read image")

    image = image.astype(np.float32)

    # Estimate the local noise residual by removing low-frequency structure.
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    residual = np.abs(image - blurred)

    mean_noise = float(np.mean(residual))

    # Divide the image into blocks and measure local noise variation.
    height, width = residual.shape
    block_size = 16

    block_scores = []

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = residual[
                y:min(y + block_size, height),
                x:min(x + block_size, width),
            ]

            if block.size == 0:
                continue

            block_scores.append(float(np.mean(block)))

    if not block_scores:
        return {
            "technique": "noise",
            "score": 0.0,
            "mean_noise": 0.0,
            "noise_variation": 0.0,
            "localized": False,
            "evidence_found": False,
        }

    noise_variation = float(np.std(block_scores))

    # This is a heuristic forensic signal, not a probability.
    normalized_score = min(noise_variation / 10.0, 1.0)

    return {
        "technique": "noise",
        "score": round(normalized_score, 4),
        "mean_noise": round(mean_noise, 4),
        "noise_variation": round(noise_variation, 4),
        "localized": False,
        "evidence_found": normalized_score >= 0.30,
    }
