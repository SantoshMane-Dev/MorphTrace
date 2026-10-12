from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance


FEATURE_VERSION = 1
IMAGE_SIZE = (256, 256)


def extract_features(image_path):
    """Extract image statistics and forensic indicators."""
    image_path = Path(image_path)

    with Image.open(image_path) as source:
        rgb = source.convert("RGB")
        resized = rgb.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

        rgb_array = np.asarray(resized, dtype=np.float32) / 255.0
        gray = cv2.cvtColor(
            (rgb_array * 255).astype(np.uint8),
            cv2.COLOR_RGB2GRAY,
        )

        features = []

        # Global color statistics.
        for channel in range(3):
            values = rgb_array[:, :, channel]
            features.extend([
                float(values.mean()),
                float(values.std()),
                float(np.percentile(values, 10)),
                float(np.percentile(values, 90)),
            ])

            histogram, _ = np.histogram(
                values,
                bins=16,
                range=(0.0, 1.0),
                density=False,
            )
            histogram = histogram.astype(np.float32)
            histogram /= max(float(histogram.sum()), 1.0)
            features.extend(histogram.tolist())

        # Grayscale distribution.
        gray_float = gray.astype(np.float32) / 255.0

        features.extend([
            float(gray_float.mean()),
            float(gray_float.std()),
        ])

        histogram, _ = np.histogram(
            gray_float,
            bins=32,
            range=(0.0, 1.0),
        )
        histogram = histogram.astype(np.float32)
        histogram /= max(float(histogram.sum()), 1.0)
        features.extend(histogram.tolist())

        # Edge and high-frequency statistics.
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        edges = cv2.Canny(gray, 100, 200)

        features.extend([
            float(laplacian.var()),
            float(np.mean(np.abs(laplacian))),
            float(np.mean(edges > 0)),
        ])

        # Noise residual statistics.
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)
        residual = gray.astype(np.float32) - denoised.astype(np.float32)

        features.extend([
            float(residual.mean()),
            float(residual.std()),
            float(np.mean(np.abs(residual))),
            float(np.percentile(np.abs(residual), 90)),
        ])

        # Local block variation.
        block_variances = []
        height, width = gray.shape

        for row in range(8):
            for column in range(8):
                y0, y1 = row * height // 8, (row + 1) * height // 8
                x0, x1 = column * width // 8, (column + 1) * width // 8

                block = gray[y0:y1, x0:x1]
                block_variances.append(float(block.var()))

        features.extend(block_variances)

        # Error Level Analysis statistics.
        buffer = BytesIO()
        resized.save(buffer, format="JPEG", quality=90)
        buffer.seek(0)

        with Image.open(buffer) as recompressed:
            difference = ImageChops.difference(resized, recompressed)
            difference = ImageEnhance.Brightness(difference).enhance(1.0)
            difference_array = np.asarray(
                difference,
                dtype=np.float32,
            )

        ela_gray = difference_array.mean(axis=2)

        features.extend([
            float(ela_gray.mean()),
            float(ela_gray.std()),
            float(ela_gray.max()),
            float(np.percentile(ela_gray, 95)),
        ])

        ela_histogram, _ = np.histogram(
            ela_gray,
            bins=16,
            range=(0.0, 256.0),
        )
        ela_histogram = ela_histogram.astype(np.float32)
        ela_histogram /= max(float(ela_histogram.sum()), 1.0)
        features.extend(ela_histogram.tolist())

    result = np.asarray(features, dtype=np.float32)

    if not np.all(np.isfinite(result)):
        raise ValueError(f"Invalid feature values for image: {image_path.name}")

    return result