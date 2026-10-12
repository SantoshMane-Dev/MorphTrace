from io import BytesIO
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance


FEATURE_VERSION = 1
IMAGE_SIZE = (256, 256)


def _prepare_image(image_source):
    """Normalize image-like inputs into an RGB PIL image."""
    if isinstance(image_source, Image.Image):
        return image_source.convert("RGB")

    if isinstance(image_source, np.ndarray):
        array = np.asarray(image_source)

        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        elif array.ndim == 3 and array.shape[2] == 1:
            array = np.repeat(array, 3, axis=2)
        elif array.ndim == 3 and array.shape[2] == 4:
            array = array[:, :, :3]

        if array.dtype != np.uint8:
            if np.max(array) <= 1.0:
                array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
            else:
                array = np.clip(array, 0, 255).astype(np.uint8)

        return Image.fromarray(array, mode="RGB")

    if isinstance(image_source, (str, os.PathLike)):
        return Image.open(Path(image_source)).convert("RGB")

    raise TypeError(
        "Image input must be a file path, pathlib.Path, PIL image, "
        "or a NumPy array."
    )


def _extract_feature_vector(source):
    """Compute the feature vector for a PIL image."""
    rgb = source.convert("RGB")
    resized = rgb.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

    rgb_array = np.asarray(resized, dtype=np.float32) / 255.0
    gray = cv2.cvtColor(
        (rgb_array * 255).astype(np.uint8),
        cv2.COLOR_RGB2GRAY,
    )

    features = []

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

    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    edges = cv2.Canny(gray, 100, 200)

    features.extend([
        float(laplacian.var()),
        float(np.mean(np.abs(laplacian))),
        float(np.mean(edges > 0)),
    ])

    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    residual = gray.astype(np.float32) - denoised.astype(np.float32)

    features.extend([
        float(residual.mean()),
        float(residual.std()),
        float(np.mean(np.abs(residual))),
        float(np.percentile(np.abs(residual), 90)),
    ])

    block_variances = []
    height, width = gray.shape

    for row in range(8):
        for column in range(8):
            y0, y1 = row * height // 8, (row + 1) * height // 8
            x0, x1 = column * width // 8, (column + 1) * width // 8
            block = gray[y0:y1, x0:x1]
            block_variances.append(float(block.var()))

    features.extend(block_variances)

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
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result


def extract_features(image_source):
    """Extract image statistics and forensic indicators."""
    if isinstance(image_source, (str, os.PathLike)):
        image_path = Path(image_source)
        with Image.open(image_path) as source:
            result = _extract_feature_vector(source)
        if not np.all(np.isfinite(result)):
            raise ValueError(f"Invalid feature values for image: {image_path.name}")
        return result

    source = _prepare_image(image_source)
    result = _extract_feature_vector(source)

    if not np.all(np.isfinite(result)):
        raise ValueError("Invalid feature values for image input.")

    return result