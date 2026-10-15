from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "data" / "demo"
AUTHENTIC_DIR = DATASET_DIR / "Au"
MANIPULATED_DIR = DATASET_DIR / "Tp"


def make_authentic_image(seed: int, width: int = 256, height: int = 256):
    rng = np.random.default_rng(seed)
    array = np.zeros((height, width, 3), dtype=np.uint8)

    base_r = rng.integers(0, 255)
    base_g = rng.integers(0, 255)
    base_b = rng.integers(0, 255)

    for y in range(height):
        for x in range(width):
            array[y, x, 0] = (base_r + x // 3 + y // 4) % 256
            array[y, x, 1] = (base_g + y // 2) % 256
            array[y, x, 2] = (base_b + x // 5 + y // 6) % 256

    # Add a soft blended texture for realism.
    noise = rng.integers(0, 24, size=(height, width, 3), dtype=np.uint8)
    array = np.clip(array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def make_manipulated_image(seed: int, width: int = 256, height: int = 256):
    rng = np.random.default_rng(seed)
    image = make_authentic_image(seed)
    array = np.array(image)

    patch_w = width // 4
    patch_h = height // 4
    x0 = rng.integers(0, width - patch_w)
    y0 = rng.integers(0, height - patch_h)
    x1 = x0 + patch_w
    y1 = y0 + patch_h

    patch = array[y0:y1, x0:x1].copy()
    patch = np.clip(patch + 35, 0, 255).astype(np.uint8)
    array[y0:y1, x0:x1] = patch

    # Simulate a copy-move or local tamper by blending in a second region.
    x2 = (x0 + patch_w // 2) % width
    y2 = (y0 + patch_h // 2) % height
    x3 = min(width, x2 + patch_w)
    y3 = min(height, y2 + patch_h)
    array[y2:y3, x2:x3] = np.clip(array[y2:y3, x2:x3] - 20, 0, 255).astype(np.uint8)

    # Apply blur to one section to simulate post-processing.
    blurred = image.crop((x0, y0, x1, y1)).filter(ImageFilter.GaussianBlur(radius=2))
    array[y0:y1, x0:x1] = np.array(blurred)

    return Image.fromarray(array, mode="RGB")


def save_images():
    AUTHENTIC_DIR.mkdir(parents=True, exist_ok=True)
    MANIPULATED_DIR.mkdir(parents=True, exist_ok=True)

    for index in range(20):
        authentic = make_authentic_image(index * 13 + 1)
        authentic.save(AUTHENTIC_DIR / f"authentic_{index:02d}.png")

        manipulated = make_manipulated_image(index * 17 + 7)
        manipulated.save(MANIPULATED_DIR / f"manipulated_{index:02d}.png")

    print(f"Created synthetic dataset under {DATASET_DIR}")
    print(f"Authentic images: {len(list(AUTHENTIC_DIR.glob('*')))}")
    print(f"Manipulated images: {len(list(MANIPULATED_DIR.glob('*')))}")


if __name__ == "__main__":
    save_images()
