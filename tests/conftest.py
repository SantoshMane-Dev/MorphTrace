
import numpy as np
import pytest
from PIL import Image, ImageDraw


@pytest.fixture
def sample_image(tmp_path):
    """Create a deterministic, textured image for analyzer tests."""
    rng = np.random.default_rng(42)

    pixels = rng.integers(
        0, 256, size=(256, 256, 3), dtype=np.uint8
    )
    image = Image.fromarray(pixels, mode="RGB")
    draw = ImageDraw.Draw(image)

    # Add structured features alongside the texture.
    for x in range(16, 256, 32):
        draw.rectangle(
            (x, 20, min(x + 15, 255), 235),
            outline=(255, 255, 255),
            width=2,
        )

    path = tmp_path / "sample.jpg"
    image.save(path, format="JPEG", quality=95)
    return path
