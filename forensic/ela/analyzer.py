from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops


def analyze_ela(image_path, output_path, quality=90):
    image_path = Path(image_path)
    output_path = Path(output_path)

    with Image.open(image_path) as original:
        is_jpeg = original.format == "JPEG"
        original = original.convert("RGB")

        recompressed_buffer = BytesIO()
        original.save(recompressed_buffer, "JPEG", quality=quality)
        recompressed_buffer.seek(0)

        with Image.open(recompressed_buffer) as recompressed:
            difference = ImageChops.difference(original, recompressed)

        array = np.asarray(difference, dtype=np.float32)
        max_difference = int(array.max())
        display_scale = float(np.percentile(array, 99.5))
        if display_scale <= 0:
            display_scale = 1.0
        ela_array = np.clip(array * (255.0 / display_scale), 0, 255).astype(np.uint8)
        Image.fromarray(ela_array, mode="RGB").save(output_path)

        mean_difference = float(array.mean())
        normalized_score = min(mean_difference / 255.0, 1.0) if is_jpeg else 0.0

        return {
            "technique": "ELA",
            "score": round(normalized_score, 4),
            "mean_difference": round(mean_difference, 4),
            "max_difference": max_difference,
            "output": str(output_path),
            "localized": True,
            "applicable": is_jpeg,
            "note": (
                "ELA is most informative for JPEG images and remains a visual clue, not proof."
                if is_jpeg
                else "JPEG recompression map only; ELA is not scored as evidence for this non-JPEG image."
            ),
        }
