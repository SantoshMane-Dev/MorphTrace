from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageEnhance


def analyze_ela(image_path, output_path, quality=90):
    image_path = Path(image_path)
    output_path = Path(output_path)

    with Image.open(image_path) as original:
        original = original.convert("RGB")

        temp_path = output_path.with_suffix(".ela_temp.jpg")
        original.save(temp_path, "JPEG", quality=quality)

        with Image.open(temp_path) as recompressed:
            difference = ImageChops.difference(original, recompressed)

        extrema = difference.getextrema()
        max_difference = max(
            channel_max for _, channel_max in extrema
        )

        scale = 255.0 / max_difference if max_difference > 0 else 1.0

        ela_image = ImageEnhance.Brightness(difference).enhance(scale)
        ela_image.save(output_path)

        temp_path.unlink(missing_ok=True)

        array = np.asarray(difference, dtype=np.float32)

        mean_difference = float(array.mean())
        normalized_score = min(mean_difference / 255.0, 1.0)

        return {
            "technique": "ELA",
            "score": round(normalized_score, 4),
            "mean_difference": round(mean_difference, 4),
            "max_difference": int(max_difference),
            "output": str(output_path),
            "localized": True,
        }
