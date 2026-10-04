from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS


def analyze_metadata(image_path):
    path = Path(image_path)

    with Image.open(path) as image:
        exif = {}

        for tag_id, value in image.getexif().items():
            tag_name = TAGS.get(tag_id, str(tag_id))

            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")

            exif[tag_name] = str(value)

        evidence = []

        if not exif:
            evidence.append({
                "type": "metadata",
                "finding": "No EXIF metadata was found",
                "severity": "informational",
                "localized": False
            })

        software = exif.get("Software")

        if software:
            evidence.append({
                "type": "metadata",
                "finding": f"Software metadata detected: {software}",
                "severity": "review",
                "localized": False
            })

        return {
            "filename": path.name,
            "format": image.format,
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "exif": exif,
            "evidence": evidence
        }
