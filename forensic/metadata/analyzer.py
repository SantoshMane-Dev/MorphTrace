from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS


def _format_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip("\x00")
    if isinstance(value, (tuple, list)):
        return " × ".join(str(part) for part in value)
    return str(value)


def analyze_metadata(image_path):
    path = Path(image_path)

    with Image.open(path) as image:
        info = image.info
        exif = {}

        for tag_id, value in image.getexif().items():
            tag_name = TAGS.get(tag_id, str(tag_id))

            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")

            exif[tag_name] = str(value)

        container_metadata = {
            "File size": f"{path.stat().st_size:,} bytes",
            "EXIF": "Present" if exif else "Not embedded",
            "XMP": "Present" if info.get("xmp") else "Not found",
            "ICC profile": (
                f"Present ({len(info['icc_profile']):,} bytes)"
                if info.get("icc_profile")
                else "Not embedded"
            ),
        }

        jfif_version = info.get("jfif_version")
        if jfif_version:
            container_metadata["JFIF version"] = ".".join(
                str(part) for part in jfif_version
            )

        jfif_density = info.get("jfif_density")
        if jfif_density:
            density_unit = {
                0: "unitless",
                1: "dpi",
                2: "dpcm",
            }.get(info.get("jfif_unit", 0), "unknown unit")
            container_metadata["JFIF density"] = (
                f"{_format_value(jfif_density)} {density_unit}"
            )

        dpi = info.get("dpi")
        if dpi:
            container_metadata["Resolution"] = f"{_format_value(dpi)} dpi"

        if "progressive" in info or "progression" in info:
            container_metadata["Progressive JPEG"] = (
                "Yes" if info.get("progressive") or info.get("progression") else "No"
            )

        comment = info.get("comment")
        if comment:
            container_metadata["Comment"] = _format_value(comment)[:200]

        text_metadata = getattr(image, "text", {})
        if isinstance(text_metadata, dict):
            for key, value in text_metadata.items():
                if isinstance(value, (str, bytes)):
                    container_metadata[str(key)] = _format_value(value)[:200]

        evidence = []

        if not exif:
            evidence.append({
                "type": "metadata",
                "finding": (
                    "No EXIF metadata was found; other container metadata "
                    "may still be present."
                ),
                "severity": "informational",
                "localized": False
            })

        software = exif.get("Software")

        if software:
            evidence.append({
                "type": "metadata",
                "finding": f"Software metadata detected: {software}",
                "severity": "informational",
                "localized": False
            })

        return {
            "filename": path.name,
            "format": image.format,
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "exif": exif,
            "container_metadata": container_metadata,
            "evidence": evidence
        }
