from pathlib import Path

from PIL import Image
from PIL.ExifTags import GPSTAGS, IFD, TAGS


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
        raw_exif = image.getexif()

        for tag_id, value in raw_exif.items():
            tag_name = TAGS.get(tag_id, str(tag_id))
            if tag_name in {"ExifOffset", "GPSInfo"}:
                continue

            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")

            exif[tag_name] = str(value)

        for ifd_id, prefix, tag_names in (
            (IFD.Exif, "EXIF", TAGS),
            (IFD.GPSInfo, "GPS", GPSTAGS),
        ):
            try:
                ifd_values = raw_exif.get_ifd(ifd_id)
            except (KeyError, OSError, TypeError, ValueError):
                continue

            for tag_id, value in ifd_values.items():
                tag_name = tag_names.get(tag_id, str(tag_id))
                exif[f"{prefix} {tag_name}"] = _format_value(value)

        xmp_present = any(
            "xmp" in str(key).casefold() and bool(value)
            for key, value in info.items()
        )

        container_metadata = {
            "File size": f"{path.stat().st_size:,} bytes",
            "EXIF": "Present" if exif else "Not embedded",
            "XMP": "Present" if xmp_present else "Not found",
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
