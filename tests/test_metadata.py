from pathlib import Path

from forensic.metadata.analyzer import analyze_metadata


def test_metadata_analysis():
    uploads = Path("uploads")

    source = next(
        path for path in uploads.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    result = analyze_metadata(source)

    assert result["format"] in {"JPEG", "PNG"}
    assert result["width"] > 0
    assert result["height"] > 0
    assert "exif" in result
    assert "evidence" in result
