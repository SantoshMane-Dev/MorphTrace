from pathlib import Path

from forensic.metadata.analyzer import analyze_metadata


def test_metadata_analysis():
    image_path = Path("tests/sample.jpg")

    result = analyze_metadata(image_path)

    assert result["format"] == "JPEG"
    assert result["width"] > 0
    assert result["height"] > 0
    assert "exif" in result
