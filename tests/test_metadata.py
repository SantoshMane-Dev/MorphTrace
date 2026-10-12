
from forensic.metadata.analyzer import analyze_metadata


def test_metadata_analysis(sample_image):
    result = analyze_metadata(sample_image)

    assert result["format"] == "JPEG"
    assert result["width"] == 256
    assert result["height"] == 256
    assert isinstance(result["exif"], dict)
    assert isinstance(result["evidence"], list)
