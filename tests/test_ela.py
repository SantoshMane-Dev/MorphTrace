
from PIL import Image

from forensic.ela.analyzer import analyze_ela


def test_ela_analysis(sample_image, tmp_path):
    output_path = tmp_path / "ela.png"

    result = analyze_ela(sample_image, output_path)

    assert result["technique"] == "ELA"
    assert 0 <= result["score"] <= 1
    assert result["localized"] is True
    assert output_path.is_file()
    assert result["applicable"] is True


def test_ela_does_not_score_png_recompression_as_manipulation(tmp_path):
    image_path = tmp_path / "source.png"
    output_path = tmp_path / "ela.png"
    Image.new("RGB", (64, 64), "blue").save(image_path)

    result = analyze_ela(image_path, output_path)

    assert result["applicable"] is False
    assert result["score"] == 0
    assert "not scored as evidence" in result["note"]
    assert output_path.is_file()
