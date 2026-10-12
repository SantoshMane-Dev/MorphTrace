
from forensic.ela.analyzer import analyze_ela


def test_ela_analysis(sample_image, tmp_path):
    output_path = tmp_path / "ela.png"

    result = analyze_ela(sample_image, output_path)

    assert result["technique"] == "ELA"
    assert 0 <= result["score"] <= 1
    assert result["localized"] is True
    assert output_path.is_file()
