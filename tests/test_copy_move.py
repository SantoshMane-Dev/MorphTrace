
from forensic.copy_move.analyzer import analyze_copy_move


def test_copy_move_analysis(sample_image, tmp_path):
    output_path = tmp_path / "copy_move.png"

    result = analyze_copy_move(sample_image, output_path)

    assert result["technique"] == "copy-move"
    assert 0 <= result["score"] <= 1
    assert result["matches"] >= 0
    assert isinstance(result["suspicious_regions"], list)
    assert output_path.is_file()
