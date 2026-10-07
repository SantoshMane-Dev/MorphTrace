from pathlib import Path

from forensic.ela.analyzer import analyze_ela


def test_ela_analysis():
    image_path = Path("uploads")

    source = next(
        path for path in image_path.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    output_path = Path("results/test_ela.png")

    result = analyze_ela(source, output_path)

    assert result["technique"] == "ELA"
    assert 0 <= result["score"] <= 1
    assert result["localized"] is True
    assert output_path.exists()

    output_path.unlink()
