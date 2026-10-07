from pathlib import Path

from forensic.noise.analyzer import analyze_noise


def test_noise_analysis():
    uploads = Path("uploads")

    source = next(
        path for path in uploads.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    result = analyze_noise(source)

    assert result["technique"] == "noise"
    assert 0 <= result["score"] <= 1
    assert "mean_noise" in result
    assert "noise_variation" in result
    assert "evidence_found" in result
