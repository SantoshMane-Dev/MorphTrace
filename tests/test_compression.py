from pathlib import Path

from forensic.compression.analyzer import analyze_compression


def test_compression_analysis():
    uploads = Path("uploads")

    source = next(
        path for path in uploads.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    result = analyze_compression(source)

    assert result["technique"] == "compression"
    assert 0 <= result["score"] <= 1
    assert "mean_high_frequency" in result
    assert "evidence_found" in result
