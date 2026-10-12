
from forensic.compression.analyzer import analyze_compression


def test_compression_analysis(sample_image):
    result = analyze_compression(sample_image)

    assert result["technique"] == "compression"
    assert 0 <= result["score"] <= 1
    assert "mean_high_frequency" in result
    assert isinstance(result["evidence_found"], bool)
