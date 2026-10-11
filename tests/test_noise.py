
from forensic.noise.analyzer import analyze_noise


def test_noise_analysis(sample_image):
    result = analyze_noise(sample_image)

    assert result["technique"] == "noise"
    assert 0 <= result["score"] <= 1
    assert "mean_noise" in result
    assert "noise_variation" in result
    assert isinstance(result["evidence_found"], bool)
