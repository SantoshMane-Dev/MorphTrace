from pathlib import Path

from forensic.copy_move.analyzer import analyze_copy_move


def test_copy_move_analysis():
    uploads = Path("uploads")

    source = next(
        path for path in uploads.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    output = Path("results/test_copy_move.png")

    result = analyze_copy_move(source, output)

    assert result["technique"] == "copy-move"
    assert 0 <= result["score"] <= 1
    assert result["localized"] is True
    assert output.exists()

    output.unlink()
