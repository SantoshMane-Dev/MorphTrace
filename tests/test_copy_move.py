
import numpy as np
from PIL import Image, ImageDraw

from forensic.copy_move.analyzer import analyze_copy_move


def test_copy_move_analysis(sample_image, tmp_path):
    output_path = tmp_path / "copy_move.png"

    result = analyze_copy_move(sample_image, output_path)

    assert result["technique"] == "copy-move"
    assert 0 <= result["score"] <= 1
    assert result["matches"] >= 0
    assert isinstance(result["suspicious_regions"], list)
    assert output_path.is_file()


def test_copy_move_finds_a_duplicated_textured_region(tmp_path):
    rng = np.random.default_rng(13)
    pixels = rng.integers(0, 256, (600, 800, 3), dtype=np.uint8)
    pixels[280:480, 480:680] = pixels[90:290, 80:280].copy()
    image_path = tmp_path / "copied.png"
    output_path = tmp_path / "copy_move.png"
    Image.fromarray(pixels).save(image_path)

    result = analyze_copy_move(image_path, output_path)

    assert result["matches"] >= 8
    assert result["score"] > 0
    assert result["suspicious_regions"]


def test_copy_move_rejects_unrelated_random_features(tmp_path):
    rng = np.random.default_rng(13)
    pixels = rng.integers(0, 256, (600, 800, 3), dtype=np.uint8)
    image_path = tmp_path / "clean.png"
    output_path = tmp_path / "copy_move.png"
    Image.fromarray(pixels).save(image_path)

    result = analyze_copy_move(image_path, output_path)

    assert result["matches"] < 8
    assert result["score"] == 0


def test_copy_move_rejects_clean_repetitive_patterns(tmp_path):
    image = Image.new("RGB", (640, 480), "white")
    draw = ImageDraw.Draw(image)
    for y in range(20, 480, 40):
        for x in range(20, 640, 40):
            draw.ellipse((x - 10, y - 10, x + 10, y + 10), outline="black", width=3)
            draw.line((x - 7, y, x + 7, y), fill="black", width=2)
    image_path = tmp_path / "repeated.png"
    output_path = tmp_path / "copy_move.png"
    image.save(image_path)

    result = analyze_copy_move(image_path, output_path)

    assert result["matches"] < 8
    assert result["score"] == 0
