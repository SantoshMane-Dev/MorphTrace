
from PIL import ExifTags, Image, PngImagePlugin

from forensic.metadata.analyzer import analyze_metadata


def test_metadata_analysis(sample_image):
    result = analyze_metadata(sample_image)

    assert result["format"] == "JPEG"
    assert result["width"] == 256
    assert result["height"] == 256
    assert isinstance(result["exif"], dict)
    assert isinstance(result["evidence"], list)
    assert result["container_metadata"]["EXIF"] == "Not embedded"
    assert result["container_metadata"]["File size"].endswith("bytes")


def test_metadata_reads_nested_exif_and_gps(tmp_path):
    exif = Image.Exif()
    exif[270] = "Visible description"
    exif[34665] = {36867: "2025:01:02 03:04:05", 37386: 50}
    exif[34853] = {
        1: "N",
        2: (37.0, 46.0, 29.0),
        3: "W",
        4: (122.0, 25.0, 10.0),
    }
    image_path = tmp_path / "nested.jpg"
    Image.new("RGB", (80, 60), "white").save(image_path, exif=exif)

    result = analyze_metadata(image_path)

    assert result["exif"]["ImageDescription"] == "Visible description"
    assert result["exif"]["EXIF DateTimeOriginal"] == "2025:01:02 03:04:05"
    assert "GPS GPSLatitude" in result["exif"]
    assert "GPS GPSLongitude" in result["exif"]


def test_metadata_reads_png_text_and_xmp(tmp_path):
    png_info = PngImagePlugin.PngInfo()
    png_info.add_text("Author", "MorphTrace test author")
    png_info.add_itxt("XML:com.adobe.xmp", "<x:xmpmeta>sample</x:xmpmeta>")
    image_path = tmp_path / "metadata.png"
    Image.new("RGB", (80, 60), "white").save(image_path, pnginfo=png_info)

    result = analyze_metadata(image_path)

    assert result["container_metadata"]["Author"] == "MorphTrace test author"
    assert result["container_metadata"]["XMP"] == "Present"
