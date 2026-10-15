
import io

import joblib
import numpy as np
import pytest
from PIL import Image
from sklearn.dummy import DummyClassifier

from app import create_app
from forensic.ml.features import FEATURE_VERSION, extract_features


@pytest.fixture
def client(tmp_path):
    app = create_app({
        "TESTING": True,
        "UPLOAD_DIR": str(tmp_path / "uploads"),
        "RESULTS_DIR": str(tmp_path / "results"),
        "DATASET_DIR": str(tmp_path / "dataset"),
        "MAX_CONTENT_LENGTH": 16 * 1024 * 1024,
        "TRAINING_TOOLS_ENABLED": True,
    })
    return app.test_client()


def make_image(image_format="JPEG"):
    image = Image.new("RGB", (128, 128), color="red")
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    buffer.seek(0)
    return buffer


def test_homepage_loads(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Format converter" in response.data
    assert b"Image forensics" in response.data
    assert b"Training tools" not in response.data
    assert b"IMAGE INTEGRITY / FILE UTILITIES" not in response.data
    assert b"LOCAL WORKSPACE" not in response.data


def test_inspect_page_loads(client):
    response = client.get("/inspect")
    assert response.status_code == 200
    assert b"Drop a frame for review" in response.data


def test_roadmap_page_loads(client):
    response = client.get("/roadmap")
    assert response.status_code == 200
    assert b"Video integrity monitor" in response.data


def test_converter_page_loads(client):
    response = client.get("/convert")
    assert response.status_code == 200
    assert b"Convert and download" in response.data
    assert b"PDF" in response.data


def test_training_workspace_loads(client):
    response = client.get("/training")
    assert response.status_code == 200
    assert b"Add original images" in response.data
    assert b"Add manipulated images" in response.data


def test_training_tools_are_hidden_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("MORPHTRACE_ENABLE_TRAINING_TOOLS", raising=False)
    app = create_app({
        "TESTING": True,
        "UPLOAD_DIR": str(tmp_path / "uploads"),
        "RESULTS_DIR": str(tmp_path / "results"),
        "DATASET_DIR": str(tmp_path / "dataset"),
    })
    client = app.test_client()

    assert client.get("/training").status_code == 404
    assert client.post("/dataset/upload/original").status_code == 404


def test_converter_converts_jpeg_to_png(client):
    response = client.post(
        "/convert",
        data={"image": (make_image(), "sample.jpg"), "output_format": "png"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.mimetype == "image/png"
    assert Image.open(io.BytesIO(response.data)).format == "PNG"


def test_converter_converts_webp_to_jpeg(client):
    response = client.post(
        "/convert",
        data={"image": (make_image("WEBP"), "sample.webp"), "output_format": "jpg"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.mimetype == "image/jpeg"
    assert Image.open(io.BytesIO(response.data)).format == "JPEG"


def test_converter_exports_image_as_pdf(client):
    response = client.post(
        "/convert",
        data={"image": (make_image(), "sample.jpg"), "output_format": "pdf"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.mimetype == "application/pdf"
    assert response.data.startswith(b"%PDF")


def test_converter_rejects_extension_content_mismatch(client):
    response = client.post(
        "/convert",
        data={"image": (make_image("PNG"), "sample.jpg"), "output_format": "webp"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400


def test_upload_accepts_valid_jpeg(client):
    response = client.post(
        "/upload",
        data={"image": (make_image(), "sample.jpg")},
        content_type="multipart/form-data",
        follow_redirects=False,
    )

    assert response.status_code == 302
    assert "/analyze/" in response.headers["Location"]


def test_upload_rejects_unsupported_extension(client):
    response = client.post(
        "/upload",
        data={
            "image": (io.BytesIO(b"not an image"), "sample.gif")
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400


def test_upload_rejects_invalid_image_content(client):
    response = client.post(
        "/upload",
        data={
            "image": (io.BytesIO(b"not an image"), "sample.jpg")
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400


def test_dataset_upload_stores_valid_images_in_selected_class(client, tmp_path):
    response = client.post(
        "/dataset/upload/original",
        data={"images": [(make_image(), "source.jpg")]},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.json["stored"] == 1
    assert response.json["count"] == 1
    stored_files = list((tmp_path / "dataset" / "Au").glob("*.jpg"))
    assert len(stored_files) == 1


def test_dataset_upload_rejects_extension_content_mismatch(client, tmp_path):
    response = client.post(
        "/dataset/upload/manipulated",
        data={"images": [(make_image(), "source.png")]},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert not list((tmp_path / "dataset" / "Tp").glob("*"))


def test_analysis_rejects_invalid_filename(client):
    response = client.get("/analyze/../../etc/passwd")
    assert response.status_code in (400, 404)


def test_analysis_rejects_missing_upload(client):
    response = client.get("/analyze/" + "a" * 32 + ".jpg")
    assert response.status_code == 404


def test_extract_features_accepts_pil_image():
    image = Image.new("RGB", (32, 32), color="blue")
    features = extract_features(image)

    assert features.shape == (505,)
    assert np.isfinite(features).all()


def test_analysis_includes_ml_result_when_model_exists(tmp_path):
    model_path = tmp_path / "manipulation_classifier.joblib"
    model = DummyClassifier(strategy="most_frequent")
    model.fit(
        np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]]),
        np.array(["authentic", "manipulated", "authentic", "manipulated"]),
    )
    joblib.dump({"model": model, "feature_version": FEATURE_VERSION}, model_path)

    app = create_app({
        "TESTING": True,
        "UPLOAD_DIR": str(tmp_path / "uploads"),
        "RESULTS_DIR": str(tmp_path / "results"),
        "MODEL_PATH": str(model_path),
        "MAX_CONTENT_LENGTH": 16 * 1024 * 1024,
    })
    client = app.test_client()

    upload_response = client.post(
        "/upload",
        data={"image": (make_image(), "sample.jpg")},
        content_type="multipart/form-data",
        follow_redirects=False,
    )

    analysis_path = upload_response.headers["Location"]
    response = client.get(analysis_path)

    assert response.status_code == 200
    assert b"ML insight" in response.data
    assert b"Predicted class" in response.data
    assert b"Mean pixel difference" in response.data


def test_ml_manipulation_flag_prevents_authentic_verdict(tmp_path):
    model_path = tmp_path / "manipulation_classifier.joblib"
    model = DummyClassifier(strategy="constant", constant="manipulated")
    model.fit(
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        np.array(["authentic", "manipulated"]),
    )
    joblib.dump({"model": model, "feature_version": FEATURE_VERSION}, model_path)

    app = create_app({
        "TESTING": True,
        "UPLOAD_DIR": str(tmp_path / "uploads"),
        "RESULTS_DIR": str(tmp_path / "results"),
        "MODEL_PATH": str(model_path),
        "MAX_CONTENT_LENGTH": 16 * 1024 * 1024,
    })
    client = app.test_client()

    upload_response = client.post(
        "/upload",
        data={"image": (make_image(), "sample.jpg")},
        content_type="multipart/form-data",
        follow_redirects=False,
    )

    response = client.get(upload_response.headers["Location"])

    assert response.status_code == 200
    assert b"Fake" in response.data
    assert b"result-overlay verdict-manipulated" in response.data
    assert b"result-highlight--manipulated" in response.data


def test_authentic_ml_result_uses_authentic_result_style(tmp_path):
    model_path = tmp_path / "authentic_classifier.joblib"
    model = DummyClassifier(strategy="constant", constant="authentic")
    model.fit(
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        np.array(["authentic", "authentic"]),
    )
    joblib.dump({"model": model, "feature_version": FEATURE_VERSION}, model_path)

    app = create_app({
        "TESTING": True,
        "UPLOAD_DIR": str(tmp_path / "uploads"),
        "RESULTS_DIR": str(tmp_path / "results"),
        "MODEL_PATH": str(model_path),
        "MAX_CONTENT_LENGTH": 16 * 1024 * 1024,
    })
    client = app.test_client()
    upload_response = client.post(
        "/upload",
        data={"image": (make_image(), "sample.jpg")},
        content_type="multipart/form-data",
        follow_redirects=False,
    )

    response = client.get(upload_response.headers["Location"])

    assert response.status_code == 200
    assert b"Real" in response.data
    assert b"result-overlay verdict-authentic" in response.data
    assert b"result-highlight--authentic" in response.data


def test_strong_copy_move_signal_returns_fake_verdict(tmp_path):
    rng = np.random.default_rng(13)
    pixels = rng.integers(0, 256, (600, 800, 3), dtype=np.uint8)
    pixels[280:480, 480:680] = pixels[90:290, 80:280].copy()
    image_buffer = io.BytesIO()
    Image.fromarray(pixels).save(image_buffer, format="JPEG", quality=94)
    image_buffer.seek(0)

    app = create_app({
        "TESTING": True,
        "UPLOAD_DIR": str(tmp_path / "uploads"),
        "RESULTS_DIR": str(tmp_path / "results"),
        "MODEL_PATH": str(tmp_path / "no-model.joblib"),
        "MAX_CONTENT_LENGTH": 16 * 1024 * 1024,
    })
    client = app.test_client()
    upload_response = client.post(
        "/upload",
        data={"image": (image_buffer, "copied-region.jpg")},
        content_type="multipart/form-data",
        follow_redirects=False,
    )

    response = client.get(upload_response.headers["Location"])

    assert response.status_code == 200
    assert b"Fake" in response.data
    assert b"result-overlay verdict-manipulated" in response.data


def test_stale_model_is_unavailable_and_weak_signals_return_real(tmp_path):
    model_path = tmp_path / "stale-model.joblib"
    model = DummyClassifier(strategy="constant", constant="authentic")
    model.fit(np.array([[0.0], [1.0]]), np.array(["authentic", "authentic"]))
    joblib.dump({"model": model, "feature_version": FEATURE_VERSION - 1}, model_path)

    app = create_app({
        "TESTING": True,
        "UPLOAD_DIR": str(tmp_path / "uploads"),
        "RESULTS_DIR": str(tmp_path / "results"),
        "MODEL_PATH": str(model_path),
        "MAX_CONTENT_LENGTH": 16 * 1024 * 1024,
    })
    client = app.test_client()

    upload_response = client.post(
        "/upload",
        data={"image": (make_image(), "sample.jpg")},
        content_type="multipart/form-data",
        follow_redirects=False,
    )
    response = client.get(upload_response.headers["Location"])

    assert response.status_code == 200
    assert b"Real" in response.data
    assert b"outdated features" in response.data
