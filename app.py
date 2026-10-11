
import logging
import re
from pathlib import Path
from uuid import uuid4

from flask import (
    Flask,
    abort,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from forensic.aggregation.analyzer import aggregate_evidence
from forensic.compression.analyzer import analyze_compression
from forensic.copy_move.analyzer import analyze_copy_move
from forensic.ela.analyzer import analyze_ela
from forensic.metadata.analyzer import analyze_metadata
from forensic.ml.classifier import predict_image
from forensic.noise.analyzer import analyze_noise


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
ALLOWED_IMAGE_FORMATS = {
    "jpg": "JPEG",
    "jpeg": "JPEG",
    "png": "PNG",
}

MAX_UPLOAD_BYTES = 16 * 1024 * 1024
UPLOAD_NAME_PATTERN = re.compile(r"^[0-9a-f]{32}\.(jpg|jpeg|png)$")


def create_app(test_config=None):
    app = Flask(__name__)

    app.config.from_mapping(
        MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
        MODEL_PATH=None,
    )

    if test_config:
        app.config.update(test_config)

    upload_dir = Path(app.config.get("UPLOAD_DIR", UPLOAD_DIR))
    results_dir = Path(app.config.get("RESULTS_DIR", RESULTS_DIR))

    upload_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO)

    @app.errorhandler(RequestEntityTooLarge)
    def handle_large_upload(_error):
        return "Image exceeds the 16 MB upload limit.", 413

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/upload")
    def upload():
        if "image" not in request.files:
            return "No image uploaded.", 400

        uploaded_file = request.files["image"]

        if not uploaded_file.filename:
            return "No file selected.", 400

        safe_name = secure_filename(uploaded_file.filename)

        if not safe_name or "." not in safe_name:
            return "Unsupported file type. Upload a JPG or PNG image.", 400

        extension = safe_name.rsplit(".", 1)[1].lower()

        if extension not in ALLOWED_EXTENSIONS:
            return "Unsupported file type. Upload a JPG or PNG image.", 400

        unique_name = f"{uuid4().hex}.{extension}"
        file_path = upload_dir / unique_name

        try:
            uploaded_file.save(file_path)

            # Validate the actual image content, not just its extension.
            with Image.open(file_path) as image:
                actual_format = image.format
                image.verify()

            expected_format = ALLOWED_IMAGE_FORMATS[extension]

            if actual_format != expected_format:
                file_path.unlink(missing_ok=True)
                return (
                    "The file content does not match its extension.",
                    400,
                )

        except (UnidentifiedImageError, OSError, ValueError) as error:
            file_path.unlink(missing_ok=True)
            app.logger.info("Rejected invalid image upload: %s", error)
            return "The uploaded file is not a valid image.", 400

        except Exception:
            file_path.unlink(missing_ok=True)
            app.logger.exception("Unexpected error while saving upload.")
            return "Unable to process the uploaded file.", 500

        return redirect(url_for("analyze", filename=unique_name))

    @app.get("/results/<filename>")
    def result_file(filename):
        # Only serve generated PNG analysis outputs.
        if not re.fullmatch(
            r"[0-9a-f]{32}_(ela|copy_move)\.png",
            filename,
        ):
            abort(404)

        if not (results_dir / filename).is_file():
            abort(404)

        return send_from_directory(results_dir, filename)

    @app.get("/analyze/<filename>")
    def analyze(filename):
        if not UPLOAD_NAME_PATTERN.fullmatch(filename):
            abort(404)

        file_path = upload_dir / filename

        if not file_path.is_file():
            abort(404)

        generated_files = [
            results_dir / f"{file_path.stem}_ela.png",
            results_dir / f"{file_path.stem}_copy_move.png",
        ]

        try:
            metadata = analyze_metadata(file_path)

            ela = analyze_ela(file_path, generated_files[0])

            copy_move = analyze_copy_move(
                file_path,
                generated_files[1],
            )

            compression = analyze_compression(file_path)
            noise = analyze_noise(file_path)
            ml = predict_image(file_path, model_path=app.config.get("MODEL_PATH"))

            assessment = aggregate_evidence(
                metadata,
                ela,
                copy_move,
                compression,
                noise,
            )

        except (UnidentifiedImageError, OSError, ValueError) as error:
            app.logger.warning(
                "Image analysis failed for %s: %s",
                filename,
                error,
            )
            return "Unable to analyze this image. Please upload a valid JPG or PNG.", 400

        except Exception:
            app.logger.exception(
                "Unexpected analysis failure for %s.",
                filename,
            )
            return "An internal error occurred during image analysis.", 500

        return render_template(
            "results.html",
            metadata=metadata,
            ela=ela,
            copy_move=copy_move,
            compression=compression,
            noise=noise,
            assessment=assessment,
            ml=ml,
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
