import logging
import re
from pathlib import Path
from uuid import uuid4

from flask import Flask, abort, redirect, render_template, request, send_from_directory, url_for
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from aiOrNot import analyze as analyze_with_ai_or_not

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
MAX_UPLOAD_BYTES = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
EXPECTED_FORMATS = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG"}
UPLOAD_NAME_PATTERN = re.compile(r"^[0-9a-f]{32}\.(jpg|jpeg|png)$")


def create_app(test_config=None):
    app = Flask(__name__)
    app.config.from_mapping(
        MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
        UPLOAD_DIR=UPLOAD_DIR,
        DETECTOR_FUNCTION=analyze_with_ai_or_not,
    )
    if test_config:
        app.config.update(test_config)

    upload_dir = Path(app.config["UPLOAD_DIR"])
    upload_dir.mkdir(parents=True, exist_ok=True)
    # The upload form uses XHR; it follows the redirect once before the browser
    # navigates to that same result URL. Cache each response to avoid two API jobs.
    result_cache = {}
    logging.basicConfig(level=logging.INFO)

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        return "Image exceeds the 16 MB upload limit.", 413

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/inspect")
    def inspect():
        return render_template("inspect.html")

    @app.get("/roadmap")
    def roadmap():
        return render_template("roadmap.html")

    @app.route("/convert", methods=["GET", "POST"])
    def convert():
        if request.method == "GET":
            return render_template("convert.html")
        uploaded = request.files.get("image")
        if not uploaded or not uploaded.filename:
            return render_template("convert.html", error="Choose an image to convert."), 400
        safe_name = secure_filename(uploaded.filename)
        input_ext = Path(safe_name).suffix.lower()
        output_format = request.form.get("output_format", "png").lower()
        output_map = {"png": ("PNG", "png"), "jpg": ("JPEG", "jpg"), "webp": ("WEBP", "webp"), "pdf": ("PDF", "pdf")}
        if input_ext not in {".jpg", ".jpeg", ".png", ".webp"} or output_format not in output_map:
            return render_template("convert.html", error="Choose a supported image and output format."), 400
        try:
            with Image.open(uploaded.stream) as source:
                image = ImageOps.exif_transpose(source)
                target, extension = output_map[output_format]
                if target in {"JPEG", "PDF"}:
                    rgba = image.convert("RGBA")
                    flattened = Image.new("RGB", rgba.size, "white")
                    flattened.paste(rgba, mask=rgba.getchannel("A"))
                    image = flattened
                else:
                    image = image.copy()
                from io import BytesIO
                output = BytesIO()
                image.save(output, format=target)
                output.seek(0)
            from flask import send_file
            mimetype = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp", "PDF": "application/pdf"}[target]
            return send_file(output, mimetype=mimetype, as_attachment=True, download_name=f"{Path(safe_name).stem}.{extension}")
        except (UnidentifiedImageError, OSError, ValueError):
            return render_template("convert.html", error="That file could not be read as an image."), 400

    @app.post("/upload")
    def upload():
        uploaded = request.files.get("image")
        if uploaded is None or not uploaded.filename:
            return "Choose an image before continuing.", 400
        safe_name = secure_filename(uploaded.filename)
        extension = Path(safe_name).suffix.lower().lstrip(".")
        if extension not in ALLOWED_EXTENSIONS:
            return "Unsupported file type. Upload a JPG or PNG image.", 400

        filename = f"{uuid4().hex}.{extension}"
        destination = upload_dir / filename
        try:
            uploaded.save(destination)
            with Image.open(destination) as image:
                actual_format = image.format
                image.verify()
            if actual_format != EXPECTED_FORMATS[extension]:
                destination.unlink(missing_ok=True)
                return "The image content does not match its file extension.", 400
        except (UnidentifiedImageError, OSError, ValueError):
            destination.unlink(missing_ok=True)
            return "The uploaded file is not a valid image.", 400
        return redirect(url_for("detector_result", filename=filename))

    @app.get("/uploads/<filename>")
    def uploaded_file(filename):
        if not UPLOAD_NAME_PATTERN.fullmatch(filename):
            abort(404)
        if not (upload_dir / filename).is_file():
            abort(404)
        return send_from_directory(upload_dir, filename)

    @app.get("/ai-result/<filename>")
    def detector_result(filename):
        if not UPLOAD_NAME_PATTERN.fullmatch(filename):
            abort(404)
        image_path = upload_dir / filename
        if not image_path.is_file():
            abort(404)

        image_url = url_for("uploaded_file", filename=filename)
        cached = result_cache.get(filename)
        if cached is not None:
            return render_template("result.html", image_url=image_url, **cached)

        try:
            # This is the same function used by the standalone aiOrNot.py script.
            response = app.config["DETECTOR_FUNCTION"](str(image_path))
            if not isinstance(response, dict):
                raise RuntimeError("The detector returned an unexpected response.")
            details = response.get("result_details") or {}
            if not isinstance(details, dict):
                details = {}
            if details.get("error_message"):
                raise RuntimeError(str(details["error_message"]))
            verdict = details.get("final_result") or response.get("final_result") or response.get("label")
            if verdict is None:
                verdict = response.get("result")
            if verdict is None:
                raise RuntimeError("The detector response did not contain a result.")

            confidence = details.get("final_label_confidence")
            if confidence is None:
                confidence = details.get("confidence")
            warnings = details.get("warnings") or []
            checks = []
            for key, label in (("metadata", "Metadata"), ("ocr", "OCR"), ("synthid", "SynthID")):
                value = details.get(key)
                if value not in (None, [], ""):
                    checks.append({"label": label, "value": value})

            result_data = {
                "verdict": verdict,
                "confidence": confidence,
                "warnings": warnings if isinstance(warnings, list) else [],
                "checks": checks,
                "ml_model": details.get("ml_model"),
                "raw_score": response.get("result"),
            }
        except Exception as error:
            app.logger.exception("Image detector failed for %s", filename)
            result_data = {"error_message": str(error)}

        result_cache[filename] = result_data
        status_code = 502 if "error_message" in result_data else 200
        return render_template("result.html", image_url=image_url, **result_data), status_code

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
