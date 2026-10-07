from pathlib import Path
from uuid import uuid4

from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from forensic.aggregation.analyzer import aggregate_evidence
from forensic.metadata.analyzer import analyze_metadata
from forensic.ela.analyzer import analyze_ela
from forensic.copy_move.analyzer import analyze_copy_move
from forensic.compression.analyzer import analyze_compression
from forensic.noise.analyzer import analyze_noise


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


def create_app():
    app = Flask(__name__)

    UPLOAD_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/upload")
    def upload():
        if "image" not in request.files:
            return "No image uploaded", 400

        file = request.files["image"]

        if file.filename == "":
            return "No file selected", 400

        filename = secure_filename(file.filename)

        if "." not in filename:
            return "Unsupported file type", 400

        extension = filename.rsplit(".", 1)[1].lower()

        if extension not in ALLOWED_EXTENSIONS:
            return "Unsupported file type", 400

        unique_name = f"{uuid4().hex}.{extension}"
        file_path = UPLOAD_DIR / unique_name
        file.save(file_path)

        return redirect(url_for("analyze", filename=unique_name))

    @app.get("/results/<filename>")
    def result_file(filename):
        file_path = RESULTS_DIR / filename

        if not file_path.exists():
            return "Result file not found", 404

        return send_from_directory(RESULTS_DIR, filename)


    @app.get("/analyze/<filename>")
    def analyze(filename):
        file_path = UPLOAD_DIR / filename

        if not file_path.exists():
            return "Image not found", 404

        try:
            metadata = analyze_metadata(file_path)

            ela_path = RESULTS_DIR / f"{file_path.stem}_ela.png"
            ela = analyze_ela(file_path, ela_path)

            copy_move_path = RESULTS_DIR / f"{file_path.stem}_copy_move.png"
            copy_move = analyze_copy_move(
                file_path,
                copy_move_path
            )

            compression = analyze_compression(file_path)
            noise = analyze_noise(file_path)

            aggregation = aggregate_evidence(
                metadata,
                ela,
                copy_move,
                compression,
                noise
            )

        except Exception:
            return "Unable to analyze image", 400

        return render_template(
            "results.html",
            metadata=metadata,
            ela=ela,
            copy_move=copy_move,
            compression=compression,
            noise=noise,
            assessment=aggregation,
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
