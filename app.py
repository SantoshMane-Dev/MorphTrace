from pathlib import Path
from uuid import uuid4

from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


def create_app():
    app = Flask(__name__)

    UPLOAD_DIR.mkdir(exist_ok=True)

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
        file.save(UPLOAD_DIR / unique_name)

        return redirect(url_for("index"))

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
