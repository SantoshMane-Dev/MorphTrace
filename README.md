# MorphTrace

MorphTrace is a Flask interface for the image detector in `aiOrNot.py`.

## Run locally

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000.

## Image analysis flow

1. The visitor selects a JPG, JPEG, or PNG image on `/inspect`.
2. Flask validates the upload and saves it under `uploads/`.
3. Flask calls the existing `analyze()` function from `aiOrNot.py`.
4. The result page displays the detector's `final_result`, `final_label_confidence`, and returned optional details/warnings.

`aiOrNot.py` is retained as provided. Its network requests and response handling are not rewritten by MorphTrace. The result page does not calculate a second classification or invent confidence values.

The original `static/css/style.css`, `static/js/app.js`, homepage HTML, and image assets are retained from the supplied project. The CSS has not been redesigned or replaced.

## Notes

- Upload limit: 16 MB.
- Supported detector uploads: JPG, JPEG, PNG.
- Uploaded files remain in `uploads/`; remove them periodically or add a retention policy before public deployment.
- The detector must be reachable from the machine running Flask.
