from pathlib import Path

import joblib

from forensic.ml.features import FEATURE_VERSION, extract_features


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "manipulation_classifier.joblib"


def predict_image(image_path, model_path=None):
    """Classify an image using a previously trained model."""
    model_path = Path(model_path or DEFAULT_MODEL_PATH)

    if not model_path.is_file():
        return {
            "available": False,
            "message": (
                "The ML model has not been trained yet. "
                "Train the model before using ML classification."
            ),
        }

    artifact = joblib.load(model_path)

    if not isinstance(artifact, dict):
        raise ValueError("Invalid ML model artifact.")

    if artifact.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Model feature version does not match the extractor.")

    model = artifact.get("model")
    if model is None:
        raise ValueError("ML model artifact does not contain a classifier.")

    features = extract_features(image_path).reshape(1, -1)

    predicted_class = str(model.predict(features)[0])
    class_scores = model.predict_proba(features)[0]

    scores = {
        str(label): round(float(score) * 100, 2)
        for label, score in zip(model.classes_, class_scores)
    }

    return {
        "available": True,
        "predicted_class": predicted_class,
        "model_scores": scores,
        "model_score_percent": scores.get(predicted_class, 0.0),
        "score_interpretation": (
            "Uncalibrated model score; not a validated probability "
            "of image manipulation."
        ),
    }