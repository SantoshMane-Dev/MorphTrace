import argparse
from pathlib import Path

import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from forensic.ml.features import FEATURE_VERSION, extract_features


BASE_DIR = Path(__file__).resolve().parents[1]
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def collect_images(dataset_dir):
    samples = []

    class_directories = {
        "authentic": dataset_dir / "Au",
        "manipulated": dataset_dir / "Tp",
    }

    for label, directory in class_directories.items():
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Dataset folder not found: {directory}"
            )

        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                samples.append((path, label))

    counts = {
        label: sum(1 for _, item_label in samples if item_label == label)
        for label in class_directories
    }

    if min(counts.values()) < 10:
        raise ValueError(
            "At least 10 readable images per class are required. "
            f"Found: {counts}"
        )

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Train the MorphTrace image manipulation classifier."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=BASE_DIR / "data" / "casia",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "models" / "manipulation_classifier.joblib",
    )
    args = parser.parse_args()

    samples = collect_images(args.dataset)
    features = []
    labels = []
    failures = []

    for index, (path, label) in enumerate(samples, start=1):
        try:
            vector = extract_features(path)
            features.append(vector)
            labels.append(label)
        except Exception as error:
            failures.append((path.name, str(error)))

        if index % 100 == 0:
            print(f"Processed {index}/{len(samples)} images")

    if failures:
        print(f"Skipped {len(failures)} unreadable images.")
        for filename, error in failures[:10]:
            print(f"  {filename}: {error}")

    if len(set(labels)) != 2:
        raise ValueError("Both classes must contain readable images.")

    X = np.vstack(features)
    y = np.asarray(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )

    print(f"Training on {len(y_train)} images...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\n========== MORPHTRACE ML EVALUATION ==========")
    print(f"Training images: {len(y_train)}")
    print(f"Test images: {len(y_test)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(
        "Manipulated-image recall: "
        f"{recall_score(y_test, predictions, pos_label='manipulated'):.4f}"
    )
    print(
        "Manipulated-image F1: "
        f"{f1_score(y_test, predictions, pos_label='manipulated'):.4f}"
    )

    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    print("Confusion matrix (rows=true, columns=predicted)")
    print("Label order: authentic, manipulated")
    print(confusion_matrix(
        y_test,
        predictions,
        labels=["authentic", "manipulated"],
    ))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "feature_version": FEATURE_VERSION,
            "classes": list(model.classes_),
            "training_samples": len(y_train),
            "test_samples": len(y_test),
        },
        args.output,
    )

    print(f"\nModel saved to: {args.output}")
    print(
        "NOTE: These metrics are an initial baseline. "
        "Related images across train/test splits can inflate performance. "
        "Validate on independent images before relying on predictions."
    )


if __name__ == "__main__":
    main()
