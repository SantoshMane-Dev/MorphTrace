import argparse
import re
import sys
from pathlib import Path

import numpy as np
import joblib

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from forensic.ml.features import FEATURE_VERSION, extract_features


BASE_DIR = Path(__file__).resolve().parents[1]
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
MIN_TRAINING_SAMPLES_PER_CLASS = 50
MIN_VALIDATION_BALANCED_ACCURACY = 0.65


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


def is_generated_demo_dataset(samples):
    expected_stems = {
        "authentic": re.compile(r"^authentic_\d{2}$"),
        "manipulated": re.compile(r"^manipulated_\d{2}$"),
    }
    return bool(samples) and all(
        expected_stems[label].fullmatch(path.stem)
        for path, label in samples
    )


def build_models():
    extra_trees = ExtraTreesClassifier(
        n_estimators=400,
        class_weight="balanced",
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    random_forest = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    rbf_svm = CalibratedClassifierCV(
        estimator=make_pipeline(
            StandardScaler(),
            SVC(
                C=1.0,
                kernel="rbf",
                class_weight="balanced",
                random_state=42,
            ),
        ),
        method="sigmoid",
        cv=3,
        ensemble=False,
    )

    return {
        "extra_trees": extra_trees,
        "random_forest": random_forest,
        "rbf_svm": rbf_svm,
        "soft_voting_ensemble": VotingClassifier(
            estimators=[
                ("extra_trees", extra_trees),
                ("random_forest", random_forest),
                ("rbf_svm", rbf_svm),
            ],
            voting="soft",
            n_jobs=1,
        ),
    }


def evaluate_models(features, labels):
    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )
    scoring = {
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
    }
    results = {}

    for name, model in build_models().items():
        scores = cross_validate(
            model,
            features,
            labels,
            cv=cross_validation,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )
        results[name] = {
            "model": model,
            "balanced_accuracy_mean": float(scores["test_balanced_accuracy"].mean()),
            "balanced_accuracy_std": float(scores["test_balanced_accuracy"].std()),
            "f1_macro_mean": float(scores["test_f1_macro"].mean()),
        }

    return results


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
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Compare algorithms with cross-validation without saving a model.",
    )
    args = parser.parse_args()

    samples = collect_images(args.dataset)
    synthetic_demo = is_generated_demo_dataset(samples)
    if synthetic_demo and not args.evaluate_only:
        raise ValueError(
            "The selected data matches MorphTrace's generated demo set, "
            "not real CASIA images. Use --evaluate-only for diagnostics; "
            "the demo set must not produce a deployed model."
        )

    counts = {
        label: sum(1 for _, sample_label in samples if sample_label == label)
        for label in ("authentic", "manipulated")
    }
    if not args.evaluate_only and min(counts.values()) < MIN_TRAINING_SAMPLES_PER_CLASS:
        raise ValueError(
            "At least 50 real images per class are required for model training. "
            f"Found: {counts}"
        )

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

    if not args.evaluate_only:
        fitted_counts = {
            label: int(np.sum(y == label))
            for label in ("authentic", "manipulated")
        }
        if min(fitted_counts.values()) < MIN_TRAINING_SAMPLES_PER_CLASS:
            raise ValueError(
                "At least 50 readable real images per class are required. "
                f"Found: {fitted_counts}"
            )

    print(f"Evaluating {len(y)} images with stratified 5-fold cross-validation...")
    results = evaluate_models(X, y)
    for name, result in results.items():
        print(
            f"{name}: balanced accuracy "
            f"{result['balanced_accuracy_mean']:.3f} "
            f"(+/- {result['balanced_accuracy_std']:.3f}), "
            f"macro F1 {result['f1_macro_mean']:.3f}"
        )

    best_name, best_result = max(
        results.items(),
        key=lambda item: item[1]["balanced_accuracy_mean"],
    )
    print(f"Best cross-validation candidate: {best_name}")

    if args.evaluate_only:
        print("Evaluation only: no model artifact was saved.")
        return

    if best_result["balanced_accuracy_mean"] < MIN_VALIDATION_BALANCED_ACCURACY:
        raise ValueError(
            "Cross-validated balanced accuracy is below 0.65; refusing to "
            "save an unreliable model."
        )

    model = best_result["model"]
    model.fit(X, y)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "feature_version": FEATURE_VERSION,
            "classes": list(model.classes_),
            "training_samples": len(y),
            "validation": {
                "method": "stratified 5-fold cross-validation",
                "balanced_accuracy_mean": best_result["balanced_accuracy_mean"],
                "balanced_accuracy_std": best_result["balanced_accuracy_std"],
                "f1_macro_mean": best_result["f1_macro_mean"],
            },
            "algorithm": best_name,
        },
        args.output,
    )

    print(f"\nModel saved to: {args.output}")
    print("NOTE: Validate again on a separate source-image test set before operational use.")


if __name__ == "__main__":
    main()
