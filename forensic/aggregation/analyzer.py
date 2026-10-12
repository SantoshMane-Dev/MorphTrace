
import math


def _bounded_score(value):
    """Convert a score to a finite float in the range [0, 1]."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0

    if not math.isfinite(score):
        return 0.0

    return max(0.0, min(score, 1.0))


def _nonnegative_int(value):
    """Safely convert a match count to a nonnegative integer."""
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return 0


def aggregate_evidence(
    metadata,
    ela,
    copy_move,
    compression=None,
    noise=None,
):
    """
    Aggregate forensic indicators into a heuristic evidence score.

    The score is not a probability of manipulation. The weights and
    thresholds are preliminary heuristics and require validation against
    a representative dataset before operational use.
    """
    metadata = metadata if isinstance(metadata, dict) else {}
    ela = ela if isinstance(ela, dict) else {}
    copy_move = copy_move if isinstance(copy_move, dict) else {}
    compression = compression if isinstance(compression, dict) else {}
    noise = noise if isinstance(noise, dict) else {}

    evidence = []

    # Metadata findings are contextual clues, not proof of manipulation.
    metadata_items = metadata.get("evidence", [])
    if isinstance(metadata_items, list):
        for item in metadata_items:
            if not isinstance(item, dict):
                continue

            finding = item.get("finding")
            if not finding:
                continue

            evidence.append({
                "technique": "metadata",
                "finding": str(finding),
                "severity": str(item.get("severity", "informational")),
                "localized": bool(item.get("localized", False)),
            })

    ela_score = _bounded_score(ela.get("score", 0.0))
    copy_move_matches = _nonnegative_int(copy_move.get("matches", 0))
    copy_move_score = _bounded_score(
        copy_move.get("score", copy_move_matches / 50.0)
    )
    compression_score = _bounded_score(compression.get("score", 0.0))
    noise_score = _bounded_score(noise.get("score", 0.0))

    if ela_score >= 0.10:
        evidence.append({
            "technique": "ELA",
            "finding": (
                "Elevated recompression difference detected. "
                "Image content and compression history may affect this result."
            ),
            "severity": "review",
            "localized": bool(ela.get("localized", False)),
        })

    if copy_move_matches >= 5:
        evidence.append({
            "technique": "copy-move",
            "finding": (
                f"{copy_move_matches} potential duplicated feature matches "
                "detected. Geometric verification is needed to assess "
                "whether these matches indicate copying."
            ),
            "severity": "review",
            "localized": bool(copy_move.get("localized", False)),
        })

    if compression.get("evidence_found", False):
        evidence.append({
            "technique": "compression",
            "finding": (
                "Compression-frequency characteristics exceeded the "
                "current heuristic threshold."
            ),
            "severity": "review",
            "localized": bool(compression.get("localized", False)),
        })

    if noise.get("evidence_found", False):
        evidence.append({
            "technique": "noise",
            "finding": (
                "Local noise variation exceeded the current heuristic "
                "threshold. Texture, lighting, and sensor processing can "
                "also affect this signal."
            ),
            "severity": "review",
            "localized": bool(noise.get("localized", False)),
        })

    # Preliminary weights: these are not statistically calibrated.
    signals = {
        "ela": ela_score,
        "copy_move": copy_move_score,
        "compression": compression_score,
        "noise": noise_score,
    }

    evidence_score = (
        ela_score * 0.30
        + copy_move_score * 0.40
        + compression_score * 0.15
        + noise_score * 0.15
    )
    evidence_score = round(_bounded_score(evidence_score), 4)

    review_count = sum(
        1 for item in evidence if item["severity"] == "review"
    )
    localized_count = sum(
        1 for item in evidence if item["localized"]
    )

    if review_count == 0:
        assessment = "No strong forensic indicators triggered review"
    elif evidence_score >= 0.60:
        assessment = "Multiple or strong indicators require forensic review"
    elif evidence_score >= 0.30:
        assessment = "Forensic indicators require review"
    else:
        assessment = "Limited forensic indicators detected; review findings"

    return {
        "evidence_score": evidence_score,
        "assessment": assessment,
        "signals": signals,
        "evidence": evidence,
        "review_count": review_count,
        "localized_evidence_count": localized_count,
        "score_interpretation": (
            "Heuristic evidence score, not a probability or proof "
            "of image manipulation."
        ),
    }
