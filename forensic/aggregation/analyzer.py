def aggregate_evidence(metadata, ela, copy_move):
    evidence = []

    for item in metadata.get("evidence", []):
        evidence.append({
            "technique": "metadata",
            "finding": item["finding"],
            "severity": item["severity"],
            "localized": item.get("localized", False),
        })

    ela_score = ela.get("score", 0.0)

    if ela_score >= 0.10:
        evidence.append({
            "technique": "ELA",
            "finding": "Elevated compression difference detected",
            "severity": "review",
            "localized": True,
        })

    copy_move_matches = copy_move.get("matches", 0)

    if copy_move_matches >= 5:
        evidence.append({
            "technique": "copy-move",
            "finding": f"{copy_move_matches} potential duplicated feature matches detected",
            "severity": "review",
            "localized": True,
        })

    # Weighted evidence score.
    ela_signal = min(ela_score, 1.0)
    copy_move_signal = min(copy_move_matches / 50.0, 1.0)

    overall_score = (
        ela_signal * 0.40
        + copy_move_signal * 0.60
    )

    if overall_score >= 0.60:
        assessment = "Potential manipulation detected"
    elif overall_score >= 0.30:
        assessment = "Manipulation evidence requires review"
    else:
        assessment = "No strong manipulation evidence detected"

    return {
        "manipulation_likelihood": round(overall_score, 4),
        "assessment": assessment,
        "evidence": evidence,
    }
