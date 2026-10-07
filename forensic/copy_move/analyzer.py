from pathlib import Path

import cv2
import numpy as np


def analyze_copy_move(image_path, output_path):
    image_path = Path(image_path)
    output_path = Path(output_path)

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError("Unable to read image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=3000)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    if descriptors is None or len(keypoints) < 10:
        cv2.imwrite(str(output_path), image)

        return {
            "technique": "copy-move",
            "score": 0.0,
            "matches": 0,
            "suspicious_regions": [],
            "localized": True,
            "output": str(output_path),
        }

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = matcher.knnMatch(descriptors, descriptors, k=2)

    good_matches = []

    for pair in matches:
        if len(pair) < 2:
            continue

        first, second = pair

        # Lowe-style ratio test.
        if first.distance >= second.distance * 0.75:
            continue

        point_a = np.array(keypoints[first.queryIdx].pt)
        point_b = np.array(keypoints[first.trainIdx].pt)

        spatial_distance = np.linalg.norm(point_a - point_b)

        # Ignore self-matches and very nearby features.
        if spatial_distance < 40:
            continue

        good_matches.append(first)

    result = image.copy()
    suspicious_regions = []

    for match in good_matches:
        point_a = tuple(
            map(int, keypoints[match.queryIdx].pt)
        )
        point_b = tuple(
            map(int, keypoints[match.trainIdx].pt)
        )

        cv2.circle(result, point_a, 10, (0, 0, 255), 2)
        cv2.circle(result, point_b, 10, (0, 0, 255), 2)
        cv2.line(result, point_a, point_b, (0, 0, 255), 1)

        suspicious_regions.append({
            "point_a": point_a,
            "point_b": point_b,
            "distance": round(float(match.distance), 2),
        })

    cv2.imwrite(str(output_path), result)

    match_count = len(good_matches)

    # This is an evidence score, not a probability of forgery.
    score = min(match_count / 50.0, 1.0)

    return {
        "technique": "copy-move",
        "score": round(score, 4),
        "matches": match_count,
        "suspicious_regions": suspicious_regions,
        "localized": True,
        "output": str(output_path),
    }
