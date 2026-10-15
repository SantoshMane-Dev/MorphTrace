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

    orb = cv2.ORB_create(nfeatures=5000, fastThreshold=7)
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

    pairs = matcher.knnMatch(
        descriptors,
        descriptors,
        k=min(4, len(descriptors)),
    )
    candidate_matches = []
    points_a = []
    points_b = []

    for pair in pairs:
        distinct_matches = [
            match for match in pair
            if match.queryIdx != match.trainIdx
        ]
        if len(distinct_matches) < 2:
            continue

        first, second = distinct_matches[:2]

        if second.distance > 0 and first.distance >= second.distance * 0.85:
            continue

        point_a = np.asarray(keypoints[first.queryIdx].pt, dtype=np.float32)
        point_b = np.asarray(keypoints[first.trainIdx].pt, dtype=np.float32)

        spatial_distance = np.linalg.norm(point_a - point_b)

        if spatial_distance < 40:
            continue

        candidate_matches.append(first)
        points_a.append(point_a)
        points_b.append(point_b)

    good_matches = []
    inlier_count = 0
    if len(candidate_matches) >= 6:
        transform, inlier_mask = cv2.estimateAffinePartial2D(
            np.asarray(points_a),
            np.asarray(points_b),
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
            maxIters=2000,
            confidence=0.99,
            refineIters=10,
        )
        if transform is not None and inlier_mask is not None:
            inlier_indices = np.flatnonzero(inlier_mask.ravel())
            inlier_count = len(inlier_indices)
            source_points = np.asarray(points_a)[inlier_indices]
            target_points = np.asarray(points_b)[inlier_indices]
            source_extent = np.ptp(source_points, axis=0)
            target_extent = np.ptp(target_points, axis=0)
            transform_scale = float(np.hypot(transform[0, 0], transform[1, 0]))
            localized_regions = (
                source_extent[0] <= image.shape[1] * 0.75
                and source_extent[1] <= image.shape[0] * 0.75
                and target_extent[0] <= image.shape[1] * 0.75
                and target_extent[1] <= image.shape[0] * 0.75
            )
            plausible_scale = 0.5 <= transform_scale <= 2.0
            if (
                inlier_count >= 8
                and inlier_count / len(candidate_matches) >= 0.08
                and localized_regions
                and plausible_scale
            ):
                good_matches = [candidate_matches[index] for index in inlier_indices]

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

    score = min(match_count / 30.0, 1.0)

    return {
        "technique": "copy-move",
        "score": round(score, 4),
        "matches": match_count,
        "candidate_matches": len(candidate_matches),
        "suspicious_regions": suspicious_regions,
        "localized": True,
        "output": str(output_path),
    }
