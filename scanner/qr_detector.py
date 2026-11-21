
import math
import cv2
import numpy as np

from scanner.finder_patterns import find_finder_patterns


def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.hypot(
        p2[0] - p1[0],
        p2[1] - p1[1]
    )


def group_finder_patterns(candidates):
    """Group finder patterns into possible QR codes."""

    from itertools import combinations

    groups = []

    for triple in combinations(candidates, 3):

        centers = [item["center"] for item in triple]

        d01 = distance(centers[0], centers[1])
        d02 = distance(centers[0], centers[2])
        d12 = distance(centers[1], centers[2])

        distances = sorted([d01, d02, d12])

        a, b, c = distances

        if a == 0 or b == 0:
            continue

        # Finder patterns form an approximate right triangle.
        pythagorean_error = abs(
            (a * a + b * b) - c * c
        ) / (c * c)

        if pythagorean_error > 0.25:
            continue

        # The two shorter sides should have similar lengths.
        if abs(a - b) / max(a, b) > 0.30:
            continue

        sizes = [item["size"] for item in triple]

        if min(sizes) == 0:
            continue

        if max(sizes) / min(sizes) > 1.8:
            continue

        groups.append(triple)

    return groups
def detect_qr_codes(frame):
    """Detect possible QR codes in an image."""

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    candidates = find_finder_patterns(binary)
    groups = group_finder_patterns(candidates)

    detections = []

    for group in groups:
        corners = estimate_qr_corners(group)

        detections.append({
            "corners": corners,
            "finder_patterns": group
        })

    return detections, candidates