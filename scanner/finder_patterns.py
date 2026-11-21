import cv2
import numpy as np

def is_square(contour):
    """Check if a contour resembles a square."""

    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        return False

    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

    if len(approx) != 4:
        return False

    if not cv2.isContourConvex(approx):
        return False

    x, y, w, h = cv2.boundingRect(approx)

    if h == 0:
        return False

    aspect_ratio = float(w) / h

    return 0.7 <= aspect_ratio <= 1.3


def find_finder_patterns(binary_image):
    """Detect potential QR finder patterns"""

    countours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    if hierarchy is None:
        return candidates

    hierarchy = hierarchy[0]

    for i, contour in enumerate(countours):

        if not is_square(contour):
            continue

        area = cv2.contourArea(contour)

        if area < 100:
            continue

        #QR finder paterns have nested squares
        child_index = hierarchy[i][2]

        if child_index == -1:
            continue

        grandchild_index = hierarchy[child_index][2]    

        if grandchild_index == -1:
            continue

        #Check that nested shapes have plausible sizes
        child_area = cv2.contourArea(countours[child_index])
        grandchild_area = cv2.contourArea(countours[grandchild_index])

        if area == 0 or child_area == 0:
            continue

        child_ratio = child_area / area
        grandchild_ratio = grandchild_area / child_area

        if not (0.1 <= child_ratio <= 0.8):
            continue

        if not (0.1 <= grandchild_ratio <= 0.8):
            continue

        moments = cv2.moments(contour)

        if moments["m00"] == 0:
            continue

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        x, y, w, h = cv2.boundingRect(contour)

        candidates.append({
            "center": (cx, cy),
            "size": max(w, h),
            "contour": contour
        })

    return candidates