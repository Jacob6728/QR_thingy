import cv2

def to_grayscale(frame):
    """Convert a BGR image to grayscale."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def gaussian_blur(gray):
    """Reduce image noise before QR detection."""
    return cv2.GaussianBlur(gray, (5, 5), 0)


def adaptive_threshold(gray):
    """Apply adaptive thresholding to a grayscale image."""
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)

def process_frame(frame):
    """Return all preprocessing stages"""
    gray = to_grayscale(frame)
    blurred = gaussian_blur(gray)
    thresholded = adaptive_threshold(blurred)

    return{
        "Original": frame,
        "Grayscale": gray,
        "Gaussian Blurred": blurred,
        "Threshold" : thresholded
    }