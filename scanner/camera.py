
import cv2


class Camera:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.capture = None

    def start(self):
        """Open the selected webcam."""
        self.capture = cv2.VideoCapture(self.camera_index)

        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            raise RuntimeError("Unable to open webcam.")

    def read_frame(self):
        """Read one frame from the webcam."""
        if self.capture is None:
            return None

        success, frame = self.capture.read()

        if not success:
            return None

        return frame

    def stop(self):
        """Release the webcam."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None
