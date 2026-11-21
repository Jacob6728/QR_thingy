
import cv2

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QFileDialog,
    QMessageBox
)

from scanner.camera import Camera
from scanner.preprocessing import process_frame
from scanner.qr_detector import detect_qr_codes

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Custom QR Vision Scanner")
        self.resize(1100, 750)

        self.camera = Camera()
        self.current_frame = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        title = QLabel("Custom QR Vision Scanner")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "font-size: 24px; font-weight: bold;"
        )
        layout.addWidget(title)

        controls = QHBoxLayout()

        self.start_button = QPushButton("Start Camera")
        self.start_button.clicked.connect(self.start_camera)
        controls.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Camera")
        self.stop_button.clicked.connect(self.stop_camera)
        controls.addWidget(self.stop_button)

        self.upload_button = QPushButton("Upload Image")
        self.upload_button.clicked.connect(self.upload_image)
        controls.addWidget(self.upload_button)

        self.mode_selector = QComboBox()
        self.mode_selector.addItems([
            "Original",
            "Grayscale",
            "Gaussian Blur",
            "Threshold",
            "QR Detection"
        ])

        self.mode_selector.currentTextChanged.connect(
            self.refresh_preview
        )
        controls.addWidget(self.mode_selector)

        layout.addLayout(controls)

        self.image_label = QLabel("Camera not started")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 420)
        self.image_label.setStyleSheet(
            "background-color: #202020; color: white;"
        )
        layout.addWidget(self.image_label, stretch=1)

        self.status_label = QLabel("Status: Ready")
        layout.addWidget(self.status_label)

    def start_camera(self):
        self.stop_camera()

        try:
            self.camera.start()
        except RuntimeError as error:
            QMessageBox.warning(self, "Camera Error", str(error))
            return

        self.timer.start(33)
        self.status_label.setText("Status: Camera running")

    def stop_camera(self):
        self.timer.stop()
        self.camera.stop()

        self.status_label.setText("Status: Camera stopped")

    def update_frame(self):
        frame = self.camera.read_frame()

        if frame is None:
            self.stop_camera()
            return

        self.current_frame = frame
        self.refresh_preview()

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select an image",
            "",
            "Images (*.png *.jpg *.jpeg)"
        )

        if not file_path:
            return

        self.stop_camera()

        frame = cv2.imread(file_path)

        if frame is None:
            QMessageBox.warning(
                self,
                "Image Error",
                "Unable to read the selected image."
            )
            return

        self.current_frame = frame
        self.refresh_preview()
        self.status_label.setText("Status: Image loaded")

    def draw_qr_detections(self, frame):
        """Draw finder patterns and QR boundaries."""

        output = frame.copy()

        detections, candidates = detect_qr_codes(frame)

        # Draw candidate finder-pattern centers.
        for candidate in candidates:
            cx, cy = candidate["center"]

            cv2.circle(
                output,
                (cx, cy),
                5,
                (0, 255, 255),
                -1
            )

        # Draw QR outlines.
        for detection in detections:
            corners = detection["corners"].astype(int)

            cv2.polylines(
                output,
                [corners],
                True,
                (0, 255, 0),
                3
            )

            x, y = corners[0]

            cv2.putText(
                output,
                "QR DETECTED",
                (int(x), int(y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        return output

    def refresh_preview(self):
        if self.current_frame is None:
            return

        stages = process_frame(self.current_frame)
        selected_mode = self.mode_selector.currentText()

        if selected_mode == "QR Detection":
            image = self.draw_qr_detections(self.current_frame)
        else:
            image = stages[selected_mode]

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width, channels = image.shape
        bytes_per_line = channels * width

        qt_image = QImage(
            image.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888
        ).copy()

        pixmap = QPixmap.fromImage(qt_image)

        self.image_label.setPixmap(
            pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()
