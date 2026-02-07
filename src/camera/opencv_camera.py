from base_camera import BaseCamera
import cv2


class OpenCVCamera(BaseCamera):

    def __init__(self):
        super().__init__()
        self.cap = None

    def start(self):
        try:
            self.cap = cv2.VideoCapture(cv2.CAP_ANY, 0)
            if not self.cap.isOpened():
                raise RuntimeError("Failed to open OpenCV camera")
            self.isOpen = True
        except Exception as e:
            raise RuntimeError(f"Failed to start OpenCV camera: {e}")

    def stop(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.isOpen = False

    def get_frame(self):
        if not self.isOpen:
            raise RuntimeError("Camera not started")
        ret, frame = self.cap.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
