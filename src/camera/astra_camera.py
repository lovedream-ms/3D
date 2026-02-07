from base_camera import BaseCamera
import astra_py
import cv2


class AstraCamera(BaseCamera):

    def __init__(self):
        super().__init__()
        self.cam = None

    def start(self):
        try:
            self.cam = astra_py.AstraCamera()
            self.cam.set_resolution(1920, 1080, 30)
            self.cam.set_outputs(
                want_color=True, want_depth=False, want_shaded=False, want_cloud=False
            )
            self.cam.start()
            self.isOpen = True
        except Exception as e:
            raise RuntimeError(f"Failed to start Astra camera: {e}")

    def stop(self):
        if self.cam is not None:
            self.cam.stop()
            self.cam = None
            self.isOpen = False

    def get_frame(self):
        if not self.isOpen:
            raise RuntimeError("Camera not started")
        frames = self.cam.get_frames()
        if not frames:
            return None
        color = frames[0]
        return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
