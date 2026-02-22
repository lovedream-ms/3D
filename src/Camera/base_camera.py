from abc import ABC, abstractmethod
import numpy as np


class BaseCamera(ABC):

    def __init__(self):
        self.isOpen = False

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def get_frame(self):
        pass

    def capture_frame(self, save_path):
        if not self.isOpen:
            print("Camera is not open. Cannot capture frame.")
            return

        rgbFrame, depthFrame, pointCloudFrame = self.get_frame()
        np.savez(
            save_path,
            rgbFrame=rgbFrame,
            depthFrame=depthFrame,
            pointCloudFrame=pointCloudFrame,
        )
