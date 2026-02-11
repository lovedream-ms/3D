from abc import ABC, abstractmethod


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
