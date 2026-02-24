# VisionPipeline.py
import numpy as np
import glob
from collections import Counter
from YoloModel import YoloModel
from dataclasses import dataclass, asdict
import json

from ultralytics.engine.results import Results

from Config import ZHUOZI_MODEL_PATH


@dataclass
class DetectionConfig:
    conf_thres: float = 0.25
    iou_thres: float = 0.45

    def save_to_json(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def load_from_json(cls, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


class DetectionPipeline:

    def __init__(self, config: DetectionConfig = DetectionConfig()):
        self.config = config if config else {}

        self.conf_thres = self.config.conf_thres
        self.iou_thres = self.config.iou_thres

        self.model = YoloModel(model_path=ZHUOZI_MODEL_PATH, task="detect")

    def detect_frames(self, tableNum) -> Counter:
        detectionResults = []

        imagePaths = glob.glob(f"results/machine/camera/T{tableNum}-*.npz")
        print(f"DetectionPipeline: Found {len(imagePaths)} images for T{tableNum}")

        for imagePath in imagePaths:
            with np.load(imagePath, allow_pickle=True) as data:
                rgbFrame, depthFrame, pointCloudFrame = (
                    data["rgbFrame"],
                    data["depthFrame"],
                    data["pointCloudFrame"],
                )

            detectionResult, self.resultFrame = self.detect_frame(
                rgbFrame, depthFrame, pointCloudFrame
            )
            filename = imagePath.split("/")[-1].replace(".npz", ".npy")
            np.save(f"results/machine/detection/{filename}", self.resultFrame)
            detectionResults.append(detectionResult)

        return self.fusion_results(detectionResults)

    def detect_frame(
        self, rgbFrame, depthFrame, pointCloudFrame
    ) -> tuple[Results, np.ndarray]:
        # TODO: 使用者需要将这里的类型Results注明,适合自定义一种类型,以便后续开发和维护
        detectionResult: Results = self.model.predict(rgbFrame)
        return detectionResult, detectionResult.plot()

    def fusion_results(self, detectionResults) -> Counter:
        # TODO: 替换为真实的融合逻辑
        return Counter({"CA001": 2, "CA002": 1})
