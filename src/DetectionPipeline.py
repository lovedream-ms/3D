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

    def detect_frames(self, path) -> Counter:
        detectionResults = []

        imagePaths = glob.glob(f"{path}/*.npz")
        print(f"DetectionPipeline: Found {len(imagePaths)} images for path {path}")

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
            filePath = imagePath.replace("camera", "detection").replace(".npz", ".npy")
            np.save(filePath, self.resultFrame)
            detectionResults.append(detectionResult)

        return self.fusion_results(detectionResults)

    def detect_frame(
        self, rgbFrame, depthFrame, pointCloudFrame
    ) -> tuple[Results, np.ndarray]:
        # TODO: 使用者需要将这里的类型Results注明,适合自定义一种类型,以便后续开发和维护
        detectionResult: Results = self.model.predict(rgbFrame)
        return detectionResult, detectionResult.plot()

    def fusion_results(self, detectionResults: list[Results]) -> Counter:
        final_counter = Counter()
        frame_counts = []

        for result in detectionResults:
            current_frame_counter = Counter()

            if result.boxes is None or len(result.boxes) == 0:
                frame_counts.append(current_frame_counter)
                continue

            classes = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            names = result.names  # 字典: {0: 'CA001', 1: 'CA002', ...}

            for cls_id, conf in zip(classes, confs):
                if conf >= self.config.conf_thres:
                    class_name = names[cls_id]
                    current_frame_counter[class_name] += 1

            frame_counts.append(current_frame_counter)

        for frame_counter in frame_counts:
            for cls_name, count in frame_counter.items():
                if count > final_counter[cls_name]:
                    final_counter[cls_name] = count

        return final_counter
