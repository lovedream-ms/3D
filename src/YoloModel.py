import torch
import numpy as np

from ultralytics.models.yolo.classify import ClassificationPredictor
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.models.yolo.obb import OBBPredictor
from ultralytics.models.yolo.segment import SegmentationPredictor
from ultralytics.models.yolo.pose import PosePredictor

from ultralytics.engine.results import Results


class OmEngine:
    """把昇腾推理封装成类似 PyTorch nn.Module 的接口"""

    def __init__(self, model_path):
        from ais_bench.infer.interface import InferSession

        self.session = InferSession(device_id=0, model_path=model_path)

    def __call__(self, im):
        raw_outputs = self.session.infer(feeds=[im.cpu().numpy()], mode="static")
        return [torch.from_numpy(np.array(o)) for o in raw_outputs]


class YoloModel:
    def __init__(self, model_path, task="detect", conf=0.25, iou=0.45, max_det=20):
        self.task = task
        self.is_om = model_path.endswith(".om")

        # 定义所有任务对应的预测器类
        self.pred_map = {
            "class": ClassificationPredictor,
            "detect": DetectionPredictor,
            "obb": OBBPredictor,
            "segment": SegmentationPredictor,
            "pose": PosePredictor,
        }

        if self.is_om:
            self.engine = OmEngine(model_path)

            overrides = {"task": task, "conf": conf, "iou": iou, "max_det": max_det}
            self.predictor = self.pred_map[task](overrides=overrides)
            self.predictor.model = self.engine
            self.predictor.setup_model(model=None)
        else:
            from ultralytics import YOLO

            self.model = YOLO(model_path)

        self.model.predict(np.zeros((64, 64, 3), dtype=np.uint8), task=task)
        self.predictor = self.model.predictor

    def predict(self, frame) -> Results:
        im = self.predictor.preprocess([frame])
        preds = self.predictor.model(im)
        return self.predictor.postprocess(preds, im, [frame])[0]


if __name__ == "__main__":
    import cv2

    # 一次支持 pt/om, cls/detect/obb/segment/pose 全任务，其余任务均可以使用
    model = YoloModel("models/yolov8n.pt", task="detect")
    # model = YoloModel("models/yolov8n.om", task="detect")
    result = model.predict(cv2.imread("data/test/0-0-0-0-0-1-180121.png"))
    result.show()
