import torch
import numpy as np
import cv2
import requests
import pickle

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


class WebModel:
    def __init__(self, model_url, task="detect"):
        self.url = model_url
        self.task = task

    def predict(self, frame) -> Results:
        success, img_encoded = cv2.imencode(".jpg", frame)
        if not success:
            raise ValueError("图像编码失败")

        files = {"file": ("image.jpg", img_encoded.tobytes(), "image/jpeg")}
        response = requests.post(self.url, files=files)

        if response.status_code != 200:
            raise RuntimeError(f"Web API 错误: {response.text}")

        result: Results = pickle.loads(response.content)
        result.orig_img = frame

        return result


if __name__ == "__main__":
    import cv2

    # 一次支持 pt/om, cls/detect/obb/segment/pose 全任务，其余任务均可以使用
    # model = YoloModel("models/yolov8n.pt", task="detect")
    model = WebModel("http://127.0.0.1:8000/predict", task="detect")
    # model = YoloModel("models/yolov8n.om", task="detect")
    result = model.predict(cv2.imread("data/test/0-0-0-0-0-1-180121.png"))
    result.show()
