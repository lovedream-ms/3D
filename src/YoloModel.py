import cv2
import numpy as np
import torch
from abc import ABC, abstractmethod
from PIL import Image, ImageDraw, ImageFont

from ais_bench.infer.interface import InferSession
from ultralytics.utils.nms import non_max_suppression

from config import CLASSES, colors

FONT_PATH = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
font = ImageFont.truetype(FONT_PATH, 20)


def drawBoundingBox(img, classId, confidence, x1, y1, x2, y2):
    color = colors[classId]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    label = f"{list(CLASSES.keys())[classId]} ({confidence:.2f})"
    text_color_pil = tuple(color.astype(int)[::-1])

    text_x = max(0, x1)
    text_y = max(0, y1 - 25)

    draw.text((text_x, text_y), label, font=font, fill=text_color_pil)

    np.copyto(img, cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR))


class BaseYoloModel(ABC):
    def __init__(self, modelPath, scoreMap=None, conf=0.25, iou=0.45, maxDet=20):
        self.modelPath = modelPath
        self.scoreMap = scoreMap
        self.conf = conf
        self.iou = iou
        self.maxDet = maxDet
        self._load_model()

    @abstractmethod
    def _load_model(self):
        self.model = InferSession(device_id=0, model_path=self.modelPath)

    def warm_up(self):
        testImage = np.zeros((640, 640, 3), dtype=np.uint8)
        self.infer(testImage)

    def _preprocess(self, originalImage, imgsz=640):
        h, w, _ = originalImage.shape
        length = max((h, w))

        # 填充 (Padding) 为正方形并调整至 imgsz x imgsz
        image = cv2.copyMakeBorder(
            originalImage,
            0,
            length - h,
            0,
            length - w,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        # 归一化并转换为 blob (NCHW)
        blob = cv2.dnn.blobFromImage(
            image, scalefactor=1 / 255.0, size=(imgsz, imgsz), swapRB=True, crop=False
        )

        scale = length / imgsz
        return blob, scale

    def _postprocess(self, outputs, originalImage, scale):
        """
        后处理：包含 NMS 和坐标还原
        """
        preds = torch.from_numpy(outputs[0][0]).unsqueeze(0)

        detections = non_max_suppression(
            preds,
            conf_thres=self.conf,
            iou_thres=self.iou,
            classes=None,
            agnostic=False,
            multi_label=False,
            max_det=self.maxDet,
        )

        detectionFrame = originalImage.copy()
        results = []

        for det in detections:
            if len(det) == 0:
                continue

            det[:, :4] *= scale

            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                class_id = int(cls)
                score = float(conf)

                drawBoundingBox(detectionFrame, class_id, score, x1, y1, x2, y2)

                results.append(
                    {
                        "classId": class_id,
                        "className": list(CLASSES.keys())[class_id],
                        "score": score,
                        "box": [x1, y1, x2, y2],
                    }
                )

        return [d["className"] for d in results], detectionFrame

    def infer(self, rgbFrame):
        blob, scale = self._preprocess(rgbFrame)
        outputs = self.model.infer(feeds=[blob], mode="static")
        outputs = [np.array(output) for output in outputs]
        classNames, detectionFrame = self._postprocess(outputs, rgbFrame, scale)
        return classNames, detectionFrame
