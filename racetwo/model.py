import os
import sys
import cv2
import numpy as np
from time import time
from ais_bench.infer.interface import InferSession
import torch

# 确保本地仓库内的 `ultralytics/ultralytics` 包优先被导入（与 YoloModelom.py 保持一致）
_repo_ultra_root = os.path.join(os.path.dirname(__file__), 'ultralytics')
_pkg_dir = os.path.join(_repo_ultra_root, 'ultralytics')
if os.path.isdir(_pkg_dir) and _repo_ultra_root not in sys.path:
    sys.path.insert(0, _repo_ultra_root)
from ultralytics.utils.ops import non_max_suppression

from config import (
    ZHUOZI_MODEL_PATH, WUPIN_MODEL_PATH,
    DEVICE_ID, CLASSES,
    OVERALL_IMG_HEIGHT, OVERALL_IMG_WIDTH,
    LOCAL_IMG_HEIGHT, LOCAL_IMG_WIDTH,
    PRED_CONF_THRES, PRED_IOU_THRES,
    M1_CLASS_SCORE_THRESHOLDS, M2_CLASS_SCORE_THRESHOLDS,
    INFER_DEBUG,
)


def _xywh2xyxy(x):
    y = np.zeros_like(x, dtype=np.float32)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
    return y


def _clip_boxes(boxes, h, w):
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
    return boxes


def _nms(boxes, scores, classes, iou_thres=0.7):
    if boxes.size == 0:
        return np.array([], dtype=np.int32)
    # 按类别分别做 NMS，保持与 Ultralytics 默认一致（class-aware）
    keep_indices = []
    unique_classes = np.unique(classes)
    for c in unique_classes:
        idxs = np.where(classes == c)[0]
        if idxs.size == 0:
            continue
        b = boxes[idxs].astype(np.float32)
        s = scores[idxs].astype(np.float32)
        order = s.argsort()[::-1]
        while order.size > 0:
            i = order[0]
            keep_indices.append(idxs[i])
            if order.size == 1:
                break
            xx1 = np.maximum(b[i, 0], b[order[1:], 0])
            yy1 = np.maximum(b[i, 1], b[order[1:], 1])
            xx2 = np.minimum(b[i, 2], b[order[1:], 2])
            yy2 = np.minimum(b[i, 3], b[order[1:], 3])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            area_i = (b[i, 2] - b[i, 0]) * (b[i, 3] - b[i, 1])
            area_others = (b[order[1:], 2] - b[order[1:], 0]) * (b[order[1:], 3] - b[order[1:], 1])
            iou = inter / (area_i + area_others - inter + 1e-9)
            remain = np.where(iou <= float(iou_thres))[0]
            order = order[remain + 1]
    return np.array(keep_indices, dtype=np.int32)


def _letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    h0, w0 = img.shape[:2]
    new_h, new_w = int(new_shape[0]), int(new_shape[1])
    r = min(new_h / h0, new_w / w0)
    if r != 1.0:
        interp = cv2.INTER_LINEAR if r > 1.0 else cv2.INTER_AREA
        img_resized = cv2.resize(img, (int(round(w0 * r)), int(round(h0 * r))), interpolation=interp)
    else:
        img_resized = img
    h, w = img_resized.shape[:2]
    pad_w = new_w - w
    pad_h = new_h - h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img_padded, r, (left, top)


def _scale_boxes_to_original(boxes_xyxy, ratio, pad, orig_h, orig_w):
    if boxes_xyxy.size == 0:
        return boxes_xyxy
    boxes = boxes_xyxy.copy().astype(np.float32)
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes[:, :4] /= float(ratio)
    boxes = _clip_boxes(boxes, orig_h, orig_w)
    return boxes




class Overall_Detector():
    """模型1：全图检测器（OM 推理，YoloModelom 同款预/后处理）"""
    def __init__(self, model_path: str, model_name: str = "检测器"):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"{model_name}模型文件不存在: {model_path}")
        ext = os.path.splitext(model_path)[1].lower()
        if ext != '.om':
            raise ValueError(f"{model_name}需要 OM 模型，当前: {ext}")
        self.model_path = model_path
        self.model_name = model_name
        self.classes = CLASSES
        self.conf = float(PRED_CONF_THRES)
        self.iou = float(PRED_IOU_THRES)
        self.max_det = 100
        # 初始化 OM 推理会话
        try:
            self.session = InferSession(device_id=int(DEVICE_ID) if DEVICE_ID is not None else 0, model_path=model_path)
        except Exception as e:
            raise RuntimeError(f"OM 模型加载失败: {e}")
        self.colors = (np.random.uniform(0, 255, size=(max(1, len(self.classes)), 3))).astype(np.uint8)
        print(f"✅ {self.model_name}加载成功(OM): {model_path} | device_id={DEVICE_ID}")

    def preprocess_single(self, image):
        """与 YoloModelom.py 同款预处理：方形填充到正方形，resize 到 640x640，blobFromImage 归一化。"""
        h0, w0 = image.shape[:2]
        if image.ndim == 3 and image.shape[2] == 3:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            bgr = image
        length = max(h0, w0)
        img_sq = cv2.copyMakeBorder(bgr, 0, length - h0, 0, length - w0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        blob = cv2.dnn.blobFromImage(img_sq, scalefactor=1/255.0, size=(640, 640), swapRB=True, crop=False)
        scale = max(h0, w0) / 640.0
        return blob, scale, (h0, w0)

    def _postprocess(self, outputs, scale):
        """与 YoloModelom.py 一致的后处理：ultralytics NMS，输出 numpy。"""
        try:
            tensor = torch.from_numpy(outputs[0][0]).unsqueeze(0)
        except Exception:
            tensor = torch.from_numpy(np.array(outputs)).unsqueeze(0) if outputs is not None else None
        if tensor is None:
            return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int32)

        dets = non_max_suppression(
            tensor,
            conf_thres=self.conf,
            iou_thres=self.iou,
            classes=None,
            agnostic=False,
            multi_label=False,
            max_det=self.max_det,
        )
        if not dets or len(dets[0]) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int32)

        det = dets[0].clone()
        det[:, :4] = det[:, :4] * float(scale)
        boxes = det[:, :4].cpu().numpy().astype(np.float32)
        confs = det[:, 4].cpu().numpy().astype(np.float32)
        clses = det[:, 5].cpu().numpy().astype(np.int32)
        return boxes, confs, clses

    def _class_thresholds(self, cls_ids: np.ndarray) -> np.ndarray:
        class_names = np.array([self.classes[i] if 0 <= i < len(self.classes) else 'unknown' for i in cls_ids])
        return np.array([float(M1_CLASS_SCORE_THRESHOLDS.get(n, PRED_CONF_THRES)) for n in class_names], dtype=np.float32)

    def _reverse_letterbox(self, boxes_xyxy: np.ndarray, ratio_pad, orig_hw) -> np.ndarray:
        ratio, pad = ratio_pad
        orig_h, orig_w = orig_hw
        return _scale_boxes_to_original(boxes_xyxy, ratio, pad, orig_h, orig_w)

    def _nms_by_class(self, boxes_xyxy: np.ndarray, scores: np.ndarray, cls_ids: np.ndarray) -> np.ndarray:
        return _nms(boxes_xyxy, scores, cls_ids, iou_thres=float(PRED_IOU_THRES))

    def infer(self, img):
        """执行推理（与 YoloModelom 同款预处理/后处理，固定 640x640）。"""
        blob, scale, _ = self.preprocess_single(img)
        outs = self.session.infer(feeds=[blob], mode="static")
        return self._postprocess(outs, scale)

    def is_available(self):
        """检查模型是否可用"""
        return self.session is not None    

class Local_Detector():
    """模型2：区域检测器（OM 推理，YoloModelom 同款预/后处理）"""
    def __init__(self, model_path: str, model_name: str = "检测器"):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"{model_name}模型文件不存在: {model_path}")
        ext = os.path.splitext(model_path)[1].lower()
        if ext != '.om':
            raise ValueError(f"{model_name}需要 OM 模型，当前: {ext}")
        self.model_path = model_path
        self.model_name = model_name
        self.classes = CLASSES
        self.conf = float(PRED_CONF_THRES)
        self.iou = float(PRED_IOU_THRES)
        self.max_det = 100
        # 初始化 OM 推理会话
        try:
            self.session = InferSession(device_id=int(DEVICE_ID) if DEVICE_ID is not None else 0, model_path=model_path)
        except Exception as e:
            raise RuntimeError(f"OM 模型加载失败: {e}")
        self.colors = (np.random.uniform(0, 255, size=(max(1, len(self.classes)), 3))).astype(np.uint8)
        print(f"✅ {self.model_name}加载成功(OM): {model_path} | device_id={DEVICE_ID}")

    def preprocess_single(self, image):
        """与 YoloModelom.py 一致：方形填充，640x640，blobFromImage。"""
        h0, w0 = image.shape[:2]
        if image.ndim == 3 and image.shape[2] == 3:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            bgr = image
        length = max(h0, w0)
        img_sq = cv2.copyMakeBorder(bgr, 0, length - h0, 0, length - w0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        blob = cv2.dnn.blobFromImage(img_sq, scalefactor=1/255.0, size=(640, 640), swapRB=True, crop=False)
        scale = max(h0, w0) / 640.0
        return blob, scale, (h0, w0)

    def _postprocess(self, outputs, scale):
        try:
            tensor = torch.from_numpy(outputs[0][0]).unsqueeze(0)
        except Exception:
            tensor = torch.from_numpy(np.array(outputs)).unsqueeze(0) if outputs is not None else None
        if tensor is None:
            return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int32)

        dets = non_max_suppression(
            tensor,
            conf_thres=self.conf,
            iou_thres=self.iou,
            classes=None,
            agnostic=False,
            multi_label=False,
            max_det=self.max_det,
        )
        if not dets or len(dets[0]) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int32)

        det = dets[0].clone()
        det[:, :4] = det[:, :4] * float(scale)
        boxes = det[:, :4].cpu().numpy().astype(np.float32)
        confs = det[:, 4].cpu().numpy().astype(np.float32)
        clses = det[:, 5].cpu().numpy().astype(np.int32)
        return boxes, confs, clses

    def _class_thresholds(self, cls_ids: np.ndarray) -> np.ndarray:
        class_names = np.array([self.classes[i] if 0 <= i < len(self.classes) else 'unknown' for i in cls_ids])
        return np.array([float(M2_CLASS_SCORE_THRESHOLDS.get(n, PRED_CONF_THRES)) for n in class_names], dtype=np.float32)

    def _reverse_letterbox(self, boxes_xyxy: np.ndarray, ratio_pad, orig_hw) -> np.ndarray:
        ratio, pad = ratio_pad
        orig_h, orig_w = orig_hw
        return _scale_boxes_to_original(boxes_xyxy, ratio, pad, orig_h, orig_w)

    def _nms_by_class(self, boxes_xyxy: np.ndarray, scores: np.ndarray, cls_ids: np.ndarray) -> np.ndarray:
        return _nms(boxes_xyxy, scores, cls_ids, iou_thres=float(PRED_IOU_THRES))

    def infer(self, img):
        """执行推理（与 YoloModelom 同款预处理/后处理，固定 640x640）。"""
        blob, scale, _ = self.preprocess_single(img)
        outs = self.session.infer(feeds=[blob], mode="static")
        return self._postprocess(outs, scale)

    def is_available(self):
        """检查模型是否可用"""
        return self.session is not None



class ModelManager:
    """模型管理器，负责所有模型的初始化和管理"""

    def __init__(self):
        print("正在初始化模型管理器...")
        try:
            self.overall_detector = Overall_Detector(ZHUOZI_MODEL_PATH, "模型1：全图检测器")
        except Exception as e:
            print(f"❌ 模型1加载失败: {e}")
            raise
        try:
            self.local_detector = Local_Detector(WUPIN_MODEL_PATH, "模型2：区域检测器")
        except Exception as e:
            print(f"❌ 模型2加载失败: {e}")
            raise
                          
    #接口函数     
    def get_overall_detector(self):
        """获取全图检测器"""
        return self.overall_detector

    def get_local_detector(self):
        """获取区域检测器"""
        return self.local_detector