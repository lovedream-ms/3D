#!/usr/bin/env python3
"""
MAP 计算脚本

从以下目录读取 YOLO 标签：
- 预测：test/inferresult/inferlabels
- 真实：test/initial/labels

支持：
- IoU 阈值可配（默认 0.5）
- 采用 VOC 风格的插值积分法计算 AP（对 P-R 曲线做单调包络再积分）

提供函数：
- map(iou_threshold=0.5) -> (mAP, per_class_ap)
"""

import os
from typing import Dict, List, Tuple
from config import CLASSES as CLASS_NAMES  # 用于打印类别名称


class YoloDet:
    def __init__(
        self, class_id: int, cx: float, cy: float, w: float, h: float, conf: float = 1.0
    ):
        self.class_id = int(class_id)
        self.cx = float(cx)
        self.cy = float(cy)
        self.w = float(w)
        self.h = float(h)
        self.conf = float(conf)

    def xyxy_norm(self) -> Tuple[float, float, float, float]:
        x1 = self.cx - self.w / 2.0
        y1 = self.cy - self.h / 2.0
        x2 = self.cx + self.w / 2.0
        y2 = self.cy + self.h / 2.0
        return x1, y1, x2, y2


def _parse_yolo_file(file_path: str) -> List[YoloDet]:
    dets: List[YoloDet] = []
    if not os.path.exists(file_path):
        return dets
    try:
        # 优先用 utf-8-sig 去除 BOM，失败再回退 utf-8 并忽略错误
        with open(file_path, "r", encoding="utf-8-sig") as f:
            text = f.read()
    except Exception:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            return dets

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            conf = float(parts[5]) if len(parts) > 5 else 1.0
            dets.append(YoloDet(cls_id, cx, cy, w, h, conf))
        except Exception as e:
            print(f"    Parse error on line '{line}': {e}")
            continue

    return dets


def _load_dataset(
    gt_dir: str, pred_dir: str
) -> Tuple[Dict[str, List[YoloDet]], Dict[str, List[YoloDet]]]:
    gt: Dict[str, List[YoloDet]] = {}
    pred: Dict[str, List[YoloDet]] = {}

    if os.path.isdir(gt_dir):
        for name in os.listdir(gt_dir):
            if name.endswith(".txt"):
                stem = os.path.splitext(name)[0]
                gt[stem] = _parse_yolo_file(os.path.join(gt_dir, name))

    if os.path.isdir(pred_dir):
        for name in os.listdir(pred_dir):
            if name.endswith(".txt"):
                stem = os.path.splitext(name)[0]
                pred[stem] = _parse_yolo_file(os.path.join(pred_dir, name))

    return gt, pred


def _iou_xyxy(
    a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    bb = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def _compute_ap(rec: List[float], prec: List[float]) -> float:
    # 单调包络
    mrec = [0.0] + rec + [1.0]
    mpre = [0.0] + prec + [0.0]
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    # PR 曲线分段积分
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def _evaluate_class(
    class_id: int,
    gt_by_img: Dict[str, List[YoloDet]],
    pred_by_img: Dict[str, List[YoloDet]],
    iou_thr: float,
) -> float:
    # 收集该类的 GT 与 Pred
    gt_records = {}
    num_gt = 0
    for img, gts in gt_by_img.items():
        cls_gts = [g for g in gts if g.class_id == class_id]
        if cls_gts:
            gt_records[img] = {
                "boxes": [g.xyxy_norm() for g in cls_gts],
                "matched": [False] * len(cls_gts),
            }
            num_gt += len(cls_gts)

    if num_gt == 0:
        return -1.0  # 该类没有 GT，不计入 mAP

    preds: List[Tuple[str, float, Tuple[float, float, float, float]]] = []
    for img, preds_list in pred_by_img.items():
        for p in preds_list:
            if p.class_id != class_id:
                continue
            preds.append((img, p.conf, p.xyxy_norm()))

    if not preds:
        return 0.0

    # 按置信度排序
    preds.sort(key=lambda x: x[1], reverse=True)

    tp = [0] * len(preds)
    fp = [0] * len(preds)

    for idx, (img, conf, pbox) in enumerate(preds):
        record = gt_records.get(img)
        if record is None:
            fp[idx] = 1
            continue
        boxes = record["boxes"]
        matched = record["matched"]

        best_iou = 0.0
        best_j = -1
        for j, gbox in enumerate(boxes):
            iou = _iou_xyxy(pbox, gbox)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou >= iou_thr and best_j >= 0 and not matched[best_j]:
            tp[idx] = 1
            matched[best_j] = True
        else:
            fp[idx] = 1

    # 累积
    cum_tp = []
    cum_fp = []
    s_tp = 0
    s_fp = 0
    for i in range(len(preds)):
        s_tp += tp[i]
        s_fp += fp[i]
        cum_tp.append(s_tp)
        cum_fp.append(s_fp)

    rec = [ct / num_gt for ct in cum_tp]
    prec = [cum_tp[i] / max(1, (cum_tp[i] + cum_fp[i])) for i in range(len(preds))]
    return _compute_ap(rec, prec)


def map(iou_threshold: float = 0.5) -> Tuple[float, Dict[int, float]]:
    # 使用固定的绝对路径
    gt_dir = r"C:\Users\lqx94\Desktop\raceone\test\initial\labels"
    pred_dir = r"C:\Users\lqx94\Desktop\raceone\test\inferresult\inferlabels"

    gt_by_img, pred_by_img = _load_dataset(gt_dir, pred_dir)
    # 统一图片键集合（使用文件名保持一致）
    all_imgs = set(gt_by_img.keys()) | set(pred_by_img.keys())
    # 缺失文件填空
    for k in all_imgs:
        gt_by_img.setdefault(k, [])
        pred_by_img.setdefault(k, [])

    # 类别集合（从 GT 获取主集合）
    class_ids = set()
    for lst in gt_by_img.values():
        for d in lst:
            class_ids.add(d.class_id)
    # 若 GT 没有类，则从预测补充
    if not class_ids:
        for lst in pred_by_img.values():
            for d in lst:
                class_ids.add(d.class_id)

    per_class_ap: Dict[int, float] = {}
    valid_aps: List[float] = []
    for cid in sorted(class_ids):
        ap = _evaluate_class(cid, gt_by_img, pred_by_img, float(iou_threshold))
        per_class_ap[cid] = max(0.0, ap) if ap >= 0.0 else -1.0
        if ap >= 0.0:
            valid_aps.append(max(0.0, ap))

    mAP = sum(valid_aps) / len(valid_aps) if valid_aps else 0.0
    return mAP, per_class_ap


def map_range(
    iou_start: float = 0.5, iou_end: float = 0.95, iou_step: float = 0.05
) -> Tuple[float, Dict[int, float]]:
    """COCO 风格 mAP@[start:end]（步长 step，闭区间采样）。返回总体 mAP 以及各类的均值 AP。"""
    thresholds = []
    t = iou_start
    while t <= (iou_end + 1e-9):
        thresholds.append(round(t, 2))
        t += iou_step

    # 使用固定的绝对路径
    gt_dir = r"C:\Users\lqx94\Desktop\raceone\test\initial\labels"
    pred_dir = r"C:\Users\lqx94\Desktop\raceone\test\inferresult\inferlabels"

    gt_by_img, pred_by_img = _load_dataset(gt_dir, pred_dir)
    # 类别集合
    class_ids = set()
    for lst in gt_by_img.values():
        for d in lst:
            class_ids.add(d.class_id)
    if not class_ids:
        for lst in pred_by_img.values():
            for d in lst:
                class_ids.add(d.class_id)

    # 为每类累积 AP
    ap_sums: Dict[int, float] = {cid: 0.0 for cid in class_ids}
    ap_counts: Dict[int, int] = {cid: 0 for cid in class_ids}

    for thr in thresholds:
        for cid in class_ids:
            ap = _evaluate_class(cid, gt_by_img, pred_by_img, float(thr))
            if ap >= 0.0:  # 仅统计有 GT 的类
                ap_sums[cid] += max(0.0, ap)
                ap_counts[cid] += 1

    per_class = {}
    vals = []
    for cid in class_ids:
        if ap_counts[cid] > 0:
            val = ap_sums[cid] / ap_counts[cid]
            per_class[cid] = val
            vals.append(val)
        else:
            per_class[cid] = -1.0
    overall = sum(vals) / len(vals) if vals else 0.0
    return overall, per_class


def main():
    # 固定路径，无需命令行参数
    print("=" * 60)
    print("YOLO mAP计算工具")
    print("=" * 60)
    print(f"真值目录: C:\\Users\\lqx94\\Desktop\\raceone\\test\\initial\\labels")
    print(
        f"推理目录: C:\\Users\\lqx94\\Desktop\\raceone\\test\\inferresult\\inferlabels"
    )
    print("=" * 60)

    # 计算mAP@0.5
    m50, per50 = map(0.5)
    print(f"\nmAP@0.5: {m50:.4f}")
    if per50:
        print("\n各类别AP@0.5:")
        for cid in sorted(per50.keys()):
            ap = per50[cid]
            cname = CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else str(cid)
            if ap < 0:
                print(f"  类别 {cid:>2} ({cname:>6}): N/A (无真值)")
            else:
                print(f"  类别 {cid:>2} ({cname:>6}): {ap:.4f}")

    # 计算mAP@0.5:0.95
    print("\n" + "-" * 40)
    m9595, per9595 = map_range(0.5, 0.95, 0.05)
    print(f"\nmAP@0.5:0.95: {m9595:.4f}")
    if per9595:
        print("\n各类别AP@0.5:0.95:")
        for cid in sorted(per9595.keys()):
            ap = per9595[cid]
            cname = CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else str(cid)
            if ap < 0:
                print(f"  类别 {cid:>2} ({cname:>6}): N/A (无真值)")
            else:
                print(f"  类别 {cid:>2} ({cname:>6}): {ap:.4f}")

    print("\n" + "=" * 60)
    print("计算完成！")


if __name__ == "__main__":
    main()
