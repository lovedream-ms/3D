# detect.py
import os
from collections import defaultdict
# 可选导入 PyQt6：缺失时跳过 GUI 刷新
try:
    from PyQt6.QtWidgets import QApplication   # 用于 processEvents
except Exception:
    QApplication = None

from config import (
    CLASS_MAX_COUNTS, RESULT_FOLDER, DESK_EXPANSION_RATIO, PRED_CONF_THRES,
    M2_CLASS_SCORE_THRESHOLDS, CROSS_MODEL_MATCH_IOU,
    M1_CLASS_SCORE_THRESHOLDS, MERGE_CLASS_PREFERENCE,
    SUPPLEMENT_ALLOWED_SOURCES,
)
import numpy as np
import cv2

# ---------- 辅助函数 (从 deskadres.py 合并) ----------
def box_random_color(name):
    """为给定类别名生成稳定的随机颜色（BGR）。"""
    try:
        seed = np.uint32(abs(hash(str(name))) & 0xFFFFFFFF)
        rs = np.random.RandomState(seed)
        b, g, r = rs.randint(0, 256, size=3).tolist()
        return int(b), int(g), int(r)
    except Exception:
        return (0, 255, 0)

BOX_THICKNESS = 2  # 稍微加粗


def Merge_box(frame, crop_xyxy, boxes2, confs2, clses2, boxes1_all, confs1_all, clses1_all,
              overall_detector, local_detector):
    """模型1和模型2的检测结果融合

    Args:
        frame: 原图图像（用于绘制结果）
        crop_xyxy: 裁剪区域坐标 (x1, y1, x2, y2)
        boxes2: 模型2检测框坐标
        confs2: 模型2置信度
        clses2: 模型2类别ID
        boxes1_all: 模型1的所有检测框
        confs1_all: 模型1的所有置信度
        clses1_all: 模型1的所有类别ID
        overall_detector: 全图检测器（模型1）
        local_detector: 区域检测器（模型2）

    Returns:
        tuple: (stage3_img, boxes_xyxy, confs, clses, desk_boxes)
               - stage3_img: 在原图上绘制融合结果的图像
               - boxes_xyxy: 融合后的检测框坐标（原图坐标系）
               - confs: 融合后的置信度
               - clses: 融合后的类别ID
               - desk_boxes: 融合结果中的桌子检测框
    """
    # 如果没有模型1的数据，直接处理模型2的结果
    if boxes1_all is None or confs1_all is None or clses1_all is None or overall_detector is None:
        # 将模型2检测结果从裁剪坐标系回归到原图坐标系
        boxes_xyxy = None
        if boxes2 is not None and boxes2.size > 0:
            cx1, cy1, cx2, cy2 = crop_xyxy
            boxes_xyxy = boxes2.copy()
            boxes_xyxy[:, 0] += cx1  # x1
            boxes_xyxy[:, 1] += cy1  # y1
            boxes_xyxy[:, 2] += cx1  # x2
            boxes_xyxy[:, 3] += cy1  # y2

        # 绘制模型2结果图像（在原图上）
        stage3_img = frame.copy()
        if boxes_xyxy is not None and boxes_xyxy.size > 0:
            for (x1, y1, x2, y2), score, cid in zip(boxes_xyxy.tolist(), confs2.tolist(), clses2.tolist()):
                cname = local_detector.classes[cid] if 0 <= cid < len(local_detector.classes) else str(cid)
                color = box_random_color(cname)
                cv2.rectangle(stage3_img, (int(x1), int(y1)), (int(x2), int(y2)), color, BOX_THICKNESS)
                label = f"{cname}:{score:.2f}"
                cv2.putText(stage3_img, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 提取模型2中的桌子框（如果有的话）
        desk_boxes = None
        desk_ids = [i for i, name in enumerate(local_detector.classes) if name == 'desk']
        if len(desk_ids) > 0 and boxes_xyxy is not None and len(boxes_xyxy) > 0:
            desk_mask = np.isin(clses2, np.array(desk_ids, dtype=np.int32))
            if desk_mask.any():
                desk_boxes = boxes_xyxy[desk_mask]

        return stage3_img, boxes_xyxy, confs2, clses2, desk_boxes

    try:
        # 1) 收集模型1的非桌子预测，并做阈值过滤
        non_desk_mask_m1 = np.logical_not(np.isin(clses1_all, np.array([i for i, name in enumerate(overall_detector.classes) if name == 'desk'], dtype=np.int32)))
        boxes1_filtered = boxes1_all[non_desk_mask_m1]
        confs1_filtered = confs1_all[non_desk_mask_m1]
        clses1_filtered = clses1_all[non_desk_mask_m1]

        m1_boxes_in_crop = []
        m1_confs_in_crop = []
        m1_clses_in_crop = []

        if boxes1_filtered is not None and len(boxes1_filtered) > 0:
            cx1, cy1, cx2, cy2 = crop_xyxy
            crop_w = max(0, cx2 - cx1)
            crop_h = max(0, cy2 - cy1)

            for (bx1, by1, bx2, by2), bconf, bcid in zip(boxes1_filtered.tolist(), confs1_filtered.tolist(), clses1_filtered.tolist()):
                bname = overall_detector.classes[bcid] if 0 <= int(bcid) < len(overall_detector.classes) else str(bcid)
                if bname == 'desk':
                    continue
                thr1 = float(M1_CLASS_SCORE_THRESHOLDS.get(bname, M2_CLASS_SCORE_THRESHOLDS.get(bname, PRED_CONF_THRES)))
                if float(bconf) < thr1:
                    continue

                # 将模型1框从全图坐标变换到裁剪坐标系
                tx1 = max(0, min(crop_w, int(round(bx1 - cx1))))
                ty1 = max(0, min(crop_h, int(round(by1 - cy1))))
                tx2 = max(0, min(crop_w, int(round(bx2 - cx1))))
                ty2 = max(0, min(crop_h, int(round(by2 - cy1))))
                if tx2 - tx1 <= 1 or ty2 - ty1 <= 1:
                    continue

                # 将模型1类别名映射到模型2的类别索引
                try:
                    m2_cid = local_detector.classes.index(bname)
                except Exception:
                    continue

                m1_boxes_in_crop.append([tx1, ty1, tx2, ty2])
                m1_confs_in_crop.append(float(bconf))
                m1_clses_in_crop.append(int(m2_cid))

        # 2) 与模型2做IoU匹配（并根据冲突策略决定保留谁）
        def _iou_xyxy(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter = iw * ih
            if inter <= 0:
                return 0.0
            aa = max(0, (ax2 - ax1)) * max(0, (ay2 - ay1))
            bb = max(0, (bx2 - bx1)) * max(0, (by2 - by1))
            union = aa + bb - inter
            return float(inter) / float(union) if union > 0 else 0.0

        boxes2_list = boxes2.tolist() if boxes2 is not None and boxes2.size > 0 else []
        confs2_list = confs2.tolist() if confs2 is not None and confs2.size > 0 else []
        clses2_list = clses2.tolist() if clses2 is not None and clses2.size > 0 else []

        matched_m2_indices = set()

        if len(m1_boxes_in_crop) > 0:
            for m1_box, m1_conf, m1_cid in zip(m1_boxes_in_crop, m1_confs_in_crop, m1_clses_in_crop):
                best_iou = 0.0
                best_j = -1
                for j, m2_box in enumerate(boxes2_list):
                    iou = _iou_xyxy(m1_box, m2_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j
                if best_iou >= float(CROSS_MODEL_MATCH_IOU) and best_j >= 0:
                    matched_m2_indices.add(best_j)
                    # 已有匹配：处理类别冲突
                    try:
                        m1_name = local_detector.classes[int(m1_cid)]
                    except Exception:
                        m1_name = str(int(m1_cid))
                    try:
                        m2_name = local_detector.classes[int(clses2_list[best_j])]
                    except Exception:
                        m2_name = str(int(clses2_list[best_j]))

                    # 若类别一致，直接继续
                    if str(m1_name) == str(m2_name):
                        continue

                    # 根据配置决定取 overall 或 local（both 情况下维持模型2即可）
                    pref = str(MERGE_CLASS_PREFERENCE.get(m1_name) or MERGE_CLASS_PREFERENCE.get(m2_name) or "")
                    if pref.lower() == 'overall':
                        # 用模型1替换模型2的该匹配框
                        boxes2_list[best_j] = [int(m1_box[0]), int(m1_box[1]), int(m1_box[2]), int(m1_box[3])]
                        confs2_list[best_j] = float(m1_conf)
                        clses2_list[best_j] = int(m1_cid)
                    elif pref.lower() == 'local':
                        # 保留模型2原结果
                        pass
                    else:
                        # 未配置或 both 时，保持模型2
                        pass
                    continue

                # 无匹配：根据补框策略决定是否从模型1补框到模型2
                try:
                    m1_name_unmatched = local_detector.classes[int(m1_cid)]
                except Exception:
                    m1_name_unmatched = str(int(m1_cid))
                
                # 检查补框策略
                supplement_allowed = str(SUPPLEMENT_ALLOWED_SOURCES.get(m1_name_unmatched, "both"))
                if supplement_allowed.lower() in ('overall', 'both'):
                    boxes2_list.append([int(m1_box[0]), int(m1_box[1]), int(m1_box[2]), int(m1_box[3])])
                    confs2_list.append(float(m1_conf))
                    clses2_list.append(int(m1_cid))
                # 如果是 'local'，则不补框

        # 过滤模型2中"仅模型2检测到"的框：根据补框策略决定是否保留
        if len(boxes2_list) > 0:
            keep_idx = []
            for j in range(len(boxes2_list)):
                if j in matched_m2_indices:
                    # 已匹配的框总是保留
                    keep_idx.append(j)
                    continue
                
                # 对于仅模型2检测到的框，检查补框策略
                try:
                    cname_j = local_detector.classes[int(clses2_list[j])]
                except Exception:
                    cname_j = str(int(clses2_list[j]))
                
                if cname_j == 'desk':
                    # desk 总是保留
                    keep_idx.append(j)
                    continue
                
                supplement_allowed = str(SUPPLEMENT_ALLOWED_SOURCES.get(cname_j, "both"))
                if supplement_allowed.lower() in ('local', 'both'):
                    keep_idx.append(j)
                # 如果是 'overall'，则不保留仅模型2的框
            if len(keep_idx) < len(boxes2_list):
                boxes2_list = [boxes2_list[k] for k in keep_idx]
                confs2_list = [confs2_list[k] for k in keep_idx]
                clses2_list = [clses2_list[k] for k in keep_idx]

        # 回写合并结果
        if len(boxes2_list) == 0:
            boxes2 = boxes2[:0]
            confs2 = confs2[:0]
            clses2 = clses2[:0]
        else:
            boxes2 = np.array(boxes2_list, dtype=np.float32)
            confs2 = np.array(confs2_list, dtype=np.float32)
            clses2 = np.array(clses2_list, dtype=np.int32)
    except Exception:
        pass  # 融合失败时，继续使用原模型2结果

    # 将融合后的检测结果从裁剪坐标系回归到原图坐标系
    boxes_xyxy = None
    if boxes2 is not None and boxes2.size > 0:
        cx1, cy1, cx2, cy2 = crop_xyxy
        boxes_xyxy = boxes2.copy()
        boxes_xyxy[:, 0] += cx1  # x1
        boxes_xyxy[:, 1] += cy1  # y1
        boxes_xyxy[:, 2] += cx1  # x2
        boxes_xyxy[:, 3] += cy1  # y2

    # 绘制融合结果图像（在原图上）
    stage3_img = frame.copy()
    if boxes_xyxy is not None and boxes_xyxy.size > 0:
        for (x1, y1, x2, y2), score, cid in zip(boxes_xyxy.tolist(), confs2.tolist(), clses2.tolist()):
            cname = local_detector.classes[cid] if 0 <= cid < len(local_detector.classes) else str(cid)
            color = box_random_color(cname)
            cv2.rectangle(stage3_img, (int(x1), int(y1)), (int(x2), int(y2)), color, BOX_THICKNESS)
            label = f"{cname}:{score:.2f}"
            cv2.putText(stage3_img, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 提取融合结果中的桌子框（如果有的话）
    desk_boxes = None
    desk_ids = [i for i, name in enumerate(local_detector.classes) if name == 'desk']
    if len(desk_ids) > 0 and boxes_xyxy is not None and len(boxes_xyxy) > 0:
        desk_mask = np.isin(clses2, np.array(desk_ids, dtype=np.int32))
        if desk_mask.any():
            desk_boxes = boxes_xyxy[desk_mask]

    return stage3_img, boxes_xyxy, confs2, clses2, desk_boxes

def Cut_Desk_Extend(img, boxes_xyxy, expansion_ratio=0.10):
    """根据传入的桌子框集合，合并并外扩后返回裁剪区域与坐标。

    参数 boxes_xyxy 仅应包含桌子类的框。
    返回: (crop_img, (x1,y1,x2,y2)) 或 (None, None) 如果没有桌子。
    """
    H, W = img.shape[:2]
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return None, None

    x1 = float(np.min(boxes_xyxy[:, 0]))
    y1 = float(np.min(boxes_xyxy[:, 1]))
    x2 = float(np.max(boxes_xyxy[:, 2]))
    y2 = float(np.max(boxes_xyxy[:, 3]))
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1 or bh <= 1:
        return None, None
    x1 -= bw * expansion_ratio
    y1 -= bh * expansion_ratio
    x2 += bw * expansion_ratio
    y2 += bh * expansion_ratio
    x1 = max(0.0, min(W, x1))
    y1 = max(0.0, min(H, y1))
    x2 = max(0.0, min(W, x2))
    y2 = max(0.0, min(H, y2))
    ix1, iy1, ix2, iy2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    if ix2 - ix1 <= 1 or iy2 - iy1 <= 1:
        return None, None
    crop = img[iy1:iy2, ix1:ix2, :]
    return crop.copy(), (ix1, iy1, ix2, iy2)

def Overall_Address(frame, overall_detector):
    """模型1：全图检测器处理逻辑
    
    Args:
        frame: 输入图像帧
        overall_detector: 全图检测器
    
    Returns:
        tuple: (stage1_img, boxes_xyxy, confs, clses, desk_boxes)
               - stage1_img: 绘制了检测结果的图像
               - boxes_xyxy: 检测框坐标
               - confs: 置信度
               - clses: 类别ID
               - desk_boxes: 桌子检测框
    """
    # 模型1检测
    boxes_xyxy, confs, clses = overall_detector.infer(frame.copy())

    # 可视化模型1结果
    stage1_img = frame.copy()
    try:
        if boxes_xyxy is not None and len(boxes_xyxy) > 0:
            for (x1, y1, x2, y2), score, cid in zip(boxes_xyxy.tolist(), confs.tolist(), clses.tolist()):
                cname = overall_detector.classes[cid] if 0 <= cid < len(overall_detector.classes) else str(cid)
                thr = float(M1_CLASS_SCORE_THRESHOLDS.get(cname, M2_CLASS_SCORE_THRESHOLDS.get(cname, PRED_CONF_THRES)))
                if float(score) < thr:
                    continue
                color = box_random_color(cname)
                cv2.rectangle(stage1_img, (int(x1), int(y1)), (int(x2), int(y2)), color, BOX_THICKNESS)
                label = f"{cname}:{score:.2f}"
                cv2.putText(stage1_img, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    except Exception:
        stage1_img = frame.copy()

    # 提取桌子框
    desk_boxes = None
    desk_ids = [i for i, name in enumerate(overall_detector.classes) if name == 'desk']
    if len(desk_ids) > 0 and boxes_xyxy is not None and len(boxes_xyxy) > 0:
        desk_mask = np.isin(clses, np.array(desk_ids, dtype=np.int32))
        if desk_mask.any():
            desk_boxes = boxes_xyxy[desk_mask]

    return stage1_img, boxes_xyxy, confs, clses, desk_boxes

def Local_Address(frame, crop_img, crop_xyxy, local_detector, boxes1_all=None, confs1_all=None, clses1_all=None, overall_detector=None, perobj_save_dir=None):
    """模型2：区域检测器处理逻辑（包含融合）
    
    Args:
        frame: 原图图像（用于绘制结果）
        crop_img: 裁剪后的图像
        crop_xyxy: 裁剪区域坐标 (x1, y1, x2, y2)
        local_detector: 区域检测器
        boxes1_all: 模型1的非桌子框 (可选，用于融合)
        confs1_all: 模型1的置信度 (可选，用于融合)
        clses1_all: 模型1的类别ID (可选，用于融合)
        overall_detector: 全图检测器 (可选，用于融合)
    
    Returns:
        tuple: (local_img, merge_img, stage3_img, boxes_xyxy, confs, clses, desk_boxes)
               - local_img: Local_Address检测图片（模型2原始检测结果）
               - merge_img: Merge_box后检测图片（融合结果）
               - stage3_img: 在原图上绘制了融合结果的图像
               - boxes_xyxy: 回归到原图坐标系的检测框坐标
               - confs: 融合后的置信度
               - clses: 融合后的类别ID
               - desk_boxes: 融合结果中的桌子检测框
    """
    # 模型2检测（检测器预处理期望 BGR，此处从 RGB 转 BGR 再传入）
    # 统一使用 RGB 输入
    crop_rgb = crop_img
    boxes2, confs2, clses2 = local_detector.infer(crop_rgb)

    # 基于类别阈值过滤
    keep = []
    for i, (cid, sc) in enumerate(zip(clses2.tolist(), confs2.tolist())):
        cname = local_detector.classes[cid] if 0 <= cid < len(local_detector.classes) else str(cid)
        thr = float(M2_CLASS_SCORE_THRESHOLDS.get(cname, PRED_CONF_THRES))
        if float(sc) >= thr:
            keep.append(i)

    if len(keep) > 0:
        keep = np.array(keep, dtype=np.int64)
        boxes2, confs2, clses2 = boxes2[keep], confs2[keep], clses2[keep]
    else:
        boxes2 = boxes2[:0]
        confs2 = confs2[:0]
        clses2 = clses2[:0]

    # 创建Local_Address检测图片（模型2原始检测结果）
    local_img = frame.copy()
    if boxes2 is not None and boxes2.size > 0:
        # 将模型2检测结果从裁剪坐标系回归到原图坐标系
        cx1, cy1, cx2, cy2 = crop_xyxy
        local_boxes_xyxy = boxes2.copy()
        local_boxes_xyxy[:, 0] += cx1  # x1
        local_boxes_xyxy[:, 1] += cy1  # y1
        local_boxes_xyxy[:, 2] += cx1  # x2
        local_boxes_xyxy[:, 3] += cy1  # y2

        # 在原图上绘制模型2的原始检测结果
        for (x1, y1, x2, y2), score, cid in zip(local_boxes_xyxy.tolist(), confs2.tolist(), clses2.tolist()):
            cname = local_detector.classes[cid] if 0 <= cid < len(local_detector.classes) else str(cid)
            color = box_random_color(cname)
            cv2.rectangle(local_img, (int(x1), int(y1)), (int(x2), int(y2)), color, BOX_THICKNESS)
            label = f"{cname}:{score:.2f}"
            cv2.putText(local_img, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 将模型2逐目标置信度写入指定目录（local）
    if perobj_save_dir is not None:
        try:
            os.makedirs(perobj_save_dir, exist_ok=True)
            for i, (cid, sc) in enumerate(zip(clses2.tolist(), confs2.tolist())):
                cname = local_detector.classes[cid] if 0 <= cid < len(local_detector.classes) else str(cid)
                if cname == 'desk':
                    continue
                with open(os.path.join(perobj_save_dir, f"obj_{i:03d}.txt"), 'w', encoding='utf-8') as f:
                    f.write(f"det_label={cname};det_conf={float(sc):.4f}\n")
        except Exception:
            pass

    # 使用融合函数处理模型1和模型2的结果融合
    merge_img, boxes_xyxy, confs2, clses2, desk_boxes = Merge_box(
        frame, crop_xyxy, boxes2, confs2, clses2, boxes1_all, confs1_all, clses1_all,
        overall_detector, local_detector
    )

    # 创建最终绘制的基底图：使用原始帧的干净拷贝
    stage3_img = frame.copy()

    return local_img, merge_img, stage3_img, boxes_xyxy, confs2, clses2, desk_boxes

def detect_round1(detector, frame_provider, update_cb,
                  team_name="TEAM", round_id=1,
                  overall_detector=None, local_detector=None,
                  cls_model=None,
                  session_dir=None,
                  session_dir_result=None):
    """两阶段检测主流程（重构版）

    使用分离的函数处理不同模型的逻辑：
    1) Overall_Address: 模型1全图检测
    2) Local_Address: 模型2区域检测和模型融合
    3) Cls_Address: 模型3分类验证和结果处理
    Args:
        detector: 默认检测器
        frame_provider: 图像帧提供函数
        update_cb: UI更新回调
        team_name: 队伍名称
        round_id: 轮次ID
        overall_detector: 全图检测器（模型1）
        local_detector: 区域检测器（模型2）
        cls_model: 分类模型（模型3）
        session_dir: 保存裁剪图片的目录
        session_dir_result: 保存分类结果的目录
    Returns:
        tuple: (overall_img, local_img, merge_img, final_img, cls_dir, clsresult_dir, final_counts)
               - overall_img: Overall_Address检测图片
               - local_img: Local_Address检测图片
               - merge_img: Merge_box后检测图片
               - final_img: Cls_Address处理后的原图检测图片
               - cls_dir: cls文件夹路径
               - clsresult_dir: clsresult文件夹路径
               - final_counts: 最终计数结果
    """
    
    frame = frame_provider()

    # 双模型路径（模型1 + 模型2 + 模型3）
    if overall_detector is not None and local_detector is not None:
        # 1. 模型1：全图检测
        stage1_img, boxes1, confs1, clses1, desk_boxes = Overall_Address(frame, overall_detector)

        # 将模型1逐目标置信度写入指定目录（overall）
        if session_dir is not None:
            try:
                os.makedirs(session_dir, exist_ok=True)
                if boxes1 is not None and len(boxes1) > 0:
                    for i, (cid, sc) in enumerate(zip(clses1.tolist(), confs1.tolist())):
                        name = overall_detector.classes[cid] if 0 <= cid < len(overall_detector.classes) else str(cid)
                        if name == 'desk':
                            continue
                        with open(os.path.join(session_dir, f"obj_{i:03d}.txt"), 'w', encoding='utf-8') as f:
                            f.write(f"det_label={name};det_conf={float(sc):.4f}\n")
            except Exception:
                pass

        # 检查是否有桌子检测结果
        if desk_boxes is not None:
            # 2. 桌子裁剪和扩展
            crop_img, crop_xyxy = Cut_Desk_Extend(frame, desk_boxes, expansion_ratio=DESK_EXPANSION_RATIO)

            if crop_img is not None:
                # 3. 模型2：区域检测和模型融合
                local_img, merge_img, stage3_img, boxes2, confs2, clses2, desk_boxes_m2 = Local_Address(
                    frame, crop_img, crop_xyxy, local_detector,
                    boxes1, confs1, clses1, overall_detector,
                    perobj_save_dir=session_dir_result
                )

                # 4. 使用融合结果作为最终展示图（优先 merge_img，其次 local_img，最后回退原图）
                out_img = merge_img if merge_img is not None else (local_img if local_img is not None else stage3_img)
                final_counts = cal_classes_num(local_detector, clses2)

                # UI更新
                if update_cb:
                    update_cb(out_img)
                try:
                    if QApplication is not None:
                        QApplication.processEvents()
                except Exception:
                    pass

                return stage1_img, local_img, merge_img, out_img, session_dir, session_dir_result, final_counts
        else:
            # 没有检测到桌子，直接返回Overall的结果
            print("⚠️ 未检测到桌子，仅返回Overall检测结果")
            empty_counts = defaultdict(int)
            # 计算Overall检测结果的计数（排除桌子）
            if boxes1 is not None and len(boxes1) > 0:
                for cid in clses1.tolist():
                    name = overall_detector.classes[cid] if 0 <= cid < len(overall_detector.classes) else str(cid)
                    if name != 'desk':
                        empty_counts[name] += 1
            
            # UI更新
            if update_cb:
                update_cb(stage1_img)
            try:
                if QApplication is not None:
                    QApplication.processEvents()
            except Exception:
                pass
            
            return stage1_img, None, None, stage1_img, None, None, empty_counts

    # 单模型路径（降级处理）- 已移除，使用空结果返回
    print("⚠️ 检测器不可用，返回空结果")
    empty_counts = defaultdict(int)
    empty_img = frame.copy()
    return empty_img, None, None, empty_img, None, None, empty_counts

def cal_classes_num(detector, clses):
    cnt = defaultdict(int)
    for cid in clses.tolist():
        name = detector.classes[cid] if 0 <= cid < len(detector.classes) else str(cid)
        cnt[name] += 1
    return cnt

def _write_txt(counts, rid, team, table: int = 1, num=None):
    """写单轮（或单桌）结果到 result_r，格式：
    START\nGoal_ID=CA001;Num=2;Table=1\n...\nEND
    """
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    def _lines_for(counts_map, table_id: int):
        return ["START"] + [
            f"Goal_ID={k};Num={int(v)};Table={int(table_id)}" for k, v in counts_map.items() if k != "desk"
        ] + ["END"]

    # 常规写入
    if num is None:
        lines = _lines_for(counts, table)
        path = os.path.join(RESULT_FOLDER, f"{team}-R{rid}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"✅ 结果写入: {path}")
        return

    # 兼容旧的 "math" 分支（修正等号判断）
    if num == "math":
        lines = _lines_for(counts, table)
        # 保持历史附加项（如需）
        lines = lines[:-1] + ["Goal_ID=W001;Num=1;Table={}".format(int(table)),
                              "Goal_ID=W002;Num=1;Table={}".format(int(table))] + ["END"]
        path = os.path.join(RESULT_FOLDER, f"{team}-R{rid}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"✅ 结果写入: {path}")





