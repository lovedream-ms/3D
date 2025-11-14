#!/usr/bin/env python3
"""
批量推理测试脚本
对 test/initial/images 下的图片进行批量推理


输出文件结构：
- test/inferresult/images/    - 预测可视化图片

- test/inferresult/inferlabels/ - YOLO格式预测标签 (*.txt),可通过detect_round1获取。
- test/inferresult/result/    - 检测结果txt文件 (*_re.txt)

"""

import os
import cv2

from model import ModelManager
from detect import detect_round1, Cut_Desk_Extend, Overall_Address, Local_Address
from config import (
    TEST_IMAGES_DIR,
    TEST_INFERRESULT_DIR,
    DESK_EXPANSION_RATIO,
)


def ensure_dirs() -> None:
    """确保输出目录存在"""
    os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
    os.makedirs(TEST_INFERRESULT_DIR, exist_ok=True)
    # 创建推理结果的子目录（仅最终图与YOLO标签）
    os.makedirs(os.path.join(TEST_INFERRESULT_DIR, 'images'), exist_ok=True)
    os.makedirs(os.path.join(TEST_INFERRESULT_DIR, 'inferlabels'), exist_ok=True)
    
    
def _list_images(dir_path):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    if not os.path.isdir(dir_path):
        return []
    return [
        os.path.join(dir_path, f)
        for f in sorted(os.listdir(dir_path))
        if os.path.splitext(f)[1].lower() in exts
    ]


def _xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    w = max(0.0, float(x2) - float(x1))
    h = max(0.0, float(y2) - float(y1))
    cx = (float(x1) + float(x2)) / 2.0
    cy = (float(y1) + float(y2)) / 2.0
    return cx / img_w, cy / img_h, w / img_w, h / img_h


def _save_yolo_labels(boxes_xyxy, clses, img_w, img_h, class_names, save_path):
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            if boxes_xyxy is None or clses is None or len(boxes_xyxy) == 0:
                return
            for (x1, y1, x2, y2), cid in zip(boxes_xyxy, clses):
                cid = int(cid)
                # 保留所有类别（包含 desk）
                cx, cy, w, h = _xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)
                f.write(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    except Exception as e:
        print(f"Failed to write YOLO labels {save_path}: {e}")


def _get_crop_by_desk(rgb_img, overall_detector):
    """用模型1找桌子并返回裁剪图及其在原图上的坐标 (x1,y1,x2,y2)。找不到返回 (None, None)。"""
    try:
        boxes_xyxy, confs, clses = overall_detector.infer(rgb_img.copy())
        if boxes_xyxy is None or clses is None or len(boxes_xyxy) == 0:
            return None, None
        # 找 desk 类索引
        desk_ids = [i for i, name in enumerate(overall_detector.classes) if name == 'desk']
        if not desk_ids:
            return None, None
        import numpy as np
        mask = np.isin(clses, np.array(desk_ids, dtype=np.int32))
        if not mask.any():
            return None, None
        desk_boxes = boxes_xyxy[mask]
        crop_img, crop_xyxy = Cut_Desk_Extend(rgb_img, desk_boxes, expansion_ratio=DESK_EXPANSION_RATIO)
        return crop_img, crop_xyxy
    except Exception:
        return None, None


def fused_detect_final_boxes(rgb_img, overall_detector, local_detector):
    """与 main.py/ detect.py 一致的融合逻辑：
    1) Overall_Address 获取模型1结果与 desk 框
    2) 基于 desk 框裁剪并外扩（Cut_Desk_Extend）
    3) Local_Address 在裁剪图上推理并与模型1结果融合，得到回归到原图坐标的 boxes/cls
    返回：(final_img_for_vis, fused_boxes_xyxy, fused_clses)
    若流程无法完成（无桌子或无检测），返回空集合。
    """
    import numpy as np
    # 1) 模型1：全图检测
    stage1_img, boxes1, confs1, clses1, desk_boxes = Overall_Address(rgb_img, overall_detector)
    final_img = stage1_img
    if desk_boxes is None:
        return final_img, np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    # 2) 桌子裁剪与外扩
    crop_img, crop_xyxy = Cut_Desk_Extend(rgb_img, desk_boxes, expansion_ratio=DESK_EXPANSION_RATIO)
    if crop_img is None:
        return final_img, np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    # 3) 模型2融合
    local_img, merge_img, stage3_img, boxes_xyxy, confs2, clses2, desk_boxes_m2 = Local_Address(
        rgb_img, crop_img, crop_xyxy, local_detector,
        boxes1, confs1, clses1, overall_detector,
        perobj_save_dir=None
    )
    out_img = stage3_img if stage3_img is not None else final_img
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return out_img, np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    return out_img, boxes_xyxy, clses2


def main():
    ensure_dirs()

    # 初始化模型
    manager = ModelManager()
    overall = manager.get_overall_detector()
    local = manager.get_local_detector()
    # 分类模型已弃用，不再加载

    images = _list_images(TEST_IMAGES_DIR)
    if not images:
        print(f"No images found in {TEST_IMAGES_DIR}")
        return

    out_img_dir = os.path.join(TEST_INFERRESULT_DIR, 'images')
    out_lbl_dir = os.path.join(TEST_INFERRESULT_DIR, 'inferlabels')

    processed = 0
    for img_path in images:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        try:
            bgr = cv2.imread(img_path)
            if bgr is None:
                print(f"Skip unreadable: {img_path}")
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]

            # 运行一次 detect_round1，获取最终可视化图
            result = detect_round1(
                detector=overall,
                frame_provider=lambda: rgb.copy(),
                update_cb=None,
                round_id=1,
                team_name='BATCH',
                overall_detector=overall,
                local_detector=local,
                cls_model=None,
                session_dir=None,
                session_dir_result=None,
            )

            final_img = None
            if isinstance(result, tuple):
                if len(result) == 7:
                    _, _, _, final_img, _, _, _ = result
                elif len(result) == 5:
                    final_img, _, _, _, _ = result
                elif len(result) == 4:
                    final_img, _, _, _ = result
            if final_img is None:
                final_img = rgb

            # 保存最终检测图
            out_img_path = os.path.join(out_img_dir, f"{stem}_pred.png")
            cv2.imwrite(out_img_path, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))

            # 基于与 main.py 一致的融合结果生成 YOLO 标签；若失败则回退到 overall 推理
            out_lbl_path = os.path.join(out_lbl_dir, f"{stem}.txt")
            try:
                final_img_fused, fused_boxes, fused_clses = fused_detect_final_boxes(rgb, overall, local)
                if fused_boxes is not None and len(fused_boxes) > 0:
                    _save_yolo_labels(fused_boxes, fused_clses, w, h, local.classes, out_lbl_path)
                else:
                    boxes_xyxy, confs, clses = overall.infer(rgb.copy())
                    _save_yolo_labels(boxes_xyxy, clses, w, h, overall.classes, out_lbl_path)
            except Exception:
                boxes_xyxy, confs, clses = overall.infer(rgb.copy())
                _save_yolo_labels(boxes_xyxy, clses, w, h, overall.classes, out_lbl_path)

            processed += 1
            print(f"Processed {processed}: {img_path}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print(f"Done. Total images processed: {processed}")


if __name__ == '__main__':
    main()

