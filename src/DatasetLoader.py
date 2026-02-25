import json
import os
import glob
from dataclasses import dataclass
from collections import Counter


@dataclass
class Scenario:
    name: str  # 场景名称 (如 "scene_001_table1_easy")
    data_dir: str  # 该场景数据的绝对或相对路径
    ground_truth: Counter  # 真实的物体数量标签


def load_dataset(dataset_root: str = "dataset") -> list[Scenario]:
    scenarios = []

    for scene_name in os.listdir(dataset_root):
        scene_dir = os.path.join(dataset_root, scene_name)

        if not os.path.isdir(scene_dir):
            continue

        gt_path = os.path.join(scene_dir, "gt.json")

        if not os.path.exists(gt_path):
            print(f"警告: 场景 {scene_name} 缺少 gt.json，已跳过。")
            continue

        with open(gt_path, "r", encoding="utf-8") as f:
            gt_dict = json.load(f)
            gt_counter = Counter(gt_dict)

        scenarios.append(
            Scenario(name=scene_name, data_dir=scene_dir, ground_truth=gt_counter)
        )

    print(f"成功加载 {len(scenarios)} 个测试场景！")
    return scenarios


def calculate_error(pred: Counter, gt: Counter) -> float:
    error = 0
    all_classes = set(pred.keys()).union(set(gt.keys()))

    for cls_name in all_classes:
        pred_count = pred.get(cls_name, 0)
        gt_count = gt.get(cls_name, 0)
        error += abs(pred_count - gt_count)

    return error
