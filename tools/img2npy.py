import cv2
import numpy as np


if __name__ == "__main__":
    import glob
    import os
    import json
    from ultralytics import YOLO
    from collections import Counter

    model = YOLO("models/yolov8n.pt")
    root = "datasets/race"
    for base in os.listdir(root):
        for img_path in glob.glob(os.path.join(root, base, "*.png")):
            npz_path = img_path.replace(".png", ".npz")
            rgbFrame = cv2.imread(img_path)
            np.savez(
                npz_path,
                rgbFrame=rgbFrame,
                depthFrame=None,
                pointCloudFrame=None,
            )

            result = model.predict(rgbFrame, verbose=False)[0]

            class_indices = result.boxes.cls.cpu().numpy().astype(int)
            class_names = [result.names[idx] for idx in class_indices]

            count_dict = dict(Counter(class_names))

            json_path = os.path.join(root, base, "gt.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(count_dict, f, indent=4, ensure_ascii=False)

            print(f"已处理: {img_path} -> 识别结果: {count_dict}")
