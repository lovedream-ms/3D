#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将源目录中的 BGR 图片批量转换为 RGB 并保存到目标目录。

默认：
- 源目录: /home/HwHiAiUser/Desktop/raceone/data/capture
- 目标目录: /home/HwHiAiUser/Desktop/raceone/data/RGB

用法示例：
  python convert_bgr_to_rgb.py
  python convert_bgr_to_rgb.py --src /path/to/capture --dst /path/to/RGB
"""

import os
import sys
import argparse
from typing import Tuple

import cv2


DEFAULT_SRC = r"C:\Users\lqx94\Desktop\raceone\data\capture"
DEFAULT_DST = r"C:\Users\lqx94\Desktop\raceone\data\RGB"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def is_image_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename)
    return ext.lower() in IMAGE_EXTS


def convert_one(src_path: str, dst_path: str) -> Tuple[bool, str]:
    try:
        bgr = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if bgr is None:
            return False, f"skip (unreadable): {src_path}"
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        ok = cv2.imwrite(dst_path, rgb)
        if not ok:
            return False, f"write failed: {dst_path}"
        return True, dst_path
    except Exception as e:
        return False, f"error: {src_path} -> {e}"


def convert_dir(src_root: str, dst_root: str) -> Tuple[int, int]:
    total = 0
    ok_count = 0
    for root, _, files in os.walk(src_root):
        rel = os.path.relpath(root, src_root)
        for fname in files:
            if not is_image_file(fname):
                continue
            total += 1
            src_path = os.path.join(root, fname)
            dst_path = os.path.join(dst_root, rel, fname)
            ok, msg = convert_one(src_path, dst_path)
            if ok:
                ok_count += 1
                print(f"[OK] {src_path} -> {msg}")
            else:
                print(f"[FAIL] {msg}")
    return ok_count, total


def main():
    parser = argparse.ArgumentParser(description="Convert BGR images to RGB and save to target directory.")
    parser.add_argument("--src", default=DEFAULT_SRC, help="Source directory (BGR images)")
    parser.add_argument("--dst", default=DEFAULT_DST, help="Destination directory for RGB images")
    args = parser.parse_args()

    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)

    if not os.path.isdir(src):
        print(f"Source directory not found: {src}")
        sys.exit(1)

    os.makedirs(dst, exist_ok=True)
    ok_count, total = convert_dir(src, dst)
    print(f"Done. Converted {ok_count}/{total} images. Output: {dst}")


if __name__ == "__main__":
    main()


