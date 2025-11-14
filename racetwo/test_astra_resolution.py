import os
import time
import sys
import cv2

# 让 Windows 能找到 astra 相关 DLL（必须在 import astra_py 之前）
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_CAM_ROOT = os.path.join(_BASE_DIR, 'camera')
if os.path.isdir(_CAM_ROOT):
    if _CAM_ROOT not in sys.path:
        sys.path.insert(0, _CAM_ROOT)
    for _root, _dirs, _files in os.walk(_CAM_ROOT):
        if _root not in sys.path:
            sys.path.insert(0, _root)
import astra_py


def main():
    cam = astra_py.AstraCamera()

    # 在 start() 之前设置分辨率（示例：1280x720@30；设备不支持会回退到最接近模式）
    cam.set_resolution(1920, 1080, 30)
    # 仅启用彩色输出，其余流关闭（start 时只开需要的流）
    cam.set_outputs(want_color=True, want_depth=False, want_shaded=False, want_cloud=False)

    cam.start()

    # 拉取 30 帧做简单验证（仅彩色）
    for i in range(30):
        frames = cam.get_frames()  # 无参，按 set_outputs 的配置返回
        (color,) = frames  # 只返回一个元素：color
        print('color shape:', color.shape, 'dtype:', color.dtype)
        cv2.imwrite(f'color_{i}.png', color)
        time.sleep(0.03)

    cam.stop()
    print('Done.')


if __name__ == '__main__':
    main()


