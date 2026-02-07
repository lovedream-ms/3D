import os, sys
from Socket import JudgeBoxClient  # JudgeBoxClient
from config import (
    RESULT_FOLDER,
    TEAM_SHORT_NAME,
    JUDGE_BOX_IP,
    JUDGE_BOX_PORT,
    CAPTURE_SAVE_DIR,
    WARMUP_IMAGE_PATH,
    ENV_GOAL_MAP,
)


# ---- 全局裁判盒客户端（仿照 tongxin.py：懒加载 + 统一实例） ----
_client = None


def _get_judge_client():
    global _client
    try:
        if _client is None:
            _client = JudgeBoxClient(ip=JUDGE_BOX_IP, port=JUDGE_BOX_PORT)
        return _client
    except Exception:
        return None


# 提前发送 Start1（在导入 PyQt/相机等重依赖前），确保即便后续 GUI 启动失败，也能发出
try:
    _c_early = _get_judge_client()
    if _c_early is not None:
        _c_early.send_start(1)
        print("[INFO] Early start signal (R1) sent before GUI init.")
except Exception as _e:
    try:
        print(f"[WARN] Early start signal failed: {_e}")
    except Exception:
        pass

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_CAM_ROOT = os.path.join(_BASE_DIR, "camera")
if os.path.isdir(_CAM_ROOT):
    if _CAM_ROOT not in sys.path:
        sys.path.insert(0, _CAM_ROOT)
    for _root, _dirs, _files in os.walk(_CAM_ROOT):
        if _root not in sys.path:
            sys.path.insert(0, _root)
import astra_py

# 避免外部环境变量干扰 Qt 插件搜索（在部分 Windows 环境下可避免加载错误）
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
os.environ.pop("QT_PLUGIN_PATH", None)
from YoloModel import ModelManager
from detect import detect_round1, _write_txt
import time

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QGroupBox,
    QTextEdit,
    QFrame,
    QFileDialog,
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QTimer
import numpy as np


import cv2
import camera


class MainWindow(QMainWindow):

    def __init__(self, round_id=None, env_name: str = ""):

        super().__init__()
        self.version = "0.0.1"
        self.setWindowTitle(f"3D Detection-{self.version}")
        self.setGeometry(100, 100, 984, 600)
        # 相机管理器
        self.camera_manager = camera.OpenCVCamera()

        self.timer = QTimer()

        self._warmup_done = False
        self.model_manager = ModelManager()
        self.overall_detector = self.model_manager.get_overall_detector()
        self.local_detector = self.model_manager.get_local_detector()

        self.latest_frame = None
        self.captured_image = None
        self.camera_active = False

        self._init_ui()

        # 相机管理器与事件接管
        self.timer.timeout.connect(self.camera_manager.get_frame)
        self.cameraButton.clicked.connect(self.toggle_camera)

        # 保存外部传入的环境名（用于结果额外写入）
        self.cli_env_name = env_name or ""

    def toggle_camera(
        self,
    ):
        if self.cap.isOpen:
            self.cap.stop()
            self.updateFrameTimer.stop()
            self.cameraButton.setText("打开相机")
            self.imageLabel.clear()  # 清除图像显示
        else:
            self.cap.start()
            time.sleep(1)  # 等待相机初始化
            self.updateFrameTimer.start(30)  # 每30毫秒更新一次帧
            self.cameraButton.setText("关闭相机")

    def _init_ui(self):
        """构建界面组件、布局，并绑定按钮信号。"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Header
        header = QLabel("仙道杀招-五指拳心剑")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet(
            "font-size: 22px; font-weight: bold; padding: 12px; background-color: #2c3e50; color: white;"
        )
        main_layout.addWidget(header)

        # Image Display
        image_group = QGroupBox("Camera View")
        image_group.setStyleSheet("QGroupBox { font-size: 16px; font-weight: bold; }")
        image_layout = QVBoxLayout()
        self.image_display = QLabel("No Image")
        self.image_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_display.setFixedSize(1000, 560)
        self.image_display.setStyleSheet(
            "border: 1px solid #ccc; background-color: #f0f0f0;"
        )
        image_layout.addWidget(self.image_display)
        image_group.setLayout(image_layout)
        main_layout.addWidget(image_group)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        main_layout.addWidget(line)

        # Bottom Layout
        bottom_layout = QHBoxLayout()
        main_layout.addLayout(bottom_layout)

        # Control Panel
        control_group = QGroupBox("Control Panel")
        control_group.setStyleSheet("QGroupBox { font-size: 16px; font-weight: bold; }")
        control_layout = QVBoxLayout()

        self.status_label = QLabel("Status: Disconnected")
        self.status_label.setStyleSheet("color: gray; padding: 4px;")
        control_layout.addWidget(self.status_label)

        self.cameraButton = QPushButton("打开摄像头")
        self.Button_Round1 = QPushButton("Start1")
        self.Button_Capture = QPushButton("获取图片")
        self.Button_Inference = QPushButton("图片推理")

        for btn in [
            self.cameraButton,
            self.Button_Capture,
            self.Button_Inference,
            self.Button_Round1,
        ]:
            btn.setFixedHeight(38)
            btn.setStyleSheet(
                "QPushButton { background-color: #3498db; color: white; border-radius: 6px; }"
                "QPushButton:hover { background-color: #2980b9; }"
            )
            control_layout.addWidget(btn)

        control_layout.addStretch()
        control_group.setLayout(control_layout)
        bottom_layout.addWidget(control_group, stretch=1)

        # Recognition Result - now displays image
        result_group = QGroupBox("Recognition Result")
        result_group.setStyleSheet("QGroupBox { font-size: 16px; font-weight: bold; }")
        result_layout = QVBoxLayout()
        self.result_image_display = QLabel(
            "Recognition result image will appear here..."
        )
        self.result_image_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_image_display.setFixedSize(400, 280)
        self.result_image_display.setStyleSheet(
            "background-color: #fdfdfd; border: 1px solid #ccc;"
        )
        result_layout.addWidget(self.result_image_display)
        result_group.setLayout(result_layout)
        bottom_layout.addWidget(result_group, stretch=2)

        # Log Output
        log_group = QGroupBox("System Log")
        log_group.setStyleSheet("QGroupBox { font-size: 16px; font-weight: bold; }")
        log_layout = QVBoxLayout()
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("background-color: #f4f4f4;")
        log_layout.addWidget(self.log_output)
        log_group.setLayout(log_layout)
        bottom_layout.addWidget(log_group, stretch=2)

        # 事件绑定（相机按钮在 __init__ 中改接 Camera.handle_camera_toggle）
        try:
            self.cameraButton.clicked.disconnect()
        except Exception:
            pass
        self.cameraButton.clicked.connect(self.camera_manager.handle_camera_toggle)
        self.Button_Round1.clicked.connect(self.start1Button)
        self.Button_Capture.clicked.connect(self.camera_manager.captureImage)
        self.Button_Inference.clicked.connect(self.inferButton)

    def _log(self, message):
        """将消息附带时间戳写入右侧日志框。"""
        timestamp = time.strftime("[%H:%M:%S]")
        if hasattr(self, "log_output"):
            self.log_output.append(f"{timestamp} {message}")
        else:
            print(f"{timestamp} {message}")

    def log(self, message):
        self._log(message)

    def captureImage(self):
        """抓拍当前帧，仅保存到 data/capture。"""
        if not self.camera_active:
            self.log("Camera not active. Cannot capture image.")
            return
        if self.latest_frame is None:
            self.log("No frame available to capture.")
            return
        try:
            os.makedirs(CAPTURE_SAVE_DIR, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            save_path = os.path.join(CAPTURE_SAVE_DIR, f"captured-{timestamp}.png")
            # 直接以 RGB 保存
            rgb = np.ascontiguousarray(self.latest_frame)
            cv2.imwrite(save_path, rgb)
            self.captured_image = self.latest_frame.copy()
            self.last_captured_path = save_path
            self.log(f"Image captured: {save_path} (仅保存，不自动处理)")
        except Exception as e:
            self.log(f"Failed to capture image: {str(e)}")

    def _file_picker(self):
        """弹出文件选择框（默认 data/），返回选中图片路径列表；取消返回空列表。支持多选。"""
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(data_dir, exist_ok=True)
        filters = "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All Files (*)"
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择用于推理的图片（可多选）", data_dir, filters
        )
        return paths

    def inferButton(self):
        """从 data/ 目录选择图片并执行图片推理。"""
        paths = self._file_picker()
        if not paths:
            return
        self._inference_images_show(paths)

    def start1Button(self, round_id=1):
        # 开始信号仅在 __init__ 创建 client 后发送一次，这里不再重复
        # 等待稳定
        time.sleep(0.5)
        # 1) 抓拍3张 BGR 到 data/raceimage/{NNN}
        raceimage_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "data", "raceimage")
        )
        os.makedirs(raceimage_root, exist_ok=True)
        # 生成递增会话目录（与 allresult 同步编号方式）
        try:
            entries = [
                d
                for d in os.listdir(raceimage_root)
                if os.path.isdir(os.path.join(raceimage_root, d))
            ]
        except Exception:
            entries = []
        nums = []
        for name in entries:
            if len(name) == 3 and name.isdigit():
                try:
                    nums.append(int(name))
                except Exception:
                    pass
        next_num = (max(nums) + 1) if nums else 1
        capture_dir = os.path.join(raceimage_root, f"{next_num:03d}")
        os.makedirs(capture_dir, exist_ok=True)
        self.log("Round 1: Capturing 3 images with 0.3s interval...")
        saved = self._captured_three_image(capture_dir, num_shots=3)  # 修改为拍摄3张
        if not saved:
            self.log("No images captured; abort Round 1.")
            return
        self.log(f"Captured {len(saved)} images successfully.")

        # 2) 批量转为 RGB 到 data/raceRGB/{NNN}（内部转换逻辑）
        rgb_session_dir, rgb_paths = self._convert_bgr_to_rgb_session(
            saved, capture_dir
        )
        if not rgb_paths:
            self.log("No RGB images prepared; abort Round 1.")
            return
        # 取最后2张进行推理
        if len(rgb_paths) >= 2:
            rgb_paths = rgb_paths[-2:]
            self.log(
                f"Using last 2 images from total {len(saved)} captured for inference."
            )
        else:
            self.log(
                f"Using all {len(rgb_paths)} images for inference (less than 2 captured)."
            )

        # 执行检测（输入转换后的 RGB 图片路径）
        self.status_label.setText("Status: Image Detecting")
        self.status_label.setStyleSheet("color: orange; padding: 4px;")
        session_dir, per_image_counts, last_final_img = self._infer_images(
            rgb_paths, round_id=1, update_status_ui=True, paths_are_rgb=True
        )
        # 融合多图计数并写入 result_r/DUTWUQXJ-R1.txt
        try:
            self.log(f"Using union strategy for {len(per_image_counts)} images...")
            final_counts = self._multiframe_fusion(per_image_counts)
            self.log(f"Union result: {final_counts}")
            _write_txt(final_counts or {}, 1, TEAM_SHORT_NAME)
            # 追加环境映射指定的额外结果（在 END 前插入）
            self._append_env_goal_before_end()
        except Exception as e:
            self.log(f"Write result failed: {str(e)}")

        # 发送结果文件（统一通过全局客户端）
        try:
            result_path = os.path.join(RESULT_FOLDER, f"{TEAM_SHORT_NAME}-R1.txt")
            _c = _get_judge_client()
            if _c is not None and os.path.isfile(result_path):
                _c.send_result(result_path)
                self.log(f"Result sent to judge box: {result_path}")
            elif not os.path.isfile(result_path):
                self.log(f"Result file not found: {result_path}")
        except Exception as e:
            self.log(f"Send result failed: {str(e)}")
        # 完成
        self.status_label.setText("Status: Finished")
        self.status_label.setStyleSheet("color: green; padding: 4px;")
        self.log("Camera Round 1 finished.")

    def closeEvent(self, event):
        """窗口关闭：释放相机与裁判盒连接。"""
        try:
            self.timer.stop()
        except Exception:
            pass

        try:
            _c = _get_judge_client()
            if _c:
                _c.close()
        except Exception:
            pass
        try:
            self.log("Application closed.")
        except Exception:
            pass
        event.accept()

    def _warmup_models(self):
        """已取消预热，保留空方法以兼容旧调用。"""
        self._warmup_done = True

    def _inference_images_show(self, image_paths):
        """专门处理图片推理的核心逻辑，不包含相机相关的操作。"""
        if not image_paths:
            return
        self.status_label.setText("Status: Image Detecting")
        self.status_label.setStyleSheet("color: orange; padding: 4px;")

        session_dir, per_image_counts, last_final_img = self._infer_images(
            image_paths, round_id=1, update_status_ui=False, paths_are_rgb=False
        )

        # 在主画面显示最后一张原始输入图（未画框）
        if image_paths:
            last_image_path = image_paths[-1]
            try:
                raw_bgr = cv2.imread(last_image_path)
                if raw_bgr is not None:
                    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
                    h0, w0 = raw_rgb.shape[:2]
                    q_img0 = QImage(raw_rgb.data, w0, h0, QImage.Format.Format_RGB888)
                    pixmap0 = QPixmap.fromImage(q_img0)
                    self.image_display.setPixmap(
                        pixmap0.scaled(
                            self.image_display.width(),
                            self.image_display.height(),
                            Qt.AspectRatioMode.KeepAspectRatio,
                        )
                    )
            except Exception:
                pass

        # 在结果区显示推理结果
        if last_final_img is not None:
            try:
                img = np.ascontiguousarray(last_final_img)
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8, copy=False)
                bytes_per_line = img.shape[1] * 3
                q_img = QImage(
                    img.data,
                    img.shape[1],
                    img.shape[0],
                    bytes_per_line,
                    QImage.Format.Format_RGB888,
                ).copy()
                pix = QPixmap.fromImage(q_img)
                if hasattr(self, "result_image_display"):
                    self.result_image_display.setPixmap(
                        pix.scaled(
                            self.result_image_display.width(),
                            self.result_image_display.height(),
                            Qt.AspectRatioMode.KeepAspectRatio,
                        )
                    )
            except Exception:
                pass

        self.status_label.setText("Status: Finished")
        self.status_label.setStyleSheet("color: green; padding: 4px;")

    def _show_recognition_frame(self, img):
        """在右侧结果区域显示一帧 RGB 图像（检测中的中间/最终帧）。"""
        img = np.ascontiguousarray(img)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8, copy=False)
        bytes_per_line = img.shape[1] * 3
        q_img = QImage(
            img.data,
            img.shape[1],
            img.shape[0],
            bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()
        pix = QPixmap.fromImage(q_img)
        # 同时更新主显示区和结果显示区
        self.image_display.setPixmap(
            pix.scaled(
                self.image_display.width(),
                self.image_display.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
            )
        )
        if hasattr(self, "result_image_display"):
            self.result_image_display.setPixmap(
                pix.scaled(
                    self.result_image_display.width(),
                    self.result_image_display.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                )
            )

    def _captured_three_image(self, session_dir, num_shots=3):
        """使用 Astra 相机连续抓拍 num_shots 张并保存到会话目录，返回路径列表。"""
        paths = []
        cam = None
        try:
            cam = astra_py.AstraCamera()
            # 可选：按需设定分辨率与输出；设备不支持会回退
            try:
                cam.set_resolution(1920, 1080, 30)
            except Exception:
                pass
            cam.set_outputs(
                want_color=True, want_depth=False, want_shaded=False, want_cloud=False
            )
            cam.start()
            time.sleep(0.3)
            for i in range(int(num_shots)):
                try:
                    frames = cam.get_frames()
                    if not frames:
                        time.sleep(0.05)
                        continue
                    color = frames[0]  # BGR(HWC)
                    # 直接保存为 PNG
                    save_path = os.path.join(session_dir, f"frame_{i+1:02d}.png")
                    cv2.imwrite(save_path, color)
                    paths.append(save_path)
                except Exception:
                    pass
                time.sleep(0.3)  # 修改间隔时间为0.3秒
        except Exception as e:
            self.log(f"Astra capture failed: {str(e)}")
        finally:
            try:
                if cam is not None:
                    cam.stop()
            except Exception:
                pass
        return paths

    def _convert_bgr_to_rgb_session(self, saved_paths, session_dir):
        """将一次会话中的 BGR 图片批量转换为 RGB 并保存到 data/raceRGB/{会话号}。
        返回 (rgb_session_dir, rgb_paths)
        """
        rgb_paths = []
        try:
            rgb_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "data", "raceRGB")
            )
            os.makedirs(rgb_root, exist_ok=True)
            sess_name = os.path.basename(session_dir)
            rgb_session_dir = os.path.join(rgb_root, sess_name)
            os.makedirs(rgb_session_dir, exist_ok=True)
            for p in saved_paths:
                try:
                    bgr = cv2.imread(p, cv2.IMREAD_COLOR)
                    if bgr is None:
                        continue
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    dst = os.path.join(rgb_session_dir, os.path.basename(p))
                    cv2.imwrite(dst, rgb)
                    rgb_paths.append(dst)
                except Exception:
                    pass
            return rgb_session_dir, rgb_paths
        except Exception as e:
            self.log(f"Convert to RGB failed: {str(e)}")
            return None, []

    def _read_perimage_results(self, result_txt_path):
        """解析 allresult/NNN/result.txt，返回每张图片的计数字典列表。格式: image_i: k=v, k=v"""
        counts_list = []
        try:
            with open(result_txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # 允许前缀 'image_i:'
                    try:
                        parts = line.split(":", 1)
                        rhs = parts[1] if len(parts) > 1 else line
                    except Exception:
                        rhs = line
                    d = {}
                    for kv in rhs.split(","):
                        kv = kv.strip()
                        if not kv:
                            continue
                        if "=" in kv:
                            k, v = kv.split("=", 1)
                            k = k.strip()
                            try:
                                d[k] = int(float(v))
                            except Exception:
                                # 非法数值忽略
                                continue
                    counts_list.append(d)
        except Exception as e:
            self.log(f"Parse result.txt failed: {str(e)}")
        return counts_list

    def _infer_images(
        self, image_paths, round_id=1, update_status_ui=False, paths_are_rgb=False
    ):
        """通用多图推理：保存到 allresult/NNN，并返回 (session_dir, per_image_counts, last_final_img)。"""
        session_dir = self._create_allresult_save_dir()
        per_image_counts = []
        last_final_img = None

        for path in image_paths:
            if paths_are_rgb:
                ok, rgb = self._load_and_display_image_rgbfile(path)
            else:
                ok, rgb = self._load_and_display_image(path)
            if not ok:
                per_image_counts.append({})
                continue
            has_desk = self._check_image_has_desk(rgb)
            cls_dir, clsresult_dir = self._create_overall(session_dir, has_desk)

            start_t = time.time()
            result_tuple = detect_round1(
                detector=self.overall_detector,
                frame_provider=lambda: rgb.copy(),
                update_cb=self._show_recognition_frame if update_status_ui else None,
                round_id=1,
                team_name=TEAM_SHORT_NAME,
                overall_detector=self.overall_detector,
                local_detector=self.local_detector,
                cls_model=None,
                session_dir=cls_dir,
                session_dir_result=clsresult_dir,
            )
            try:
                infer_ms = (time.time() - start_t) * 1000.0
                print(f"[Infer] {os.path.basename(path)}: {infer_ms:.1f} ms")
            except Exception:
                pass

            if isinstance(result_tuple, tuple) and len(result_tuple) == 7:
                (
                    overall_img,
                    local_img,
                    merge_img,
                    final_img,
                    out_cls_dir,
                    out_clsres_dir,
                    counts_frame,
                ) = result_tuple
            elif isinstance(result_tuple, tuple) and len(result_tuple) == 5:
                final_img, crop_img, stage1_img, merge_img, counts_frame = result_tuple
                overall_img, local_img, merge_img, out_cls_dir, out_clsres_dir = (
                    stage1_img,
                    merge_img,
                    merge_img,
                    None,
                    None,
                )
            else:
                final_img, crop_img, stage1_img, merge_img = result_tuple
                counts_frame = None
                overall_img, local_img, merge_img, out_cls_dir, out_clsres_dir = (
                    stage1_img,
                    merge_img,
                    merge_img,
                    None,
                    None,
                )

            self._save_inference_results(
                session_dir,
                path,
                has_desk,
                overall_img,
                local_img,
                merge_img,
                final_img,
            )
            per_image_counts.append(counts_frame or {})
            # 改为使用融合图作为最终展示图
            last_final_img = merge_img if merge_img is not None else last_final_img

        return session_dir, per_image_counts, last_final_img

    def _load_and_display_image_rgbfile(self, image_path):
        """加载已为RGB编码的图片并显示，返回 (ok, rgb)。优先使用 PIL，缺失则回退 OpenCV。"""
        try:
            try:
                from PIL import Image

                im = Image.open(image_path).convert("RGB")
                rgb = np.array(im)
            except Exception:
                bgr = cv2.imread(image_path)
                if bgr is None:
                    self.log(f"Failed to load image: {image_path}")
                    return False, None
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            q_img = QImage(rgb.data, w, h, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.image_display.setPixmap(
                pixmap.scaled(
                    self.image_display.width(),
                    self.image_display.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                )
            )
            return True, rgb
        except Exception as e:
            self.log(f"Error loading RGB image {image_path}: {str(e)}")
            return False, None

    def _multiframe_fusion(self, per_image_counts):
        """使用并集策略融合多张图片的计数结果。
        - 类别存在条件：在任意一帧中出现即保留
        - 数量判定：取两帧中该类别的最大值
        """
        from collections import defaultdict

        # 统计每个类别的所有出现的数量
        class_occurrences = defaultdict(list)  # {class_name: [count1, count2, ...]}

        for counts in per_image_counts:
            if not counts:
                continue
            for class_name, count in counts.items():
                if class_name == "desk":
                    continue
                try:
                    class_occurrences[class_name].append(int(count))
                except Exception:
                    pass

        # 计算最终结果：存在即保留，数量取最大值
        final_counts = {}
        for class_name, count_list in class_occurrences.items():
            final_counts[class_name] = max(count_list)
            self.log(
                f"  {class_name}: counts from frames {count_list} -> final: {max(count_list)}"
            )

        return final_counts

    def _create_allresult_save_dir(self):
        # 统一写入项目根目录的 allresult 目录
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "allresult"))
        os.makedirs(base_dir, exist_ok=True)
        try:
            entries = [
                d
                for d in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, d))
            ]
        except Exception:
            entries = []
        nums = []
        for name in entries:
            if len(name) == 3 and name.isdigit():
                try:
                    nums.append(int(name))
                except Exception:
                    pass
        next_num = (max(nums) + 1) if nums else 1
        dirname = f"{next_num:03d}"
        session_dir = os.path.join(base_dir, dirname)
        os.makedirs(session_dir, exist_ok=True)
        return session_dir

    def _append_env_goal_before_end(self):
        """根据外部环境名，在结果文件 END 前插入额外行。
        ENV_GOAL_MAP 可为二元组 (goal_id, num) 或三元组 (goal_id, num, table)。
        优先来源：self.cli_env_name；若为空，回退读取环境变量 RACE_ORIG_ENV。
        仅在映射命中且结果文件存在时执行。
        """
        try:
            env_name = (
                getattr(self, "cli_env_name", "")
                or os.environ.get("RACE_ORIG_ENV", "")
                or ""
            ).strip()
            if not env_name:
                return
            goal_tuple = ENV_GOAL_MAP.get(env_name)
            if not goal_tuple:
                return
            # 兼容 2/3 元组
            if isinstance(goal_tuple, (list, tuple)) and len(goal_tuple) >= 2:
                goal_id = goal_tuple[0]
                num = int(goal_tuple[1])
                table = int(goal_tuple[2]) if len(goal_tuple) >= 3 else 1
            else:
                return
            result_path = os.path.join(RESULT_FOLDER, f"{TEAM_SHORT_NAME}-R1.txt")
            if not os.path.isfile(result_path):
                return
            with open(result_path, "r", encoding="utf-8") as f:
                lines = [ln.rstrip("\n") for ln in f.readlines()]
            # 在 END 前插入
            out_lines = []
            inserted = False
            for ln in lines:
                if ln.strip() == "END" and not inserted:
                    out_lines.append(
                        f"Goal_ID={goal_id};Num={int(num)};Table={int(table)}"
                    )
                    out_lines.append("END")
                    inserted = True
                else:
                    out_lines.append(ln)
            if not inserted:
                out_lines.append(f"Goal_ID={goal_id};Num={int(num)};Table={int(table)}")
                out_lines.append("END")
            with open(result_path, "w", encoding="utf-8") as f:
                f.write("\n".join(out_lines) + "\n")
            self.log(
                f"Extra env goal appended for env '{env_name}': {goal_id}={int(num)}, table={int(table)}"
            )
        except Exception as e:
            self.log(f"Append env goal failed: {str(e)}")

    def _load_and_display_image(self, image_path):
        try:
            bgr = cv2.imread(image_path)
            if bgr is None:
                self.log(f"Failed to load image: {image_path}")
                return False, None
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            q_img = QImage(rgb.data, w, h, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.image_display.setPixmap(
                pixmap.scaled(
                    self.image_display.width(),
                    self.image_display.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                )
            )
            return True, rgb
        except Exception as e:
            self.log(f"Error loading image {image_path}: {str(e)}")
            return False, None

    def _check_image_has_desk(self, image):
        try:
            boxes_xyxy, confs, clses = self.overall_detector.infer(image.copy())
            desk_ids = [
                i
                for i, name in enumerate(self.overall_detector.classes)
                if name == "desk"
            ]
            if boxes_xyxy is not None and len(boxes_xyxy) > 0 and len(desk_ids) > 0:
                import numpy as np

                return np.isin(clses, np.array(desk_ids, dtype=np.int32)).any()
        except Exception as e:
            self.log(f"桌子检测失败: {str(e)}")
        return False

    def _create_overall(self, session_dir, has_desk):
        # 统一改为 overall/local，并始终创建，分别用于模型1/模型2逐目标置信度
        session_overall_dir = os.path.join(session_dir, "overall")
        session_local_dir = os.path.join(session_dir, "local")
        try:
            os.makedirs(session_overall_dir, exist_ok=True)
            os.makedirs(session_local_dir, exist_ok=True)
            return session_overall_dir, session_local_dir
        except Exception as e:
            self.log(f"Failed to create directories: {str(e)}")
            return None, None

    def _save_inference_results(
        self,
        session_dir,
        image_path,
        has_desk,
        overall_img,
        local_img,
        merge_img,
        final_img,
    ):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        try:
            if overall_img is not None:
                m1_path = os.path.join(session_dir, f"{stem}_overall.png")
                bgr1 = cv2.cvtColor(overall_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(m1_path, bgr1)
            if has_desk:
                if local_img is not None:
                    local_path = os.path.join(session_dir, f"{stem}_Local.png")
                    cv2.imwrite(local_path, cv2.cvtColor(local_img, cv2.COLOR_RGB2BGR))
                if merge_img is not None:
                    merge_path = os.path.join(session_dir, f"{stem}_merge.png")
                    cv2.imwrite(merge_path, cv2.cvtColor(merge_img, cv2.COLOR_RGB2BGR))
            # 停止保存 realresult 图片，结果展示统一使用融合图
        except Exception as e:
            self.log(f"Error saving results for {image_path}: {str(e)}")


def auto_run_by_arg():
    """命令行自动模式：python main.py num → 启动主窗体、开启摄像头并调用 start1Button。"""
    if len(sys.argv) < 2:
        return
    # 参数1：外部环境名（来自 run.sh）
    env_name_arg = sys.argv[1].strip()

    app = QApplication(sys.argv)
    window = MainWindow(env_name=env_name_arg)
    window.show()

    # 稍等片刻后调用第一轮检测按钮逻辑（不自动退出）
    QTimer.singleShot(1000, window.start1Button)
    sys.exit(app.exec())


if __name__ == "__main__":
    # 入口：有任意参数走自动模式；否则正常启动 GUI
    if len(sys.argv) > 1:
        auto_run_by_arg()
    else:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
