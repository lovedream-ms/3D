import os, sys
import warnings
import time

# ========== 最早期：立即建立通信并发送 Start2 ==========
# 在导入任何重量级模块前执行，减少启动延迟
_start_time = time.time()
print(f"[DEBUG] Script started at {time.strftime('%H:%M:%S', time.localtime(_start_time))}")

from mysocket import JudgeBoxClient
from config import JUDGE_BOX_IP, JUDGE_BOX_PORT, TEAM_SHORT_NAME

_client = None
_START_SENT = False

def _get_judge_client():
    global _client
    try:
        if _client is None:
            _client = JudgeBoxClient(ip=JUDGE_BOX_IP, port=JUDGE_BOX_PORT)
        return _client
    except Exception:
        return None

# 立即发送 Start2 信号
try:
    _c_early = _get_judge_client()
    if _c_early is not None:
        _c_early.send_start(2)
        _START_SENT = True
        _conn_time = time.time()
        print(f"[INFO] Early start signal (R2) sent at {time.strftime('%H:%M:%S', time.localtime(_conn_time))}, took {_conn_time - _start_time:.2f}s")
except Exception as _e:
    try:
        print(f"[WARN] Early start signal failed: {_e}")
    except Exception:
        pass

# ========== 之后再导入重量级模块 ==========
print(f"[DEBUG] Starting heavy imports at {time.strftime('%H:%M:%S', time.localtime())}")
from config import (
    RESULT_FOLDER, CAPTURE_SAVE_DIR,
    WARMUP_IMAGE_PATH, ENV_GOAL_MAP,
)
# # 避免外部环境变量干扰 Qt 插件搜索（在部分 Windows 环境下可避免加载错误）
# os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
# os.environ.pop("QT_PLUGIN_PATH", None)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_CAM_ROOT = os.path.join(_BASE_DIR, 'camera')
if os.path.isdir(_CAM_ROOT):
    if _CAM_ROOT not in sys.path:
        sys.path.insert(0, _CAM_ROOT)
    for _root, _dirs, _files in os.walk(_CAM_ROOT):
        if _root not in sys.path:
            sys.path.insert(0, _root)

# 导入重量级模块
import astra_py
try:
    from config import ROTATE_WAIT_SECONDS
except Exception as e:
    warnings.warn(f"Failed to import ROTATE_WAIT_SECONDS: {e}")
    ROTATE_WAIT_SECONDS = 0
import cv2


from model import ModelManager
from detect import detect_round1, write_txt_tables
import time

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QGroupBox, QTextEdit, QFrame, QFileDialog,
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QTimer

import numpy as np

class MainWindow(QMainWindow):
    
    def __init__(self, round_id=None):
        # 统一通信由全局客户端负责；不在实例内创建连接
        super().__init__()
        self.version = "0.0.1"
        self.setWindowTitle(f"3D Detection-{self.version}")
        self.setGeometry(100, 100, 984, 600)
        # 相机管理器
        self.camera_manager = Camera(self)
        # 初始化计时器（不再进行裁判盒预连接）
        self.timer = QTimer()

        # 模型与服务
        self.model_manager = ModelManager()
        
        self.overall_detector = self.model_manager.get_overall_detector()
        self.local_detector = self.model_manager.get_local_detector()
        

        # 状态
        self.latest_frame = None
        self.captured_image = None
        self.camera_active = False
        self.is_sending = False
        self.is_detecting = False
        
        # UI
        self._init_ui()

        # 相机管理器与事件接管
        
        try:
            self.timer.timeout.disconnect()
        except Exception:
            warnings.warn("Failed to disconnect timer timeout handler.")
        self.timer.timeout.connect(self.camera_manager.refresh_frame)
        try:
            self.Button_Camera.clicked.disconnect()
        except Exception:
            warnings.warn("Failed to disconnect existing camera button handler.")
        self.Button_Camera.clicked.connect(self.camera_manager.handle_camera_toggle)

        # 启动即发轮次：改为使用全局客户端；若上方已早发，则这里不再重复
        if round_id is not None:
            try:
                _c = _get_judge_client()
                if _c is not None and not _START_SENT:
                    _c.send_start(int(round_id))
                    self.is_sending = True
            except Exception:
                warnings.warn("Failed to send start signal at init.")

    def _init_ui(self):
        """构建界面组件、布局，并绑定按钮信号。"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Header
        header = QLabel("仙道杀招-五指拳心剑")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("font-size: 22px; font-weight: bold; padding: 12px; background-color: #2c3e50; color: white;")
        main_layout.addWidget(header)

        # Image Display
        image_group = QGroupBox("Camera View")
        image_group.setStyleSheet("QGroupBox { font-size: 16px; font-weight: bold; }")
        image_layout = QVBoxLayout()
        self.image_display = QLabel("No Image")
        self.image_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_display.setFixedSize(1000, 560)
        self.image_display.setStyleSheet("border: 1px solid #ccc; background-color: #f0f0f0;")
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

        self.Button_Camera = QPushButton("打开摄像头")
        self.Button_Round2 = QPushButton("Start2")
        self.Button_Trans = QPushButton("Trans")
        self.Button_Capture = QPushButton("获取图片")
        self.Button_Inference = QPushButton("图片推理")
        
        for btn in [self.Button_Camera, self.Button_Capture, self.Button_Trans, self.Button_Inference, self.Button_Round2]:
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
        self.result_image_display = QLabel("Recognition result image will appear here...")
        self.result_image_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_image_display.setFixedSize(400, 280)
        self.result_image_display.setStyleSheet("background-color: #fdfdfd; border: 1px solid #ccc;")
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
            self.Button_Camera.clicked.disconnect()
        except Exception:
            warnings.warn("Failed to disconnect camera button in _init_ui.")
        self.Button_Camera.clicked.connect(self.camera_manager.handle_camera_toggle)
        self.Button_Round2.clicked.connect(self.start2Button)
        self.Button_Capture.clicked.connect(self.camera_manager.captureImage)
        self.Button_Inference.clicked.connect(self.inferButton)
        self.Button_Trans.clicked.connect(self.handle_trans_convert)
 

    def _set_interaction_enabled(self, enabled: bool):
        """启用/禁用交互控件（检测过程中禁用所有按钮）。"""
        try:
            for btn in [getattr(self, 'Button_Camera', None),
                        getattr(self, 'Button_Capture', None),
                        getattr(self, 'Button_Inference', None),
                        getattr(self, 'Button_OCR', None),
                        getattr(self, 'Button_Round2', None)]:
                if btn is not None:
                    btn.setEnabled(enabled)
        except Exception:
            warnings.warn("Failed to toggle interaction enabled state.")

    def closeEvent(self, event):
        """检测过程中不允许关闭窗口。"""
        if getattr(self, 'is_detecting', False):
            self.log("Detection in progress. Close ignored.")
            event.ignore()
        else:
            event.accept()

    def _log(self, message):
        """将消息附带时间戳写入右侧日志框。"""
        timestamp = time.strftime("[%H:%M:%S]")
        if hasattr(self, 'log_output'):
            self.log_output.append(f"{timestamp} {message}")
        # 同步输出到终端
        print(f"{timestamp} {message}")

    def log(self, message):
        self._log(message)
            
    def captureImage(self):
        """使用 astra_py 独立抓拍一张到 data/capture；astra 不可用时回退使用 latest_frame。"""
        try:
            os.makedirs(CAPTURE_SAVE_DIR, exist_ok=True)
            timestamp = time.strftime('%Y%m%d-%H%M%S')
            save_path = os.path.join(CAPTURE_SAVE_DIR, f'captured-{timestamp}.png')

            if astra_py is not None:
                cam = None
                try:
                    cam = astra_py.AstraCamera()
                    try:
                        cam.set_resolution(1920, 1080, 30)
                    except Exception:
                        pass
                    cam.set_outputs(want_color=True, want_depth=False, want_shaded=False, want_cloud=False)
                    cam.start()
                    time.sleep(0.2)
                    frames = cam.get_frames()
                    if frames:
                        color = frames[0]  # BGR(HWC)
                        cv2.imwrite(save_path, color)
                        # 转为 RGB 仅用于 UI 内存显示
                        rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                        self.captured_image = rgb.copy()
                        self.last_captured_path = save_path
                        self.log(f"Image captured (Astra): {save_path}")
                        return
                except Exception as e:
                    self.log(f"Astra capture failed, fallback: {str(e)}")
                finally:
                    try:
                        if cam is not None:
                            cam.stop()
                    except Exception:
                        pass

            # 回退：使用当前 UI 帧
            if self.latest_frame is None:
                self.log("No frame available to capture (fallback).")
                return
            bgr = cv2.cvtColor(self.latest_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, bgr)
            self.captured_image = self.latest_frame.copy()
            self.last_captured_path = save_path
            self.log(f"Image captured: {save_path} (fallback)")
        except Exception as e:
            self.log(f"Failed to capture image: {str(e)}")

    def _file_picker(self):
        """弹出文件选择框（默认 data/），返回选中图片路径列表；取消返回空列表。支持多选。"""
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(data_dir, exist_ok=True)
        filters = "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All Files (*)"
        paths, _ = QFileDialog.getOpenFileNames(self, "选择用于推理的图片（可多选）", data_dir, filters)
        return paths

    def inferButton(self):
        """从 data/ 目录选择图片并执行图片推理。"""
        paths = self._file_picker()
        if not paths:
            return
        self._inference_images_show(paths)

    def handle_trans_convert(self):
        """将 data/capture 中的 BGR 图片批量转换为 RGB 到 data/RGB，并删除源图片。"""
        src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'capture'))
        dst_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'RGB'))
        if not os.path.isdir(src_dir):
            self.log(f"源目录不存在：{src_dir}")
            return
        os.makedirs(dst_dir, exist_ok=True)
        ok_count, total = 0, 0
        for root, _, files in os.walk(src_dir):
            rel = os.path.relpath(root, src_dir)
            out_root = os.path.join(dst_dir, rel) if rel != '.' else dst_dir
            os.makedirs(out_root, exist_ok=True)
            for fname in files:
                lower = fname.lower()
                if not (lower.endswith('.png') or lower.endswith('.jpg') or lower.endswith('.jpeg') or lower.endswith('.bmp') or lower.endswith('.webp')):
                    continue
                total += 1
                fpath = os.path.join(root, fname)
                try:
                    bgr = cv2.imread(fpath, cv2.IMREAD_COLOR)
                    if bgr is None:
                        continue
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    out_path = os.path.join(out_root, fname)
                    cv2.imwrite(out_path, rgb)
                    ok_count += 1
                except Exception:
                    pass
        self.log(f"转换完成：{ok_count}/{total}")
        # 删除源目录中的所有图片文件（递归）
        removed = 0
        for root, _, files in os.walk(src_dir):
            for fname in files:
                lower = fname.lower()
                if not (lower.endswith('.png') or lower.endswith('.jpg') or lower.endswith('.jpeg') or lower.endswith('.bmp') or lower.endswith('.webp')):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    os.remove(fpath)
                    removed += 1
                except Exception as e:
                    self.log(f"删除失败：{fpath} -> {e}")
        self.log(f"已删除 capture 源图片数量：{removed}")

    def _inference_images_show(self, image_paths):
        """专门处理图片推理的核心逻辑，不包含相机相关的操作。"""
        if not image_paths:
            return

        # 锁定交互
        self.is_detecting = True
        self._set_interaction_enabled(False)
        try:
            self.status_label.setText("Status: Image Detecting")
            self.status_label.setStyleSheet("color: orange; padding: 4px;")

            session_dir, per_image_counts, last_final_img = self._infer_images(image_paths, round_id=1, update_status_ui=False)

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
                            pixmap0.scaled(self.image_display.width(), self.image_display.height(), Qt.AspectRatioMode.KeepAspectRatio)
                        )
                except Exception:
                    warnings.warn("Failed to update main image display after inference.")

            # 在结果区显示推理结果
            if last_final_img is not None:
                try:
                    img = np.ascontiguousarray(last_final_img)
                    if img.dtype != np.uint8:
                        img = img.astype(np.uint8, copy=False)
                    bytes_per_line = img.shape[1] * 3
                    q_img = QImage(img.data, img.shape[1], img.shape[0], bytes_per_line, QImage.Format.Format_RGB888).copy()
                    pix = QPixmap.fromImage(q_img)
                    if hasattr(self, 'result_image_display'):
                        self.result_image_display.setPixmap(
                            pix.scaled(self.result_image_display.width(), self.result_image_display.height(), Qt.AspectRatioMode.KeepAspectRatio)
                        )
                except Exception:
                    warnings.warn("Failed to update result image display after inference.")

            self.status_label.setText("Status: Finished")
            self.status_label.setStyleSheet("color: green; padding: 4px;")
        finally:
            self.is_detecting = False
            self._set_interaction_enabled(True)

    def _show_recognition_frame(self, img):
            """在右侧结果区域显示一帧 RGB 图像（检测中的中间/最终帧）。"""
            img = np.ascontiguousarray(img)
            if img.dtype != np.uint8:
                img = img.astype(np.uint8, copy=False)
            bytes_per_line = img.shape[1] * 3
            q_img = QImage(img.data, img.shape[1], img.shape[0], bytes_per_line,
                    QImage.Format.Format_RGB888).copy()
            pix = QPixmap.fromImage(q_img)
            # 同时更新主显示区和结果显示区
            self.image_display.setPixmap(
                pix.scaled(self.image_display.width(),
                        self.image_display.height(),
                        Qt.AspectRatioMode.KeepAspectRatio)
            )
            if hasattr(self, 'result_image_display'):
                self.result_image_display.setPixmap(
                    pix.scaled(self.result_image_display.width(),
                            self.result_image_display.height(),
                            Qt.AspectRatioMode.KeepAspectRatio)
                )

    def _show_rotating_and_wait(self, seconds):
        """在 UI 上提示相机正在转动，并在等待期间保持界面可响应。"""
        try:
            total = max(0.0, float(seconds))
        except Exception:
            total = 0.0
        start_ts = time.time()
        # 初始提示
        self.status_label.setText(f"Status: Rotating camera... ({int(total)}s)")
        self.status_label.setStyleSheet("color: #3498db; padding: 4px;")
        QApplication.processEvents()
        # 循环等待并刷新 UI
        while True:
            elapsed = time.time() - start_ts
            if elapsed >= total:
                break
            remain = int(max(0, total - elapsed))
            self.status_label.setText(f"Status: Rotating camera... ({remain}s)")
            QApplication.processEvents()
            time.sleep(0.1)
        
    def start2Button(self, checked=False):
        round_id = 2
        # 锁定交互
        self.is_detecting = True
        self._set_interaction_enabled(False)
        
        # 停止UI相机，避免与检测相机冲突
        was_camera_active = self.camera_active
        if self.camera_active:
            self.log("Stopping UI camera for detection...")
            self.camera_manager.handle_camera_toggle()
            QApplication.processEvents()  # 处理事件队列
            import time as time_module
            time_module.sleep(1.0)  # 增加延迟，确保相机完全释放
        
        try:
            # 向裁判盒发送第二轮开始
            if not self.is_sending:
                try:
                    _c = _get_judge_client()
                    if _c is not None:
                        _c.send_start(int(round_id))
                        self.is_sending = True
                except Exception:
                    warnings.warn("Failed to send start signal for Round 2.")

            tables_results = []  # [(counts, table)]

            # Table=1：拍摄3张，间隔0.3s，取后两张推理，结果取并集
            self.status_label.setText("Status: Camera Round 2 - Table 1")
            self.status_label.setStyleSheet("color: orange; padding: 4px;")
            self.log("Table 1: Capturing 3 images with 0.3s interval...")
            
            # 拍摄3张图片，间隔0.3秒
            bgr_session, bgr_paths = self._captured_multiple_images_session(num_images=3, interval=0.3)
            
            # 转换BGR到RGB - 只转换最后2张
            if len(bgr_paths) >= 2:
                bgr_paths_to_convert = bgr_paths[-2:]  # 取最后2张
                self.log(f"Converting last 2 images from {len(bgr_paths)} total captured...")
            else:
                bgr_paths_to_convert = bgr_paths
                self.log(f"Converting all {len(bgr_paths)} images (less than 2 captured)...")
            
            _rgb_dir, rgb_paths = self._convert_bgr_to_rgb_session(bgr_paths_to_convert, bgr_session)
            
            # 推理所有转换后的图片（最多2张）
            session_dir, per_image_counts, _ = self._infer_images(rgb_paths, round_id=int(round_id), update_status_ui=True)
            
            # 使用并集策略进行结果统计
            self.log(f"Using union strategy for {len(per_image_counts)} images...")
            final_counts = self._table_union_strategy(per_image_counts)
            
            # 记录统计结果
            self.log(f"Table 1 union result: {final_counts}")
            tables_results.append((final_counts or {}, 1))

            # 发送旋转并等待
            try:
                _c = _get_judge_client()
                if _c is not None:
                    _c.send_start_rotate_camera()
            except Exception:
                warnings.warn("Failed to send rotate command (first).")
            self._show_rotating_and_wait(ROTATE_WAIT_SECONDS)

            # Table=2：拍摄3张，间隔0.3s，取后两张推理，结果取并集
            self.status_label.setText("Status: Camera Round 2 - Table 2")
            self.status_label.setStyleSheet("color: orange; padding: 4px;")
            self.log("Table 2: Capturing 3 images with 0.3s interval...")
            
            # 拍摄3张图片，间隔0.3秒
            bgr_session, bgr_paths = self._captured_multiple_images_session(num_images=3, interval=0.3)
            
            # 转换BGR到RGB - 只转换最后2张
            if len(bgr_paths) >= 2:
                bgr_paths_to_convert = bgr_paths[-2:]  # 取最后2张
                self.log(f"Converting last 2 images from {len(bgr_paths)} total captured...")
            else:
                bgr_paths_to_convert = bgr_paths
                self.log(f"Converting all {len(bgr_paths)} images (less than 2 captured)...")
            
            _rgb_dir, rgb_paths = self._convert_bgr_to_rgb_session(bgr_paths_to_convert, bgr_session)
            
            # 推理所有转换后的图片（最多2张）
            session_dir, per_image_counts, _ = self._infer_images(rgb_paths, round_id=int(round_id), update_status_ui=True)
            
            # 使用并集策略进行结果统计
            self.log(f"Using union strategy for {len(per_image_counts)} images...")
            final_counts = self._table_union_strategy(per_image_counts)
            
            # 记录统计结果
            self.log(f"Table 2 union result: {final_counts}")
            tables_results.append((final_counts or {}, 2))
 

            # 再次旋转并等待
            try:
                _c = _get_judge_client()
                if _c is not None:
                    _c.send_start_rotate_camera()
            except Exception:
                warnings.warn("Failed to send rotate command (second).")
            self._show_rotating_and_wait(ROTATE_WAIT_SECONDS)

            # Table=3：使用修改后的投票策略 - 拍摄9张，使用最后8张推理
            self.status_label.setText("Status: Camera Round 2 - Table 3")
            self.status_label.setStyleSheet("color: orange; padding: 4px;")
            self.log("Table 3: Capturing 9 images with 0.6s interval...")
            
            # 拍摄9张图片，间隔0.6秒
            bgr_session, bgr_paths = self._captured_multiple_images_session(num_images=9, interval=0.6)
            
            # 转换BGR到RGB - 只转换最后8张
            if len(bgr_paths) >= 8:
                bgr_paths_to_convert = bgr_paths[-8:]  # 取最后8张
                self.log(f"Converting last 8 images from {len(bgr_paths)} total captured...")
            else:
                bgr_paths_to_convert = bgr_paths
                self.log(f"Converting all {len(bgr_paths)} images (less than 8 captured)...")
            
            _rgb_dir, rgb_paths = self._convert_bgr_to_rgb_session(bgr_paths_to_convert, bgr_session)
            
            # 推理所有转换后的图片（最多8张）
            session_dir, per_image_counts, _ = self._infer_images(rgb_paths, round_id=int(round_id), update_status_ui=True)
            
            # 使用新的投票策略进行结果统计
            self.log(f"Using new voting strategy for {len(per_image_counts)} images...")
            final_counts = self._table3_voting_strategy(per_image_counts)
            
            # 记录统计结果
            self.log(f"Table 3 voting result: {final_counts}")
            tables_results.append((final_counts or {}, 3))
 

            # 写合并结果
            self.log(f"Writing result file with {len(tables_results)} tables...")
            write_txt_tables(tables_results, int(round_id), TEAM_SHORT_NAME)
            expected_path = os.path.join(RESULT_FOLDER, f"{TEAM_SHORT_NAME}-R{int(round_id)}.txt")
            self.log(f"Round 2 three tables done. Expected result file: {expected_path}")
            self.status_label.setText("Status: Finished")
            self.status_label.setStyleSheet("color: green; padding: 4px;")

            # 列出result_r目录的所有文件用于调试
            try:
                if os.path.exists(RESULT_FOLDER):
                    files = os.listdir(RESULT_FOLDER)
                    self.log(f"Files in {RESULT_FOLDER}: {files}")
            except Exception:
                pass

            # 发送合并结果到裁判盒
            try:
                result_path = os.path.join(RESULT_FOLDER, f"{TEAM_SHORT_NAME}-R{int(round_id)}.txt")
                if os.path.isfile(result_path):
                    file_size = os.path.getsize(result_path)
                    self.log(f"Found result file: {result_path} (size: {file_size} bytes)")
                    _c = _get_judge_client()
                    if _c is not None:
                        _c.send_result(result_path)
                        self.log(f"Result sent to judge box: {result_path}")
                    else:
                        self.log("Judge client is None, cannot send result")
                else:
                    self.log(f"Result file not found: {result_path}")
                    # 再次检查文件是否存在（可能有延迟）
                    import time as _time
                    _time.sleep(0.1)
                    if os.path.isfile(result_path):
                        self.log(f"File appeared after delay: {result_path}")
            except Exception as e:
                self.log(f"Send result failed: {str(e)}")
        finally:
            self.is_sending = False
            self.is_detecting = False
            self._set_interaction_enabled(True)
            
            # 恢复UI相机（如果之前是开启的）
            if was_camera_active and not self.camera_active:
                self.log("Restarting UI camera...")
                self.camera_manager.handle_camera_toggle()

    def _table_union_strategy(self, per_image_counts):
        """计算多张图片检测结果的并集：对于每个目标ID，取所有出现过的最大数量"""
        union_counts = {}
        for counts in per_image_counts:
            for goal_id, num in counts.items():
                if goal_id not in union_counts or num > union_counts[goal_id]:
                    union_counts[goal_id] = num
        return union_counts
    
    def _captured_multiple_images_session(self, num_images=4, interval=0.3):
        """使用 astra_py 独立抓拍多张图片，与 UI 绑定相机无关。
        参数:
            num_images: 要拍摄的图片数量
            interval: 拍摄间隔(秒)
        返回:
            (会话目录, 图片路径列表)
        """
        base_capture = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'raceimage'))
        os.makedirs(base_capture, exist_ok=True)
        # 为会话分配下一个编号
        next_idx = 1
        try:
            entries = [d for d in os.listdir(base_capture) if os.path.isdir(os.path.join(base_capture, d))]
            nums = [int(x) for x in entries if len(x) == 3 and x.isdigit()]
            next_idx = (max(nums) + 1) if nums else 1
        except Exception:
            warnings.warn("Failed to enumerate capture sessions, fallback to 001.")
        session_dir = os.path.join(base_capture, f"{next_idx:03d}")
        os.makedirs(session_dir, exist_ok=True)

        paths = []
        if astra_py is None:
            self.log("astra_py not available; cannot capture via Astra camera.")
            return session_dir, paths
        
        # 确保之前的相机资源已经完全释放
        time.sleep(0.5)
        
        cam = None
        try:
            cam = astra_py.AstraCamera()
            try:
                cam.set_resolution(1920, 1080, 30)
            except Exception:
                pass
            cam.set_outputs(want_color=True, want_depth=False, want_shaded=False, want_cloud=False)
            cam.start()
            time.sleep(1.5)
            
            self.log(f"Starting to capture {num_images} images with {interval}s interval...")
            
            for i in range(num_images):
                try:
                    frames = cam.get_frames()
                    if not frames:
                        time.sleep(0.05)
                        continue
                    color = frames[0]  # BGR(HWC)
                    timestamp = time.strftime('%Y%m%d-%H%M%S-%f')[:-3]  # 包含毫秒
                    save_path = os.path.join(session_dir, f'captured_{i+1:03d}_{timestamp}.png')
                    cv2.imwrite(save_path, color)
                    paths.append(save_path)
                    
                    # 更新状态
                    if i % 5 == 0 or i == num_images - 1:  # 每5张或最后一张时更新状态
                        self.status_label.setText(f"Status: Capturing images... ({i+1}/{num_images})")
                        QApplication.processEvents()
                        
                except Exception as e:
                    self.log(f"Capture failed: {str(e)}")
                
                # 等待间隔时间（最后一张不需要等待）
                if i < num_images - 1:
                    time.sleep(interval)
                    
        except Exception as e:
            self.log(f"Astra capture init failed: {str(e)}")
        finally:
            try:
                if cam is not None:
                    cam.stop()
            except Exception:
                pass
                    
        self.log(f"Captured {len(paths)} images in session {session_dir}")
        return session_dir, paths

    def _convert_bgr_to_rgb_session(self, saved_paths, session_dir):
        """将一次会话中的 BGR 图片批量转换为 RGB 并保存到 data/raceRGB/{会话号}。
        返回 (rgb_session_dir, rgb_paths)
        """
        rgb_paths = []
        try:
            rgb_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'raceRGB'))
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

 
 
    def _infer_images(self, image_paths, round_id=1, update_status_ui=False):
        """通用多图推理：保存到 allresult/NNN，并返回 (session_dir, per_image_counts, last_final_img)。"""
        session_dir = self._create_allresult_save_dir()
        per_image_counts = []
        last_final_img = None

        for path in image_paths:
            ok, rgb = self._load_and_display_image(path)
            if not ok:
                per_image_counts.append({})
                continue
            has_desk = self._check_image_has_desk(rgb)
            cls_dir, clsresult_dir = self._create_overall(session_dir, has_desk)

            result_tuple = detect_round1(
                detector=self.overall_detector,
                frame_provider=lambda: rgb.copy(),
                update_cb=self._show_recognition_frame if update_status_ui else None,
                round_id=int(round_id),
                team_name=TEAM_SHORT_NAME,
                overall_detector=self.overall_detector,
                local_detector=self.local_detector,
                cls_model=None,
                session_dir=cls_dir,
                session_dir_result=clsresult_dir,
            )

            if isinstance(result_tuple, tuple) and len(result_tuple) == 7:
                overall_img, local_img, merge_img, final_img, out_cls_dir, out_clsres_dir, counts_frame = result_tuple
            elif isinstance(result_tuple, tuple) and len(result_tuple) == 5:
                final_img, crop_img, stage1_img, merge_img, counts_frame = result_tuple
                overall_img, local_img, merge_img, out_cls_dir, out_clsres_dir = stage1_img, merge_img, merge_img, None, None
            else:
                final_img, crop_img, stage1_img, merge_img = result_tuple
                counts_frame = None
                overall_img, local_img, merge_img, out_cls_dir, out_clsres_dir = stage1_img, merge_img, merge_img, None, None

            self._save_inference_results(session_dir, path, has_desk, overall_img, local_img, merge_img, final_img)
            per_image_counts.append(counts_frame or {})

            # 为每张 merge 图片生成对应的 txt，记录该张图片的合并结果（不含 desk）
            try:
                counts_map = counts_frame or {}
                lines = []
                for k, v in (counts_map or {}).items():
                    if k == 'desk':
                        continue
                    try:
                        lines.append(f"Goal_ID={k};Num={int(v)}")
                    except Exception:
                        pass
                if lines:
                    stem = os.path.splitext(os.path.basename(path))[0]
                    out_txt = os.path.join(session_dir, f"{stem}_merge.txt")
                    with open(out_txt, 'w', encoding='utf-8') as f:
                        f.write("\n".join(lines) + "\n")
            except Exception:
                pass
            # 结果展示优先使用融合图
            last_final_img = merge_img if merge_img is not None else last_final_img

        return session_dir, per_image_counts, last_final_img

    def _multiframe_fusion(self, per_image_counts):
        """融合多张图片的计数结果：按类别取最大值（忽略 desk）。"""
        from collections import defaultdict
        fused = defaultdict(int)
        if not per_image_counts:
            return {}
        for counts in per_image_counts:
            if not counts:
                continue
            for k, v in counts.items():
                if k == 'desk':
                    continue
                try:
                    fused[k] = max(fused[k], int(v))
                except Exception:
                    warnings.warn("Failed to fuse counts; skip.")
        return dict(fused)
    
    def _table12_voting_strategy(self, per_image_counts):
        """
        Table 1 和 Table 2 专用投票策略：
        - 类别存在条件：在6张图片中至少出现3次
        - 数量判定：如果数量2出现至少3次，则为2；否则为1
        """
        from collections import defaultdict
        
        # 统计每个类别的出现次数和数量分布
        class_occurrences = defaultdict(list)  # {class_name: [count1, count2, ...]}
        
        for counts in per_image_counts:
            if not counts:
                continue
            for class_name, count in counts.items():
                if class_name == 'desk':
                    continue
                try:
                    class_occurrences[class_name].append(int(count))
                except Exception:
                    pass
        
        # 计算最终结果
        final_counts = {}
        
        for class_name, count_list in class_occurrences.items():
            # 类别至少出现3次才认为存在
            if len(count_list) < 3:
                self.log(f"  {class_name}: appeared {len(count_list)} times (< 3), ignored")
                continue
            
            # 统计数量2出现的次数
            count_2_occurrences = sum(1 for c in count_list if c == 2)
            
            # 如果数量2出现至少3次，则认为数量是2，否则是1
            if count_2_occurrences >= 3:
                final_counts[class_name] = 2
                self.log(f"  {class_name}: appeared {len(count_list)} times, count=2 appeared {count_2_occurrences} times -> final: 2")
            else:
                final_counts[class_name] = 1
                self.log(f"  {class_name}: appeared {len(count_list)} times, count=2 appeared {count_2_occurrences} times -> final: 1")
        
        return final_counts
    
    def _table3_voting_strategy(self, per_image_counts):
        """
        Table 3 专用投票策略（新规则）：
        - 基于最后8张图片推理结果。
        - 类别存在条件：在8张图片中至少出现3次。
        - 数量判定：如果数量2出现至少3次，则为2；否则为1。
        """
        from collections import defaultdict
        
        # 统计每个类别的出现次数和数量分布
        class_occurrences = defaultdict(list)  # {class_name: [count1, count2, ...]}
        
        for counts in per_image_counts:
            if not counts:
                continue
            for class_name, count in counts.items():
                if class_name == 'desk':
                    continue
                try:
                    class_occurrences[class_name].append(int(count))
                except Exception:
                    pass
        
        # 计算最终结果
        final_counts = {}
        
        for class_name, count_list in class_occurrences.items():
            # 类别至少出现3次才认为存在
            if len(count_list) < 3:
                self.log(f"  {class_name}: appeared {len(count_list)} times (< 3), ignored")
                continue
            
            # 统计数量2出现的次数
            count_2_occurrences = sum(1 for c in count_list if c == 2)
            
            # 如果数量2出现至少3次，则认为数量是2，否则是1
            if count_2_occurrences >= 3:
                final_counts[class_name] = 2
                self.log(f"  {class_name}: appeared {len(count_list)} times, count=2 appeared {count_2_occurrences} times -> final: 2")
            else:
                final_counts[class_name] = 1
                self.log(f"  {class_name}: appeared {len(count_list)} times, count=2 appeared {count_2_occurrences} times -> final: 1")
        
        return final_counts
    
   
    def _create_allresult_save_dir(self):
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'allresult'))
            os.makedirs(base_dir, exist_ok=True)
            try:
                entries = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
            except Exception as e:
                warnings.warn(f"Failed to list allresult directory: {e}")
                entries = []
            nums = []
            for name in entries:
                if len(name) == 3 and name.isdigit():
                    try:
                        nums.append(int(name))
                    except Exception as e:
                        warnings.warn(f"Invalid allresult subdir '{name}': {e}")
            next_num = (max(nums) + 1) if nums else 1
            dirname = f"{next_num:03d}"
            session_dir = os.path.join(base_dir, dirname)
            os.makedirs(session_dir, exist_ok=True)
            return session_dir

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
                pixmap.scaled(self.image_display.width(), self.image_display.height(), Qt.AspectRatioMode.KeepAspectRatio)
            )
            return True, rgb
        except Exception as e:
            self.log(f"Error loading image {image_path}: {str(e)}")
            return False, None

    def _check_image_has_desk(self, image):
        try:
            boxes_xyxy, confs, clses = self.overall_detector.infer(image.copy())
            desk_ids = [i for i, name in enumerate(self.overall_detector.classes) if name == 'desk']
            if boxes_xyxy is not None and len(boxes_xyxy) > 0 and len(desk_ids) > 0:
                import numpy as np
                return np.isin(clses, np.array(desk_ids, dtype=np.int32)).any()
        except Exception as e:
            self.log(f"桌子检测失败: {str(e)}")
        return False

    def _create_overall(self, session_dir, has_desk):
        """对齐 raceone：始终创建 overall/local 两个目录，用于逐目标置信度记录。"""
        session_overall_dir = os.path.join(session_dir, 'overall')
        session_local_dir = os.path.join(session_dir, 'local')
        try:
            os.makedirs(session_overall_dir, exist_ok=True)
            os.makedirs(session_local_dir, exist_ok=True)
            return session_overall_dir, session_local_dir
        except Exception as e:
            self.log(f"Failed to create directories: {str(e)}")
            return None, None

    def _save_inference_results(self, session_dir, image_path, has_desk, overall_img, local_img, merge_img, final_img):
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
        except Exception as e:
            self.log(f"Error saving results for {image_path}: {str(e)}")
 
class Camera:

    def __init__(self, gui_instance):
        self.gui = gui_instance

    def handle_camera_toggle(self):
        """处理摄像头开关切换"""
        if self.gui.camera_active:
            # 关闭摄像头
            self._close_camera()
        else:
            # 开启摄像头
            self._open_camera()

    def _open_camera(self):
        """开启摄像头：使用 astra_py.AstraCamera 与 raceone 一致。"""
        if astra_py is None:
            self.gui.image_display.setText("Camera Error: astra_py not available")
            self.gui.status_label.setText("Status: Connection Failed")
            self.gui.status_label.setStyleSheet("color: red; padding: 4px;")
            self.gui.log("astra_py not available; cannot open Astra camera")
            return
        try:
            cam = astra_py.AstraCamera()
            try:
                cam.set_resolution(1920, 1080, 30)
            except Exception:
                pass
            cam.set_outputs(want_color=True, want_depth=False, want_shaded=False, want_cloud=False)
            cam.start()
            self.gui.astra_cam = cam
        except Exception as e:
            self.gui.image_display.setText("Camera Error: Cannot open Astra camera")
            self.gui.status_label.setText("Status: Connection Failed")
            self.gui.status_label.setStyleSheet("color: red; padding: 4px;")
            self.gui.log(f"Failed to open Astra camera: {e}")
            return
        self.gui.camera_active = True
        self.gui.timer.start(30)
        self.gui.status_label.setText("Status: Connected")
        self.gui.status_label.setStyleSheet("color: green; padding: 4px;")
        self.gui.Button_Camera.setText("关闭摄像头")
        self.gui.log("Astra camera started successfully.")

    def _close_camera(self):
        """关闭摄像头"""
        self.gui.timer.stop()
        # 关闭 Astra 相机
        if hasattr(self.gui, 'astra_cam') and self.gui.astra_cam:
            try:
                self.gui.astra_cam.stop()
            except Exception:
                pass
            self.gui.astra_cam = None
        self.gui.camera_active = False
        self.gui.Button_Camera.setText("开启摄像头")
        self.gui.status_label.setText("Status: Disconnected")
        self.gui.status_label.setStyleSheet("color: red; padding: 4px;")
        self.gui.log("Camera stopped.")

    def refresh_frame(self):
        """刷新摄像头帧"""
        if hasattr(self.gui, 'astra_cam') and self.gui.astra_cam:
            try:
                frames = self.gui.astra_cam.get_frames()
                if not frames:
                    return
                color = frames[0]  # BGR(HWC)
                rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                self.gui.latest_frame = rgb
                h, w = rgb.shape[:2]
                q_img = QImage(rgb.data, w, h, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                self.gui.image_display.setPixmap(
                    pixmap.scaled(self.gui.image_display.width(), self.gui.image_display.height(), Qt.AspectRatioMode.KeepAspectRatio)
                )
            except Exception:
                pass

    def captureImage(self):
        """抓拍当前帧，仅保存到 data/capture。"""
        if not self.gui.camera_active:
            self.gui.log("Camera not active. Cannot capture image.")
            return
        if self.gui.latest_frame is None:
            self.gui.log("No frame available to capture.")
            return
        try:
            capture_dir = os.path.join(os.path.dirname(__file__), 'data', 'capture')
            os.makedirs(capture_dir, exist_ok=True)
            timestamp = time.strftime('%Y%m%d-%H%M%S')
            save_path = os.path.join(capture_dir, f'captured-{timestamp}.png')
            bgr = cv2.cvtColor(self.gui.latest_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, bgr)
            self.gui.captured_image = self.gui.latest_frame.copy()
            self.gui.last_captured_path = save_path
            self.gui.log(f"Image captured: {save_path} (仅保存，不自动处理)")
        except Exception as e:
            self.gui.log(f"Failed to capture image: {str(e)}")

    def _load_camera_pref(self):
        """已改为 Astra 相机，保留兼容接口。"""
        return None, None


def auto_run_by_arg():
    """命令行自动模式：python main1.py num → 启动主窗体、开启摄像头并调用 Start2。"""
    if len(sys.argv) < 2:
        return
    # 接受一个参数 num（当前逻辑不直接使用，仅作为触发自动流程的开关）
    _ = sys.argv[1].strip()

    app = QApplication(sys.argv)
    window = MainWindow()
    # 自动运行模式，不再透传 CLI 参数
    window.show()

    # 打开相机（交由 Camera 管理器）
    window.camera_manager.handle_camera_toggle()
    if not window.camera_active:
        print("Camera failed to start, abort auto-run.")
        sys.exit(1)

    # 稍等片刻后调用第二轮按钮逻辑（不自动退出）
    QTimer.singleShot(1000, window.start2Button)
    sys.exit(app.exec())


if __name__ == '__main__':
    # 入口：有任意参数走自动模式；否则正常启动 GUI
    if len(sys.argv) > 1:
        auto_run_by_arg()
    else:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())