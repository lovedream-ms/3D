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
        # 相机管理器
        self.camera_manager = camera.OpenCVCamera()

        self.updateFrameTimer = QTimer()

        self._warmup_done = False
        self.model_manager = ModelManager()
        self.overall_detector = self.model_manager.get_overall_detector()
        self.local_detector = self.model_manager.get_local_detector()

        self.latest_frame = None
        self.captured_image = None
        self.camera_active = False

        self._init_ui()

        # 相机管理器与事件接管
        self.updateFrameTimer.timeout.connect(self.camera_manager.get_frame)
        self.cameraButton.clicked.connect(self.toggle_camera)

        # 保存外部传入的环境名（用于结果额外写入）
        self.cli_env_name = env_name or ""

    def toggle_camera(self):
        if self.camera_manager.isOpen:
            self.camera_manager.stop()
            self.updateFrameTimer.stop()
            self.cameraButton.setText("打开相机")
            self.image_display.clear()
        else:
            self.camera_manager.start()
            time.sleep(1)
            self.updateFrameTimer.start(30)
            self.cameraButton.setText("关闭相机")

    def _init_ui(self):
        """构建界面组件、布局，并绑定按钮信号。"""
        self.setWindowTitle(f"3D Detection-{self.version}")
        self.setGeometry(100, 100, 984, 600)

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
        self.cameraButton.clicked.connect(self.camera_manager.get_frame)
        self.Button_Round1.clicked.connect(self.start1Button)
        self.Button_Capture.clicked.connect(self.captureImage)
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

    def capture_frame(self):
        if not self.cameraManager.isOpen:
            self.log("Camera not active. Cannot capture image.")
            return

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        # TODO: 测试时间戳目录的性能
        savePath = f"./results/machine/camera/{timestamp}/"
        os.makedirs(savePath)

        for key, value in videoStreams.items():
            (
                np.save(savePath + f"{key}Frame.npy", eval(f"self.{key}Frame"))
                if value
                else None
            )

        self.log(f"Image captured and saved to: {savePath}")

    def start_detection(self):
        iterations = 3 if roundNum == 2 else 1

        for _ in range(iterations):
            self.capture_frame()
            if roundNum == 2:
                self.socketClient.send_rotate()

            _, perImageCounts, _ = self.detectImages()
            finalCounts = self.multiframe_fusion(perImageCounts)

            _write_txt(finalCounts or {}, 2, teamShortName)

        # 追加文件 start 和 end
        resultPath = f"results/machine/{teamShortName}-R{roundNum}.txt"
        self.socketClient.send_result(resultPath)
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
