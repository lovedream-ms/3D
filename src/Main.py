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
    QTableWidgetItem,
    QTableWidget,
    QHeaderView,
)
from PyQt6.QtGui import QPixmap, QImage, QColor
from PyQt6.QtCore import Qt, QTimer, pyqtSignal

import numpy as np
import time
import sys
import glob
import threading

from collections import Counter
from ultralytics.engine.results import Results

import Camera
from Socket import Socket
from DetectionPipeline import DetectionPipeline, DetectionConfig
from Utils import _write_txt
from Config import *


class MainWindow(QMainWindow):
    detectionFinishedSignal = pyqtSignal()
    updateTableSignal = pyqtSignal(list)

    def __init__(self):
        self.socketClient = Socket(JUDGE_BOX_IP, JUDGE_BOX_PORT)
        self.socketClient.send_start(1)

        super().__init__()

        self.cameraManager = Camera.OpenCVCamera()

        try:
            cfg = DetectionConfig.load_from_json("assets/best-param.json")
            self.log("成功加载最优超参数配置")
        except FileNotFoundError:
            cfg = DetectionConfig()
            self.log("未找到优化配置，使用默认参数")

        self.pipeline = DetectionPipeline(cfg)

        self.status = "idle"
        self.resultFrame = np.zeros((480, 640, 3), dtype=np.uint8)

        self.init_ui()

    def toggle_camera(self):
        if self.cameraManager.isOpen:
            self.cameraManager.stop()
            self.updateFrameTimer.stop()
            self.cameraButton.setText("打开相机")
            self.imageDisplay.clear()
        else:
            self.cameraManager.start()
            time.sleep(1)
            self.updateFrameTimer.start(30)
            self.cameraButton.setText("关闭相机")

    def log(self, message):
        timestamp = time.strftime("[%H:%M:%S]")
        print(f"{timestamp} {message}")

    def init_ui(self):
        # Window setup
        self.setWindowTitle(f"{raceName}-{teamName}-R{roundNum}-{version}")
        self.resize(1000, 800)

        mainWidget = QWidget()
        self.setCentralWidget(mainWidget)

        # Create components
        headerLabel = QLabel(f"{raceName}-{teamName}-R{roundNum}-{version}")
        headerLabel.setObjectName("headerLabel")
        headerLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)

        image_group = QGroupBox("Camera View")
        self.imageDisplay = QLabel("No Image")
        self.imageDisplay.setObjectName("displayScreen")
        self.imageDisplay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.imageDisplay.setMinimumSize(640, 480)

        line = QFrame()
        line.setObjectName("separatorLine")

        control_group = QGroupBox("Control Panel")
        self.statusLabel = QLabel("Status: Disconnected")

        self.cameraButton = QPushButton("打开摄像头")
        self.round1Button = QPushButton("开始检测")
        self.captureButton = QPushButton("录制图片")
        self.exitButton = QPushButton("退出程序")

        controlButtons = [
            self.cameraButton,
            self.round1Button,
            self.captureButton,
            self.exitButton,
        ]

        result_group = QGroupBox("Recognition Result")

        self.resultTable = QTableWidget(0, 2)
        self.resultTable.setObjectName("resultTable")
        self.resultTable.setHorizontalHeaderLabels(["名称", "个数"])
        self.resultTable.verticalHeader().setVisible(False)
        self.resultTable.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.resultTable.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.resultTable.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        header = self.resultTable.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

        log_group = QGroupBox("System Log")
        self.logOutput = QTextEdit()
        self.logOutput.setObjectName("logOutput")
        self.logOutput.setReadOnly(True)

        # Build layout tree
        main_layout = QVBoxLayout(mainWidget)
        main_layout.addWidget(headerLabel)

        image_layout = QVBoxLayout()
        image_layout.addWidget(self.imageDisplay)
        image_group.setLayout(image_layout)
        main_layout.addWidget(image_group)

        main_layout.addWidget(line)

        bottom_layout = QHBoxLayout()

        # 1. 控制面板装配
        control_layout = QVBoxLayout()
        control_layout.addWidget(self.statusLabel)

        control_layout.addStretch(1)
        for btn in controlButtons:
            btn.setFixedHeight(45)
            control_layout.addWidget(btn)
            control_layout.addStretch(1)

        control_group.setLayout(control_layout)
        bottom_layout.addWidget(control_group, stretch=1)

        # 2. 结果区装配
        result_layout = QVBoxLayout()
        result_layout.addWidget(self.resultTable)
        result_group.setLayout(result_layout)
        bottom_layout.addWidget(result_group, stretch=2)

        # 3. 日志区装配
        log_layout = QVBoxLayout()
        log_layout.addWidget(self.logOutput)
        log_group.setLayout(log_layout)
        bottom_layout.addWidget(log_group, stretch=2)

        main_layout.addLayout(bottom_layout)

        self.updateFrameTimer = QTimer()

        # 信号绑定
        self.updateFrameTimer.timeout.connect(self.update_frame)
        self.captureButton.clicked.connect(self.capture_frame)
        self.round1Button.clicked.connect(self.start_detection)
        self.cameraButton.clicked.connect(self.toggle_camera)
        self.exitButton.clicked.connect(self.close)
        self.detectionFinishedSignal.connect(self.close)
        self.updateTableSignal.connect(self.update_result_table)

    def update_frame(self):
        if self.cameraManager.isOpen:
            self.rgbFrame, self.depthFrame, self.pointCloudFrame = (
                self.cameraManager.get_frame()
            )

        if self.status == "detection":
            self.statusLabel.setText(f"Status: {self.status} active")
            self.statusLabel.setStyleSheet("color: orange; padding: 4px;")
            showFrame = self.resultFrame
        elif self.cameraManager.isOpen:
            self.statusLabel.setText("Status: Camera active")
            self.statusLabel.setStyleSheet("color: green; padding: 4px;")
            showFrame = self.rgbFrame
        else:
            return

        q_image = QImage(
            showFrame.tobytes(),
            showFrame.shape[1],
            showFrame.shape[0],
            showFrame.shape[1] * 3,
            QImage.Format.Format_RGB888,
        )
        scaled_pixmap = q_image.scaled(
            self.imageDisplay.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.imageDisplay.setPixmap(QPixmap(scaled_pixmap))

    def capture_frame(self, tableNum):
        # TODO: 是否应该支持视频?
        # 不需要 太相近的图片识别没有意义
        # 选择1: timestamp/rgbFrame.npy、depthFrame.npy、pointCloudFrame.npy | 测试时间戳目录的性能
        # 选择2: timestamp.npy(包含三者)
        if not self.cameraManager.isOpen:
            self.log("Camera not active. Cannot capture image.")
            return

        timestamp = time.strftime("%m%d-%H:%M:%S")
        filePath = f"results/machine/camera/T{tableNum}/{timestamp}.npz"
        self.cameraManager.capture_frame(filePath)

        self.log(f"Image captured and saved to: {filePath}")

    def start_detection(self):
        self.status = "detection"
        self.round1Button.setEnabled(False)
        self.updateFrameTimer.stop()
        self.log("Starting detection thread...")

        if not self.cameraManager.isOpen:
            self.log("Camera is not open. auto open.")
            self.toggle_camera()

        self.detection_thread = threading.Thread(target=self.detect, args=())
        self.detection_thread.start()

    def detect(self):
        results: list[Counter] = []
        tableNum = 3 if roundNum == 2 else 1

        for n in range(tableNum):
            self.capture_frame(n + 1)
            self.socketClient.send_rotate() if roundNum == 2 else None
            results.append(
                self.pipeline.detect_frames(f"results/machine/camera/T{n + 1}")
            )

        _write_txt(results)
        self.updateTableSignal.emit(results)
        self.socketClient.send_result(resultPath)
        self.detectionFinishedSignal.emit()

    def update_result_table(self, resultCounter: list[Counter]):
        total_rows = sum(len(c) for c in resultCounter)
        self.resultTable.setRowCount(total_rows)

        color_palette = [
            QColor("#00ff00"),
            QColor("#00ffff"),
            QColor("#ffaa00"),
            QColor("#ff00ff"),
            QColor("#ff7700"),
            QColor("#b088ff"),
        ]

        row = 0

        for idx, counter_data in enumerate(resultCounter):
            current_color = color_palette[idx % len(color_palette)]

            for name, count in counter_data.items():
                item_name = QTableWidgetItem(str(name))
                item_name.setForeground(current_color)
                item_name.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.resultTable.setItem(row, 0, item_name)

                item_count = QTableWidgetItem(str(count))
                item_count.setForeground(current_color)
                item_count.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.resultTable.setItem(row, 1, item_count)

                row += 1

    def check_status_label(self):
        self.statusLabel.setText("Status: Image Detecting")
        self.statusLabel.setStyleSheet("color: orange; padding: 4px;")

        self.statusLabel.setText("Status: Finished")
        self.statusLabel.setStyleSheet("color: green; padding: 4px;")

        self.statusLabel.setText("Status: Image Detecting")
        self.statusLabel.setStyleSheet("color: orange; padding: 4px;")

        self.statusLabel.setText("Status: Finished")
        self.statusLabel.setStyleSheet("color: green; padding: 4px;")

    def closeEvent(self, event):
        self.log("Application closed.")

        self.updateFrameTimer.stop()
        self.socketClient.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    with open("assets/Skeuomorphic.qss", "r", encoding="utf-8") as f:
        app.setStyleSheet(f.read())
    window = MainWindow()
    QTimer.singleShot(0, window.start_detection)
    window.show()
    sys.exit(app.exec())
