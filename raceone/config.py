# config.py

import os

# 项目根目录
BASE_DIR = os.path.dirname(__file__)


# ONNX 模型路径
ZHUOZI_MODEL_PATH = os.path.join(BASE_DIR, "model", "v8_v1_overall.om")
WUPIN_MODEL_PATH = os.path.join(BASE_DIR, "model", "v8_v2_local.om")

# 推理设备与调试
DEVICE_ID = 0


# 预热图片路径（模型加载后进行一次预推理）
WARMUP_IMAGE_PATH = os.path.join(BASE_DIR, 'data', 'test', '0-0-0-0-0-1-180121.png')

CLASSES = [
    'desk',
    'CA001','CA002','CA003','CA004',
    'CB001','CB002','CB003','CB004',
    'CC001','CC002','CC003','CC004',
    'CD001','CD002','CD003','CD004'
]

# -------- 通用检测阈值 --------
# 统一推理阈值（与 Ultralytics predict 示例一致）
PRED_IOU_THRES = 0.66
# 统一置信度阈值（未命中各模型细粒度阈值时回退）
PRED_CONF_THRES = 0.68

# 模型2（local_detector）各类别最低置信度阈值（未配置使用 PRED_CONF_THRES）
M2_CLASS_SCORE_THRESHOLDS = {
    'desk': 0.2,
    'CA001': 0.69, 'CA002': 0.7, 'CA003': 0.7, 'CA004': 0.7,
    'CB001': 0.68, 'CB002': 0.86, 'CB003': 0.66 ,'CB004': 0.85,
    'CC001': 0.7, 'CC002': 0.7, 'CC003': 0.7, 'CC004': 0.7,
    'CD001': 0.7, 'CD002': 0.7, 'CD003': 0.7, 'CD004': 0.7,
}

# 模型1（overall_detector）类别置信度阈值（用于模型1可视化/补框前筛选）
M1_CLASS_SCORE_THRESHOLDS = {
    'desk': 0.2,
    'CA001': 0.66, 'CA002': 0.68, 'CA003': 0.71, 'CA004': 0.7,
    'CB001': 0.7, 'CB002': 0.85, 'CB003': 0.66, 'CB004': 0.85,
    'CC001': 0.7, 'CC002': 0.7, 'CC003': 0.7, 'CC004': 0.7,
    'CD001': 0.7, 'CD002': 0.7, 'CD003': 0.7, 'CD004': 0.7,
}

# -------- 推理调试开关 --------
INFER_DEBUG = False

# 每个类别允许的最多数量
CLASS_MAX_COUNTS = {
    'CA001': 2,
    'CA002': 2,
    'CA003': 2,
    'CA004': 2,
    
    'CB001': 2,
    'CB002': 1,
    'CB003': 2,
    'CB004': 1,
    
    'CC001': 2,
    'CC002': 2,
    'CC003': 2,
    'CC004': 2,
    
    'CD001': 2,
    'CD002': 2,
    'CD003': 2,
    'CD004': 2,
    
    'desk': 1,
}

# 保存结果路径（写入裁判箱所需的 TXT 文件目录）
# 改为固定保存到桌面目录，便于查找/导出
RESULT_FOLDER = "/home/HwHiAiUser/Desktop/result_r"
# 确保目录存在
try:
    os.makedirs(RESULT_FOLDER, exist_ok=True)
except Exception:
    pass
TEAM_SHORT_NAME = "DUT-WZQXJ"
JUDGE_BOX_IP = "192.168.1.88"
JUDGE_BOX_PORT = 6666

# -------- 采集与推理控制（第一轮 Start1 使用） --------
# 采集模式：'frames' 表示抓取多帧图片；'video' 表示录制一段视频
CAPTURE_MODE = 'frames'

# 多帧图片模式下，抓取的帧数
CAPTURE_FRAME_COUNT = 3



# -------- 相机抓拍保存路径 --------
# 原始抓拍保存目录
CAPTURE_SAVE_DIR = os.path.join(BASE_DIR, 'data', 'capture')

# 桌子外扩比例（用于裁剪后扩大区域，再交给模型2）
DESK_EXPANSION_RATIO = 0.20

# -------- 前处理输入尺寸（保持与 Ultralytics 一致的 letterbox 流程） --------
# 注意：尺寸需为 32 的倍数
OVERALL_IMG_HEIGHT = 640
OVERALL_IMG_WIDTH = 640

LOCAL_IMG_HEIGHT = 640
LOCAL_IMG_WIDTH = 640

# -------- 跨模型对齐与补框参数 --------
# 模型1（全图）与模型2（桌面裁剪）在坐标对应时用于匹配的 IoU 阈值
# 当 IoU >= 该阈值时，认为两者检测到同一目标；若标签不同，以模型2为主。
# 当 IoU < 该阈值时，认为模型1存在“额外检测”，将其补充到模型2的最终结果中。
CROSS_MODEL_MATCH_IOU = 0.30

# -------- 融合策略：类别优先选择的模型来源 --------
# key 为类别名，value 为 'overall' 或 'local'，用于当模型1与模型2对同一目标给出不同类别时的优先级。
# 例：{'CA001': 'overall', 'CA002': 'local'} 表示 CA001 冲突时以模型1为准，CA002 以模型2为准。
MERGE_CLASS_PREFERENCE = {
    # 示例：可按需在此处增加或修改
    'CA001': 'both',
    'CA002': 'local',
    'CA003': 'overall',
    'CA004': 'both',
    'CB001': 'local',
    'CB002': 'overall',
    'CB003': 'local',
    'CB004': 'local',
    'CC001': 'overall',
    'CC002': 'overall',
    'CC003': 'local',
    'CC004': 'overall',
    'CD001': 'local',
    'CD002': 'local',
    'CD003': 'overall',
    'CD004': 'both',
}

# -------- 补框策略：允许哪个模型进行“额外检测补框” --------
# key 为类别名；value 取值：'overall' | 'local' | 'both'
# - 'overall': 仅允许模型1(全图)未匹配框补到最终结果
# - 'local': 仅允许模型2(局部)未匹配框保留（不接收模型1的补框）
# - 'both': 两边都可（默认行为）
# 例：仅允许模型2对 'CB001' 的补框：{'CB001': 'local'}
SUPPLEMENT_ALLOWED_SOURCES = {
    'CA001': 'both',
    'CA002': 'local',
    'CA003': 'overall',
    'CA004': 'both',
    'CB001': 'local',
    'CB002': 'overall',
    'CB003': 'local',
    'CB004': 'local',
    'CC001': 'overall',
    'CC002': 'overall',
    'CC003': 'local',
    'CC004': 'overall',
    'CD001': 'local',
    'CD002': 'local',
    'CD003': 'overall',
    'CD004': 'both',
}

 
# -------- 运行环境到目标映射（用于 run.sh 后处理写入） --------
# 键：conda 环境名；值：(Goal_ID, Num)
# 可按需修改，例如：'math': ('W002', 1), 'chinese': ('W001', 1)
ENV_GOAL_MAP = {
    'math': ('W002', 1),
    'chinese': ('W001', 1),
}
