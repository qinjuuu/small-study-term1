"""系统化配置中心 —— 所有路径、参数、全局设置统一管理"""
import os
import sys
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ═══════════ 路径 ═══════════
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
CHART_DIR = os.path.join(ROOT, "output", "charts")
REPORT_DIR = os.path.join(ROOT, "output", "reports")

# ═══════════ 随机种子（确保可复现）═══════════
RANDOM_SEED = 42

# ═══════════ 数据集 ═══════════
N_SAMPLES = 8000
TEST_SIZE = 0.2

# ═══════════ KNN 参数 ═══════════
KNN_RANGE = range(1, 31)
KNN_CV_FOLDS = 5

# ═══════════ K-means 参数 ═══════════
KMEANS_RANGE = range(1, 11)
KMEANS_N_INIT = 10

# ═══════════ 图表 ═══════════
CHART_DPI = 150
# 调色板
PALETTE = {
    "blue":   "#5B9BD5",
    "orange": "#ED7D31",
    "gray":   "#A5A5A5",
    "gold":   "#FFC000",
    "green":  "#70AD47",
    "red":    "#FF6384",
}
USAGE_COLORS = {
    "Basic":        "#5B9BD5",
    "Student":      "#ED7D31",
    "Professional": "#A5A5A5",
    "Gaming":       "#FFC000",
}

# ═══════════ 特征字段 ═══════════
# 用于 IQR 异常检测的数值列
NUMERIC_COLS = [
    "CPU_Cores", "CPU_Frequency_GHz", "RAM_GB", "Storage_GB",
    "CPU_Performance_Score", "GPU_Performance_Score",
    "Overall_Performance_Score", "Price_USD",
    "Price_Performance_Ratio", "Perf_Per_Core",
    "RAM_Storage_Ratio", "GPU_Weight",
]

# 聚类画像展示列
PROFILE_COLS = [
    "CPU_Cores", "CPU_Frequency_GHz", "RAM_GB", "Storage_GB",
    "CPU_Performance_Score", "GPU_Performance_Score",
    "Overall_Performance_Score", "Price_USD",
    "Price_Performance_Ratio",
]

# ═══════════ 文件路径 ═══════════
RAW_DATA = os.path.join(DATA_DIR, "raw_laptops.csv")
SAMPLED_DATA = os.path.join(DATA_DIR, "sampled_laptops.csv")
PROCESSED_DATA = os.path.join(DATA_DIR, "processed.csv")
CLUSTERED_DATA = os.path.join(DATA_DIR, "clustered.csv")
SCALER_PATH = os.path.join(DATA_DIR, "scaler.pkl")


# ═══════════ 初始化函数 ═══════════
_initialized = False


def init_env():
    """初始化环境：创建目录、设置字体、配置编码（幂等）"""
    global _initialized
    if _initialized:
        return
    _initialized = True

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CHART_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    # UTF-8 输出
    if hasattr(sys.stdout, "buffer") and not isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    # 中文字体
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
