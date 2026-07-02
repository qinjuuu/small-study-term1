"""真实数据集适配器
将 Kaggle Global Laptop Dataset 映射到项目流水线标准格式。
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

from config import ROOT, DATA_DIR, RANDOM_SEED

np.random.seed(RANDOM_SEED)

KAGGLE_DIR = os.path.join(
    os.path.expanduser("~"),
    ".cache/kagglehub/datasets/fa23bst011/"
    "global-laptop-dataset-with-cpu-gpu-and-price/versions/1"
)
SRC_FILE = os.path.join(KAGGLE_DIR, "global_laptop_dataset_1.2M.csv")

# ── 输出路径 ──
FULL_DATA = os.path.join(DATA_DIR, "real_full.csv")          # 全量 1.2M
ANALYSIS_DATA = os.path.join(DATA_DIR, "real_analysis.csv")  # 分析用 50k 采样
PROCESSED_DATA = os.path.join(DATA_DIR, "processed.csv")     # 预处理后
CLUSTERED_DATA = os.path.join(DATA_DIR, "clustered.csv")     # 聚类后
SCALER_PATH = os.path.join(DATA_DIR, "scaler.pkl")

# ── 屏幕尺寸估算：按使用类型 ──
SCREEN_BY_USAGE = {
    "Student":       (14.0, 1.0),   # (均值, 标准差)
    "Basic":         (14.5, 1.2),
    "Professional":  (15.6, 1.0),
    "Gaming":        (16.0, 1.0),
    "High-End Gaming": (16.5, 0.8),
}

# ── 品牌名称映射（标准化）────
BRAND_MAP = {
    "Acer": "Acer", "Lenovo": "Lenovo", "HP": "HP",
    "Asus": "Asus", "Dell": "Dell", "MSI": "MSI", "Apple": "Apple",
}

# ── 使用类型映射 ──
USAGE_MAP = {
    "Student":          "Student",
    "Basic":            "Basic",
    "Professional":     "Professional",
    "High-End Gaming":  "Gaming",
}


def load_and_map():
    """加载 Kaggle 数据集，映射为项目标准列名"""
    print(f"📥 加载 {SRC_FILE} ...")
    df = pd.read_csv(SRC_FILE)
    print(f"   完成: {len(df):,} 行 × {len(df.columns)} 列")

    mapped = pd.DataFrame()

    # 直接映射
    mapped["Brand"] = df["Laptop_Brand"].map(BRAND_MAP).fillna("Other")
    mapped["Usage_Type"] = df["Usage_Type"].map(USAGE_MAP).fillna("Basic")
    mapped["CPU_Brand"] = df["CPU_Brand"]
    mapped["GPU_Brand"] = df["GPU_Brand"]
    mapped["GPU_Model"] = df["GPU_Model"]
    mapped["CPU_Cores"] = df["Cores"].astype(int)
    mapped["CPU_Frequency_GHz"] = df["Base_Clock"].round(2)  # 基频
    mapped["RAM_GB"] = df["RAM_GB"].astype(int)
    mapped["Storage_GB"] = df["Storage_GB"].astype(int)
    mapped["CPU_Performance_Score"] = df["CPU_Performance"].astype(int)
    mapped["GPU_Performance_Score"] = df["GPU_Performance"].astype(int)
    mapped["Overall_Performance_Score"] = df["Total_Performance"].astype(int)
    mapped["Price_USD"] = df["Price_USD"].round(2)

    # 派生字段
    mapped["Price_Performance_Ratio"] = (
        mapped["Overall_Performance_Score"] /
        mapped["Price_USD"].replace(0, 1)
    ).round(2)

    # 按使用类型估算屏幕尺寸（原数据无此字段）
    mapped["Screen_Size"] = mapped["Usage_Type"].apply(
        lambda t: np.random.normal(*SCREEN_BY_USAGE.get(t, (15.0, 1.5)))
    ).clip(11.0, 18.0).round(1)

    # 清理
    mapped["CPU_Frequency_GHz"] = mapped["CPU_Frequency_GHz"].clip(0.8, 6.0)
    mapped["Storage_GB"] = mapped["Storage_GB"].clip(64, 4096)
    mapped = mapped.dropna()

    print(f"   映射后: {len(mapped):,} 行 × {len(mapped.columns)} 列")
    print(f"   品牌: {mapped['Brand'].unique().tolist()}")
    print(f"   用途: {sorted(mapped['Usage_Type'].unique().tolist())}")
    print(f"   价格: ${mapped['Price_USD'].min():.0f} ~ ${mapped['Price_USD'].max():.0f}")
    return mapped


def generate():
    """主入口：生成完整/采样/预处理三份数据"""
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. 全量保存
    df = load_and_map()
    df.to_csv(FULL_DATA, index=False)
    print(f"✅ 全量数据: {FULL_DATA}  ({len(df):,} 行)")

    # 2. 分析用采样（50k，保持用途分布均衡）
    n_sample = min(50000, len(df))
    samples = []
    for usage in df["Usage_Type"].unique():
        sub = df[df["Usage_Type"] == usage]
        n = max(1, int(n_sample * len(sub) / len(df)))
        samples.append(sub.sample(n=min(n, len(sub)), random_state=RANDOM_SEED))
    df_sample = pd.concat(samples, ignore_index=True)
    df_sample.to_csv(ANALYSIS_DATA, index=False)
    print(f"✅ 分析采样: {ANALYSIS_DATA}  ({len(df_sample):,} 行)")

    # 3. 数据处理报告
    print("\n=== 真实数据概况 ===")
    print(f"  总行数:   {len(df):,}")
    print(f"  国家数:   10  (Brazil, Germany, Canada, UK, ...)")
    print(f"  品牌数:   {df['Brand'].nunique()}  ({', '.join(df['Brand'].unique())})")
    print(f"  用途类型: {sorted(df['Usage_Type'].unique())}")
    print(f"  价格区间: ${df['Price_USD'].min():.0f} ~ ${df['Price_USD'].max():.0f}")

    # 用途分布
    print("\n  用途分布:")
    for u in sorted(df["Usage_Type"].unique()):
        cnt = len(df[df["Usage_Type"] == u])
        print(f"    {u}: {cnt:,}  ({cnt/len(df)*100:.1f}%)")

    return df_sample


if __name__ == "__main__":
    generate()
