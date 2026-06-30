"""阶段1：数据加载与初步探索"""
import pandas as pd
import numpy as np
import os

# 项目根目录
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)


def download_dataset():
    """通过 kagglehub 下载 Global Laptop Dataset"""
    import kagglehub
    print("[1/4] 正在下载数据集...")
    path = kagglehub.dataset_download(
        "fa23bst011/global-laptop-dataset-with-cpu-gpu-and-price"
    )
    # 找到 CSV 文件
    for f in os.listdir(path):
        if f.endswith(".csv"):
            src = os.path.join(path, f)
            dst = os.path.join(DATA_DIR, "raw_laptops.csv")
            import shutil
            shutil.copy(src, dst)
            print(f"[1/4] 已保存到: {dst}")
            return dst
    raise FileNotFoundError("未找到CSV文件")


def load_and_sample(csv_path, n=8000, random_state=42):
    """加载全量数据并随机抽样"""
    print(f"[2/4] 正在加载数据...")
    df = pd.read_csv(csv_path)
    print(f"  全量数据: {df.shape[0]} 行, {df.shape[1]} 列")

    if len(df) > n:
        df = df.sample(n=n, random_state=random_state).reset_index(drop=True)
        print(f"  抽样后: {df.shape[0]} 行")
    return df


def basic_explore(df):
    """基础数据探索"""
    print(f"\n[3/4] === 数据概览 ===")
    print(f"形状: {df.shape}")
    print(f"\n列名: {list(df.columns)}")
    print(f"\n数据类型:\n{df.dtypes}")
    print(f"\n缺失值统计:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\n基本统计:\n{df.describe().to_string()}")
    return df


def save_sampled(df):
    """保存抽样数据"""
    path = os.path.join(DATA_DIR, "sampled_laptops.csv")
    df.to_csv(path, index=False)
    print(f"\n[4/4] 抽样数据已保存: {path}")


if __name__ == "__main__":
    csv_path = download_dataset()
    df = load_and_sample(csv_path, n=8000)
    basic_explore(df)
    save_sampled(df)
    print("\n✅ 阶段1完成")
