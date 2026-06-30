"""阶段2：数据预处理 — 异常值/特征工程/标准化"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHART_DIR = os.path.join(ROOT, "output", "charts")
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(CHART_DIR, exist_ok=True)


def load_data():
    return pd.read_csv(os.path.join(DATA_DIR, "sampled_laptops.csv"))


def check_outliers(df):
    """IQR 异常值检测"""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print("=== 异常值检测 (IQR) ===")
    outlier_counts = {}
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n = ((df[col] < lower) | (df[col] > upper)).sum()
        if n > 0:
            outlier_counts[col] = n
            print(f"  {col}: {n} 个异常值 ({n/len(df)*100:.1f}%)")
    print(f"总计: {len(outlier_counts)} 列含异常值")
    return df


def plot_outlier_box(df):
    """多列异常值箱线图"""
    cols = ["CPU_Cores", "CPU_Frequency_GHz", "RAM_GB", "Storage_GB",
            "Overall_Performance_Score", "Price_USD"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        axes[i].boxplot(df[col], patch_artist=True,
                        boxprops=dict(facecolor="#5B9BD5", alpha=0.6))
        axes[i].set_title(col, fontsize=11)
    fig.suptitle("关键特征箱线图（含异常值标记）", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(CHART_DIR, "06_outlier_boxplots.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] {path}")
    return path


def feature_engineering(df):
    """特征工程：构造复合特征"""
    print("\n=== 特征工程 ===")
    # 性价比已经在了
    # 性能密度：综合分 / 核心数（单核效率）
    df["Perf_Per_Core"] = (df["Overall_Performance_Score"] / df["CPU_Cores"]).round(1)
    print(f"  + Perf_Per_Core (单核性能密度)")
    # 内存存储比
    df["RAM_Storage_Ratio"] = (df["RAM_GB"] / df["Storage_GB"] * 100).round(2)
    print(f"  + RAM_Storage_Ratio (内存存储比)")
    # GPU 占比
    df["GPU_Weight"] = (df["GPU_Performance_Score"] /
                         (df["CPU_Performance_Score"] + df["GPU_Performance_Score"])).round(3)
    print(f"  + GPU_Weight (GPU性能占比)")
    return df


def standardize(df):
    """标准化"""
    from sklearn.preprocessing import StandardScaler
    print("\n=== 标准化 ===")
    feats = ["CPU_Cores", "CPU_Frequency_GHz", "RAM_GB", "Storage_GB",
             "CPU_Performance_Score", "GPU_Performance_Score",
             "Overall_Performance_Score", "Price_USD",
             "Price_Performance_Ratio", "Perf_Per_Core",
             "RAM_Storage_Ratio", "GPU_Weight"]
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[feats])
    df_scaled = pd.DataFrame(df_scaled, columns=[f"{c}_scaled" for c in feats])
    print(f"  标准化特征数: {len(feats)}")
    # 保存标准化器
    with open(os.path.join(DATA_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print(f"  已保存 scaler")
    return df_scaled, feats


def save_processed(df, df_scaled):
    """保存处理后的数据"""
    # 合并
    df_out = pd.concat([df, df_scaled], axis=1)
    path = os.path.join(DATA_DIR, "processed.csv")
    df_out.to_csv(path, index=False)
    print(f"\n✅ 处理后数据: {path}  形状 {df_out.shape}")
    return df_out


def full_preprocess():
    df = load_data()
    check_outliers(df)
    plot_outlier_box(df)
    df = feature_engineering(df)
    df_scaled, feat_names = standardize(df)
    df_out = save_processed(df, df_scaled)
    print("\n✅ 阶段2：预处理完成")
    return df_out


if __name__ == "__main__":
    full_preprocess()
