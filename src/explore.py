"""阶段1：数据探索与可视化"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import *


def load_data():
    return pd.read_csv(SAMPLED_DATA)


def explore(df: pd.DataFrame):
    """打印数据概览"""
    print("=" * 60)
    print("数据探索报告")
    print("=" * 60)
    print(f"样本数: {len(df)}, 特征数: {len(df.columns)}")
    print(f"\n列名及类型:")
    print(df.dtypes)
    print(f"\n缺失值:")
    miss = df.isnull().sum()
    miss = miss[miss > 0]
    print(miss if len(miss) > 0 else "无缺失值")
    print(f"\n使用类型分布:")
    print(df["Usage_Type"].value_counts())
    print(f"\n数值特征描述:")
    print(df.describe().to_string())


def plot_usage_pie(df):
    """使用类型饼图"""
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df["Usage_Type"].value_counts()
    colors = ["#5B9BD5", "#ED7D31", "#A5A5A5", "#FFC000"]
    ax.pie(counts, labels=counts.index, autopct="%1.1f%%",
           colors=colors, startangle=140, explode=(0.02,)*4)
    ax.set_title("使用类型分布", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(CHART_DIR, "01_usage_distribution.png")
    fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] {path}")
    return path


def plot_price_distribution(df):
    """价格分布直方图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # 全量
    axes[0].hist(df["Price_USD"], bins=40, color="#5B9BD5", edgecolor="white")
    axes[0].axvline(df["Price_USD"].median(), color="red", linestyle="--", label=f"中位数=${df['Price_USD'].median():.0f}")
    axes[0].set_xlabel("Price (USD)")
    axes[0].set_ylabel("数量")
    axes[0].set_title("价格分布直方图")
    axes[0].legend()
    # 按使用类型
    for ut, c in zip(["Basic", "Student", "Professional", "Gaming"],
                     ["#5B9BD5", "#ED7D31", "#A5A5A5", "#FFC000"]):
        axes[1].hist(df[df["Usage_Type"] == ut]["Price_USD"],
                     bins=30, alpha=0.6, label=ut, color=c, edgecolor="white")
    axes[1].set_xlabel("Price (USD)")
    axes[1].set_ylabel("数量")
    axes[1].set_title("各使用类型价格分布")
    axes[1].legend()
    fig.tight_layout()
    path = os.path.join(CHART_DIR, "02_price_distribution.png")
    fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] {path}")
    return path


def plot_correlation_heatmap(df):
    """特征相关性热力图"""
    num_cols = ["CPU_Cores", "CPU_Frequency_GHz", "RAM_GB", "Storage_GB",
                "CPU_Performance_Score", "GPU_Performance_Score",
                "Overall_Performance_Score", "Price_USD",
                "Price_Performance_Ratio"]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=ax, square=True,
                annot_kws={"size": 8})
    ax.set_title("数值特征相关性热力图", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(CHART_DIR, "03_correlation_heatmap.png")
    fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] {path}")
    return path


def plot_box_price_by_type(df):
    """按使用类型价格箱线图"""
    fig, ax = plt.subplots(figsize=(8, 5))
    order = ["Basic", "Student", "Professional", "Gaming"]
    colors = ["#5B9BD5", "#ED7D31", "#A5A5A5", "#FFC000"]
    bp = ax.boxplot([df[df["Usage_Type"] == t]["Price_USD"] for t in order],
                    tick_labels=order, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_ylabel("Price (USD)")
    ax.set_title("各使用类型价格箱线图", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(CHART_DIR, "04_price_boxplot.png")
    fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] {path}")
    return path


def plot_performance_scatter(df):
    """性价比散点图"""
    fig, ax = plt.subplots(figsize=(9, 6))
    colors_map = {"Basic": "#5B9BD5", "Student": "#ED7D31",
                  "Professional": "#A5A5A5", "Gaming": "#FFC000"}
    for ut in colors_map:
        subset = df[df["Usage_Type"] == ut]
        ax.scatter(subset["Price_USD"], subset["Overall_Performance_Score"],
                   c=colors_map[ut], label=ut, alpha=0.5, s=15)
    ax.set_xlabel("Price (USD)")
    ax.set_ylabel("综合性能分")
    ax.set_title("价格 vs 综合性能分（按使用类型着色）", fontsize=14, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(CHART_DIR, "05_price_vs_performance.png")
    fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] {path}")
    return path


if __name__ == "__main__":
    df = load_data()
    explore(df)
    print("\n生成可视化图表...")
    plot_usage_pie(df)
    plot_price_distribution(df)
    plot_correlation_heatmap(df)
    plot_box_price_by_type(df)
    plot_performance_scatter(df)
    print("\n✅ 阶段1：数据探索完成")
