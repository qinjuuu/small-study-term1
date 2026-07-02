"""阶段10：品牌溢价量化分析 — 控制配置后，品牌到底值多少钱？

用回归模型量化每个品牌的价格溢价系数，
回答"同配置下 Apple 比 Acer 贵多少"这类问题。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from config import *


def run(df):
    """品牌溢价分析"""
    init_env()

    print("=" * 60)
    print("阶段10：品牌溢价量化分析")
    print("=" * 60)

    df = df.copy()

    # ── 1. 品牌基础统计 ──
    print("\n  --- 品牌基础统计 ---")
    brand_stats = df.groupby("Brand").agg(
        Count=("Price_USD", "count"),
        Avg_Price=("Price_USD", "mean"),
        Median_Price=("Price_USD", "median"),
        Avg_Performance=("Overall_Performance_Score", "mean"),
        Avg_Perf_Price=("Price_Performance_Ratio", "mean"),
    ).round(1)
    brand_stats = brand_stats.sort_values("Avg_Price", ascending=False)
    print(brand_stats.to_string())

    # ── 2. 品牌溢价回归 ──
    # 特征 = 硬件配置 + 品牌one-hot
    config_feats = [
        "CPU_Cores", "CPU_Frequency_GHz", "RAM_GB", "Storage_GB",
        "Screen_Size", "CPU_Performance_Score", "GPU_Performance_Score",
    ]

    # One-hot 编码品牌
    brand_dummies = pd.get_dummies(df["Brand"], prefix="Brand", drop_first=True)
    # drop_first=True: 以第一个品牌（字母序）为基准
    baseline_brand = sorted(df["Brand"].unique())[0]

    X_config = df[config_feats].values
    X_brand = brand_dummies.values.astype(float)
    X = np.hstack([X_config, X_brand])
    y = df["Price_USD"].values

    model = Ridge(alpha=1.0, random_state=RANDOM_SEED)
    model.fit(X, y)
    r2 = model.score(X, y)

    # 提取品牌系数 = 品牌溢价
    brand_names = list(brand_dummies.columns)
    brand_coefs = {}
    for i, name in enumerate(brand_names):
        brand_real = name.replace("Brand_", "")
        brand_coefs[brand_real] = model.coef_[len(config_feats) + i]

    # 基准品牌溢价为0
    brand_coefs[baseline_brand] = 0.0

    # 按溢价排序
    brand_premium = sorted(brand_coefs.items(), key=lambda x: x[1], reverse=True)

    print(f"\n  --- 品牌溢价 (基准: {baseline_brand}, R^2={r2:.4f}) ---")
    print(f"  {'Brand':<10} {'Premium (USD)':>14} {'Avg Price':>10} {'Count':>6}")
    print("  " + "-" * 44)
    for brand, premium in brand_premium:
        avg_p = brand_stats.loc[brand, "Avg_Price"]
        cnt = int(brand_stats.loc[brand, "Count"])
        print(f"  {brand:<10} ${premium:>13.1f} ${avg_p:>9.0f} {cnt:>6}")

    # ── 3. GPU品牌溢价 ──
    gpu_stats = df.groupby("GPU_Brand").agg(
        Count=("Price_USD", "count"),
        Avg_Price=("Price_USD", "mean"),
        Avg_Performance=("Overall_Performance_Score", "mean"),
        Avg_GPU_Score=("GPU_Performance_Score", "mean"),
    ).round(1)
    print("\n  --- GPU 品牌统计 ---")
    print(gpu_stats.to_string())

    # ── 4. 图表：品牌溢价排名 ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    brands_sorted = [b for b, _ in brand_premium]
    premiums = [p for _, p in brand_premium]
    bar_colors = [PALETTE["red"] if p > 0 else PALETTE["green"]
                  for p in premiums]

    bars = ax1.barh(brands_sorted, premiums, color=bar_colors,
                    edgecolor="white", height=0.6)
    for bar, val in zip(bars, premiums):
        ax1.text(val + (5 if val >= 0 else -5), bar.get_y() + bar.get_height() / 2,
                 f"${val:+.0f}", va="center",
                 ha="left" if val >= 0 else "right", fontsize=10)
    ax1.set_xlabel("Brand Premium (USD)", fontsize=12)
    ax1.set_title(f"Brand Price Premium\n(baseline: {baseline_brand}, R^2={r2:.3f})",
                  fontsize=13, fontweight="bold")
    ax1.axvline(0, color="k", linewidth=0.8)

    # 品牌价格箱线图
    brand_order = [b for b, _ in brand_premium]
    box_data = [df[df["Brand"] == b]["Price_USD"].values for b in brand_order]
    bp = ax2.boxplot(box_data, tick_labels=brand_order, patch_artist=True,
                     widths=0.5, showfliers=False)
    for patch, brand in zip(bp["boxes"], brand_order):
        patch.set_facecolor(PALETTE["blue"])
        patch.set_alpha(0.5)
    ax2.set_ylabel("Price (USD)", fontsize=12)
    ax2.set_title("Price Distribution by Brand",
                  fontsize=13, fontweight="bold")
    ax2.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    path = os.path.join(CHART_DIR, "27_brand_premium.png")
    fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [图] {path}")

    # ── 5. 图表：品牌性价比 vs 价格 ──
    fig2, ax = plt.subplots(figsize=(10, 7))

    for brand in df["Brand"].unique():
        mask = df["Brand"] == brand
        ax.scatter(df.loc[mask, "Price_USD"],
                   df.loc[mask, "Price_Performance_Ratio"],
                   label=brand, alpha=0.4, s=12)

    ax.set_xlabel("Price (USD)", fontsize=12)
    ax.set_ylabel("Price-Performance Ratio", fontsize=12)
    ax.set_title("Brand: Price vs Value for Money",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, title="Brand", title_fontsize=10)

    fig2.tight_layout()
    path2 = os.path.join(CHART_DIR, "28_brand_value_scatter.png")
    fig2.savefig(path2, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig2)
    print(f"  [图] {path2}")

    print("\n✅ 阶段10完成\n")

    return {
        "r2": r2,
        "brand_premium": dict(brand_premium),
        "baseline_brand": baseline_brand,
    }
