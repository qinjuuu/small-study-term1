"""阶段9：价格分箱分段分析 — 不同价位段的定价逻辑差异

将市场分为 Budget / Mid / Premium / Flagship 四档，
分别建模，发现入门级"配置决定价格"、旗舰级"品牌决定价格"的差异。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from config import *


# 价格分档定义
SEGMENTS = [
    ("Budget", 0, 500),
    ("Mid-range", 500, 1000),
    ("Premium", 1000, 2000),
    ("Flagship", 2000, 10000),
]

SEGMENT_COLORS = {
    "Budget": PALETTE["green"],
    "Mid-range": PALETTE["blue"],
    "Premium": PALETTE["orange"],
    "Flagship": PALETTE["red"],
}


def run(df):
    """价格分箱分段分析"""
    init_env()

    print("=" * 60)
    print("阶段9：价格分箱分段分析")
    print("=" * 60)

    # ── 1. 分箱 ──
    df = df.copy()
    df["Price_Segment"] = "Unknown"
    for name, lo, hi in SEGMENTS:
        mask = (df["Price_USD"] >= lo) & (df["Price_USD"] < hi)
        df.loc[mask, "Price_Segment"] = name

    print("\n  --- 价格分箱统计 ---")
    seg_stats = df.groupby("Price_Segment").agg(
        Count=("Price_USD", "count"),
        Avg_Price=("Price_USD", "mean"),
        Avg_Performance=("Overall_Performance_Score", "mean"),
        Avg_GPU=("GPU_Performance_Score", "mean"),
        Avg_Perf_Price_Ratio=("Price_Performance_Ratio", "mean"),
    ).round(1)
    # 按定义顺序排列
    seg_order = [s[0] for s in SEGMENTS]
    seg_stats = seg_stats.reindex(seg_order)
    print(seg_stats.to_string())

    # ── 2. 分段回归：性能 -> 价格 ──
    reg_feats = [
        "CPU_Cores", "CPU_Frequency_GHz", "RAM_GB", "Storage_GB",
        "CPU_Performance_Score", "GPU_Performance_Score",
        "Overall_Performance_Score",
    ]

    seg_results = {}
    print("\n  --- 分段回归 (Performance -> Price) ---")
    print(f"  {'Segment':<12} {'Count':>6} {'R^2':>8} {'CV_R^2':>8} "
          f"{'Elasticity':>10} {'MAE':>8}")
    print("  " + "-" * 52)

    for name, lo, hi in SEGMENTS:
        seg_df = df[df["Price_Segment"] == name]
        if len(seg_df) < 50:
            continue

        X = seg_df[reg_feats].values
        y = seg_df["Price_USD"].values

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = model.score(X, y)

        cv_r2 = cross_val_score(model, X, y, cv=5,
                                scoring="r2", n_jobs=-1).mean()

        # 弹性系数：性能每涨1%，价格涨多少%
        # 用 Overall_Performance_Score 的系数近似
        perf_idx = reg_feats.index("Overall_Performance_Score")
        elasticity = model.coef_[perf_idx]
        mae = np.mean(np.abs(y - y_pred))

        seg_results[name] = {
            "r2": r2,
            "cv_r2": cv_r2,
            "elasticity": elasticity,
            "mae": mae,
            "count": len(seg_df),
        }
        print(f"  {name:<12} {len(seg_df):>6} {r2:>8.4f} {cv_r2:>8.4f} "
              f"{elasticity:>10.4f} ${mae:>7.0f}")

    # ── 3. 使用类型在各档的分布 ──
    print("\n  --- 使用类型 x 价格档位 交叉表 ---")
    cross = pd.crosstab(df["Usage_Type"], df["Price_Segment"],
                        margins=True)
    cross = cross.reindex(columns=seg_order + ["All"], fill_value=0)
    print(cross.to_string())

    # ── 4. 图表：分段回归对比 ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (name, lo, hi) in enumerate(SEGMENTS):
        seg_df = df[df["Price_Segment"] == name]
        if len(seg_df) < 50:
            axes[i].text(0.5, 0.5, f"{name}\nInsufficient data",
                         ha="center", va="center", fontsize=14,
                         transform=axes[i].transAxes)
            continue

        X = seg_df["Overall_Performance_Score"].values.reshape(-1, 1)
        y = seg_df["Price_USD"].values
        m = LinearRegression().fit(X, y)
        y_fit = m.predict(X)
        r2 = m.score(X, y)

        axes[i].scatter(seg_df["Overall_Performance_Score"],
                        seg_df["Price_USD"],
                        c=SEGMENT_COLORS[name], alpha=0.3, s=8)
        sort_idx = np.argsort(X.ravel())
        axes[i].plot(X.ravel()[sort_idx], y_fit[sort_idx],
                     "k-", linewidth=2, alpha=0.7)
        axes[i].set_title(f"{name} (n={len(seg_df)}, R^2={r2:.3f})",
                          fontsize=12, fontweight="bold")
        axes[i].set_xlabel("Overall Performance Score", fontsize=10)
        axes[i].set_ylabel("Price (USD)", fontsize=10)

    fig.suptitle("Price-Performance Regression by Segment",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = os.path.join(CHART_DIR, "24_segment_regression.png")
    fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [图] {path}")

    # ── 5. 图表：R^2 & 弹性系数变化 ──
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    seg_names = [s for s in seg_results.keys()]
    r2_vals = [seg_results[s]["r2"] for s in seg_names]
    cv_vals = [seg_results[s]["cv_r2"] for s in seg_names]
    elas_vals = [seg_results[s]["elasticity"] for s in seg_names]
    bar_colors = [SEGMENT_COLORS[s] for s in seg_names]

    x = np.arange(len(seg_names))
    w = 0.35
    bars1 = ax1.bar(x - w / 2, r2_vals, w, color=bar_colors,
                    edgecolor="white", label="R^2")
    bars2 = ax1.bar(x + w / 2, cv_vals, w, color=bar_colors,
                    edgecolor="white", alpha=0.5, label="CV R^2")
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{bar.get_height():.3f}", ha="center", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(seg_names, fontsize=10)
    ax1.set_ylabel("R^2", fontsize=12)
    ax1.set_title("Model Fit by Price Segment", fontsize=13, fontweight="bold")
    ax1.legend()
    ax1.set_ylim(0, 1.1)

    bars3 = ax2.bar(seg_names, elas_vals, color=bar_colors, edgecolor="white")
    for bar, val in zip(bars3, elas_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                 f"{val:.4f}", ha="center", fontsize=10)
    ax2.set_ylabel("Price Elasticity (USD per perf point)", fontsize=11)
    ax2.set_title("Price Elasticity by Segment", fontsize=13, fontweight="bold")

    fig2.tight_layout()
    path2 = os.path.join(CHART_DIR, "25_segment_elasticity.png")
    fig2.savefig(path2, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig2)
    print(f"  [图] {path2}")

    # ── 6. 图表：各档位使用类型分布 ──
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    seg_type_cross = pd.crosstab(df["Price_Segment"],
                                  df["Usage_Type"], normalize="index") * 100
    seg_type_cross = seg_type_cross.reindex(seg_order)
    seg_type_cross.plot(kind="barh", stacked=True, ax=ax3,
                        color=[USAGE_COLORS[c] for c in seg_type_cross.columns],
                        edgecolor="white", width=0.6)
    ax3.set_xlabel("Percentage (%)", fontsize=12)
    ax3.set_ylabel("Price Segment", fontsize=12)
    ax3.set_title("Usage Type Distribution by Price Segment",
                  fontsize=14, fontweight="bold")
    ax3.legend(title="Usage Type", fontsize=10, title_fontsize=11)
    ax3.invert_yaxis()

    fig3.tight_layout()
    path3 = os.path.join(CHART_DIR, "26_segment_usage_type.png")
    fig3.savefig(path3, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig3)
    print(f"  [图] {path3}")

    print("\n✅ 阶段9完成\n")

    return {
        "segments": seg_results,
        "cross_table": cross,
    }
