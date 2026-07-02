"""阶段8：异常定价检测 — Isolation Forest + 回归残差分析

找出"割韭菜"产品（同配置价格虚高）和"宝藏机"（超值）。
商业价值：帮助消费者避坑，帮助厂商发现定价异常。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from config import *


def run(df):
    """异常定价检测主流程"""
    init_env()

    print("=" * 60)
    print("阶段8：异常定价检测")
    print("=" * 60)

    # ── 1. 用配置特征预测"公允价格" ──
    # 用除价格和性价比外的数值特征预测 Price_USD
    price_pred_feats = [
        "CPU_Cores", "CPU_Frequency_GHz", "RAM_GB", "Storage_GB",
        "Screen_Size", "CPU_Performance_Score", "GPU_Performance_Score",
        "Overall_Performance_Score", "Perf_Per_Core", "GPU_Weight",
    ]
    X = df[price_pred_feats].values
    y = df["Price_USD"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred_all = reg.predict(X)
    r2 = reg.score(X_test, y_test)

    # 残差 = 实际价格 - 预测公允价格
    df = df.copy()
    df["Fair_Price"] = y_pred_all.round(0).astype(int)
    df["Price_Residual"] = df["Price_USD"] - df["Fair_Price"]
    # 防止公允价格为 0 导致的 inf
    df["Fair_Price"] = df["Fair_Price"].replace(0, 1)
    df["Residual_Pct"] = (df["Price_Residual"] / df["Fair_Price"] * 100).round(1)
    # 截断极端值防止 histogram 崩掉
    df["Residual_Pct"] = df["Residual_Pct"].clip(-500, 500)

    print(f"\n  公允价格模型 R^2 = {r2:.4f}")
    print(f"  残差均值: ${df['Price_Residual'].mean():.1f}")
    print(f"  残差标准差: ${df['Price_Residual'].std():.1f}")

    # ── 2. Isolation Forest 异常检测 ──
    iso_feats = price_pred_feats + ["Price_USD"]
    iso = IsolationForest(
        contamination=0.05, random_state=RANDOM_SEED, n_jobs=-1
    )
    df["Anomaly"] = iso.fit_predict(df[iso_feats].values)
    n_anomaly = (df["Anomaly"] == -1).sum()
    print(f"\n  Isolation Forest 检出异常: {n_anomaly} 条 ({n_anomaly/len(df)*100:.1f}%)")

    # ── 3. 分类：割韭菜 vs 宝藏 ──
    # 残差 > 1.5倍标准差 = 割韭菜（价格虚高）
    # 残差 < -1.5倍标准差 = 宝藏（超值）
    threshold = 1.5 * df["Price_Residual"].std()
    df["Pricing_Type"] = "Normal"
    df.loc[df["Price_Residual"] > threshold, "Pricing_Type"] = "Overpriced"
    df.loc[df["Price_Residual"] < -threshold, "Pricing_Type"] = "Bargain"

    n_over = (df["Pricing_Type"] == "Overpriced").sum()
    n_bargain = (df["Pricing_Type"] == "Bargain").sum()
    print(f"\n  定价分类:")
    print(f"    割韭菜 (Overpriced): {n_over} 条 — 同配置下价格偏高")
    print(f"    宝藏机 (Bargain):    {n_bargain} 条 — 同配置下价格偏低")
    print(f"    正常 (Normal):       {len(df)-n_over-n_bargain} 条")

    # ── 4. TOP 5 割韭菜 & 宝藏 ──
    print("\n  --- TOP 5 割韭菜产品 ---")
    print(f"  {'Brand':<8} {'Type':<13} {'Price':>7} {'Fair':>7} {'Overcharge%':>12}")
    over_top = df.nlargest(5, "Residual_Pct")[
        ["Brand", "Usage_Type", "Price_USD", "Fair_Price", "Residual_Pct"]
    ]
    for _, r in over_top.iterrows():
        print(f"  {r['Brand']:<8} {r['Usage_Type']:<13} "
              f"${r['Price_USD']:>6} ${r['Fair_Price']:>6} "
              f"{r['Residual_Pct']:>11.1f}%")

    print("\n  --- TOP 5 宝藏产品 ---")
    print(f"  {'Brand':<8} {'Type':<13} {'Price':>7} {'Fair':>7} {'Discount%':>12}")
    bargain_top = df.nsmallest(5, "Residual_Pct")[
        ["Brand", "Usage_Type", "Price_USD", "Fair_Price", "Residual_Pct"]
    ]
    for _, r in bargain_top.iterrows():
        print(f"  {r['Brand']:<8} {r['Usage_Type']:<13} "
              f"${r['Price_USD']:>6} ${r['Fair_Price']:>6} "
              f"{r['Residual_Pct']:>11.1f}%")

    # ── 5. 按品牌看异常率 ──
    brand_anomaly = df.groupby("Brand").agg(
        Overpriced_rate=("Pricing_Type", lambda x: (x == "Overpriced").mean() * 100),
        Bargain_rate=("Pricing_Type", lambda x: (x == "Bargain").mean() * 100),
        Count=("Brand", "count"),
    ).round(1)
    brand_anomaly = brand_anomaly.sort_values("Overpriced_rate", ascending=False)
    print("\n  --- 品牌异常定价率 ---")
    print(brand_anomaly.to_string())

    # ── 6. 图表：异常散点图 ──
    fig, ax = plt.subplots(figsize=(10, 8))
    normal = df[df["Pricing_Type"] == "Normal"]
    over = df[df["Pricing_Type"] == "Overpriced"]
    bargain = df[df["Pricing_Type"] == "Bargain"]

    ax.scatter(normal["Fair_Price"], normal["Price_USD"],
               c=PALETTE["gray"], alpha=0.3, s=8, label=f"Normal ({len(normal)})")
    ax.scatter(over["Fair_Price"], over["Price_USD"],
               c=PALETTE["red"], alpha=0.6, s=15, label=f"Overpriced ({n_over})")
    ax.scatter(bargain["Fair_Price"], bargain["Price_USD"],
               c=PALETTE["green"], alpha=0.6, s=15, label=f"Bargain ({n_bargain})")

    # 公允价格线
    lim = max(df["Fair_Price"].max(), df["Price_USD"].max())
    ax.plot([0, lim], [0, lim], "k--", alpha=0.5, label="Fair Price Line")
    ax.set_xlim(0, lim * 1.05)
    ax.set_ylim(0, lim * 1.05)
    ax.set_xlabel("Predicted Fair Price (USD)", fontsize=12)
    ax.set_ylabel("Actual Price (USD)", fontsize=12)
    ax.set_title("Anomaly Pricing Detection: Overpriced vs Bargain",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)

    fig.tight_layout()
    path = os.path.join(CHART_DIR, "21_anomaly_pricing.png")
    fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [图] {path}")

    # ── 7. 图表：品牌异常率对比 ──
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    brands = brand_anomaly.index.tolist()
    x = np.arange(len(brands))
    w = 0.35
    bars1 = ax2.bar(x - w / 2, brand_anomaly["Overpriced_rate"], w,
                    color=PALETTE["red"], edgecolor="white", label="Overpriced %")
    bars2 = ax2.bar(x + w / 2, brand_anomaly["Bargain_rate"], w,
                    color=PALETTE["green"], edgecolor="white", label="Bargain %")
    for bar in bars1:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f"{bar.get_height():.1f}", ha="center", fontsize=9)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f"{bar.get_height():.1f}", ha="center", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(brands, fontsize=10)
    ax2.set_ylabel("Rate (%)", fontsize=12)
    ax2.set_title("Brand Anomaly Pricing Rate", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)

    fig2.tight_layout()
    path2 = os.path.join(CHART_DIR, "22_brand_anomaly_rate.png")
    fig2.savefig(path2, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig2)
    print(f"  [图] {path2}")

    # ── 8. 图表：残差分布直方图 ──
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.hist(df["Residual_Pct"], bins=80, color=PALETTE["blue"],
             edgecolor="white", alpha=0.7)
    ax3.axvline(0, color="k", linestyle="--", alpha=0.5, label="Fair Price")
    over_thresh_pct = threshold / df["Fair_Price"].mean() * 100
    ax3.axvline(over_thresh_pct, color=PALETTE["red"], linestyle="--",
                alpha=0.7, label=f"Overpriced threshold (+{over_thresh_pct:.0f}%)")
    ax3.axvline(-over_thresh_pct, color=PALETTE["green"], linestyle="--",
                alpha=0.7, label=f"Bargain threshold (-{over_thresh_pct:.0f}%)")
    ax3.set_xlabel("Price Residual (%)", fontsize=12)
    ax3.set_ylabel("Count", fontsize=12)
    ax3.set_title("Price Residual Distribution", fontsize=14, fontweight="bold")
    ax3.legend(fontsize=10)

    fig3.tight_layout()
    path3 = os.path.join(CHART_DIR, "23_residual_distribution.png")
    fig3.savefig(path3, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig3)
    print(f"  [图] {path3}")

    print("\n✅ 阶段8完成\n")

    return {
        "r2": r2,
        "n_overpriced": n_over,
        "n_bargain": n_bargain,
        "n_anomaly": n_anomaly,
    }
