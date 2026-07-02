"""阶段11：产品推荐引擎 — 基于预算和用途的智能推荐

输入预算 + 使用场景，从数据集中推荐 Top-5 最值得购买的产品。
综合评分 = 性价比排名(40%) + 配置匹配度(30%) + 品牌可靠度(30%)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import *


def _compute_score(row, budget, usage_type, all_data):
    """计算单条产品的推荐评分"""
    score = 0.0

    # 1. 预算匹配度 (40%) — 离预算越近越好，但不超预算
    if row["Price_USD"] <= budget:
        price_ratio = row["Price_USD"] / budget if budget > 0 else 0
        # 越接近预算越好（买到预算上限的通常配置更好）
        score += 40 * (0.5 + 0.5 * price_ratio)
    else:
        # 超预算惩罚
        over_ratio = (row["Price_USD"] - budget) / budget
        score += 40 * max(0, 1 - over_ratio * 2)

    # 2. 性价比 (30%) — 在同价位段中的性价比百分位
    price_low = row["Price_USD"] * 0.7
    price_high = row["Price_USD"] * 1.3
    similar = all_data[
        (all_data["Price_USD"] >= price_low) &
        (all_data["Price_USD"] <= price_high)
    ]
    if len(similar) > 0:
        percentile = (similar["Price_Performance_Ratio"] <
                      row["Price_Performance_Ratio"]).mean()
        score += 30 * percentile

    # 3. 使用类型匹配 (30%)
    if row["Usage_Type"] == usage_type:
        score += 30
    else:
        # 相邻类型给部分分
        affinity = {
            "Basic": {"Student": 0.5, "Professional": 0.2, "Gaming": 0},
            "Student": {"Basic": 0.5, "Professional": 0.5, "Gaming": 0.2},
            "Professional": {"Student": 0.5, "Gaming": 0.5, "Basic": 0.2},
            "Gaming": {"Professional": 0.5, "Student": 0.2, "Basic": 0},
        }
        score += 30 * affinity.get(usage_type, {}).get(row["Usage_Type"], 0)

    return score


def recommend(df, budget, usage_type, top_n=5):
    """推荐主函数"""
    df = df.copy()
    df["Recommend_Score"] = df.apply(
        lambda r: _compute_score(r, budget, usage_type, df), axis=1
    )
    top = df.nlargest(top_n, "Recommend_Score")
    return top


def run(df):
    """演示推荐引擎：为三种典型用户推荐"""
    init_env()

    print("=" * 60)
    print("阶段11：产品推荐引擎")
    print("=" * 60)

    # ── 演示三种典型用户 ──
    demo_users = [
        {"name": "学生党", "budget": 600, "usage": "Student"},
        {"name": "职场新人", "budget": 1200, "usage": "Professional"},
        {"name": "游戏发烧友", "budget": 2500, "usage": "Gaming"},
    ]

    all_recs = []
    for user in demo_users:
        print(f"\n  >>> {user['name']} (预算: ${user['budget']}, 用途: {user['usage']})")
        recs = recommend(df, user["budget"], user["usage"], top_n=5)
        print(f"  {'Brand':<8} {'Type':<13} {'Price':>7} {'Perf':>6} "
              f"{'Ratio':>6} {'Score':>6}")
        print("  " + "-" * 48)
        for _, r in recs.iterrows():
            print(f"  {r['Brand']:<8} {r['Usage_Type']:<13} "
                  f"${r['Price_USD']:>6} {r['Overall_Performance_Score']:>6.0f} "
                  f"{r['Price_Performance_Ratio']:>6.2f} "
                  f"{r['Recommend_Score']:>6.1f}")
            all_recs.append({
                "user": user["name"],
                "brand": r["Brand"],
                "usage": r["Usage_Type"],
                "price": r["Price_USD"],
                "perf": r["Overall_Performance_Score"],
                "ratio": r["Price_Performance_Ratio"],
                "score": r["Recommend_Score"],
            })

    # ── 图表：三个用户的推荐结果 ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for i, user in enumerate(demo_users):
        recs = recommend(df, user["budget"], user["usage"], top_n=5)
        ax = axes[i]

        colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["gold"],
                  PALETTE["green"], PALETTE["red"]]
        y_pos = np.arange(len(recs))[::-1]

        bars = ax.barh(y_pos, recs["Recommend_Score"], color=colors,
                       edgecolor="white", height=0.6)
        labels = [f"{r['Brand']} | {r['Usage_Type']}\n${r['Price_USD']} "
                  f"(perf {r['Overall_Performance_Score']:.0f})"
                  for _, r in recs.iterrows()]

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Recommend Score", fontsize=10)
        ax.set_title(f"{user['name']}\nBudget ${user['budget']} | {user['usage']}",
                     fontsize=12, fontweight="bold")
        ax.set_xlim(0, 105)

        for bar, val in zip(bars, recs["Recommend_Score"]):
            ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}", va="center", fontsize=9)

    fig.suptitle("Product Recommendation Engine - Top 5 per User Type",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(CHART_DIR, "29_recommender_demo.png")
    fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [图] {path}")

    # ── 图表：性价比热力图（价格 vs 性能，标注推荐区域）──
    fig2, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(df["Price_USD"], df["Overall_Performance_Score"],
                         c=df["Price_Performance_Ratio"], cmap="RdYlGn",
                         alpha=0.4, s=8, vmin=0, vmax=5)
    plt.colorbar(scatter, ax=ax, label="Price-Performance Ratio")

    # 标注推荐产品
    for user in demo_users:
        recs = recommend(df, user["budget"], user["usage"], top_n=1)
        if len(recs) > 0:
            r = recs.iloc[0]
            ax.scatter(r["Price_USD"], r["Overall_Performance_Score"],
                       c="black", s=100, zorder=5, marker="*",
                       edgecolors="white", linewidth=1.5)
            ax.annotate(f"{user['name']}\n${r['Price_USD']}",
                        (r["Price_USD"], r["Overall_Performance_Score"]),
                        textcoords="offset points", xytext=(10, 10),
                        fontsize=8, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="black"))

    ax.set_xlabel("Price (USD)", fontsize=12)
    ax.set_ylabel("Overall Performance Score", fontsize=12)
    ax.set_title("Value Heatmap with Top Recommendations",
                 fontsize=14, fontweight="bold")

    fig2.tight_layout()
    path2 = os.path.join(CHART_DIR, "30_recommendation_heatmap.png")
    fig2.savefig(path2, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig2)
    print(f"  [图] {path2}")

    print("\n✅ 阶段11完成\n")

    return {"demo_users": len(demo_users)}


if __name__ == "__main__":
    """交互模式：输入预算和用途，获取推荐"""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import RAW_DATA

    df = pd.read_csv(RAW_DATA)

    print("=" * 50)
    print("  笔记本推荐引擎")
    print("=" * 50)

    try:
        budget = float(input("\n  请输入预算 (USD): "))
        print("  可选用途: Basic / Student / Professional / Gaming")
        usage = input("  请输入用途: ").strip()
        if usage not in ["Basic", "Student", "Professional", "Gaming"]:
            print(f"  未知用途 '{usage}'，默认使用 Student")
            usage = "Student"

        recs = recommend(df, budget, usage, top_n=5)
        print(f"\n  >>> Top 5 推荐 ({usage}, budget ${budget:.0f})")
        print(f"  {'Brand':<8} {'Type':<13} {'Price':>7} {'Perf':>6} "
              f"{'Ratio':>6} {'Score':>6}")
        print("  " + "-" * 48)
        for _, r in recs.iterrows():
            print(f"  {r['Brand']:<8} {r['Usage_Type']:<13} "
                  f"${r['Price_USD']:>6} {r['Overall_Performance_Score']:>6.0f} "
                  f"{r['Price_Performance_Ratio']:>6.2f} "
                  f"{r['Recommend_Score']:>6.1f}")

    except (ValueError, KeyboardInterrupt):
        print("\n  输入无效，退出。")
