"""GMM 高斯混合模型聚类 — 软聚类 + 边界用户分析"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from config import *


def run(df):
    """BIC/AIC 选最优组件数，GMM 软聚类，输出概率归属和边界用户"""
    init_env()

    feat_cols = [c for c in df.columns if c.endswith("_scaled")]
    X = df[feat_cols].values

    # ── BIC / AIC 曲线 ──
    n_range = range(1, 11)
    bics, aics = [], []
    for n in n_range:
        gmm = GaussianMixture(n_components=n, random_state=RANDOM_SEED, n_init=5)
        gmm.fit(X)
        bics.append(gmm.bic(X))
        aics.append(gmm.aic(X))

    best_bic_k = int(np.argmin(bics) + 1)
    best_aic_k = int(np.argmin(aics) + 1)

    print("=" * 60)
    print("阶段6b：GMM 高斯混合模型聚类")
    print("=" * 60)
    print(f"  BIC 最优组件数: {best_bic_k}")
    print(f"  AIC 最优组件数: {best_aic_k}")

    # ── BIC/AIC 曲线图 ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = [PALETTE["blue"], PALETTE["orange"]]
    labels = ["BIC", "AIC"]
    data = [bics, aics]
    best_ks = [best_bic_k, best_aic_k]

    for i, ax in enumerate(axes):
        ax.plot(list(n_range), data[i], "o-", color=colors[i], markersize=6)
        ax.axvline(best_ks[i], color="red", linestyle="--",
                   label=f"最优 K={best_ks[i]}")
        ax.set_xlabel("组件数")
        ax.set_ylabel(labels[i])
        ax.set_title(f"{labels[i]} 曲线 (越低越好)", fontsize=13, fontweight="bold")
        ax.legend()

    fig.tight_layout()
    path = os.path.join(CHART_DIR, "16_gmm_bic_aic.png")
    fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] {path}")

    # ── 使用 BIC 最优组件数训练 ──
    use_k = best_bic_k if best_bic_k >= 2 else 3
    gmm = GaussianMixture(n_components=use_k, random_state=RANDOM_SEED, n_init=10)
    cluster_labels = gmm.fit_predict(X)
    proba = gmm.predict_proba(X)

    df["GMM_Cluster"] = cluster_labels
    # 加入每个簇的概率列
    for k_idx in range(use_k):
        df[f"GMM_Prob_C{k_idx}"] = proba[:, k_idx]
    # 最大归属概率
    df["GMM_Max_Prob"] = proba.max(axis=1)

    # 轮廓系数
    sil = silhouette_score(X, cluster_labels)
    print(f"  GMM 轮廓系数: {sil:.4f}  (K-means=0.50)")

    # ── PCA 散点图 ──
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig2, ax2 = plt.subplots(figsize=(10, 7))
    cluster_colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["gold"],
                      PALETTE["green"], PALETTE["gray"], PALETTE["red"]]
    for i in range(use_k):
        mask = cluster_labels == i
        ax2.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=cluster_colors[i % len(cluster_colors)],
                    label=f"簇 {i} ({mask.sum()}人)",
                    alpha=0.5, s=10, edgecolor="none")
    ax2.set_xlabel(f"PCA1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax2.set_ylabel(f"PCA2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax2.set_title(f"GMM 聚类结果 (K={use_k}, 轮廓系数={sil:.3f})",
                  fontsize=14, fontweight="bold")
    ax2.legend()
    fig2.tight_layout()
    path2 = os.path.join(CHART_DIR, "17_gmm_clusters.png")
    fig2.savefig(path2, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig2)
    print(f"  [图] {path2}")

    # ── 软归属分布 ──
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))

    # 最大概率的直方图
    axes3[0].hist(df["GMM_Max_Prob"], bins=60, color=PALETTE["blue"],
                  edgecolor="white", alpha=0.8)
    axes3[0].axvline(0.7, color="red", linestyle="--", label="边界阈值 0.7")
    axes3[0].set_xlabel("最大归属概率")
    axes3[0].set_ylabel("样本数")
    axes3[0].set_title("软聚类 — 最大归属概率分布", fontsize=13, fontweight="bold")
    axes3[0].legend()

    # 边界用户统计
    boundary_mask = df["GMM_Max_Prob"] < 0.7
    n_boundary = boundary_mask.sum()
    n_strong = (~boundary_mask).sum()
    axes3[1].bar(["高置信归属", "边界用户(概率<0.7)"], [n_strong, n_boundary],
                 color=[PALETTE["green"], PALETTE["orange"]], edgecolor="white")
    axes3[1].set_ylabel("样本数")
    axes3[1].set_title(f"归属置信度分类 (边界={n_boundary}人, {n_boundary/len(df)*100:.1f}%)",
                       fontsize=13, fontweight="bold")
    for i, v in enumerate([n_strong, n_boundary]):
        axes3[1].text(i, v + 10, str(v), ha="center", fontsize=12, fontweight="bold")

    fig3.tight_layout()
    path3 = os.path.join(CHART_DIR, "18_gmm_soft_assignment.png")
    fig3.savefig(path3, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig3)
    print(f"  [图] {path3}")

    # ── 边界用户画像 ──
    if n_boundary > 0:
        print(f"\n=== 边界用户 (最大概率<0.7, 共{n_boundary}人) ===")
        boundary_df = df[boundary_mask]
        # 列出边界用户中占比最大的两个簇
        top_clusters = (boundary_df["GMM_Cluster"].value_counts()
                        .head(2).index.tolist())
        print(f"  主要涉及簇: {top_clusters}")
        print(f"  边界用户平均归属概率: {boundary_df['GMM_Max_Prob'].mean():.3f}")

    # ── 簇画像 ──
    print("\n=== GMM 簇画像（原始特征均值）===")
    profile = df.groupby("GMM_Cluster")[PROFILE_COLS].mean()
    profile["Count"] = df.groupby("GMM_Cluster").size()
    profile["Avg_Max_Prob"] = df.groupby("GMM_Cluster")["GMM_Max_Prob"].mean()
    print(profile.to_string())

    # ── 与原始标签交叉对比 ──
    print("\n=== GMM 聚类 vs 原始标签 ===")
    ct = pd.crosstab(df["GMM_Cluster"], df["Usage_Type"])
    print(ct.to_string())

    print("✅ 阶段6b完成\n")

    return {
        "k": use_k,
        "silhouette": sil,
        "bic_k": best_bic_k,
        "aic_k": best_aic_k,
        "n_boundary": int(n_boundary),
        "profile": profile,
        "cross_tab": ct,
    }
