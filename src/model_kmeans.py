"""阶段5：K-means聚类 —— 无监督用户群体发现"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from config import *


def run(df):
    """返回 {"k","silhouette","profile_df","cross_tab_df"}"""
    init_env()

    feat_cols = [c for c in df.columns if c.endswith("_scaled")]
    X = df[feat_cols].values

    # ── 肘部法则 ──
    inertias = []
    for k in KMEANS_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=KMEANS_N_INIT)
        km.fit(X)
        inertias.append(km.inertia_)

    # 自动找拐点（二阶差分最小处）
    diffs = np.diff(inertias)
    diffs2 = np.diff(diffs)
    elbow_k = int(np.argmin(diffs2) + 2)
    if elbow_k < 2:
        elbow_k = 4

    # ── 轮廓系数 ──
    sil_scores = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=KMEANS_N_INIT)
        labels = km.fit_predict(X)
        sil_scores.append(silhouette_score(X, labels))
    best_sil_k = int(np.argmax(sil_scores) + 2)
    best_sil = max(sil_scores)

    print("=" * 60)
    print("阶段5：K-means聚类")
    print("=" * 60)
    print(f"  肘部法建议 K={elbow_k}")
    print(f"  轮廓系数最优 K={best_sil_k}, 系数={best_sil:.4f}")

    # ── 肘部 + 轮廓 双图 ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(KMEANS_RANGE, inertias, "o-", color=PALETTE["blue"], markersize=6)
    axes[0].axvline(elbow_k, color="red", linestyle="--", label=f"拐点 K={elbow_k}")
    axes[0].set_xlabel("K 值")
    axes[0].set_ylabel("惯性 (Inertia)")
    axes[0].set_title("肘部法则 — 确定最优聚类数", fontsize=13, fontweight="bold")
    axes[0].legend()

    axes[1].plot(range(2, 11), sil_scores, "o-", color=PALETTE["orange"], markersize=6)
    axes[1].axvline(best_sil_k, color="red", linestyle="--", label=f"最优 K={best_sil_k}")
    axes[1].set_xlabel("K 值")
    axes[1].set_ylabel("轮廓系数")
    axes[1].set_title("轮廓系数 — 聚类效果评估", fontsize=13, fontweight="bold")
    axes[1].legend()
    fig.tight_layout()
    path = os.path.join(CHART_DIR, "12_kmeans_elbow_silhouette.png")
    fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] {path}")

    # ── 用最优 K 聚类 ──
    final_k = best_sil_k
    km = KMeans(n_clusters=final_k, random_state=RANDOM_SEED, n_init=KMEANS_N_INIT)
    cluster_labels = km.fit_predict(X)
    df["Cluster"] = cluster_labels
    sil = silhouette_score(X, cluster_labels)
    print(f"  最终轮廓系数: {sil:.4f}")

    # ── 聚类散点图（PCA 降维） ──
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    centers_pca = pca.transform(km.cluster_centers_)

    fig2, ax2 = plt.subplots(figsize=(9, 7))
    cluster_colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["gold"],
                      PALETTE["green"], PALETTE["gray"], PALETTE["red"]]
    for i in range(final_k):
        mask = cluster_labels == i
        ax2.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=cluster_colors[i % len(cluster_colors)], label=f"簇 {i}",
                    alpha=0.5, s=10, edgecolor="none")
    ax2.scatter(centers_pca[:, 0], centers_pca[:, 1],
                c="red", marker="X", s=200, edgecolor="black",
                linewidth=1, label="簇中心")
    ax2.set_xlabel(f"PCA1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax2.set_ylabel(f"PCA2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax2.set_title(f"K-means 聚类结果 (K={final_k}, 轮廓系数={sil:.3f})",
                  fontsize=14, fontweight="bold")
    ax2.legend()
    fig2.tight_layout()
    path2 = os.path.join(CHART_DIR, "13_kmeans_clusters.png")
    fig2.savefig(path2, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig2)
    print(f"  [图] {path2}")

    # ── 簇画像 ──
    print("\n=== 簇画像（原始特征均值）===")
    profile = df.groupby("Cluster")[PROFILE_COLS].mean()
    profile["Count"] = df.groupby("Cluster").size()
    print(profile.to_string())

    # ── 交叉对比 ──
    print("\n=== 聚类 vs 原始标签交叉表 ===")
    ct = pd.crosstab(df["Cluster"], df["Usage_Type"])
    print(ct.to_string())

    # 保存带簇标签的数据
    df.to_csv(CLUSTERED_DATA, index=False)
    print(f"  已保存: {CLUSTERED_DATA}")
    print("✅ 阶段5完成\n")

    return {
        "k": final_k,
        "silhouette": sil,
        "profile": profile,
        "cross_tab": ct,
    }
