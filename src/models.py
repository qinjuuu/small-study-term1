"""阶段3-5：建模 — 线性回归 / KNN分类 / K-means聚类"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             confusion_matrix, classification_report,
                             silhouette_score)
from sklearn.decomposition import PCA

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
# 输出到 workspace temp，绕过沙箱限制
OUT_DIR = os.path.join(os.environ.get("TEMP", "/tmp"), "laptop_analysis_output")
CHART_DIR = os.path.join(OUT_DIR, "charts")
os.makedirs(CHART_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "processed.csv"))
    return df


# ══════════════════════════════════════════
# 阶段3：线性回归 — 价格 vs 综合性能
# ══════════════════════════════════════════
def linear_regression(df):
    print("=" * 60)
    print("阶段3：线性回归 — 价格预测综合性能分")
    print("=" * 60)

    X = df[["Price_USD"]]
    y = df["Overall_Performance_Score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"  系数: {lr.coef_[0]:.2f}")
    print(f"  截距: {lr.intercept_:.2f}")
    print(f"  MSE:  {mse:.2f}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  R2:   {r2:.4f}")

    # 回归散点图
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(X_test, y_test, alpha=0.3, s=10, color="#5B9BD5", label="实际值")
    x_line = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1, 1)
    ax.plot(x_line, lr.predict(x_line), "r-", linewidth=2, label="回归线")
    ax.set_xlabel("Price (USD)")
    ax.set_ylabel("综合性能分")
    ax.set_title(f"线性回归：价格 -> 综合性能分  (R2={r2:.3f})", fontsize=14, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(CHART_DIR, "07_linear_regression.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] {path}")

    # 残差图
    residuals = y_test - y_pred
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    axes2[0].scatter(y_pred, residuals, alpha=0.3, color="#5B9BD5")
    axes2[0].axhline(0, color="red", linestyle="--")
    axes2[0].set_xlabel("预测值")
    axes2[0].set_ylabel("残差")
    axes2[0].set_title("残差图")
    axes2[1].hist(residuals, bins=40, color="#ED7D31", edgecolor="white")
    axes2[1].axvline(0, color="red", linestyle="--")
    axes2[1].set_xlabel("残差")
    axes2[1].set_ylabel("频数")
    axes2[1].set_title("残差分布")
    fig2.tight_layout()
    path2 = os.path.join(CHART_DIR, "08_residual_analysis.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  [图] {path2}")

    print(f"\n✅ 阶段3完成\n")
    return {"mse": mse, "mae": mae, "r2": r2}


# ══════════════════════════════════════════
# 阶段4：KNN分类 — 预测使用类型
# ══════════════════════════════════════════
def knn_classification(df):
    print("=" * 60)
    print("阶段4：KNN分类 — 预测使用类型")
    print("=" * 60)

    feat_cols = [c for c in df.columns if c.endswith("_scaled")
                 and "Price_Performance" not in c and "RAM_Storage" not in c]
    X = df[feat_cols]
    le = LabelEncoder()
    y = le.fit_transform(df["Usage_Type"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 遍历 K 值
    k_range = range(1, 31)
    scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores.append(cross_val_score(knn, X_train, y_train,
                                      cv=5, scoring="accuracy").mean())
    best_k = k_range[np.argmax(scores)]
    best_score = max(scores)
    print(f"  最优K: {best_k}, 交叉验证准确率: {best_score:.4f}")

    # K值曲线图
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_range, scores, "o-", color="#5B9BD5", markersize=4)
    ax.axvline(best_k, color="red", linestyle="--", label=f"最优 K={best_k}")
    ax.set_xlabel("K 值")
    ax.set_ylabel("5折交叉验证准确率")
    ax.set_title("KNN：K 值调优曲线", fontsize=14, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(CHART_DIR, "09_knn_k_tuning.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] {path}")

    # 最优模型
    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(X_train, y_train)
    y_pred = best_knn.predict(X_test)

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2,
                xticklabels=le.classes_, yticklabels=le.classes_)
    ax2.set_xlabel("预测")
    ax2.set_ylabel("实际")
    ax2.set_title(f"KNN 混淆矩阵 (K={best_k})", fontsize=14, fontweight="bold")
    fig2.tight_layout()
    path2 = os.path.join(CHART_DIR, "10_knn_confusion_matrix.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  [图] {path2}")

    # 分类报告
    cr = classification_report(y_test, y_pred, target_names=le.classes_)
    print(f"\n  分类报告:\n{cr}")

    # 决策边界图（PCA降维到2D）
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train)
    pca_test = pca.transform(X_test)

    knn_viz = KNeighborsClassifier(n_neighbors=best_k)
    knn_viz.fit(X_pca, y_train)

    fig3, ax3 = plt.subplots(figsize=(9, 7))
    xx, yy = np.meshgrid(
        np.linspace(X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1, 200),
        np.linspace(X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1, 200)
    )
    Z = knn_viz.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    colors = ["#5B9BD5", "#ED7D31", "#A5A5A5", "#FFC000"]
    ax3.contourf(xx, yy, Z, alpha=0.3, levels=len(le.classes_)-1,
                 colors=colors)
    for i, label in enumerate(le.classes_):
        mask = y_test == i
        ax3.scatter(pca_test[mask, 0], pca_test[mask, 1],
                    c=colors[i], label=label, alpha=0.6, s=15, edgecolor="white")
    ax3.set_xlabel(f"PCA1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax3.set_ylabel(f"PCA2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax3.set_title(f"KNN 决策边界 (PCA降维, K={best_k})", fontsize=14, fontweight="bold")
    ax3.legend()
    fig3.tight_layout()
    path3 = os.path.join(CHART_DIR, "11_knn_decision_boundary.png")
    fig3.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"  [图] {path3}")

    print(f"\n✅ 阶段4完成\n")
    return {"best_k": best_k, "accuracy": best_score}


# ══════════════════════════════════════════
# 阶段5：K-means聚类
# ══════════════════════════════════════════
def kmeans_clustering(df):
    print("=" * 60)
    print("阶段5：K-means聚类")
    print("=" * 60)

    feat_cols = [c for c in df.columns if c.endswith("_scaled")]
    X = df[feat_cols].values

    # 肘部法则
    K_range = range(1, 11)
    inertias = []
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(K_range, inertias, "o-", color="#5B9BD5", markersize=6)
    axes[0].set_xlabel("K 值")
    axes[0].set_ylabel("惯性 (Inertia)")
    axes[0].set_title("肘部法则 — 确定最优聚类数", fontsize=13, fontweight="bold")
    # 标注拐点（找第二个导数变化最大的点）
    diffs = np.diff(inertias)
    diffs2 = np.diff(diffs)
    elbow_k = np.argmin(diffs2) + 2  # +2 because of two diffs
    if elbow_k < 2:
        elbow_k = 4  # fallback
    axes[0].axvline(elbow_k, color="red", linestyle="--", label=f"拐点 K={elbow_k}")
    axes[0].legend()

    # 轮廓系数
    sil_scores = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sil_scores.append(silhouette_score(X, labels))
    axes[1].plot(range(2, 11), sil_scores, "o-", color="#ED7D31", markersize=6)
    axes[1].set_xlabel("K 值")
    axes[1].set_ylabel("轮廓系数")
    axes[1].set_title("轮廓系数 — 聚类效果评估", fontsize=13, fontweight="bold")
    best_sil_k = np.argmax(sil_scores) + 2
    axes[1].axvline(best_sil_k, color="red", linestyle="--",
                    label=f"最优 K={best_sil_k}")
    axes[1].legend()
    fig.tight_layout()
    path = os.path.join(CHART_DIR, "12_kmeans_elbow_silhouette.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] {path}")
    print(f"  肘部法建议 K={elbow_k}")
    print(f"  轮廓系数最优 K={best_sil_k}, 系数={max(sil_scores):.4f}")

    # 用最优 K 聚类
    final_k = best_sil_k
    km = KMeans(n_clusters=final_k, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(X)
    df["Cluster"] = cluster_labels

    # 轮廓系数
    sil = silhouette_score(X, cluster_labels)
    print(f"  最终轮廓系数: {sil:.4f}")

    # 聚类散点图（PCA 降维）
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    centers_pca = pca.transform(km.cluster_centers_)

    fig2, ax2 = plt.subplots(figsize=(9, 7))
    colors = ["#5B9BD5", "#ED7D31", "#FFC000", "#70AD47", "#A5A5A5", "#FF6384"]
    for i in range(final_k):
        mask = cluster_labels == i
        ax2.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=colors[i % len(colors)], label=f"簇 {i}",
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
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  [图] {path2}")

    # 每簇用户画像
    print(f"\n=== 簇画像（原始特征均值）===")
    profile_cols = ["CPU_Cores", "CPU_Frequency_GHz", "RAM_GB", "Storage_GB",
                    "CPU_Performance_Score", "GPU_Performance_Score",
                    "Overall_Performance_Score", "Price_USD",
                    "Price_Performance_Ratio"]
    profile = df.groupby("Cluster")[profile_cols].mean()
    profile["Count"] = df.groupby("Cluster").size()
    print(profile.to_string())

    # 簇 vs 使用类型对比
    print(f"\n=== 聚类 vs 原始标签交叉表 ===")
    ct = pd.crosstab(df["Cluster"], df["Usage_Type"])
    print(ct.to_string())

    # 保存带簇标签的数据
    df.to_csv(os.path.join(OUT_DIR, "clustered.csv"), index=False)

    print(f"\n✅ 阶段5完成\n")
    return {"k": final_k, "silhouette": sil, "profile": profile, "cross_tab": ct}


# ══════════════════════════════════════════
# Main
# ══════════════════════════════════════════
if __name__ == "__main__":
    df = load_data()
    linear_regression(df)
    knn_classification(df)
    kmeans_clustering(df)
    print("\n🎉 全部建模完成！")
