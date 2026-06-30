"""阶段4：KNN分类 —— 根据硬件特征预测使用类型"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from config import *


def run(df):
    """返回 {"best_k","accuracy","classification_report"}"""
    init_env()

    # 选标准化后的特征（排除复合特征避免共线性）
    feat_cols = [c for c in df.columns if c.endswith("_scaled")
                 and "Price_Performance" not in c and "RAM_Storage" not in c]
    X = df[feat_cols]
    le = LabelEncoder()
    y = le.fit_transform(df["Usage_Type"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    # ── K 值遍历 ──
    scores = []
    for k in KNN_RANGE:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores.append(cross_val_score(knn, X_train, y_train,
                                      cv=KNN_CV_FOLDS, scoring="accuracy").mean())
    best_k = KNN_RANGE[np.argmax(scores)]
    best_score = max(scores)

    print("=" * 60)
    print("阶段4：KNN分类 — 预测使用类型")
    print("=" * 60)
    print(f"  最优 K: {best_k}, 交叉验证准确率: {best_score:.4f}")

    # ── K 值调优曲线 ──
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(KNN_RANGE, scores, "o-", color=PALETTE["blue"], markersize=4)
    ax.axvline(best_k, color="red", linestyle="--", label=f"最优 K={best_k}")
    ax.set_xlabel("K 值")
    ax.set_ylabel(f"{KNN_CV_FOLDS}折交叉验证准确率")
    ax.set_title("KNN：K 值调优曲线", fontsize=14, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(CHART_DIR, "09_knn_k_tuning.png")
    fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] {path}")

    # ── 最优模型 ──
    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(X_train, y_train)
    y_pred = best_knn.predict(X_test)

    # ── 混淆矩阵 ──
    cm = confusion_matrix(y_test, y_pred)
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2,
                xticklabels=le.classes_, yticklabels=le.classes_)
    ax2.set_xlabel("预测")
    ax2.set_ylabel("实际")
    ax2.set_title(f"KNN 混淆矩阵 (K={best_k})", fontsize=14, fontweight="bold")
    fig2.tight_layout()
    path2 = os.path.join(CHART_DIR, "10_knn_confusion_matrix.png")
    fig2.savefig(path2, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig2)
    print(f"  [图] {path2}")

    # ── 分类报告 ──
    cr = classification_report(y_test, y_pred, target_names=le.classes_)
    print(f"\n  分类报告:\n{cr}")

    # ── 决策边界（PCA 降维） ──
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
    colors_4 = [PALETTE["blue"], PALETTE["orange"], PALETTE["gray"], PALETTE["gold"]]
    ax3.contourf(xx, yy, Z, alpha=0.3, levels=len(le.classes_)-1, colors=colors_4)
    for i, label in enumerate(le.classes_):
        mask = y_test == i
        ax3.scatter(pca_test[mask, 0], pca_test[mask, 1],
                    c=colors_4[i], label=label, alpha=0.6, s=15, edgecolor="white")
    ax3.set_xlabel(f"PCA1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax3.set_ylabel(f"PCA2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax3.set_title(f"KNN 决策边界 (PCA降维, K={best_k})", fontsize=14, fontweight="bold")
    ax3.legend()
    fig3.tight_layout()
    path3 = os.path.join(CHART_DIR, "11_knn_decision_boundary.png")
    fig3.savefig(path3, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig3)
    print(f"  [图] {path3}")
    print("✅ 阶段4完成\n")

    return {"best_k": best_k, "accuracy": best_score, "classification_report": cr}
