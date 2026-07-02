"""模型对比框架 — 回归/聚类 全模型横向评测"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from config import *


def run(df):
    """对比回归模型（5个）和聚类模型（3个），输出雷达图和对比表"""
    init_env()

    feat_cols = [c for c in df.columns if c.endswith("_scaled")]
    X_cluster = df[feat_cols].values

    # 回归用特征（不含目标变量）
    reg_feats = [c for c in feat_cols if "Overall_Performance_Score" not in c]
    X_reg = df[reg_feats].values
    y_reg = df["Overall_Performance_Score"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X_reg, y_reg, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    # ── 回归模型 ──
    reg_models = {
        "简单线性": LinearRegression(),
        "多元线性": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_SEED),
        "Lasso": Lasso(alpha=0.1, random_state=RANDOM_SEED, max_iter=5000),
        "XGBoost": None,
    }

    reg_results = {}
    for name, model in reg_models.items():
        if name == "XGBoost":
            try:
                from xgboost import XGBRegressor
                model = XGBRegressor(n_estimators=200, max_depth=6,
                                     learning_rate=0.1, random_state=RANDOM_SEED,
                                     verbosity=0)
            except ImportError:
                continue

        if name == "简单线性":
            price_idx = reg_feats.index("Price_USD_scaled")
            X_tr = X_train[:, [price_idx]]
            X_te = X_test[:, [price_idx]]
            X_all = X_reg[:, [price_idx]]
        else:
            X_tr, X_te, X_all = X_train, X_test, X_reg

        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        cv_r2 = cross_val_score(model, X_all, y_reg, cv=5,
                                scoring="r2", n_jobs=-1).mean()
        reg_results[name] = {
            "R²": r2_score(y_test, y_pred),
            "CV_R²": cv_r2,
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
        }

    # ── 聚类模型 ──
    clu_models = {}
    # K-means (K=2)
    km = KMeans(n_clusters=2, random_state=RANDOM_SEED, n_init=10)
    km_labels = km.fit_predict(X_cluster)
    clu_models["K-Means(K=2)"] = {
        "轮廓系数": silhouette_score(X_cluster, km_labels),
        "CH指数": calinski_harabasz_score(X_cluster, km_labels),
        "DB指数": davies_bouldin_score(X_cluster, km_labels),
    }

    # GMM (K=2)
    gmm = GaussianMixture(n_components=2, random_state=RANDOM_SEED, n_init=10)
    gmm_labels = gmm.fit_predict(X_cluster)
    clu_models["GMM(K=2)"] = {
        "轮廓系数": silhouette_score(X_cluster, gmm_labels),
        "CH指数": calinski_harabasz_score(X_cluster, gmm_labels),
        "DB指数": davies_bouldin_score(X_cluster, gmm_labels),
    }

    # GMM (BIC最优)
    bics = [GaussianMixture(n_components=n, random_state=RANDOM_SEED,
                            n_init=5).fit(X_cluster).bic(X_cluster)
            for n in range(1, 11)]
    best_k = int(np.argmin(bics) + 1)
    if best_k < 2:
        best_k = 2
    gmm_best = GaussianMixture(n_components=best_k, random_state=RANDOM_SEED, n_init=10)
    gmm_best_labels = gmm_best.fit_predict(X_cluster)
    clu_models[f"GMM(K={best_k})"] = {
        "轮廓系数": silhouette_score(X_cluster, gmm_best_labels),
        "CH指数": calinski_harabasz_score(X_cluster, gmm_best_labels),
        "DB指数": davies_bouldin_score(X_cluster, gmm_best_labels),
    }

    # ── 打印对比表 ──
    print("=" * 60)
    print("阶段7：全模型对比")
    print("=" * 60)
    print("\n--- 回归模型 ---")
    print(f"{'模型':<10} {'R²':>8} {'CV_R²':>8} {'MAE':>10} {'MSE':>12}")
    print("-" * 48)
    for name, m in reg_results.items():
        print(f"{name:<10} {m['R²']:>8.4f} {m['CV_R²']:>8.4f} "
              f"{m['MAE']:>10.0f} {m['MSE']:>12.0f}")

    print("\n--- 聚类模型 ---")
    print(f"{'模型':<12} {'轮廓系数':>10} {'CH指数':>12} {'DB指数':>10}")
    print("-" * 48)
    for name, m in clu_models.items():
        print(f"{name:<12} {m['轮廓系数']:>10.4f} {m['CH指数']:>12.0f} "
              f"{m['DB指数']:>10.4f}")

    # ── 雷达图 ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                              subplot_kw=dict(projection="polar"))

    # 回归雷达图 — 用 R² / CV_R² / (1 - MAE归一化)
    reg_names = list(reg_results.keys())
    reg_max_mae = max(m["MAE"] for m in reg_results.values()) or 1
    reg_values = {
        name: [
            m["R²"],
            m["CV_R²"],
            1 - m["MAE"] / reg_max_mae,
        ] for name, m in reg_results.items()
    }
    reg_angles = [0, 2 * np.pi / 3, 4 * np.pi / 3]
    reg_labels = ["R^2", "CV_R^2", "1-MAE (norm)"]

    colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["gold"],
              PALETTE["green"], PALETTE["red"]]
    for i, (name, vals) in enumerate(reg_values.items()):
        vals_plot = vals + [vals[0]]
        angles_plot = list(reg_angles) + [reg_angles[0]]
        axes[0].fill(angles_plot, vals_plot, alpha=0.1, color=colors[i % len(colors)])
        axes[0].plot(angles_plot, vals_plot, "o-", linewidth=2,
                     color=colors[i % len(colors)], label=name, markersize=5)
    axes[0].set_xticks(reg_angles)
    axes[0].set_xticklabels(reg_labels, fontsize=11)
    axes[0].set_title("回归模型雷达对比", fontsize=14, fontweight="bold", pad=20)
    axes[0].legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=9)

    # 聚类雷达图
    clu_names = list(clu_models.keys())
    max_ch = max(m["CH指数"] for m in clu_models.values()) or 1
    max_db = max(m["DB指数"] for m in clu_models.values()) or 1
    clu_values = {
        name: [
            m["轮廓系数"],
            m["CH指数"] / max_ch,
            1 - m["DB指数"] / max_db,
        ] for name, m in clu_models.items()
    }
    clu_angles = [0, 2 * np.pi / 3, 4 * np.pi / 3]
    clu_labels = ["轮廓系数", "CH(归一化)", "1-DB(归一化)"]

    for i, (name, vals) in enumerate(clu_values.items()):
        vals_plot = vals + [vals[0]]
        angles_plot = list(clu_angles) + [clu_angles[0]]
        axes[1].fill(angles_plot, vals_plot, alpha=0.1, color=colors[i % len(colors)])
        axes[1].plot(angles_plot, vals_plot, "o-", linewidth=2,
                     color=colors[i % len(colors)], label=name, markersize=5)
    axes[1].set_xticks(clu_angles)
    axes[1].set_xticklabels(clu_labels, fontsize=11)
    axes[1].set_title("聚类模型雷达对比", fontsize=14, fontweight="bold", pad=20)
    axes[1].legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=9)

    fig.tight_layout()
    path = os.path.join(CHART_DIR, "19_model_radar.png")
    fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] {path}")

    # ── 汇总条形图 ──
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    # 回归 R²
    reg_r2 = [reg_results[n]["R²"] for n in reg_names]
    reg_cv = [reg_results[n]["CV_R²"] for n in reg_names]
    x = np.arange(len(reg_names))
    w = 0.35
    bars1 = axes2[0].bar(x - w / 2, reg_r2, w, label="Test R^2",
                         color=PALETTE["blue"], edgecolor="white")
    bars2 = axes2[0].bar(x + w / 2, reg_cv, w, label="CV R^2",
                         color=PALETTE["orange"], edgecolor="white")
    for bar in bars1:
        axes2[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                      f"{bar.get_height():.3f}", ha="center", fontsize=9)
    axes2[0].set_xticks(x)
    axes2[0].set_xticklabels(reg_names, fontsize=9, rotation=15)
    axes2[0].set_ylabel("R^2")
    axes2[0].set_title("Regression R^2 Comparison", fontsize=13, fontweight="bold")
    axes2[0].legend()
    axes2[0].set_ylim(0, 1.1)

    # 聚类 轮廓系数
    clu_sil = [clu_models[n]["轮廓系数"] for n in clu_names]
    bars3 = axes2[1].bar(clu_names, clu_sil,
                         color=[PALETTE["blue"], PALETTE["orange"], PALETTE["green"]],
                         edgecolor="white")
    for bar, val in zip(bars3, clu_sil):
        axes2[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                      f"{val:.4f}", ha="center", fontsize=10)
    axes2[1].set_ylabel("轮廓系数")
    axes2[1].set_title("聚类模型轮廓系数对比", fontsize=13, fontweight="bold")
    axes2[1].set_ylim(0, max(clu_sil) * 1.2)

    fig2.tight_layout()
    path2 = os.path.join(CHART_DIR, "20_model_summary.png")
    fig2.savefig(path2, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig2)
    print(f"  [图] {path2}")

    print("✅ 阶段7完成\n")

    return {
        "regression": reg_results,
        "clustering": clu_models,
    }
