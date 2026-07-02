"""进阶回归分析：多元线性 / 岭回归 / Lasso / XGBoost 横向对比"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import *


def run(df):
    """对 Overall_Performance_Score 做回归，对比 5 个模型"""
    init_env()

    # ── 准备数据：排除目标变量自身，保留所有 scaled 数值特征 ──
    feature_cols = [c for c in df.columns
                    if c.endswith("_scaled") and "Overall_Performance_Score" not in c]
    X = df[feature_cols].values
    y = df["Overall_Performance_Score"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    # ── 定义模型 ──
    models = {
        "简单线性回归\n(仅Price)": LinearRegression(),
        "多元线性回归": LinearRegression(),
        "Ridge (α=1.0)": Ridge(alpha=1.0, random_state=RANDOM_SEED),
        "Lasso (α=0.1)": Lasso(alpha=0.1, random_state=RANDOM_SEED, max_iter=5000),
        "XGBoost": None,  # 单独处理
    }

    # 简单线性回归只用 Price_USD_scaled
    price_col_idx = feature_cols.index("Price_USD_scaled")
    X_train_simple = X_train[:, [price_col_idx]]
    X_test_simple = X_test[:, [price_col_idx]]

    results = {}
    metrics = {"MSE": [], "MAE": [], "R²": [], "CV_R²": []}
    names = []

    for name, model in models.items():
        if name == "XGBoost":
            try:
                from xgboost import XGBRegressor
                model = XGBRegressor(
                    n_estimators=200, max_depth=6, learning_rate=0.1,
                    random_state=RANDOM_SEED, verbosity=0
                )
            except ImportError:
                print(f"  [跳过] XGBoost 未安装，请 pip install xgboost")
                continue

        if name == "简单线性回归\n(仅Price)":
            X_tr, X_te = X_train_simple, X_test_simple
        else:
            X_tr, X_te = X_train, X_test

        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # 交叉验证 R²
        try:
            cv_r2 = cross_val_score(model, X, y, cv=5,
                                    scoring="r2", n_jobs=-1).mean()
        except Exception:
            cv_r2 = r2

        names.append(name.replace("\n", " "))
        metrics["MSE"].append(mse)
        metrics["MAE"].append(mae)
        metrics["R²"].append(r2)
        metrics["CV_R²"].append(cv_r2)

        results[name] = {"mse": mse, "mae": mae, "r2": r2, "cv_r2": cv_r2}

        if hasattr(model, "feature_importances_"):
            results[name]["feature_importances"] = dict(
                zip(feature_cols, model.feature_importances_)
            )

    # ── 打印 ──
    print("=" * 60)
    print("阶段4b：进阶回归 — 多模型对比")
    print("=" * 60)
    print(f"{'模型':<20} {'MSE':>10} {'MAE':>10} {'R²':>8} {'CV_R²':>8}")
    print("-" * 56)
    for i, name in enumerate(names):
        print(f"{name:<20} {metrics['MSE'][i]:>10.0f} {metrics['MAE'][i]:>10.0f} "
              f"{metrics['R²'][i]:>8.4f} {metrics['CV_R²'][i]:>8.4f}")

    # ── 对比条形图 ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # R²
    colors_r2 = ["#5B9BD5", "#5B9BD5", "#ED7D31", "#ED7D31", "#70AD47"]
    bars = axes[0].barh(names, metrics["R²"], color=colors_r2[:len(names)], edgecolor="white")
    for bar, val in zip(bars, metrics["R²"]):
        axes[0].text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                     f"{val:.4f}", va="center", fontsize=10)
    axes[0].set_xlabel("R²")
    axes[0].set_title("R² 对比 (越高越好)", fontsize=13, fontweight="bold")
    axes[0].set_xlim(0, 1.1)

    # MAE
    colors_mae = ["#5B9BD5", "#5B9BD5", "#ED7D31", "#ED7D31", "#70AD47"]
    bars2 = axes[1].barh(names, metrics["MAE"], color=colors_mae[:len(names)], edgecolor="white")
    for bar, val in zip(bars2, metrics["MAE"]):
        axes[1].text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                     f"{val:.0f}", va="center", fontsize=10)
    axes[1].set_xlabel("MAE")
    axes[1].set_title("MAE 对比 (越低越好)", fontsize=13, fontweight="bold")

    # CV_R²
    colors_cv = ["#5B9BD5", "#5B9BD5", "#ED7D31", "#ED7D31", "#70AD47"]
    bars3 = axes[2].barh(names, metrics["CV_R²"], color=colors_cv[:len(names)], edgecolor="white")
    for bar, val in zip(bars3, metrics["CV_R²"]):
        axes[2].text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                     f"{val:.4f}", va="center", fontsize=10)
    axes[2].set_xlabel("CV R^2")
    axes[2].set_title("5-fold CV R^2 (Robustness)", fontsize=13, fontweight="bold")
    axes[2].set_xlim(0, 1.1)

    fig.tight_layout()
    path = os.path.join(CHART_DIR, "14_regression_comparison.png")
    fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] {path}")

    # ── XGBoost 特征重要性 ──
    if "XGBoost" in results and "feature_importances" in results["XGBoost"]:
        fi = results["XGBoost"]["feature_importances"]
        fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)

        fig2, ax = plt.subplots(figsize=(9, 6))
        labels = [f[0].replace("_scaled", "") for f in fi_sorted[:10]]
        values = [f[1] for f in fi_sorted[:10]]
        ax.barh(range(len(labels)), values, color=PALETTE["green"], edgecolor="white")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("重要性")
        ax.set_title("XGBoost 特征重要性 Top 10", fontsize=13, fontweight="bold")
        fig2.tight_layout()
        path2 = os.path.join(CHART_DIR, "15_xgboost_feature_importance.png")
        fig2.savefig(path2, dpi=CHART_DPI, bbox_inches="tight")
        plt.close(fig2)
        print(f"  [图] {path2}")

    print("✅ 阶段4b完成\n")

    best_idx = metrics["R²"].index(max(metrics["R²"]))
    return {
        "models": names,
        "metrics": metrics,
        "results": results,
        "best_model": names[best_idx],
        "best_r2": max(metrics["R²"]),
        "n_features": len(feature_cols),
    }
