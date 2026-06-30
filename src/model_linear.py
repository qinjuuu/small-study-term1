"""阶段3：线性回归 —— 价格预测综合性能分"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import *


def run(df):
    """返回 {"mse","mae","r2","coef","intercept"}"""
    init_env()

    X = df[["Price_USD"]]
    y = df["Overall_Performance_Score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    result = {
        "coef": lr.coef_[0],
        "intercept": lr.intercept_,
        "mse": mean_squared_error(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }

    print("=" * 60)
    print("阶段3：线性回归 — 价格预测综合性能分")
    print("=" * 60)
    for k, v in result.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # ── 回归散点图 ──
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(X_test, y_test, alpha=0.3, s=10, color=PALETTE["blue"], label="实际值")
    x_line = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1, 1)
    ax.plot(x_line, lr.predict(x_line), "r-", linewidth=2, label="回归线")
    ax.set_xlabel("Price (USD)")
    ax.set_ylabel("综合性能分")
    ax.set_title(f"线性回归：价格 -> 综合性能分  (R^2={result['r2']:.3f})",
                 fontsize=14, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(CHART_DIR, "07_linear_regression.png")
    fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] {path}")

    # ── 残差分析 ──
    residuals = y_test - y_pred
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    axes2[0].scatter(y_pred, residuals, alpha=0.3, color=PALETTE["blue"])
    axes2[0].axhline(0, color="red", linestyle="--")
    axes2[0].set_xlabel("预测值")
    axes2[0].set_ylabel("残差")
    axes2[0].set_title("残差图")
    axes2[1].hist(residuals, bins=40, color=PALETTE["orange"], edgecolor="white")
    axes2[1].axvline(0, color="red", linestyle="--")
    axes2[1].set_xlabel("残差")
    axes2[1].set_ylabel("频数")
    axes2[1].set_title("残差分布")
    fig2.tight_layout()
    path2 = os.path.join(CHART_DIR, "08_residual_analysis.png")
    fig2.savefig(path2, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig2)
    print(f"  [图] {path2}")
    print("✅ 阶段3完成\n")

    return result
