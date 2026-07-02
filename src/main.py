"""主入口：一键运行全部分析流程（系统化版本）"""
import sys
import os

# 确保 src 目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import init_env


def main():
    init_env()

    # ────────── 阶段1：数据生成 ──────────
    print("=" * 60)
    print(">>> 阶段1：生成数据集")
    print("=" * 60)
    from generate_data import generate
    generate()

    # ────────── 阶段2：数据探索 ──────────
    print("\n" + "=" * 60)
    print(">>> 阶段2：数据探索与可视化")
    print("=" * 60)
    from explore import load_data, explore, plot_usage_pie, \
        plot_price_distribution, plot_correlation_heatmap, \
        plot_box_price_by_type, plot_performance_scatter
    df_explore = load_data()
    explore(df_explore)
    plot_usage_pie(df_explore)
    plot_price_distribution(df_explore)
    plot_correlation_heatmap(df_explore)
    plot_box_price_by_type(df_explore)
    plot_performance_scatter(df_explore)

    # ────────── 阶段3：数据预处理 ──────────
    print("\n" + "=" * 60)
    print(">>> 阶段3：数据预处理")
    print("=" * 60)
    from preprocess import full_preprocess
    df_processed = full_preprocess()

    # ────────── 阶段4：线性回归 ──────────
    print("\n" + "=" * 60)
    print(">>> 阶段4：线性回归")
    print("=" * 60)
    from model_linear import run as run_linear
    result_lr = run_linear(df_processed)

    # ────────── 阶段5：KNN分类 ──────────
    print("\n" + "=" * 60)
    print(">>> 阶段5：KNN分类")
    print("=" * 60)
    from model_knn import run as run_knn
    result_knn = run_knn(df_processed)

    # ────────── 阶段6：K-means聚类 ──────────
    print("\n" + "=" * 60)
    print(">>> 阶段6：K-means聚类")
    print("=" * 60)
    from model_kmeans import run as run_kmeans
    result_km = run_kmeans(df_processed)

    # ────────── 阶段4b：进阶回归对比（新增）──────
    print("\n" + "=" * 60)
    print(">>> 阶段4b：进阶回归（多元线性 / Ridge / Lasso / XGBoost）")
    print("=" * 60)
    from model_linear_advanced import run as run_linear_advanced
    result_lr_adv = run_linear_advanced(df_processed)

    # ────────── 阶段6b：GMM 聚类（新增）──────
    print("\n" + "=" * 60)
    print(">>> 阶段6b：GMM 高斯混合模型聚类")
    print("=" * 60)
    from model_gmm import run as run_gmm
    result_gmm = run_gmm(df_processed)

    # ────────── 阶段7：全模型对比（新增）──────
    print("\n" + "=" * 60)
    print(">>> 阶段7：全模型横向对比")
    print("=" * 60)
    from model_compare import run as run_compare
    result_compare = run_compare(df_processed)

    # ────────── 汇总 ──────────
    print("\n" + "=" * 60)
    print(" 全部分析流程完成！")
    print("=" * 60)
    print(f"  原线性回归        R² = {result_lr['r2']:.4f}")
    print(f"  最佳回归模型      R² = {result_lr_adv['best_r2']:.4f}  ({result_lr_adv['best_model']})")
    print(f"  KNN分类           准确率 = {result_knn['accuracy']:.4f}  (K={result_knn['best_k']})")
    print(f"  K-means           K={result_km['k']}, 轮廓系数={result_km['silhouette']:.4f}")
    print(f"  GMM               K={result_gmm['k']}, 轮廓系数={result_gmm['silhouette']:.4f}")
    print(f"  GMM 边界用户       {result_gmm['n_boundary']} 人")
    print(f"\n  图表：output/charts/  (共20张)")
    print(f"  报告：output/reports/analysis_report.md")
    print(f"  数据：data/")

    return {
        "linear_regression": result_lr,
        "linear_advanced": result_lr_adv,
        "knn": result_knn,
        "kmeans": result_km,
        "gmm": result_gmm,
        "model_compare": result_compare,
    }


if __name__ == "__main__":
    main()
