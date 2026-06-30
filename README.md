# 电脑消费行为画像 —— 基于硬件性能与价格的用户群体细分

> 实训项目 · 线性回归 + KNN 分类 + K-means 聚类  
> 数据集：模拟全球笔记本市场数据（8,000 条，15 个特征）

---

## 项目结构（系统化）

```
small-study-term1/
├── src/                          # 源码（模块化）
│   ├── config.py                 # 集中配置：路径、种子、参数、调色板
│   ├── main.py                   # 一键入口：串联全部流程
│   ├── generate_data.py          # 生成模拟数据集
│   ├── data_quality.py           # 数据质量审计（重复/缺失/异常值）
│   ├── explore.py                # 数据探索与可视化（5张图表）
│   ├── preprocess.py             # 特征工程 + 标准化
│   ├── model_linear.py           # 线性回归：价格 → 性能分
│   ├── model_knn.py              # KNN分类：硬件 → 使用类型
│   └── model_kmeans.py           # K-means聚类：无监督用户分群
├── data/                         # 数据文件（CSV + 标准化器）
├── output/
│   ├── charts/                   # 全部图表（13张PNG）
│   └── reports/                  # 分析报告
├── requirements.txt              # 依赖锁定
└── README.md
```

## 快速开始

```bash
# 1. 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. 安装依赖
pip install -r requirements.txt

# 3. 一键运行
python src/main.py
```

全部图表生成在 `output/charts/`，分析报告在 `output/reports/`。

## 分析流程

```
数据生成 → 数据探索 → 预处理(特征工程+标准化)
                ↓
    ┌───────────┼───────────┐
    ↓           ↓           ↓
 线性回归     KNN分类    K-means聚类
(价格→性能)  (配置→类型) (无监督分群)
    ↓           ↓           ↓
    └───────────┼───────────┘
                ↓
         策略报告 + 图表
```

## 核心结果

| 模型 | 关键指标 | 说明 |
|---|---|---|
| 线性回归 | R² = 0.60 | 价格能解释 60% 的性能方差 |
| KNN 分类 | 准确率 99.5%, K=6 | 硬件配置几乎完美映射用户类型 |
| K-means | 轮廓系数 0.50, K=2 | 自然发现二八分化：大众79% vs 发烧21% |

## 系统化设计要点

- **配置集中**：`config.py` 统一管理所有路径、随机种子、超参数、调色板
- **模块拆分**：三个模型独立文件，可单独运行和调试
- **可复现**：固定随机种子，锁定依赖版本
- **一键运行**：`python src/main.py` 跑通全流程

## 答辩材料

- 分析报告：`output/reports/analysis_report.md`
- 全部图表：`output/charts/` (01~13 共 13 张)
- 数据质量审计：`python src/data_quality.py`
