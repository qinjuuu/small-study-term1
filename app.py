import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import matplotlib
matplotlib.use("Agg")  # 防止本地开图窗

# ── 路径 ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
CHART_DIR = os.path.join(ROOT, "output", "charts")

SAMPLED_DATA   = os.path.join(DATA_DIR, "real_analysis.csv")
PROCESSED_DATA = os.path.join(DATA_DIR, "processed.csv")
CLUSTERED_DATA = os.path.join(DATA_DIR, "clustered.csv")

EXCHANGE_RATE = 7.25  # USD → CNY (RMB)
USAGE_COLORS = {"Basic": "#5B9BD5", "Student": "#ED7D31",
                "Professional": "#A5A5A5", "Gaming": "#FFC000"}

# ── 页面配置 ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="电脑消费行为画像 · 交互式仪表盘",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 自定义 CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* 全局字体 */
html, body, [class*="st-"] { font-family: 'Segoe UI', sans-serif; }

/* 标题渐变 */
h1 { background: linear-gradient(90deg, #5B9BD5, #ED7D31);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent;
     font-weight: 900 !important; font-size: 2.2rem !important; }

/* 指标卡片美化 */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    border-radius: 12px; padding: 10px 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
[data-testid="metric-container"] label { color: #555 !important; font-size: 0.85rem !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { font-size: 1.8rem !important; }

/* 侧边栏美化 */
section[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, #1a2a4a 0%, #2d4a7a 100%);
    color: white;
}
section[data-testid="stSidebar"] * { color: white !important; }

/* 分隔线 */
hr { border: none; height: 2px;
     background: linear-gradient(90deg, #5B9BD5, #ED7D31, #FFC000); margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ── 加载数据 ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    sampled   = pd.read_csv(SAMPLED_DATA)
    processed = pd.read_csv(PROCESSED_DATA) if os.path.exists(PROCESSED_DATA) else None
    clustered = pd.read_csv(CLUSTERED_DATA) if os.path.exists(CLUSTERED_DATA) else None
    # USD → CNY (人民币)
    if "Price_CNY" in sampled.columns:
        sampled["Price_CNY"] = sampled["Price_CNY"]
    else:
        sampled["Price_CNY"] = (sampled["Price_CNY"] * EXCHANGE_RATE).round(0).astype(int)
    return sampled, processed, clustered

df_raw, df_scaled, df_cluster = load_data()

if df_raw is None:
    st.error("❌ 数据文件未找到，请先运行 `python src/main.py` 生成数据。")
    st.stop()

# ── 全局侧边栏筛选器 ─────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💻 电脑消费行为画像")
    st.markdown("---")
    st.markdown("### 🔍 全局筛选")

    price_min, price_max = int(df_raw["Price_CNY"].min()), int(df_raw["Price_CNY"].max())
    price_range = st.slider("价格区间 (元)", price_min, price_max,
                            (price_min, price_max), step=100)

    usage_filter = st.multiselect("使用类型", df_raw["Usage_Type"].unique().tolist(),
                                  default=df_raw["Usage_Type"].unique().tolist())

    brand_filter = st.multiselect("品牌筛选", df_raw["Brand"].unique().tolist(),
                                  default=df_raw["Brand"].unique().tolist()[:8])

    # 应用筛选
    df = df_raw[
        df_raw["Price_CNY"].between(price_range[0], price_range[1]) &
        df_raw["Usage_Type"].isin(usage_filter) &
        df_raw["Brand"].isin(brand_filter)
    ].copy()

    st.markdown("---")
    st.metric("📦 筛选后样本数", f"{len(df)} / {len(df_raw)}")
    st.markdown(f"**特征数**: {len(df.columns)}")

    st.markdown("---")
    page = st.selectbox("📄 选择页面", [
        "🏠 欢迎页",
        "📊 数据总览",
        "🔍 特征探索",
        "🌐 3D 可视化",
        "📉 回归分析",
        "🎯 KNN 分类",
        "🎪 K-Means 聚类",
        "✨ GMM 软聚类",
        "🌀 聚类动画",
        "📐 模型对比",
        "🚨 异常检测",
        "🏷️ 品牌分析",
        "🎮 预测模拟器",
        "🎁 推荐引擎",
        "📥 数据导出",
    ])

# ── 辅助：找性能分列名 ────────────────────────────────────────────────────────
def get_perf_col(df_):
    candidates = [c for c in df_.columns if "Performance" in c and "Overall" in c]
    if candidates:
        return candidates[0]
    candidates2 = [c for c in df_.columns if "Performance" in c and "Score" in c]
    if candidates2:
        return candidates2[0]
    return None

PERF_COL = get_perf_col(df)

# ══════════════════════════════════════════════════════════════════════════════
# 页面 0: 欢迎页
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 欢迎页":
    st.title("💻 电脑消费行为画像")
    st.markdown("### 基于硬件性能与价格的用户群体细分 · 交互式分析平台")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📦 样本总量", len(df_raw))
    with col2:
        st.metric("🔢 特征数", len(df_raw.columns))
    with col3:
        st.metric("🏷️ 品牌数", df_raw["Brand"].nunique())
    with col4:
        st.metric("💰 均价", f"¥{df_raw['Price_CNY'].mean():.0f}")

    st.markdown("---")

    # 概览散点图（全量）
    if PERF_COL:
        fig_welcome = px.scatter(
            df_raw.sample(min(3000, len(df_raw)), random_state=42),
            x="Price_CNY", y=PERF_COL,
            color="Usage_Type",
            size="RAM_GB" if "RAM_GB" in df_raw.columns else None,
            color_discrete_map=USAGE_COLORS,
            opacity=0.6,
            hover_data=["Brand", "CPU_Cores", "GPU_Performance_Score"],
            title="全局预览：价格 vs 性能（按使用类型着色，气泡大小=内存）",
        )
        fig_welcome.update_layout(height=500,
                                   xaxis_title="价格 (元)",
                                   yaxis_title=PERF_COL,
                                   legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig_welcome, use_container_width=True)

    st.info("👈 请从左侧侧边栏选择分析页面，所有页面均支持全局筛选。")

# ══════════════════════════════════════════════════════════════════════════════
# 页面 1: 数据总览
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 数据总览":
    st.title("📊 数据总览（筛选后 {} 条）".format(len(df)))

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("样本数", len(df))
    with col2: st.metric("使用类型", df["Usage_Type"].nunique())
    with col3: st.metric("平均价格", f"¥{df['Price_CNY'].mean():.0f}")
    with col4: st.metric("平均性能分", f"{df[PERF_COL].mean():.1f}" if PERF_COL else "N/A")

    st.markdown("---")

    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("🍩 使用类型分布")
        type_counts = df["Usage_Type"].value_counts()
        fig_pie = go.Figure(data=[go.Pie(
            labels=type_counts.index,
            values=type_counts.values,
            hole=0.45,
            marker_colors=[USAGE_COLORS[t] for t in type_counts.index],
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>数量: %{value}<br>占比: %{percent}<extra></extra>",
            pull=[0.05] * len(type_counts),
        )])
        fig_pie.update_layout(height=420, showlegend=False,
                              margin=dict(t=20, b=20, l=20, r=20),
                              annotations=[dict(text="使用类型", x=0.5, y=0.5,
                                                font_size=14, showarrow=False)])
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.subheader("📈 价格分布（可交互）")
        fig_price = go.Figure()
        for ut in df["Usage_Type"].unique():
            subset = df[df["Usage_Type"] == ut]
            fig_price.add_trace(go.Violin(
                x=subset["Usage_Type"],
                y=subset["Price_CNY"],
                name=ut,
                line_color=USAGE_COLORS.get(ut, "#333"),
                fillcolor=USAGE_COLORS.get(ut, "#333"),
                opacity=0.7,
                box_visible=True,
                meanline_visible=True,
                hovertemplate=f"<b>{ut}</b><br>价格: %{{y}}$<extra></extra>",
            ))
        fig_price.update_layout(height=420, yaxis_title="价格 (元)",
                                margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_price, use_container_width=True)

    st.subheader("📦 各类型价格箱线图（可交互）")
    fig_box = go.Figure()
    for ut in df["Usage_Type"].unique():
        subset = df[df["Usage_Type"] == ut]
        fig_box.add_trace(go.Box(
            x=subset["Price_CNY"],
            name=ut,
            marker_color=USAGE_COLORS.get(ut, "#999"),
            boxmean="sd",
            notched=True,
            hovertemplate=f"<b>{ut}</b><br>价格: %{{x}}$<br>Q1: %{{lower}}$<br>中位数: %{{median}}$<br>Q3: %{{upper}}$<extra></extra>",
        ))
    fig_box.update_layout(height=450, xaxis_title="价格 (元)",
                          margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_box, use_container_width=True)

    with st.expander("📄 数据预览（前50行）"):
        st.dataframe(df.head(50), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# 页面 2: 特征探索
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 特征探索":
    st.title("🔍 特征探索")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if PERF_COL and PERF_COL not in num_cols:
        num_cols.append(PERF_COL)

    st.subheader("🔗 相关性热力图（可缩放）")
    corr = df[num_cols].corr()

    # 使用 Plotly 热力图，支持缩放和悬停
    fig_heat = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale="RdBu_r",
        zmid=0,
        zmin=-1,
        zmax=1,
        text=corr.round(2).values,
        texttemplate="%{text}",
        textfont_size=11,
        hovertemplate="%{y} vs %{x}<br>相关系数: %{z:.3f}<extra></extra>",
    ))
    fig_heat.update_layout(
        height=650,
        xaxis_tickangle=-60,
        yaxis_autorange="reversed",
        margin=dict(t=20, b=20, l=20, r=20),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

    # 平行坐标图
    st.subheader("📐 平行坐标图（按使用类型）")
    st.caption("每条线代表一台电脑，值在0~1之间归一化，可看清各类型在不同维度上的差异")

    from sklearn.preprocessing import MinMaxScaler
    pc_features = ["CPU_Cores", "RAM_GB", "Storage_GB",
                   "GPU_Performance_Score", "Price_CNY"]
    pc_features = [c for c in pc_features if c in df.columns]
    if len(pc_features) >= 3:
        df_pc = df[pc_features + ["Usage_Type"]].dropna()
        scaler_pc = MinMaxScaler()
        df_pc_scaled = df_pc.copy()
        df_pc_scaled[pc_features] = scaler_pc.fit_transform(df_pc[pc_features])

        # parallel_coordinates 不支持分类color，用数值列着色
        color_col = PERF_COL if PERF_COL and PERF_COL in df_pc_scaled.columns else "Price_CNY"
        fig_pc = px.parallel_coordinates(
            df_pc_scaled,
            dimensions=pc_features,
            color=color_col,
            labels={c: c for c in pc_features},
        )
        fig_pc.update_layout(height=520, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_pc, use_container_width=True)

    st.markdown("---")

    # 散点图矩阵
    st.subheader("🧩 散点图矩阵（SPLOM）")
    splom_features = st.multiselect(
        "选择4个特征（建议选最重要的）",
        num_cols,
        default=[c for c in ["Price_CNY", "CPU_Cores", "GPU_Performance_Score",
                              "RAM_GB"] if c in df.columns][:4],
        key="splom_feat",
    )
    if len(splom_features) >= 2:
        df_splom = df.sample(min(2000, len(df)), random_state=42)
        fig_splom = px.scatter_matrix(
            df_splom,
            dimensions=splom_features,
            color="Usage_Type",
            color_discrete_map=USAGE_COLORS,
            opacity=0.6,
        )
        fig_splom.update_layout(height=700,
                                margin=dict(t=20, b=20, l=20, r=20))
        # 缩小散点
        fig_splom.update_traces(marker=dict(size=5, opacity=0.5))
        st.plotly_chart(fig_splom, use_container_width=True)

    st.markdown("---")

    # 单特征分布
    st.subheader("📊 单特征分布（可交互）")
    col_sel, col_chart = st.columns([1, 2])
    with col_sel:
        selected_feat = st.selectbox("选择特征", num_cols)
        bin_size = st.slider("分箱数", 10, 100, 40)
        show_type = st.checkbox("按使用类型分层", value=True)

    with col_chart:
        if show_type:
            fig_hist = px.histogram(
                df, x=selected_feat, nbins=bin_size, color="Usage_Type",
                color_discrete_map=USAGE_COLORS,
                barmode="overlay", opacity=0.7,
                marginal="box",  # 上方显示箱线图
            )
        else:
            fig_hist = px.histogram(df, x=selected_feat, nbins=bin_size,
                                     marginal="box")
        fig_hist.update_layout(height=480,
                               margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_hist, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# 页面 3: 3D 可视化
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌐 3D 可视化":
    st.title("🌐 3D 可视化")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    st.subheader("🎲 3D 散点图")
    col1, col2, col3 = st.columns(3)
    with col1:
        x3d = st.selectbox("X 轴", num_cols, index=num_cols.index("Price_CNY") if "Price_CNY" in num_cols else 0, key="x3d")
    with col2:
        y3d = st.selectbox("Y 轴", num_cols, index=num_cols.index("CPU_Cores") if "CPU_Cores" in num_cols else 1, key="y3d")
    with col3:
        z3d = st.selectbox("Z 轴", num_cols, index=num_cols.index(PERF_COL) if PERF_COL and PERF_COL in num_cols else min(2, len(num_cols)-1), key="z3d")

    color_3d = st.selectbox("着色", ["Usage_Type", "Brand", "Cluster"] if "Cluster" in df.columns else ["Usage_Type", "Brand"])

    df_3d = df.sample(min(3000, len(df)), random_state=42)

    fig_3d = px.scatter_3d(
        df_3d, x=x3d, y=y3d, z=z3d,
        color=color_3d,
        color_discrete_map=USAGE_COLORS if color_3d == "Usage_Type" else None,
        size="RAM_GB" if "RAM_GB" in df.columns else None,
        opacity=0.7,
        hover_data=["Brand", "Price_CNY"],
    )
    fig_3d.update_layout(
        height=700,
        scene=dict(
            xaxis_title=x3d,
            yaxis_title=y3d,
            zaxis_title=z3d,
        ),
        margin=dict(t=20, b=20, l=20, r=20),
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    st.markdown("---")

    # PCA 3D
    st.subheader("🧬 PCA 3D 投影（无监督降维）")
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    pca_features = [c for c in num_cols if c != PERF_COL]
    if len(pca_features) >= 3:
        X_pca = df[pca_features].fillna(0).values
        scaler_pca = StandardScaler()
        X_pca_scaled = scaler_pca.fit_transform(X_pca)
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(X_pca_scaled)
        df_pca = df.copy()
        df_pca["PCA1"] = pca_result[:, 0]
        df_pca["PCA2"] = pca_result[:, 1]
        df_pca["PCA3"] = pca_result[:, 2]

        explained = pca.explained_variance_ratio_ * 100
        st.caption(f"PCA 解释方差：PC1={explained[0]:.1f}%，PC2={explained[1]:.1f}%，PC3={explained[2]:.1f}%，累计={explained.sum():.1f}%")

        fig_pca3d = px.scatter_3d(
            df_pca.sample(min(3000, len(df_pca)), random_state=42),
            x="PCA1", y="PCA2", z="PCA3",
            color="Usage_Type",
            color_discrete_map=USAGE_COLORS,
            opacity=0.7,
            title="PCA 3D 投影（按使用类型着色）",
        )
        fig_pca3d.update_layout(height=700, margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_pca3d, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# 页面 4: 回归分析
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📉 回归分析":
    st.title("📉 回归分析")

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    if not PERF_COL:
        st.warning("未找到性能分列，无法进行回归分析。")
        st.stop()

    # 多元回归特征选择
    feature_cols = [c for c in df.columns
                    if df[c].dtype in [np.float64, np.int64]
                    and c not in [PERF_COL]]

    st.subheader("🎛️ 多元线性回归（交互式特征选择）")
    st.caption("勾选特征，实时查看 R² 变化")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        use_multivariate = st.checkbox("启用多元回归", value=True)
        if use_multivariate:
            default_feats = [c for c in ["GPU_Performance_Score", "CPU_Cores",
                                         "RAM_GB", "Price_CNY"]
                             if c in df.columns]
            selected_features = st.multiselect(
                "选择回归特征", feature_cols,
                default=default_feats[:3],
                key="lr_feats",
            )
        else:
            selected_features = ["Price_CNY"]

    # 训练
    if len(selected_features) == 0:
        st.warning("请至少选择一个特征。")
        st.stop()

    X = df[selected_features].fillna(0)
    y = df[PERF_COL]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    lr = LinearRegression()
    lr.fit(X_tr, y_tr)
    y_pred = lr.predict(X_te)
    r2 = r2_score(y_te, y_pred)
    mse = mean_squared_error(y_te, y_pred)
    mae = mean_absolute_error(y_te, y_pred)

    with col_right:
        col_a, col_b, col_c = st.columns(3)
        with col_a: st.metric("R²", f"{r2:.4f}")
        with col_b: st.metric("MSE", f"{mse:.2f}")
        with col_c: st.metric("MAE", f"{mae:.2f}")

        # 回归结果图
        fig_reg = go.Figure()
        fig_reg.add_trace(go.Scatter(
            x=y_pred, y=y_te,
            mode="markers",
            marker=dict(color="#5B9BD5", size=8, opacity=0.4),
            name="预测 vs 实际",
            hovertemplate="预测: %{x:.1f}<br>实际: %{y:.1f}<extra></extra>",
        ))
        # 对角线
        v_min, v_max = min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())
        fig_reg.add_trace(go.Scatter(
            x=[v_min, v_max], y=[v_min, v_max],
            mode="lines",
            line=dict(color="red", width=2, dash="dash"),
            name="完美预测线",
        ))
        fig_reg.update_layout(
            height=420, xaxis_title="预测值", yaxis_title="实际值",
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_reg, use_container_width=True)

    # 系数解释
    if use_multivariate and len(selected_features) > 1:
        st.subheader("📋 回归系数解释")
        coef_df = pd.DataFrame({
            "特征": selected_features,
            "系数": lr.coef_,
            "绝对值": np.abs(lr.coef_),
        }).sort_values("绝对值", ascending=False)
        fig_coef = go.Figure([
            go.Bar(
                x=coef_df["特征"],
                y=coef_df["系数"],
                marker_color=["#5B9BD5" if c > 0 else "#ED7D31" for c in coef_df["系数"]],
                hovertemplate="%{x}<br>系数: %{y:.4f}<br>含义: 该特征每增加1单位，性能分%{y:+.2f}<extra></extra>",
            )
        ])
        fig_coef.update_layout(height=400, yaxis_title="系数值",
                               margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_coef, use_container_width=True)
        st.caption(f"截距 = {lr.intercept_:.2f}")

    # 残差分析
    st.subheader("📉 残差分析")
    residuals = y_te - y_pred
    fig_res = make_subplots(rows=1, cols=2,
                             subplot_titles=("残差 vs 预测值", "残差分布"))
    fig_res.add_trace(go.Scatter(
        x=y_pred, y=residuals,
        mode="markers",
        marker=dict(color="#5B9BD5", size=8, opacity=0.4),
        name="残差",
        hovertemplate="预测值: %{x:.1f}<br>残差: %{y:.1f}<extra></extra>",
    ), row=1, col=1)
    fig_res.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
    fig_res.add_trace(go.Histogram(
        x=residuals, nbinsx=50,
        marker_color="#ED7D31", opacity=0.7,
        name="残差分布",
        hovertemplate="残差: %{x:.1f}<br>频数: %{y}<extra></extra>",
    ), row=1, col=2)
    fig_res.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=2)
    fig_res.update_layout(height=420, showlegend=False,
                          margin=dict(t=40, b=20, l=20, r=20))
    st.plotly_chart(fig_res, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# 页面 5: KNN 分类
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 KNN 分类":
    st.title("🎯 KNN 分类分析")

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.preprocessing import StandardScaler

    feature_cols = [c for c in df.columns
                    if df[c].dtype in [np.float64, np.int64]
                    and c not in ([PERF_COL] if PERF_COL else [])]

    st.subheader("🎛️ 特征与参数选择")
    col1, col2 = st.columns([1, 1])
    with col1:
        selected_knn = st.multiselect(
            "选择KNN特征", feature_cols,
            default=[c for c in ["CPU_Cores", "CPU_Frequency_GHz", "RAM_GB",
                                  "GPU_Performance_Score", "Price_CNY"]
                     if c in df.columns][:4],
            key="knn_feat",
        )
    with col2:
        max_k = st.slider("最大 K 值", 3, 50, 20)
        cv_folds = st.slider("交叉验证折数", 2, 10, 5)

    if len(selected_knn) == 0:
        st.warning("请选择至少一个特征。")
        st.stop()

    X = df[selected_knn].values
    y = df["Usage_Type"]
    scaler_knn = StandardScaler()
    X_scaled_knn = scaler_knn.fit_transform(X)

    with st.spinner(f"正在对 K=1~{max_k} 做 {cv_folds}-fold 交叉验证..."):
        k_range = list(range(1, max_k + 1))
        cv_scores = []
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, X_scaled_knn, y, cv=cv_folds, scoring="accuracy")
            cv_scores.append(scores.mean())

    best_k = k_range[np.argmax(cv_scores)]
    best_acc = max(cv_scores)

    col_a, col_b, col_c = st.columns(3)
    with col_a: st.metric("🏆 最佳 K", best_k)
    with col_b: st.metric("🎯 最佳准确率", f"{best_acc:.4f}")
    with col_c: st.metric("📊 特征数", len(selected_knn))

    fig_knn = go.Figure()
    fig_knn.add_trace(go.Scatter(
        x=k_range, y=cv_scores,
        mode="lines+markers",
        marker=dict(color="#5B9BD5", size=7),
        line=dict(width=2.5),
        name="交叉验证准确率",
        hovertemplate="K=%{x}<br>准确率=%{y:.4f}<extra></extra>",
    ))
    fig_knn.add_vline(x=best_k, line_dash="dash", line_color="#FF6384",
                      annotation_text=f"🏆 K={best_k}  acc={best_acc:.4f}")
    fig_knn.update_layout(height=450, xaxis_title="K 值",
                          yaxis_title="交叉验证准确率",
                          margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_knn, use_container_width=True)

    # 混淆矩阵
    st.subheader("🧩 混淆矩阵（最佳KNN）")
    knn_best = KNeighborsClassifier(n_neighbors=best_k)
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled_knn, y, test_size=0.2, random_state=42)
    knn_best.fit(X_tr, y_tr)
    y_pred_knn = knn_best.predict(X_te)
    cm = confusion_matrix(y_te, y_pred_knn)

    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=np.unique(y),
        y=np.unique(y),
        colorscale="Blues",
        text=cm,
        texttemplate="%{text}",
        textfont_size=14,
        hovertemplate="真实: %{y}<br>预测: %{x}<br>数量: %{z}<extra></extra>",
    ))
    fig_cm.update_layout(height=480, xaxis_title="预测标签",
                         yaxis_title="真实标签",
                         margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_cm, use_container_width=True)

    # 分类报告
    with st.expander("📄 详细分类报告"):
        report = classification_report(y_te, y_pred_knn, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose().round(3))

# ══════════════════════════════════════════════════════════════════════════════
# 页面 6: K-Means 聚类
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎪 K-Means 聚类":
    st.title("🎪 K-Means 聚类分析")

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cluster_features = [c for c in num_cols if c != PERF_COL]

    st.subheader("🎛️ 聚类特征选择")
    col1, col2 = st.columns([1, 1])
    with col1:
        selected_cf = st.multiselect(
            "选择聚类特征", cluster_features,
            default=[c for c in ["CPU_Cores", "RAM_GB",
                                  "GPU_Performance_Score", "Price_CNY"]
                     if c in df.columns][:4],
            key="km_feat",
        )
    with col2:
        max_k_cluster = st.slider("最大 K", 3, 15, 10, key="km_maxk")

    if len(selected_cf) < 2:
        st.warning("请选择至少两个特征进行聚类。")
        st.stop()

    Xc = df[selected_cf].fillna(0).values
    scaler_c = StandardScaler()
    Xc_scaled = scaler_c.fit_transform(Xc)

    with st.spinner("正在计算不同 K 的聚类和轮廓系数..."):
        k_range_c = list(range(2, max_k_cluster + 1))
        inertias = []
        silhouettes = []
        for k in k_range_c:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(Xc_scaled)
            inertias.append(km.inertia_)
            silhouettes.append(silhouette_score(Xc_scaled, labels))

    fig_elbow = make_subplots(rows=1, cols=2,
                               subplot_titles=("📉 肘部法则（Inertia）", "📐 轮廓系数"))
    fig_elbow.add_trace(go.Scatter(
        x=k_range_c, y=inertias,
        mode="lines+markers",
        marker=dict(color="#5B9BD5", size=8),
        line=dict(width=2.5),
        name="Inertia",
        hovertemplate="K=%{x}<br>Inertia=%{y:.0f}<extra></extra>",
    ), row=1, col=1)
    fig_elbow.add_trace(go.Scatter(
        x=k_range_c, y=silhouettes,
        mode="lines+markers",
        marker=dict(color="#ED7D31", size=8),
        line=dict(width=2.5),
        name="Silhouette",
        hovertemplate="K=%{x}<br>轮廓系数=%{y:.4f}<extra></extra>",
    ), row=1, col=2)
    fig_elbow.add_vline(x=k_range_c[np.argmax(silhouettes)], line_dash="dash",
                        line_color="#FF6384", row=1, col=2)
    fig_elbow.update_layout(height=450, showlegend=False,
                             margin=dict(t=40, b=20, l=20, r=20))
    st.plotly_chart(fig_elbow, use_container_width=True)

    best_k = k_range_c[np.argmax(silhouettes)]
    st.info(f"👉 轮廓系数最大的 K = **{best_k}**（轮廓系数={silhouettes[np.argmax(silhouettes)]:.4f}）")

    # 聚类可视化
    km_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = km_best.fit_predict(Xc_scaled)
    df_cluster_view = df.copy()
    df_cluster_view["Cluster"] = [f"簇{i}" for i in cluster_labels]

    st.subheader("🗺️ 聚类散点图（可交互选轴）")
    col_x, col_y = st.columns(2)
    with col_x:
        x_axis = st.selectbox("X 轴", selected_cf, index=0, key="km_x")
    with col_y:
        y_axis = st.selectbox("Y 轴", selected_cf,
                               index=min(1, len(selected_cf)-1), key="km_y")

    fig_cluster = px.scatter(
        df_cluster_view, x=x_axis, y=y_axis,
        color="Cluster",
        symbol="Usage_Type" if "Usage_Type" in df.columns else None,
        color_discrete_sequence=px.colors.qualitative.Set2,
        hover_data=selected_cf + (["Usage_Type"] if "Usage_Type" in df.columns else []),
        title=f"K-Means 聚类结果（K={best_k}）",
    )
    # 画聚类中心
    centers = scaler_c.inverse_transform(km_best.cluster_centers_)
    for i, center in enumerate(centers):
        fig_cluster.add_trace(go.Scatter(
            x=[center[selected_cf.index(x_axis)]],
            y=[center[selected_cf.index(y_axis)]],
            mode="markers",
            marker=dict(symbol="x", size=15, color="black", line=dict(width=2, color="white")),
            name=f"簇{i}中心",
            hovertemplate=f"簇{i}中心<extra></extra>",
        ))
    fig_cluster.update_layout(height=550,
                              margin=dict(t=40, b=20, l=20, r=20))
    st.plotly_chart(fig_cluster, use_container_width=True)

    st.subheader("📊 各簇统计摘要")
    summary = df_cluster_view.groupby("Cluster")[selected_cf].agg(["mean", "std"]).round(2)
    st.dataframe(summary, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# 页面 7: GMM 软聚类
# ══════════════════════════════════════════════════════════════════════════════
elif page == "✨ GMM 软聚类":
    st.title("✨ GMM 高斯混合模型（软聚类）")

    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cluster_features = [c for c in num_cols if c != PERF_COL]

    selected_gmm = st.multiselect(
        "选择GMM特征", cluster_features,
        default=[c for c in ["CPU_Cores", "RAM_GB",
                              "GPU_Performance_Score", "Price_CNY"]
                 if c in df.columns][:3],
        key="gmm_feat",
    )

    if len(selected_gmm) < 2:
        st.warning("请选择至少两个特征。")
        st.stop()

    Xg = df[selected_gmm].fillna(0).values
    scaler_g = StandardScaler()
    Xg_scaled = scaler_g.fit_transform(Xg)

    with st.spinner("正在计算 BIC/AIC..."):
        k_range_g = list(range(1, 11))
        bics = []
        aics = []
        for k in k_range_g:
            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm.fit(Xg_scaled)
            bics.append(gmm.bic(Xg_scaled))
            aics.append(gmm.aic(Xg_scaled))

    fig_bic = make_subplots(rows=1, cols=2, subplot_titles=("📉 BIC（越小越好）", "📉 AIC（越小越好）"))
    fig_bic.add_trace(go.Scatter(x=k_range_g, y=bics, mode="lines+markers",
                                  name="BIC", marker_color="#5B9BD5",
                                  line=dict(width=2.5)), row=1, col=1)
    fig_bic.add_trace(go.Scatter(x=k_range_g, y=aics, mode="lines+markers",
                                  name="AIC", marker_color="#ED7D31",
                                  line=dict(width=2.5)), row=1, col=2)
    best_k_gmm = k_range_g[np.argmin(bics)]
    fig_bic.add_vline(x=best_k_gmm, line_dash="dash", line_color="#FF6384", row=1, col=1)
    fig_bic.add_vline(x=best_k_gmm, line_dash="dash", line_color="#FF6384", row=1, col=2)
    fig_bic.update_layout(height=450, showlegend=False,
                          margin=dict(t=40, b=20, l=20, r=20))
    st.plotly_chart(fig_bic, use_container_width=True)

    st.info(f"👉 BIC 最小的 K = **{best_k_gmm}**")

    # GMM 结果
    gmm_best = GaussianMixture(n_components=best_k_gmm, random_state=42)
    gmm_labels = gmm_best.fit_predict(Xg_scaled)
    gmm_proba = gmm_best.predict_proba(Xg_scaled)

    df_gmm = df.copy()
    df_gmm["GMM_Cluster"] = [f"簇{i}" for i in gmm_labels]
    df_gmm["GMM_Confidence"] = gmm_proba.max(axis=1)

    st.subheader("🗺️ 软聚类结果（气泡大小 = 置信度）")
    col_x, col_y = st.columns(2)
    with col_x:
        x_axis_g = st.selectbox("X 轴", selected_gmm, index=0, key="gmm_x")
    with col_y:
        y_axis_g = st.selectbox("Y 轴", selected_gmm,
                                index=min(1, len(selected_gmm)-1), key="gmm_y")

    fig_gmm = px.scatter(
        df_gmm, x=x_axis_g, y=y_axis_g,
        color="GMM_Cluster",
        size="GMM_Confidence",
        size_max=20,
        color_discrete_sequence=px.colors.qualitative.Set2,
        hover_data=selected_gmm + ["GMM_Confidence"],
        title="气泡大小 = 聚类置信度（越大 = 模型越确定）",
    )
    fig_gmm.update_layout(height=550,
                          margin=dict(t=40, b=20, l=20, r=20))
    st.plotly_chart(fig_gmm, use_container_width=True)

    # 边界用户
    st.subheader("🚧 边界用户分析（置信度低的样本 = 摇摆用户）")
    threshold = st.slider("置信度阈值", 0.0, 1.0, 0.6, 0.05, key="gmm_thresh")
    boundary_users = df_gmm[df_gmm["GMM_Confidence"] < threshold]
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("边界用户数", f"{len(boundary_users)} / {len(df_gmm)}")
    with col_b:
        st.metric("占比", f"{len(boundary_users)/len(df_gmm)*100:.1f}%")

    if len(boundary_users) > 0:
        st.dataframe(
            boundary_users[selected_gmm + ["GMM_Cluster", "GMM_Confidence"]].head(20),
            use_container_width=True,
        )

    # 各簇概率分布
    st.subheader("📊 各簇的概率分布（软聚类核心优势）")
    prob_df = pd.DataFrame(gmm_proba, columns=[f"簇{i}_概率" for i in range(best_k_gmm)])
    prob_df["主要归属"] = df_gmm["GMM_Cluster"]
    fig_prob = go.Figure()
    for i in range(best_k_gmm):
        fig_prob.add_trace(go.Box(
            y=prob_df[f"簇{i}_概率"],
            name=f"簇{i}",
            boxmean="sd",
        ))
    fig_prob.update_layout(height=420, yaxis_title="归属概率",
                           margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_prob, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# 页面 8: 聚类动画
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌀 聚类动画":
    st.title("🌀 K-Means 迭代过程动画")
    st.caption("观看 K-Means 如何一步步收敛——质心移动 + 样本归属变化")

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cluster_features = [c for c in num_cols if c != PERF_COL]

    selected_ani = st.multiselect(
        "选择两个特征（动画只展示2D）",
        cluster_features,
        default=[cluster_features[0], cluster_features[1]] if len(cluster_features) >= 2 else cluster_features,
        key="ani_feat",
    )

    if len(selected_ani) != 2:
        st.warning("请选择恰好两个特征。")
        st.stop()

    k_ani = st.slider("K 值", 2, 6, 3, key="ani_k")
    max_iter_ani = st.slider("最大迭代次数", 5, 50, 20, key="ani_iter")

    if st.button("🎬 生成动画", type="primary"):
        Xa = df[selected_ani].fillna(0).values
        scaler_a = StandardScaler()
        Xa_scaled = scaler_a.fit_transform(Xa)

        # 手动迭代 K-Means 记录每一步
        km_ani = KMeans(n_clusters=k_ani, init="random",
                        max_iter=max_iter_ani, n_init=1, random_state=42)
        frames = []

        # 初始化质心
        centers = Xa_scaled[np.random.choice(len(Xa_scaled), k_ani, replace=False)]
        for iteration in range(max_iter_ani):
            # E-step: 分配样本到最近质心
            distances = np.linalg.norm(Xa_scaled[:, np.newaxis] - centers, axis=2)
            labels = np.argmin(distances, axis=1)
            frames.append({
                "iteration": iteration,
                "centers": centers.copy(),
                "labels": labels.copy(),
            })
            # M-step: 更新质心
            new_centers = np.array([
                Xa_scaled[labels == i].mean(axis=0) if np.sum(labels == i) > 0 else centers[i]
                for i in range(k_ani)
            ])
            if np.allclose(new_centers, centers):
                break
            centers = new_centers

        # 用 Plotly 做出动画散点图
        ani_data = []
        for frame in frames:
            inv_centers = scaler_a.inverse_transform(frame["centers"])
            inv_points = scaler_a.inverse_transform(Xa_scaled)
            for i in range(k_ani):
                # 质心
                ani_data.append(dict(
                    iteration=frame["iteration"],
                    type="center",
                    cluster=i,
                    x=inv_centers[i][0],
                    y=inv_centers[i][1],
                ))
            for pt_idx, (x_val, y_val, label) in enumerate(zip(inv_points[:, 0], inv_points[:, 1], frame["labels"])):
                ani_data.append(dict(
                    iteration=frame["iteration"],
                    type="point",
                    cluster=label,
                    x=x_val,
                    y=y_val,
                    idx=pt_idx,
                ))

        df_ani = pd.DataFrame(ani_data)

        # 质心轨迹
        centers_df = df_ani[df_ani["type"] == "center"]
        fig_ani = px.scatter(
            df_ani[df_ani["type"] == "point"],
            x="x", y="y",
            color="cluster",
            animation_frame="iteration",
            animation_group="idx",
            color_discrete_sequence=px.colors.qualitative.Set2,
            opacity=0.5,
            title="K-Means 迭代动画（播放按钮在右上角 ▶）",
        )
        # 叠加质心
        for iter_val in df_ani["iteration"].unique():
            c_df = centers_df[centers_df["iteration"] == iter_val]
            fig_ani.add_trace(go.Scatter(
                x=c_df["x"], y=c_df["y"],
                mode="markers",
                marker=dict(symbol="x", size=14, color="black",
                             line=dict(width=2, color="white")),
                name="质心" if iter_val == 0 else None,
                showlegend=bool(iter_val == 0),
                legendgroup="centers",
            ))

        fig_ani.update_layout(
            height=600,
            xaxis_title=selected_ani[0],
            yaxis_title=selected_ani[1],
            margin=dict(t=40, b=20, l=20, r=20),
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig_ani, use_container_width=True)
        st.success(f"✅ 动画生成完成！共 {len(frames)} 次迭代。点击右上角播放按钮观看。")

# ══════════════════════════════════════════════════════════════════════════════
# 页面 9: 模型对比
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📐 模型对比":
    st.title("📐 模型对比")

    st.info("💡 各模型关键指标对比，可从已有输出图表中查看，也可在此页面做交互式对比。")

    # 尝试读取已有静态图
    radar_path = os.path.join(CHART_DIR, "19_model_radar.png")
    summary_path = os.path.join(CHART_DIR, "20_model_summary.png")
    compare_path = os.path.join(ROOT, "output", "reports", "model_compare.csv")

    if os.path.exists(radar_path):
        st.subheader("📊 模型雷达图（静态）")
        st.image(radar_path)

    if os.path.exists(summary_path):
        st.subheader("📋 模型汇总（静态）")
        st.image(summary_path)

    if os.path.exists(compare_path):
        st.subheader("📄 模型对比数据")
        df_compare = pd.read_csv(compare_path)
        st.dataframe(df_compare, use_container_width=True)

    # 如果没有数据，做一个简单的交互式对比
    if not os.path.exists(compare_path):
        st.subheader("🧪 快速模型对比（线性回归 vs Ridge vs Lasso）")
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler

        if PERF_COL:
            Xc = df[["Price_CNY", "GPU_Performance_Score", "CPU_Cores"]].fillna(0)
            yc = df[PERF_COL]

            models = {
                "LinearRegression": LinearRegression(),
                "Ridge": Ridge(alpha=1.0),
                "Lasso": Lasso(alpha=0.1, max_iter=5000),
            }
            results = []
            for name, model in models.items():
                scores = cross_val_score(model, Xc, yc, cv=5, scoring="r2")
                results.append(dict(
                    模型=name,
                    R2均值=f"{scores.mean():.4f}",
                    R2标准差=f"{scores.std():.4f}",
                    最高=f"{scores.max():.4f}",
                    最低=f"{scores.min():.4f}",
                ))
            st.dataframe(pd.DataFrame(results), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# 页面 10: 异常检测
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚨 异常检测":
    st.title("🚨 异常定价检测（Isolation Forest）")

    from sklearn.ensemble import IsolationForest

    if not PERF_COL:
        st.warning("未找到性能分列。")
        st.stop()

    iso_features = st.multiselect(
        "选择异常检测特征",
        ["Price_CNY", PERF_COL],
        default=["Price_CNY", PERF_COL] if PERF_COL else ["Price_CNY"],
        key="iso_feat",
    )

    if len(iso_features) < 1:
        st.warning("请选择至少一个特征。")
        st.stop()

    contamination = st.slider("异常比例（contamination）", 0.01, 0.20, 0.05, 0.01)

    X_iso = df[iso_features].values
    iso = IsolationForest(contamination=contamination, random_state=42)
    anomalies = iso.fit_predict(X_iso)
    df_iso = df.copy()
    df_iso["Is_Anomaly"] = anomalies == -1

    n_anomalies = df_iso["Is_Anomaly"].sum()
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("🚨 异常产品数", n_anomalies)
    with col_b:
        st.metric("📊 异常比例", f"{n_anomalies/len(df)*100:.1f}%")
    with col_c:
        st.metric("💰 正常均价", f"¥{df_iso[~df_iso['Is_Anomaly']]['Price_CNY'].mean():.0f}")

    st.subheader("🗺️ 异常检测结果（红色 = 异常）")
    fig_iso = px.scatter(
        df_iso, x="Price_CNY", y=PERF_COL,
        color="Is_Anomaly",
        size="RAM_GB" if "RAM_GB" in df.columns else None,
        color_discrete_map={False: "#5B9BD5", True: "#FF6384"},
        symbol="Is_Anomaly",
        hover_data=["Brand", "Usage_Type", "CPU_Cores"],
        title="异常定价检测（红色异常点 = 割韭菜产品 or 宝藏机）",
    )
    fig_iso.update_layout(height=550,
                          legend_title="是否异常",
                          margin=dict(t=40, b=20, l=20, r=20))
    st.plotly_chart(fig_iso, use_container_width=True)

    st.subheader("🚨 异常产品列表（按价格排序）")
    st.dataframe(
        df_iso[df_iso["Is_Anomaly"]].sort_values("Price_CNY", ascending=False),
        use_container_width=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# 页面 11: 品牌分析
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏷️ 品牌分析":
    st.title("🏷️ 品牌分析")

    if "Brand" not in df.columns:
        st.warning("数据中无 Brand 列。")
        st.stop()

    st.subheader("🏆 各品牌统计")
    brand_stats = df.groupby("Brand").agg(
        产品数=("Price_CNY", "count"),
        平均价格=("Price_CNY", "mean"),
        价格标准差=("Price_CNY", "std"),
    )
    if PERF_COL:
        brand_stats["平均性能"] = df.groupby("Brand")[PERF_COL].mean()
        brand_stats["性能标准差"] = df.groupby("Brand")[PERF_COL].std()

    brand_stats = brand_stats.sort_values("产品数", ascending=False).round(1)
    st.dataframe(brand_stats, use_container_width=True)

    st.subheader("📊 品牌均价 vs 平均性能（气泡大小 = 产品数）")
    if PERF_COL:
        fig_brand = px.scatter(
            brand_stats, x="平均价格", y="平均性能",
            size="产品数",
            text=brand_stats.index,
            color="平均性能",
            color_continuous_scale="RdYlBu_r",
            title="品牌定位图（右上 = 高端高性能，左下 = 入门）",
        )
        fig_brand.update_traces(textposition="top center",
                                marker=dict(line=dict(width=1, color="white")))
        fig_brand.update_layout(height=550,
                                margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_brand, use_container_width=True)

    st.subheader("📈 各品牌价格分布（小提琴图）")
    fig_violin = go.Figure()
    for i, brand in enumerate(df["Brand"].value_counts().head(10).index):
        subset = df[df["Brand"] == brand]
        fig_violin.add_trace(go.Violin(
            x=subset["Brand"],
            y=subset["Price_CNY"],
            name=brand,
            box_visible=True,
            meanline_visible=True,
            opacity=0.7,
            hovertemplate=f"<b>{brand}</b><br>价格: %{{y}}$<extra></extra>",
        ))
    fig_violin.update_layout(height=500, yaxis_title="价格 (元)",
                              margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_violin, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# 页面 12: 预测模拟器
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎮 预测模拟器":
    st.title("🎮 预测模拟器")
    st.markdown("### 拖动滑块，实时预测这台电脑的使用类型和性能分！")
    st.caption("现场演示利器 —— 让老师输入配置，看模型实时出结果")

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    # 训练KNN和回归模型
    knn_features = ["CPU_Cores", "CPU_Frequency_GHz", "RAM_GB",
                    "Storage_GB", "GPU_Performance_Score", "Price_CNY"]
    knn_features = [c for c in knn_features if c in df.columns]

    if len(knn_features) < 2:
        st.warning("特征不足，无法启动预测模拟器。")
        st.stop()

    # 分类模型
    X_knn_sim = df[knn_features].fillna(0).values
    scaler_sim = StandardScaler()
    X_scaled_sim = scaler_sim.fit_transform(X_knn_sim)
    knn_sim = KNeighborsClassifier(n_neighbors=6)
    knn_sim.fit(X_scaled_sim, df["Usage_Type"])

    # 回归模型
    if PERF_COL:
        lr_sim = LinearRegression()
        lr_sim.fit(X_scaled_sim, df[PERF_COL])

    st.markdown("---")
    st.subheader("🔧 输入配置（拖动滑块）")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 硬件配置")
        cpu_cores = st.slider("CPU 核心数", 1, 16, 4)
        cpu_freq  = st.slider("CPU 频率 (GHz)", 1.0, 5.0, 3.0, 0.1)
        ram_gb    = st.slider("内存 (GB)", 4, 128, 16, 4)
        storage_gb = st.slider("存储 (GB)", 128, 4096, 512, 128)

    with col2:
        st.markdown("#### 性能 & 价格")
        gpu_score = st.slider("GPU 性能分", 0, 100, 50)
        price_cny = st.slider("价格 (元)", 2000, 50000, 10000, 500)

    # 预测
    input_vec = np.array([[cpu_cores, cpu_freq, ram_gb, storage_gb, gpu_score, price_cny / 7.25]])
    input_scaled = scaler_sim.transform(input_vec)

    pred_type = knn_sim.predict(input_scaled)[0]
    pred_proba = knn_sim.predict_proba(input_scaled)[0]
    type_labels = knn_sim.classes_

    if PERF_COL:
        pred_perf = lr_sim.predict(input_scaled)[0]
    else:
        pred_perf = None

    st.markdown("---")
    st.subheader("🎯 预测结果")

    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        st.metric("🏷️ 预测使用类型", pred_type)
    with col_r2:
        if pred_perf is not None:
            st.metric("📊 预测性能分", f"{pred_perf:.1f}")
    with col_r3:
        confidence = pred_proba.max()
        st.metric("🎯 置信度", f"{confidence:.1%}")

    # 概率条形图
    st.subheader("📊 各类别概率分布")
    prob_df = pd.DataFrame({
        "使用类型": type_labels,
        "概率": pred_proba,
    })
    fig_proba = go.Figure(data=[go.Bar(
        x=prob_df["使用类型"],
        y=prob_df["概率"],
        marker_color=[USAGE_COLORS.get(t, "#999") for t in prob_df["使用类型"]],
        text=prob_df["概率"].apply(lambda x: f"{x:.1%}"),
        textposition="outside",
        hovertemplate="%{x}<br>概率: %{y:.4f}<extra></extra>",
    )])
    fig_proba.add_hline(y=confidence, line_dash="dash", line_color="red",
                        annotation_text=f"最高概率: {confidence:.1%}")
    fig_proba.update_layout(height=420, yaxis_title="概率", yaxis_range=[0, 1],
                             margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_proba, use_container_width=True)

    # 和真实数据的对比
    st.subheader("📍 你的配置在数据集中的位置")
    if PERF_COL:
        fig_compare = px.scatter(
            df.sample(min(3000, len(df)), random_state=42),
            x="Price_CNY", y=PERF_COL,
            color="Usage_Type",
            color_discrete_map=USAGE_COLORS,
            opacity=0.3,
            title="你的配置（★） vs 真实数据",
        )
        fig_compare.add_trace(go.Scatter(
            x=[price_cny],
            y=[pred_perf] if pred_perf else [0],
            mode="markers",
            marker=dict(symbol="star", size=20, color="red",
                         line=dict(width=2, color="white")),
            name="你的配置",
        ))
        fig_compare.update_layout(height=500,
                                   margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_compare, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# 页面 13: 推荐引擎
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎁 推荐引擎":
    st.title("🎁 电脑推荐引擎")
    st.markdown("### 输入预算和需求，获取 Top 推荐！")

    col1, col2, col3 = st.columns(3)
    with col1:
        budget = st.slider("💰 预算（元）", 5000, 60000, 30000, 1000)
    with col2:
        usage_type_rec = st.selectbox("🎯 使用类型", ["Any", "Basic", "Student", "Professional", "Gaming"])
    with col3:
        min_perf_rec = st.slider("⚡ 最低性能分", 0, 100, 20 if PERF_COL else 0)

    st.markdown("---")

    st.markdown("**配件权重（影响推荐排序）**")
    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        w_cpu = st.slider("CPU 权重", 0, 100, 25, key="rec_wcpu")
    with s2:
        w_gpu = st.slider("GPU 权重", 0, 100, 25, key="rec_wgpu")
    with s3:
        w_ram = st.slider("内存 权重", 0, 100, 20, key="rec_wram")
    with s4:
        w_sto = st.slider("存储 权重", 0, 100, 20, key="rec_wsto")
    with s5:
        w_scr = st.slider("屏幕 权重", 0, 100, 10, key="rec_wscr")


    df_rec = df.copy()
    df_rec = df_rec[df_rec["Price_CNY"] <= budget]
    if usage_type_rec != "Any":
        df_rec = df_rec[df_rec["Usage_Type"] == usage_type_rec]
    if PERF_COL:
        df_rec = df_rec[df_rec[PERF_COL] >= min_perf_rec]

    if len(df_rec) == 0:
        st.warning("😢 没有符合条件的产品，请放宽筛选条件。")
    else:
        # 加权评分排序
        total_w = w_cpu + w_gpu + w_ram + w_sto + w_scr + 0.001
        if PERF_COL and all(c in df_rec.columns for c in ["CPU_Performance_Score", "GPU_Performance_Score", "RAM_GB", "Storage_GB"]):
            cpu_n = df_rec["CPU_Performance_Score"] / (df_rec["CPU_Performance_Score"].max() + 0.001)
            gpu_n = df_rec["GPU_Performance_Score"] / (df_rec["GPU_Performance_Score"].max() + 0.001)
            ram_n = df_rec["RAM_GB"] / (df_rec["RAM_GB"].max() + 0.001)
            sto_n = df_rec["Storage_GB"] / (df_rec["Storage_GB"].max() + 0.001)
            scr_n = df_rec["Screen_Size"] / (df_rec["Screen_Size"].max() + 0.001) if "Screen_Size" in df_rec.columns else 0.5
            budget_n = 1 - abs(df_rec["Price_CNY"] - budget) / (budget + 0.001)
            budget_n = budget_n.clip(0, 1)
            df_rec["_rec_score"] = (
                cpu_n * w_cpu / total_w +
                gpu_n * w_gpu / total_w +
                ram_n * w_ram / total_w +
                sto_n * w_sto / total_w +
                scr_n * w_scr / total_w +
                budget_n * 40 / total_w
            ) * 100
            df_rec = df_rec.sort_values("_rec_score", ascending=False)
        elif "Price_Performance_Ratio" in df_rec.columns:
            df_rec = df_rec.sort_values("Price_Performance_Ratio", ascending=False)

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("✅ 符合条件产品数", len(df_rec))
        with col_b:
            if "_rec_score" in df_rec.columns:
                st.metric("🏆 最高推荐分", f"{df_rec['_rec_score'].iloc[0]:.1f}")
            elif "Price_Performance_Ratio" in df_rec.columns:
                st.metric("🏆 最高性价比", f"{df_rec['Price_Performance_Ratio'].max():.2f}")

        show_cols = [c for c in ["Brand", "Usage_Type", "Price_CNY",
                                  PERF_COL, "Price_Performance_Ratio",
                                  "CPU_Cores", "RAM_GB", "GPU_Performance_Score"]
                     if c in df_rec.columns]
        st.dataframe(df_rec[show_cols].head(20), use_container_width=True)

        # 可视化
        if PERF_COL:
            st.subheader("🗺️ 推荐结果可视化（Top 50）")
            fig_rec = px.scatter(
                df_rec.head(50), x="Price_CNY", y=PERF_COL,
                size="Price_Performance_Ratio" if "Price_Performance_Ratio" in df_rec.columns else None,
                color="Usage_Type" if usage_type_rec == "Any" else None,
                hover_data=show_cols,
                title="推荐结果（气泡越大 = 性价比越高）",
            )
            fig_rec.update_layout(height=500,
                                  margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig_rec, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# 页面 14: 数据导出
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📥 数据导出":
    st.title("📥 数据导出")

    st.subheader("当前筛选后的数据")
    st.dataframe(df.head(100), use_container_width=True)
    st.caption(f"共 {len(df)} 条，显示前 100 条")

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 下载筛选后数据（CSV）",
        data=csv_data,
        file_name="filtered_laptop_data.csv",
        mime="text/csv",
    )

    # 统计摘要
    st.subheader("📊 统计摘要")
    st.dataframe(df.describe().round(2), use_container_width=True)

    summary_csv = df.describe().to_csv().encode("utf-8")
    st.download_button(
        label="📥 下载统计摘要（CSV）",
        data=summary_csv,
        file_name="data_summary.csv",
        mime="text/csv",
    )
