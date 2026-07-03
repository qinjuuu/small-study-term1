import re

with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# 推荐引擎页面新内容
new_rec_page = '''
# ════════════════════════════════════════════════════════════════════════════
# 页面 13: 推荐引擎
# ════════════════════════════════════════════════════════════════════════════
elif page == "🎁 推荐引擎":
    st.title("🎁 电脑推荐引擎")
    st.markdown("### 设置预算、需求和配件权重，获取个性化 Top 推荐！")

    # 预设方案
    presets = {
        "自定义":   [25, 25, 20, 20, 10],
        "均衡型":   [25, 25, 20, 20, 10],
        "游戏型":   [15, 45, 15, 15, 10],
        "办公型":   [40, 10, 25, 20, 5],
        "AI/渲染":  [30, 40, 15, 10, 5],
    }
    preset_name = st.selectbox("⚙️ 快捷预设", list(presets.keys()))
    if preset_name != "自定义":
        st.info(f"已加载「{preset_name}」预设，可手动微调下方滑块")

    col1, col2, col3 = st.columns(3)
    with col1:
        budget = st.slider("💰 预算（元）", 5000, 60000, 30000, 1000)
    with col2:
        usage_type_rec = st.selectbox("🎯 使用类型", ["Any", "Basic", "Student", "Professional", "Gaming"])
    with col3:
        min_perf_rec = st.slider("⚡ 最低性能分", 0, 100, 20 if PERF_COL else 0)

    st.markdown("**配件权重（影响推荐排序）**")
    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        w_cpu = st.slider("CPU 权重", 0, 100, presets[preset_name][0], key="rec_wcpu")
    with s2:
        w_gpu = st.slider("GPU 权重", 0, 100, presets[preset_name][1], key="rec_wgpu")
    with s3:
        w_ram = st.slider("内存 权重", 0, 100, presets[preset_name][2], key="rec_wram")
    with s4:
        w_sto = st.slider("存储 权重", 0, 100, presets[preset_name][3], key="rec_wsto")
    with s5:
        w_scr = st.slider("屏幕 权重", 0, 100, presets[preset_name][4], key="rec_wscr")

    st.markdown("---")

    df_rec = df.copy()
    df_rec = df_rec[df_rec["Price_CNY"] <= budget]
    if usage_type_rec != "Any":
        df_rec = df_rec[df_rec["Usage_Type"] == usage_type_rec]
    if PERF_COL:
        df_rec = df_rec[df_rec[PERF_COL] >= min_perf_rec]

    if len(df_rec) == 0:
        st.warning("😢 没有符合条件的产品，请放宽筛选条件。")
    else:
        # 计算加权评分
        total_w = w_cpu + w_gpu + w_ram + w_sto + w_scr + 0.001
        if PERF_COL and all(c in df_rec.columns for c in ["CPU_Performance_Score", "GPU_Performance_Score", "RAM_GB", "Storage_GB"]):
            cpu_n = df_rec["CPU_Performance_Score"] / (df_rec["CPU_Performance_Score"].max() + 0.001)
            gpu_n = df_rec["GPU_Performance_Score"] / (df_rec["GPU_Performance_Score"].max() + 0.001)
            ram_n = df_rec["RAM_GB"] / (df_rec["RAM_GB"].max() + 0.001)
            sto_n = df_rec["Storage_GB"] / (df_rec["Storage_GB"].max() + 0.001)
            scr_n = 0.5
            if "Screen_Size" in df_rec.columns:
                scr_n = df_rec["Screen_Size"] / (df_rec["Screen_Size"].max() + 0.001)
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
'''

# 找到推荐引擎页面的起止位置
# 起始：elif page == "🎁 推荐引擎":
# 结束：下一个 # ════ 或 elif page == 或文件末尾
start_marker = '# ════════════════════════════════════════════════════════════════════════════\n# 页面 13: 推荐引擎'
end_marker = '# ════════════════════════════════════════════════════════════════════════════\n# 页面 14:'

start_idx = content.find(start_marker)
if start_idx == -1:
    print("找不到起始标记")
else:
    end_idx = content.find(end_marker, start_idx + len(start_marker))
    if end_idx == -1:
        print("找不到结束标记，尝试找下一个 elif")
        end_idx = content.find('\nelif page == "', start_idx + len(start_marker))
    if end_idx == -1:
        print("找不到结束位置")
    else:
        new_content = content[:start_idx] + new_rec_page + "\n" + content[end_idx:]
        with open("app.py", "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"替换成功！起始位置：{start_idx}，结束位置：{end_idx}")
'''
print("脚本写入完成")
