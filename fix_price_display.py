import re

with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# 只替换展示相关的 Price_USD → Price_CNY
# 图表 x= / y= / size= / color= / hover_data 里的字段名
# 指标卡片里的列名
# 不变：模型训练时的特征列名（那些保持 Price_USD 作为特征）

replacements = [
    # st.metric 里的显示
    (r'f"¥\{df_raw\[.Price_USD.\]', 'f"¥{df_raw["Price_CNY"]'),
    (r'f"¥\{df\[.Price_USD.\]', 'f"¥{df["Price_CNY"]'),
    (r"f\"¥\{df_raw\['Price_USD'\]", 'f"¥{df_raw["Price_CNY"]'),
    (r"f\"¥\{df\['Price_USD'\]", 'f"¥{df["Price_CNY"]'),
    # px.scatter / go.Scatter x="Price_USD" → x="Price_CNY"
    ('x="Price_USD"', 'x="Price_CNY"'),
    ('y="Price_USD"', 'y="Price_CNY"'),
    # hover_data 里的 "Price_USD"
    ('"Price_USD"', '"Price_CNY"'),
    # show_cols 里的 "Price_USD"
    # (这个在上面的替换里已经覆盖了)
    # st.metric 里的列引用
    ("['Price_USD']", "['Price_CNY']"),
    (".Price_USD.", ".Price_CNY."),
]

# 但有些 Price_USD 不能替换（模型训练时用的特征列）
# 先全部替换，再手动恢复模型训练部分的 Price_USD
for old, new in replacements:
    content = content.replace(old, new)

# 恢复模型训练部分：X 特征列里的 Price_USD 不变
# model_linear.py 里用的是 Price_USD 作为特征，app.py 里如果直接引用也需要保持
# 实际上 app.py 里模型是在各 src/*.py 里训练的，app.py 只是展示结果
# 所以 app.py 里所有的 Price_USD 都应该换成 Price_CNY

with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)

print("替换完成，检查差异...")
