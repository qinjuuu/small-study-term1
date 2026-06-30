"""数据质量审计：重复、不真实值、无效值"""
import pandas as pd
import numpy as np
from config import RAW_DATA

df = pd.read_csv(RAW_DATA)
N = len(df)
print(f"总记录数: {N}")
print(f"总字段数: {df.shape[1]}")
print(f"字段列表: {list(df.columns)}\n")

# ========== 1. 重复 ==========
print("=" * 60)
print("【1】重复数据")
print("=" * 60)

# 完全重复的行
dup_all = df.duplicated()
dup_all_count = dup_all.sum()
print(f"完全重复行数: {dup_all_count} ({dup_all_count/N*100:.2f}%)")

# 关键字段重复（同一个型号出现多次）
key_cols = [c for c in ["Brand", "Model", "CPU", "GPU", "RAM_GB", "Storage_GB"] if c in df.columns]
if key_cols:
    dup_key = df.duplicated(subset=key_cols, keep=False)
    dup_key_count = dup_key.sum()
    # 去重后唯一组合数
    unique_key_count = df[key_cols].drop_duplicates().shape[0]
    print(f"关键字段重复记录数: {dup_key_count} ({dup_key_count/N*100:.2f}%)")
    print(f"关键字段唯一组合数: {unique_key_count}")
    # 展示几个重复的例子
    if dup_key_count > 0:
        dup_examples = df[dup_key].groupby(key_cols).size().sort_values(ascending=False).head(5)
        print(f"重复最多的前5组:")
        for idx, cnt in dup_examples.items():
            print(f"  {idx} → 出现 {cnt} 次")

print()

# ========== 2. 无效/缺失值 ==========
print("=" * 60)
print("【2】缺失值与无效值")
print("=" * 60)

null_counts = df.isnull().sum()
null_cols = null_counts[null_counts > 0]
if len(null_cols) == 0:
    print("无缺失值")
else:
    for col, cnt in null_cols.items():
        print(f"  {col}: {cnt} 缺失 ({cnt/N*100:.2f}%)")

# 特殊无效值：空字符串、0、负值等
for col in df.columns:
    if df[col].dtype == object:
        empty_count = (df[col].astype(str).str.strip() == "").sum()
        zero_str_count = (df[col].astype(str).str.strip() == "0").sum()
        if empty_count > 0:
            print(f"  {col}: {empty_count} 个空字符串")
        if zero_str_count > 0:
            print(f"  {col}: {zero_str_count} 个值为 '0' 的字符串")

print()

# ========== 3. 不真实/异常值 ==========
print("=" * 60)
print("【3】不真实值 / 异常值检测")
print("=" * 60)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"数值字段: {numeric_cols}")

for col in numeric_cols:
    vals = df[col].dropna()
    if len(vals) == 0:
        continue
    q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = (vals < lower) | (vals > upper)
    n_out = outliers.sum()
    
    # 业务规则不真实值
    unrealistic = 0
    if col == "Price_USD":
        unrealistic = int((vals < 0).sum() + (vals > 20000).sum())
    elif "RAM" in col:
        unrealistic = int((vals <= 0).sum() + (vals > 512).sum())
    elif "Storage" in col:
        unrealistic = int((vals <= 0).sum() + (vals > 10000).sum())
    elif "Performance" in col or "Score" in col:
        unrealistic = int((vals < 0).sum())
    
    print(f"\n  {col}:")
    print(f"    范围: [{vals.min():.2f}, {vals.max():.2f}]  均值: {vals.mean():.2f}  中位数: {vals.median():.2f}")
    print(f"    IQR异常值: {n_out} ({n_out/len(vals)*100:.2f}%)  [<{lower:.1f} 或 >{upper:.1f}]")
    if unrealistic > 0:
        print(f"    业务不合理值: {unrealistic} ({unrealistic/len(vals)*100:.2f}%)")

# 文本字段的不合理值
print()
for col in df.select_dtypes(include=[object]).columns:
    vals = df[col].dropna().astype(str)
    unique = vals.nunique()
    top3 = vals.value_counts().head(3)
    print(f"  {col}: {unique} 个唯一值  top3={dict(top3)}")

# ========== 4. 汇总 ==========
print()
print("=" * 60)
print("【汇总】")
print("=" * 60)

total_issues = dup_all_count
total_issues += null_counts.sum()
# 统计所有数值列的 IQR 异常值总数
total_outliers = 0
for col in numeric_cols:
    vals = df[col].dropna()
    if len(vals) == 0:
        continue
    q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
    iqr = q3 - q1
    total_outliers += int(((vals < q1 - 1.5*iqr) | (vals > q3 + 1.5*iqr)).sum())

total_issues += total_outliers

print(f"完全重复行:         {dup_all_count}")
print(f"缺失值总数:         {int(null_counts.sum())}")
print(f"IQR 异常值总数:     {total_outliers}")
print(f"问题记录数(合计):   {total_issues}")
print(f"问题率:             {total_issues/(N*len(df.columns))*100:.2f}%  (按字段数加权)")
print(f"数据整体健康度:     需要根据业务场景判断")
print(f"\n注: 异常值不一定是'错误'的 — Gaming笔记本价格高、性能强是真实特征，")
print(f"    IQR 方法会将其标记为异常，但这是业务意义上的正常数据。")
