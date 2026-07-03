"""replace_usd_to_cny.py — 把 app.py 里所有 USD/$ 显示替换成 元/¥"""
import re

with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# 标签 / 标题
content = content.replace("Price (USD)", "价格 (元)")
content = content.replace("price (USD)", "价格 (元)")
content = content.replace('"USD"', '"元"')
content = content.replace("'USD'", "'元'")

# hovertemplate 里的 $%{x} / $%{y}
content = content.replace("$%{x}", "¥%{x}")
content = content.replace("$%{y}", "¥%{y}")
content = content.replace("$%{value}", "¥%{value}")

# slider / 侧边栏标签
content = content.replace("价格区间 (USD)", "价格区间 (元)")
content = content.replace("预算（USD）", "预算（元）")
content = content.replace("预算 (USD)", "预算 (元)")

# f-string 里的 ${  -> ¥{
content = content.replace('f"${', 'f"¥{')
content = content.replace("f'${", "f'¥{")

# 独立 $ 符号（在字符串里）
content = content.replace('"$" +', '"¥" +')
content = content.replace("'$' +", "'¥' +")

with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)

print("✅ 替换完成")
