import re

with open("app.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

# 找推荐引擎页面中 "---" 分隔符的位置，在其后插入权重滑块代码
output = []
in_rec_page = False
inserted = False

i = 0
while i < len(lines):
    line = lines[i]
    
    # 检测是否进入推荐引擎页面
    if "推荐引擎" in line and line.strip().startswith("elif"):
        in_rec_page = True
    
    # 在推荐引擎页面中找到 "---" 分隔符，在其后插入权重滑块
    if in_rec_page and not inserted and "st.markdown(" in line and "---" in line:
        output.append(line)  # 保留 "---" 那一行
        # 插入权重滑块代码
        output.append('\n')
        output.append('    st.markdown("**配件权重（影响推荐排序）**")\n')
        output.append('    s1, s2, s3, s4, s5 = st.columns(5)\n')
        output.append('    with s1:\n')
        output.append('        w_cpu = st.slider("CPU 权重", 0, 100, 25, key="rec_wcpu")\n')
        output.append('    with s2:\n')
        output.append('        w_gpu = st.slider("GPU 权重", 0, 100, 25, key="rec_wgpu")\n')
        output.append('    with s3:\n')
        output.append('        w_ram = st.slider("内存 权重", 0, 100, 20, key="rec_wram")\n')
        output.append('    with s4:\n')
        output.append('        w_sto = st.slider("存储 权重", 0, 100, 20, key="rec_wsto")\n')
        output.append('    with s5:\n')
        output.append('        w_scr = st.slider("屏幕 权重", 0, 100, 10, key="rec_wscr")\n')
        output.append('\n')
        inserted = True
        i += 1
        continue
    
    # 在推荐引擎页面中，找到排序逻辑，替换为加权评分
    if in_rec_page and inserted and not inserted and "按性价比排序" in line:
        # 跳过原来的排序逻辑，后面会加新逻辑
        pass
    
    output.append(line)
    i += 1

with open("app.py", "w", encoding="utf-8") as f:
    f.writelines(output)

print("权重滑块插入完成")
