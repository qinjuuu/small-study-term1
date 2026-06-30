"""主入口：一键运行全部分析流程"""
import subprocess
import sys
import os

SCRIPTS = [
    ("generate_data.py", "生成数据集"),
    ("explore.py", "数据探索与可视化"),
    ("preprocess.py", "数据预处理"),
    ("models.py", "建模（线性回归/KNN/K-means）"),
]

def main():
    src_dir = os.path.dirname(os.path.abspath(__file__))
    python = sys.executable

    for script, desc in SCRIPTS:
        print(f"\n{'='*60}")
        print(f">>> {desc}")
        print(f"{'='*60}")
        path = os.path.join(src_dir, script)
        result = subprocess.run([python, path], capture_output=False)
        if result.returncode != 0:
            print(f"❌ {script} 执行失败，退出")
            sys.exit(1)

    print("\n" + "="*60)
    print("🎉 全部分析流程完成！")
    print(f"图表：output/charts/")
    print(f"报告：output/reports/analysis_report.md")
    print("="*60)


if __name__ == "__main__":
    main()
