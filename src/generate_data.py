"""生成模拟笔记本电脑数据集
模拟市场真实分布：不同用途对应不同的硬件配置和价格区间
"""
import pandas as pd
import numpy as np
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

np.random.seed(42)
N = 8000


def generate():
    # ── 使用类型分布 ──
    usage_types = np.random.choice(
        ["Basic", "Student", "Professional", "Gaming"],
        size=N,
        p=[0.25, 0.30, 0.25, 0.20]
    )

    # ── 按使用类型生成特征 ──
    cpu_cores = np.zeros(N, dtype=int)
    cpu_freq = np.zeros(N)
    ram_gb = np.zeros(N, dtype=int)
    storage_gb = np.zeros(N)
    gpu_score = np.zeros(N)
    price = np.zeros(N)

    for i, ut in enumerate(usage_types):
        if ut == "Basic":
            cpu_cores[i] = np.random.choice([2, 4], p=[0.6, 0.4])
            cpu_freq[i] = np.random.uniform(1.6, 2.8)
            ram_gb[i] = np.random.choice([4, 8], p=[0.7, 0.3])
            storage_gb[i] = np.random.choice([128, 256, 512], p=[0.5, 0.4, 0.1])
            gpu_score[i] = np.random.uniform(200, 800)
            price[i] = np.random.uniform(200, 500)

        elif ut == "Student":
            cpu_cores[i] = np.random.choice([4, 6], p=[0.5, 0.5])
            cpu_freq[i] = np.random.uniform(2.0, 3.5)
            ram_gb[i] = np.random.choice([8, 16], p=[0.7, 0.3])
            storage_gb[i] = np.random.choice([256, 512], p=[0.5, 0.5])
            gpu_score[i] = np.random.uniform(500, 1500)
            price[i] = np.random.uniform(400, 900)

        elif ut == "Professional":
            cpu_cores[i] = np.random.choice([6, 8, 12], p=[0.3, 0.5, 0.2])
            cpu_freq[i] = np.random.uniform(2.5, 4.5)
            ram_gb[i] = np.random.choice([16, 32], p=[0.6, 0.4])
            storage_gb[i] = np.random.choice([512, 1024], p=[0.6, 0.4])
            gpu_score[i] = np.random.uniform(1000, 4000)
            price[i] = np.random.uniform(800, 2500)

        else:  # Gaming
            cpu_cores[i] = np.random.choice([8, 12, 16], p=[0.4, 0.4, 0.2])
            cpu_freq[i] = np.random.uniform(3.0, 5.5)
            ram_gb[i] = np.random.choice([16, 32], p=[0.5, 0.5])
            storage_gb[i] = np.random.choice([512, 1024, 2048], p=[0.4, 0.4, 0.2])
            gpu_score[i] = np.random.uniform(5000, 15000)
            price[i] = np.random.uniform(1000, 4000)

    # ── CPU 性能分 = 核心数 × 主频 × 100 ──
    cpu_score = cpu_cores * cpu_freq * 100

    # ── 综合性能分 = CPU分 × 0.4 + GPU分 × 0.6 ──
    overall_score = cpu_score * 0.4 + gpu_score * 0.6

    # ── 品牌 ──
    brands_cpu = ["Intel", "AMD"]
    brands_gpu_intel = ["Intel UHD", "Intel Iris Xe"]
    brands_gpu_nvidia = ["NVIDIA GeForce", "NVIDIA RTX"]
    brands_gpu_amd = ["AMD Radeon"]

    cpu_brand = np.random.choice(brands_cpu, size=N, p=[0.7, 0.3])
    gpu_brand = []
    gpu_model = []
    for i in range(N):
        if usage_types[i] in ["Basic", "Student"]:
            if np.random.random() < 0.5:
                gpu_brand.append("Intel")
                gpu_model.append(np.random.choice(brands_gpu_intel))
            else:
                gpu_brand.append("AMD")
                gpu_model.append(np.random.choice(brands_gpu_amd))
        else:
            if np.random.random() < 0.6:
                gpu_brand.append("NVIDIA")
                gpu_model.append(np.random.choice(brands_gpu_nvidia))
            else:
                gpu_brand.append("AMD")
                gpu_model.append(np.random.choice(brands_gpu_amd))

    # ── 品牌名 ──
    mfr = np.random.choice(
        ["Dell", "HP", "Lenovo", "ASUS", "Acer", "Apple", "MSI", "Razer"],
        size=N,
        p=[0.20, 0.20, 0.20, 0.15, 0.10, 0.05, 0.05, 0.05]
    )

    # ── 屏幕尺寸 ──
    screen_size = np.random.choice([13.3, 14.0, 15.6, 16.0, 17.3], size=N,
                                   p=[0.15, 0.20, 0.40, 0.15, 0.10])

    # ── 组装 DataFrame ──
    df = pd.DataFrame({
        "Brand": mfr,
        "Usage_Type": usage_types,
        "CPU_Brand": cpu_brand,
        "CPU_Cores": cpu_cores,
        "CPU_Frequency_GHz": cpu_freq.round(1),
        "GPU_Brand": gpu_brand,
        "GPU_Model": gpu_model,
        "RAM_GB": ram_gb,
        "Storage_GB": storage_gb,
        "Screen_Size": screen_size,
        "CPU_Performance_Score": cpu_score.round(0).astype(int),
        "GPU_Performance_Score": gpu_score.round(0).astype(int),
        "Overall_Performance_Score": overall_score.round(0).astype(int),
        "Price_USD": price.round(0).astype(int),
    })

    # 加一点噪声让数据更真实
    df["Price_USD"] += np.random.normal(0, 50, N).round(0).astype(int)
    df["Price_USD"] = df["Price_USD"].clip(150, 5000)

    # ── 性价比特征 ──
    df["Price_Performance_Ratio"] = (
        df["Overall_Performance_Score"] / df["Price_USD"]
    ).round(2)

    # 打乱
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # ── 保存 ──
    raw_path = os.path.join(DATA_DIR, "raw_laptops.csv")
    sample_path = os.path.join(DATA_DIR, "sampled_laptops.csv")
    df.to_csv(raw_path, index=False)
    df.to_csv(sample_path, index=False)

    print(f"✅ 生成 {len(df)} 条数据")
    print(f"   原始数据: {raw_path}")
    print(f"   抽样数据: {sample_path}")
    print(f"\n使用类型分布:")
    print(df["Usage_Type"].value_counts())
    print(f"\n描述统计:")
    print(df.describe().to_string())

    return df


if __name__ == "__main__":
    generate()
