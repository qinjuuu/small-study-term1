#!/usr/bin/env python
"""DIY 电脑配置推荐工具

输入预算、品牌、用途 + 组件优先级 → 智能推荐
直接运行: python src/diy_tool.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd

from config import SAMPLED_DATA

RATE = 7.25  # USD → RMB


# ════════════════════════════════════════════
#  推荐引擎
# ════════════════════════════════════════════

class Recommender:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.brands = sorted(self.df["Brand"].unique().tolist())
        self.usage_types = sorted(self.df["Usage_Type"].unique().tolist())
        self._cpu_max = max(self.df["CPU_Performance_Score"].max(), 1)
        self._gpu_max = max(self.df["GPU_Performance_Score"].max(), 1)
        self._perf_max = max(self.df["Overall_Performance_Score"].max(), 1)
        self._ram_max = max(self.df["RAM_GB"].max(), 1)
        self._sto_max = max(self.df["Storage_GB"].max(), 1)
        self._scr_max = max(self.df["Screen_Size"].max(), 1)
        self._ratio_max = max(self.df["Price_Performance_Ratio"].max(), 1)
        self.price_min = int(self.df["Price_USD"].min() * RATE)
        self.price_max = int(self.df["Price_USD"].max() * RATE) + 400

    def recommend(self, total, usage, brand,
                  cpu_w, gpu_w, ram_w, sto_w, scr_w, top_n=8):
        df = self.df.copy()
        # 预算转 USD 做内部匹配（数据存的是美元）
        total_usd = total / RATE
        if usage and usage != "不限":
            df = df[df["Usage_Type"] == usage]
        if brand and brand != "不限":
            df = df[df["Brand"] == brand]
        if len(df) == 0:
            return []

        w_sum = cpu_w + gpu_w + ram_w + sto_w + scr_w or 1

        def score(row):
            s = 0.0
            # 预算匹配 30% — 越接近预算越好，超预算猛扣
            p = row["Price_USD"]
            if p <= total_usd:
                s += 30 * (0.2 + 0.8 * p / max(total_usd, 1))
            else:
                over = (p - total_usd) / max(total_usd, 1)
                s += 30 * max(0, 1 - over * 4)

            # 组件优先级 40%
            s += 40 * (
                cpu_w / w_sum * row["CPU_Performance_Score"] / self._cpu_max +
                gpu_w / w_sum * row["GPU_Performance_Score"] / self._gpu_max +
                ram_w / w_sum * row["RAM_GB"] / self._ram_max +
                sto_w / w_sum * row["Storage_GB"] / self._sto_max +
                scr_w / w_sum * row["Screen_Size"] / self._scr_max
            )
            # 性价比 15%
            s += 15 * row["Price_Performance_Ratio"] / self._ratio_max
            # 综合性能 15%
            s += 15 * row["Overall_Performance_Score"] / self._perf_max
            return round(s, 1)

        df["_score"] = df.apply(score, axis=1)
        return df.nlargest(top_n, "_score")


# ════════════════════════════════════════════
#  GUI — Arknights 暗色主题
# ════════════════════════════════════════════

C0 = "#0d1117"   # 最深底
C1 = "#161b22"   # 卡片
C2 = "#21262d"   # 分隔 / 输入框
C3 = "#30363d"   # 边框
C4 = "#8b949e"   # 次要文字
C5 = "#c9d1d9"   # 主文字
C6 = "#e6c44d"   # 金 — 主强调
C7 = "#4493f8"   # 蓝 — 链接/按钮
C8 = "#3fb950"   # 绿 — 成功
C9 = "#f85149"   # 红


class ArknightsScale(tk.Frame):
    """自定义滑块：标签 + 数字 + 滑动条一体化，风格统一"""

    def __init__(self, parent, label, from_=0, to=100, default=50,
                 color=C6, **kw):
        super().__init__(parent, bg=C1, **kw)
        self.var = tk.IntVar(value=default)

        self._label = tk.Label(self, text=label, font=("Microsoft YaHei", 11),
                               bg=C1, fg=C5, anchor="w", width=8)
        self._label.pack(side="left")

        self._scale = tk.Scale(
            self, from_=from_, to=to, orient="horizontal",
            variable=self.var, bg=C1, fg=color, troughcolor=C2,
            highlightthickness=0, bd=0, length=280,
            command=lambda _: self._update_val()
        )
        self._scale.pack(side="left", padx=(4, 8))

        self._val_label = tk.Label(self, text=str(default),
                                   font=("Consolas", 12, "bold"),
                                   bg=C1, fg=color, width=4, anchor="e")
        self._val_label.pack(side="left")

    def _update_val(self):
        self._val_label.config(text=str(self.var.get()))

    def get(self):
        return self.var.get()

    def set(self, v):
        self.var.set(v)
        self._update_val()


class DiyTool:
    def __init__(self, rec):
        self.rec = rec
        self.root = tk.Tk()
        self.root.title("DIY 配置推荐")
        self.root.geometry("980x720")
        self.root.configure(bg=C0)
        self.root.minsize(800, 600)

        self.style = ttk.Style()
        self.style.theme_use("clam")
        self._setup_style()

        self._setup_keybindings()
        self._build_header()
        self._build_body()
        self._debounce_id = None

    def _setup_style(self):
        s = self.style
        s.configure("Dark.TCombobox",
                    fieldbackground=C2, background=C2,
                    foreground=C5, arrowcolor=C5,
                    bordercolor=C3, lightcolor=C2, darkcolor=C2)
        s.map("Dark.TCombobox",
              fieldbackground=[("readonly", C2)],
              foreground=[("readonly", C5)])
        self.root.option_add("*TCombobox*Listbox.background", C2)
        self.root.option_add("*TCombobox*Listbox.foreground", C5)
        self.root.option_add("*TCombobox*Listbox.selectBackground", C7)
        self.root.option_add("*TCombobox*Listbox.selectForeground", "white")
        self.root.option_add("*TCombobox*Listbox.font",
                             ("Microsoft YaHei", 11))

        s.configure("Treeview",
                    background=C1, foreground=C5,
                    fieldbackground=C1, bordercolor=C3,
                    rowheight=32)
        s.configure("Treeview.Heading",
                    background=C0, foreground=C6,
                    font=("Microsoft YaHei", 10, "bold"),
                    borderwidth=0)
        s.map("Treeview.Heading", background=[("active", C2)])
        s.map("Treeview",
              background=[("selected", C7)],
              foreground=[("selected", "white")])

    def _setup_keybindings(self):
        self.root.bind("<Control-f>", lambda e: self._on_search())

    def _build_header(self):
        h = tk.Frame(self.root, bg=C0, height=56)
        h.pack(fill="x")
        h.pack_propagate(False)

        tk.Label(h, text="▎DIY 配置推荐",
                 font=("Microsoft YaHei", 16, "bold"),
                 bg=C0, fg=C6).pack(side="left", padx=(20, 0))
        tk.Label(h, text=f"基于 {len(self.rec.df)} 条数据  ·  Ctrl+F 搜索",
                 font=("Microsoft YaHei", 9),
                 bg=C0, fg=C4).pack(side="right", padx=20)

        # 分割线
        sep = tk.Frame(self.root, bg=C3, height=1)
        sep.pack(fill="x", padx=12)

    def _build_body(self):
        body = tk.Frame(self.root, bg=C0)
        body.pack(fill="both", expand=True, padx=12, pady=10)

        # ── 顶部设置栏 ──
        card = tk.Frame(body, bg=C1, highlightthickness=0,
                        padx=16, pady=14)
        card.pack(fill="x")

        # 行1：用途 + 品牌 + 预算
        row1 = tk.Frame(card, bg=C1)
        row1.pack(fill="x", pady=(0, 10))

        self._make_field(row1, "场景", 0)
        self.usage_var = tk.StringVar(value="不限")
        self.usage_cb = ttk.Combobox(
            row1, textvariable=self.usage_var,
            values=["不限"] + self.rec.usage_types,
            state="readonly", width=12, style="Dark.TCombobox",
            font=("Microsoft YaHei", 11))
        self.usage_cb.pack(side="left", padx=(0, 16))
        self.usage_cb.bind("<<ComboboxSelected>>", lambda e: self._debounce_search())

        self._make_field(row1, "品牌", 0)
        self.brand_var = tk.StringVar(value="不限")
        self.brand_cb = ttk.Combobox(
            row1, textvariable=self.brand_var,
            values=["不限"] + self.rec.brands,
            state="readonly", width=12, style="Dark.TCombobox",
            font=("Microsoft YaHei", 11))
        self.brand_cb.pack(side="left", padx=(0, 16))
        self.brand_cb.bind("<<ComboboxSelected>>", lambda e: self._debounce_search())

        self._make_field(row1, "总预算", 0)
        self.budget_var = tk.IntVar(value=1000)

        self.budget_label = tk.Label(
            row1, text="", font=("Consolas", 20, "bold"),
            bg=C1, fg=C6, width=7, anchor="e")
        self.budget_label.pack(side="left", padx=(0, 8))
        self._update_budget_label()

        self.budget_scale = tk.Scale(
            row1, from_=self.rec.price_min, to=self.rec.price_max,
            resolution=50, orient="horizontal", variable=self.budget_var,
            bg=C1, fg=C6, troughcolor=C2, highlightthickness=0,
            bd=0, length=240,
            command=lambda _: self._on_budget_drag())
        self.budget_scale.pack(side="left", padx=(0, 4))
        tk.Label(row1, text="RMB", font=("Microsoft YaHei", 9),
                 bg=C1, fg=C4).pack(side="left")

        # 行2：组件优先级
        row2 = tk.Frame(card, bg=C1)
        row2.pack(fill="x")

        tk.Label(row2, text="组件优先级",
                 font=("Microsoft YaHei", 10, "bold"),
                 bg=C1, fg=C6).pack(side="left", padx=(0, 12))

        self.cpu_w = ArknightsScale(row2, "CPU", 0, 100, 50, C9)
        self.cpu_w.pack(side="left", padx=(0, 8))
        self.cpu_w._scale.bind("<ButtonRelease-1>", lambda e: self._debounce_search())

        self.gpu_w = ArknightsScale(row2, "GPU", 0, 100, 50, "#e6c44d")
        self.gpu_w.pack(side="left", padx=(0, 8))
        self.gpu_w._scale.bind("<ButtonRelease-1>", lambda e: self._debounce_search())

        self.ram_w = ArknightsScale(row2, "MEM", 0, 100, 50, C7)
        self.ram_w.pack(side="left", padx=(0, 8))
        self.ram_w._scale.bind("<ButtonRelease-1>", lambda e: self._debounce_search())

        self.sto_w = ArknightsScale(row2, "STO", 0, 100, 50, C8)
        self.sto_w.pack(side="left", padx=(0, 8))
        self.sto_w._scale.bind("<ButtonRelease-1>", lambda e: self._debounce_search())

        self.scr_w = ArknightsScale(row2, "SCR", 0, 100, 50, "#a371f7")
        self.scr_w.pack(side="left", padx=(0, 8))
        self.scr_w._scale.bind("<ButtonRelease-1>", lambda e: self._debounce_search())

        # 快捷按钮行
        row3 = tk.Frame(card, bg=C1)
        row3.pack(fill="x", pady=(10, 0))

        for label, cpu, gpu, ram, sto, scr in [
            ("均衡", 50, 50, 50, 50, 50),
            ("游戏", 30, 80, 40, 30, 60),
            ("办公", 70, 20, 50, 40, 30),
            ("AI/渲染", 60, 60, 60, 20, 30),
        ]:
            btn = tk.Button(row3, text=label,
                            font=("Microsoft YaHei", 9),
                            bg=C2, fg=C5, relief="flat",
                            padx=12, pady=3, cursor="hand2",
                            activebackground=C3, activeforeground=C6,
                            bd=0,
                            command=lambda c=cpu, g=gpu, r=ram, s=sto,
                            sc=scr: self._preset(c, g, r, s, sc))
            btn.pack(side="left", padx=(0, 6))

        self.status_label = tk.Label(row3, text="",
                                     font=("Microsoft YaHei", 9),
                                     bg=C1, fg=C4)
        self.status_label.pack(side="right")

        # ── 分割 ──
        tk.Frame(body, bg=C3, height=1).pack(fill="x", pady=12)

        # ── 结果表格 ──
        self._build_table(body)

    def _make_field(self, parent, text, pady):
        tk.Label(parent, text=text, font=("Microsoft YaHei", 10),
                 bg=C1, fg=C4).pack(side="left", padx=(0, 6))

    def _build_table(self, parent):
        tf = tk.Frame(parent, bg=C1)
        tf.pack(fill="both", expand=True)

        columns = ("#", "品牌", "场景", "价格", "CPU", "GPU",
                   "内存", "存储", "屏幕", "性能", "性价比", "匹配")

        self.tree = ttk.Treeview(tf, columns=columns,
                                 show="headings", height=12)
        widths = [28, 70, 80, 70, 100, 110, 55, 55, 50, 65, 55, 50]

        for col, w in zip(columns, widths):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=w, anchor="center",
                             minwidth=w)

        vsb = ttk.Scrollbar(tf, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)

        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        # 标签
        self.tree.tag_configure("rank1",
                                font=("Microsoft YaHei", 10, "bold"),
                                background="#1a2332")
        self.tree.tag_configure("rank2",
                                background="#141c26")
        self.tree.tag_configure("rank3",
                                background="#111820")

        self.tree.bind("<Double-1>", self._on_detail)

    def _preset(self, cpu, gpu, ram, sto, scr):
        self.cpu_w.set(cpu)
        self.gpu_w.set(gpu)
        self.ram_w.set(ram)
        self.sto_w.set(sto)
        self.scr_w.set(scr)
        self._on_search()

    def _on_budget_drag(self):
        self._update_budget_label()
        self._debounce_search()

    def _update_budget_label(self):
        v = self.budget_var.get()
        self.budget_label.config(text=f"¥{v}")

    def _debounce_search(self):
        if self._debounce_id:
            self.root.after_cancel(self._debounce_id)
        self._debounce_id = self.root.after(300, self._on_search)

    def _on_search(self, *_):
        total = self.budget_var.get()
        usage = self.usage_var.get()
        brand = self.brand_var.get()
        cpu_w = self.cpu_w.get()
        gpu_w = self.gpu_w.get()
        ram_w = self.ram_w.get()
        sto_w = self.sto_w.get()
        scr_w = self.scr_w.get()

        results = self.rec.recommend(total, usage, brand,
                                     cpu_w, gpu_w, ram_w, sto_w, scr_w)

        for item in self.tree.get_children():
            self.tree.delete(item)

        if len(results) == 0:
            self.status_label.config(text="无匹配，请放宽条件")
            return

        self.status_label.config(
            text=f"从 {len(self.rec.df)} 条数据中推荐 {len(results)} 条")

        for i, (_, r) in enumerate(results.iterrows()):
            tag = "rank1" if i == 0 else ("rank2" if i < 3 else "rank3")
            price_str = f"¥{int(r['Price_USD'] * RATE)}"
            cpu_str = f"{r['CPU_Brand']} {int(r['CPU_Cores'])}C {r['CPU_Frequency_GHz']:.1f}G"
            gpu_str = f"{r['GPU_Model']}"
            ram_str = f"{int(r['RAM_GB'])}GB"
            sto_str = f"{int(r['Storage_GB'])}GB"
            screen_str = f"{r['Screen_Size']:.1f}\""

            self.tree.insert("", "end", values=(
                i + 1, r["Brand"], r["Usage_Type"],
                price_str, cpu_str, gpu_str,
                ram_str, sto_str, screen_str,
                int(r["Overall_Performance_Score"]),
                f"{r['Price_Performance_Ratio']:.2f}",
                r["_score"],
            ), tags=(tag,))

    def _on_detail(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        vals = self.tree.item(sel[0], "values")
        msg = (
            f"品牌: {vals[1]}\n"
            f"场景: {vals[2]}\n"
            f"价格: {vals[3]}\n"
            f"CPU:  {vals[4]}\n"
            f"GPU:  {vals[5]}\n"
            f"内存: {vals[6]}  存储: {vals[7]}\n"
            f"屏幕: {vals[8]}  性能分: {vals[9]}\n"
            f"性价比: {vals[10]}  匹配分: {vals[11]}"
        )
        messagebox.showinfo("详细配置", msg)

    def run(self):
        self._on_search()
        self.root.mainloop()


# ════════════════════════════════════════════

def main():
    if not os.path.exists(SAMPLED_DATA):
        print(f"❌ 数据文件不存在: {SAMPLED_DATA}")
        print("   请先运行 python src/main.py")
        return

    rec = Recommender(SAMPLED_DATA)
    print(f"✅ 数据: {len(rec.df)} 条 | "
          f"价格范围 ¥{rec.price_min} ~ ¥{rec.price_max}")
    DiyTool(rec).run()


if __name__ == "__main__":
    main()
