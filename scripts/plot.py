#!/usr/bin/env python3
"""任务2 出图脚本：读取 C++ 导出的 fit_data.csv，生成 3 张结果图。

用法：python3 scripts/plot.py
输入：result/task2_fit/fit_data.csv
输出：result/task2_fit/{fit_comparison,angular_velocity,residuals}.png
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")   # 无显示环境用 Agg 后端
import matplotlib.pyplot as plt

CSV_PATH = "result/task2_fit/fit_data.csv"
OUT_DIR = "result/task2_fit/"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 读 CSV：第0列 t，第1列 omega_obs，第2列 omega_fit，第3列 residual
    data = np.loadtxt(CSV_PATH, delimiter=",", skiprows=1)
    t = data[:, 0]
    omega_obs = data[:, 1]
    omega_fit = data[:, 2]
    residual = data[:, 3]

    # ① 观测点 vs 拟合曲线（同图）
    plt.figure(figsize=(10, 5))
    plt.plot(t, omega_obs, ".", markersize=3, alpha=0.6, label="observed")
    plt.plot(t, omega_fit, "-", linewidth=1.5, label="fitted")
    plt.xlabel("Time (s)")
    plt.ylabel("Angular velocity (rad/s)")
    plt.title("Observed vs fitted angular velocity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fit_comparison.png"), dpi=150)
    plt.close()

    # ② 估计的角速度曲线
    plt.figure(figsize=(10, 5))
    plt.plot(t, omega_fit, "-", linewidth=1.5, color="tab:orange")
    plt.xlabel("Time (s)")
    plt.ylabel("Angular velocity (rad/s)")
    plt.title("Estimated angular velocity: omega(t) = b + A sin(Omega t + phi)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "angular_velocity.png"), dpi=150)
    plt.close()

    # ③ 残差曲线（围绕 0）
    plt.figure(figsize=(10, 5))
    plt.plot(t, residual, "-", linewidth=0.6, color="tab:red")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Residual (rad/s)")
    plt.title("Residuals")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "residuals.png"), dpi=150)
    plt.close()

    print("已生成 3 张图：fit_comparison.png / angular_velocity.png / residuals.png")


if __name__ == "__main__":
    main()
