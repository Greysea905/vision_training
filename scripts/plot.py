#!/usr/bin/env python3
"""任务2 出图脚本：读取 C++ 程序导出的 CSV，生成拟合对比图 / 角速度曲线 / 残差图。

计算在 C++ 里完成，结果导出成 CSV；本脚本只负责读 CSV -> 画图。

用法示例：
    python3 scripts/plot.py \
        --fit result/task2_fit/fit_data.csv \
        --omega result/task2_fit/omega_data.csv \
        --residual result/task2_fit/residual_data.csv \
        -o result/task2_fit
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无显示环境用 Agg 后端
import matplotlib.pyplot as plt


def load_csv(path, cols):
    """读取 CSV，返回 (第0列, 其余列) 两个 numpy 数组。"""
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, cols]


def plot_fit_comparison(x, obs, fitted, out_path):
    # TODO: 观测点(散点) vs 拟合曲线(线)，加图例与单位标注
    pass


def plot_angular_velocity(t, omega, out_path):
    # TODO: 角速度曲线
    pass


def plot_residuals(x, res, out_path):
    # TODO: 残差曲线（常为围绕 0 的散点/线）
    pass


def main():
    parser = argparse.ArgumentParser(description="任务2 出图")
    parser.add_argument("--fit", help="拟合对比数据 CSV")
    parser.add_argument("--omega", help="角速度数据 CSV")
    parser.add_argument("--residual", help="残差数据 CSV")
    parser.add_argument("-o", "--outdir", default="result/task2_fit")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    # TODO: 调用上面三个 plot_* 函数


if __name__ == "__main__":
    main()
