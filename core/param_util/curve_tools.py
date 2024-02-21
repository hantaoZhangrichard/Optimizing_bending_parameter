# encoding:utf-8
"""
    曲线io
"""

import numpy as np
from core.vector_tools import normalize_vector


def read_txt(filename: str) -> np.ndarray:
    """
    从txt文件读入点数据
    参数:
        filename - 文件名
    返回值:
        维度为(n, 3)的ndarray
    """
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
        t = list(map(
            lambda x: list(map(
                float, x.split()
            )),
            lines
        ))
        return np.array(t)


def write_txt(filename: str, points: np.ndarray):
    """
    将点数据写入txt
    参数:
        filename - 文件名
        points - 维度为(n, 3)的ndarray
    返回值:
        None
    """
    with open(filename, "w", encoding="utf-8") as f:
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


def calc_slopes(points):
    # 点作差，第一个点复制一份
    shifted_points = np.vstack([points[0:1], points[:-1]])
    # 计算两两之间的点的差
    delta = points - shifted_points
    # 各个点处的斜率，第一个点的斜率指定为x轴方向，即(1,0,0)
    evolvent_slopes = delta.copy()
    evolvent_slopes[0] = [1, 0, 0]
    evolvent_slopes = np.apply_along_axis(normalize_vector, 1, evolvent_slopes)
    return evolvent_slopes
