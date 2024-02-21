"""
    做一下样条插值
    目标是，输入特征线，输出重采样后的两条特征线
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve

from .curve_tools import calc_slopes, write_txt, read_txt


def cubic_spline(curve: np.ndarray):
    # print(curve)
    x, y, z = curve[:, 0], curve[:, 1], curve[:, 2]
    # plt.scatter(x, y, marker="x", s=1)
    # plt.show()
    cs = CubicSpline(x, [x, y, z], axis=1)
    return lambda t: cs(t).T


def gen_resample_line_file(input_lines: list, output_lines: list):
    curve_0 = read_txt(input_lines[0])
    curve_1 = read_txt(input_lines[1])

    cs = cubic_spline(curve_1)
    # print(cs(curve_1[:, 0]))
    slope_0 = calc_slopes(curve_0)

    ans_x_list = []
    for i in range(len(curve_1)):
        def equ(x):
            return np.dot((cs(x) - curve_0[i])[0], slope_0[i])

        ans_x = fsolve(equ, np.array([0]))
        ans_x_list.append(ans_x[0])

    feature_line_1 = cs(np.array(ans_x_list))

    write_txt(output_lines[0], curve_0)
    write_txt(output_lines[1], feature_line_1)
    return curve_0, feature_line_1
