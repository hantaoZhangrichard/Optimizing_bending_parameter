import os
import math
import random
import numpy as np
import subprocess
import automation as at
import sys

from scipy.interpolate import CubicSpline

# 生成的模具和相关文件保存在 ./data/mould_output/{mould_namt}
mould_name = sys.argv[1]
# 模具特征线采样点数量，和 gen_curve_2d 的输入中的 n 一致即可
point_num = 2000

# 可以自行调整生成曲线的算法，只要输出的约定是类似的就行
def gen_curve_2d(n, x_limit, a_limit, b_a_limit, length_limit):
    """
    随机生成一段二维椭圆曲线，长度单位都是 mm
    参数:
        n - 取点的数量
        x_limit - 椭圆的 x 坐标的范围
        a_limit - 椭圆的半长轴 a 的范围
        b_a_limit - 椭圆的半短轴 b 和半长轴 a 之比 b/a 的范围
        length_limit - 最终曲线的总长度（由于长度是离散计算的，存在误差）
    返回值:
        一个 (n, 2) 的 ndarray，第一个点应当是 (0, 0)
    """
    x = np.linspace(*x_limit, n)
    b_a, a = random.uniform(*b_a_limit), random.uniform(*a_limit)
    b = b_a * a
    y = b - b * np.sqrt(1 - (x * x) / (a * a))
    curve = np.array(list(zip(x, y)))
    # resample
    shifted_points = np.vstack([curve[0:1], curve[:-1]])
    segment_len = np.apply_along_axis(
        lambda x: math.sqrt(x[0] * x[0] + x[1] * x[1]), 1, curve-shifted_points
    )
    cumsum_len = np.cumsum(segment_len)
    len_percent = cumsum_len / length_limit  # 进行一次归一化，将长度转换成百分比
    # print(cumsum_len)
    # print(len_percent)
    # print(cumsum_len)
    # print(len_percent)
    # print(curve)
    cs = CubicSpline(len_percent, curve)
    length_sample = np.linspace(0, 1, n)
    # print(cs(length_sample))
    # print(np.hstack((length_sample.reshape((len(length_sample), 1)), cs(length_sample))))
    # print(cs(np.linspace(0, )) - curve)
    return cs(length_sample)

def write_txt(filename: str, points: np.ndarray):
    """
    将点数据写入txt
    参数:
        filename - 文件名
        points - 维度为(n, k)的ndarray
    返回值:
        None
    """
    with open(filename, "w", encoding="utf-8") as f:
        for p in points:
            for i in range(len(p) - 1):
                f.write(f"{p[i]} ")
            f.write(f"{p[-1]}\n")
    

def get_points_from_stp(stp_path):
    output_path = stp_path.replace('stp', 'txt')
    with open(stp_path, 'r', encoding="utf-8") as fp:
        lines = fp.readlines()
        flag = False
        with open(output_path, 'w', encoding="utf-8") as wp:
            points = []
            for line in lines:
                if 'CARTESIAN_POINT' in line:
                    if flag:
                        point = line[line.find(',') + 2: len(line) - 4].split(',')
                        for i in range(len(point)):
                            point[i] = float(point[i])
                        points.append(point)
                    flag = True
            points.sort()
            for i in range(len(points)):
                wp.write(f"{points[i][0]} {points[i][1]} {points[i][2]}\n")


if __name__ == "__main__":
    curve_2d = gen_curve_2d(
        n=point_num,
        x_limit=(0, 40),
        a_limit=(150, 200),
        b_a_limit=(0.1, 0.4),
        length_limit=40,
    )
    curve_3d_0 = np.hstack((curve_2d, np.zeros((curve_2d.shape[0], 1))))
    curve_3d_1 = np.hstack((curve_2d, np.zeros((curve_2d.shape[0], 1)) + 1))
    static_path = "C:/Optimizing_bending_parameter/data/mould_static/"
    recursion_path = f"C:/Optimizing_bending_parameter/data/mould_output/{mould_name}/"
    write_txt(f"C:/Optimizing_bending_parameter/data/mould_output/{mould_name}/feature_line_for_ug_0.txt", curve_3d_0)
    write_txt(f"C:/Optimizing_bending_parameter/data/mould_output/{mould_name}/feature_line_for_ug_1.txt", curve_3d_1)
    print("模具数据路径:", recursion_path)

    # 定位生成模具的 exe 路径
    gen_mould_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "external/bin/StyledSweep.exe").replace("\\", "/")
    print("模具生成程序路径:", gen_mould_bin)

    cmd = ["cmd", "/c", gen_mould_bin, os.path.abspath(static_path) + "\\",
           os.path.abspath(recursion_path) + "\\", str(point_num)]
    

    p = subprocess.Popen(
        cmd,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    try:
        outs, errs = p.communicate(timeout=600)
        with open(os.path.join(recursion_path, "build_mould_log.log"), "w", encoding="utf-8") as f:
            f.write("std out:\n")
            f.write(outs.decode('utf-8', 'replace'))
            f.write("\nstd err:\n")
            f.write(errs.decode('utf-8', 'replace'))
        print("模具生成成功")
    except TimeoutError:
        p.kill()
        print("模具生成失败，请检查代码")
        exit(-1)
    get_points_from_stp(os.path.join(recursion_path, "feature_line_from_ug_0.stp"))
    get_points_from_stp(os.path.join(recursion_path, "feature_line_from_ug_1.stp"))
