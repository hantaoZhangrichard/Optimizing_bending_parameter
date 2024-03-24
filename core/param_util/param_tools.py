import logging
import math
import os

import numpy as np

from .calc_evolvent_with_stretch import calc_evolvent_with_stretch
from .coord_convertor import param2coord_right, theta2mat
from .curve_tools import read_txt
from .resample import gen_resample_line_file
from core.vector_tools import normalize_vector, calc_vector_len, calc_angle_between_vectors

Dyz = 76
Dxy = 205


# Dxy = 135
# Dyz = 20


def calc_param_list(
        *,
        recursion_path,
        strip_length,
        pre_length,
        k,
        idx_mode="default",
        idx_config=None,
        max_step_dis=15,
        config={}
):
    '''
        计算的是，右夹钳的六个自由度
    '''
    MAX_STEP_DIS = max_step_dis
    if config["resample"]:
        logging.info("重采样")
        curve_0, curve_1 = gen_resample_line_file(input_lines=[
            os.path.join(recursion_path, "feature_line_from_ug_0.txt"),
            os.path.join(recursion_path, "feature_line_from_ug_1.txt")
        ], output_lines=[
            os.path.join(recursion_path, "feature_line_for_param_0.txt"),
            os.path.join(recursion_path, "feature_line_for_param_1.txt")
        ])
    else:
        curve_0 = read_txt(os.path.join(recursion_path, "feature_line_from_ug_0.txt"))
        curve_1 = read_txt(os.path.join(recursion_path, "feature_line_from_ug_1.txt"))
    logging.info("正在计算参数")
    length_after_pre = strip_length + pre_length
    logging.info("预拉伸后的长度为：" + str(length_after_pre))
    # 平移
    translate_vector = curve_0[0].copy()
    translate_vector[0] = 0
    translated_curve_0 = curve_0 - translate_vector
    translated_curve_1 = curve_1 - translate_vector

    # 生成渐伸线
    evolvent_points, evolvent_slopes, remained_len = calc_evolvent_with_stretch(
        translated_curve_0, length_after_pre - translated_curve_0[0][0], length_after_pre, k, need_remained_len=True
    )

    # 离散化
    idx = []
    if idx_mode == "default":
        # 1027从0.98改成了0.95
        idx = calc_idx(evolvent_points, evolvent_slopes, MAX_STEP_DIS, last_idx=round(curve_0.shape[0] * 0.98))
        idx = idx[:-1]
    elif idx_mode == "last_1800":
        idx = calc_idx(evolvent_points, evolvent_slopes, MAX_STEP_DIS, last_idx=curve_0.shape[0] - 1800)
        idx = idx[:-1]
        idx.append(curve_0.shape[0] - 1200)
        idx.append(curve_0.shape[0] - 600)
        idx.append(curve_0.shape[0] - 1)
    elif idx_mode == "gen_training_data":
        idx = calc_idx(evolvent_points, evolvent_slopes, MAX_STEP_DIS)
    elif idx_mode == "gen_training_data_v2":
        '''
            某次一拍脑袋想的，现在已经忘了
            如果倒数第一步和倒数第二步差的太小，就直接合并
        '''
        idx = calc_idx(evolvent_points, evolvent_slopes, MAX_STEP_DIS)
        print(idx)
        if len(idx) > 2 and idx[-1] - idx[-2] < round(curve_0.shape[0] * 0.05):
            del idx[-2]
        print(idx)
    elif idx_mode == "full":
        idx = calc_idx(evolvent_points, evolvent_slopes, MAX_STEP_DIS)
    elif idx_mode == "last_95%":
        idx = calc_idx(evolvent_points, evolvent_slopes, MAX_STEP_DIS, last_idx=round(curve_0.shape[0] * 0.95))[:-1]
    elif idx_mode == "last_95%_all":
        idx = calc_idx(evolvent_points, evolvent_slopes, MAX_STEP_DIS, last_idx=round(curve_0.shape[0] * 0.95))
    else:
        logging.error("暂时还不支持其他mode")
        exit(-1)
    logging.info("选取的idx为：" + str(idx))
    

    # idx = idx_noise(idx, 200)
    # print(idx)
    resampled_curve_0 = translated_curve_0[idx]
    resampled_curve_1 = translated_curve_1[idx]
    point_num = resampled_curve_0.shape[0]
    # 先生成中心线对应的凸轮轨迹
    abs_param_list = []
    for i in range(point_num):
        translate, rotate = calc_param_right(evolvent_points[idx[i]], evolvent_slopes[idx[i]])
        abs_param_list.append(translate.A.reshape(3).tolist() + rotate.tolist())
        print(translate.A.reshape(3).tolist() + rotate.tolist())

    # 记录拉弯曲线数据
    with open(os.path.join(recursion_path, "param_info.txt"), "w", encoding="utf-8") as f:
        for i in range(point_num):
            if i == 0:
                f.write(f"初始切线方向：{evolvent_slopes[idx[i]]}\n")
            else:
                angle = calc_angle_between_vectors(evolvent_slopes[idx[i]], evolvent_slopes[idx[i - 1]])
                f.write(
                    f"step_{i} 角度改变量：{angle / math.pi * 180:.3f}°\n"
                )
        pass
    # 一种新的计算转动角度的方案
    for i in range(point_num):
        # # 原点
        # origin = util.param2coord_right(*param_list[i], 0, 0, 0).T[0] \
        #     - curve_1_remained_len[i] * curve_1_evolvent_slopes[i]
        # # 两个轴
        # y_axis_1 = util.param2coord_right(*param_list[i], 0, 1, 0).T[0] \
        #     - curve_1_remained_len[i] * curve_1_evolvent_slopes[i]
        # z_axis_1 = util.param2coord_right(*param_list[i], 0, 0, 1).T[0] \
        #     - curve_1_remained_len[i] * curve_1_evolvent_slopes[i]
        # 两个曲线上的点
        curve_1_p = \
            param2coord_right(*abs_param_list[i], 0, resampled_curve_0[0][1], resampled_curve_0[0][2]).T[0] \
            - remained_len[idx[i]] * evolvent_slopes[idx[i]]
        curve_2_p = \
            param2coord_right(*abs_param_list[i], 0, resampled_curve_1[0][1], resampled_curve_1[0][2]).T[0] \
            - remained_len[idx[i]] * evolvent_slopes[idx[i]]
        # print(remained_len[idx[i]])

        # 不加x转动时两个curve对应点的相对位置
        r_21 = normalize_vector(curve_2_p - curve_1_p)
        # 实际位置
        r_21_real = normalize_vector(resampled_curve_1[i] - resampled_curve_0[i])

        # 做一个叉乘，描述了不加转动时的相对位矢应该如何转到加了转动时的相对位矢
        # 如果cross和切线方向一样，说明转过去的角度是大于0的，说明是向x正方向转动，反之则是反方向转动
        cross = np.cross(r_21, r_21_real)
        if np.abs(calc_vector_len(cross)) < 1e-10:
            theta_x = 0
        else:
            theta_x = math.asin(calc_vector_len(cross))
            theta_dir = np.dot(normalize_vector(cross), evolvent_slopes[idx[i]])
            if theta_dir < 0:
                theta_x = -theta_x

        abs_param_list[i][3] = theta_x

    # abs_param_list = param_noise(abs_param_list, 0.01)

        # 210108 注释了下面的版本，用叉乘做了一个版本
        # cos_theta_x = np.dot(r_21, r_21_real)
        # if cos_theta_x > 1:
        #     cos_theta_x = 1
        # try:
        #     theta_x = math.acos(cos_theta_x)  # 负号的问题？？？
        #     abs_param_list[i][3] = theta_x
        # except ValueError:
        #     print("math domain error")
        #     print(f"cos_theta_x = {cos_theta_x}")
        #     exit(-1)

    # 生成csv
    # print(abs_param_list)
    rel_param_list = gen_param_csv(
        param_list=abs_param_list,
        output_path=recursion_path,
        pre_length=pre_length,
        version="base",
    )

    # 添加预拉伸
    return rel_param_list


# noinspection PyPep8Naming
def calc_idx(evolvent_points, evolvent_slopes, D=13, radius=1, last_idx=None):
    """
    计算下标
    参数:
        evolvent_points - 渐伸线点集
        evolvent_slopes - 每个点对应的斜率
        D - 最大位移限制条件
        radius - 最大位移在 (D - radius, D + radius)
    返回值:
        曲线点集
    """

    def f_debug(l, r):
        translate_l, _ = calc_param_right(
            evolvent_points[l], evolvent_slopes[l])
        translate_r, _ = calc_param_right(
            evolvent_points[r], evolvent_slopes[r])
        print(l, r, [translate_r[i] - translate_l[i] for i in range(3)])

    pre_idx = 0  # 上一个找到的分割id
    idx_list = [0]  # 当前找到的id list
    if last_idx is None:
        point_num_all = evolvent_points.shape[0]  # 总点数
    else:
        point_num_all = last_idx  # 总点数
    # 简单的寻找idx的算法
    # 上一个分割点只要不是最后一个，就需要继续找下一个点
    while pre_idx < point_num_all - 1:
        ll = pre_idx  # 整个二分区间的最左端
        cur_len = point_num_all - ll  # 当前二分的左端，
        r = point_num_all - 1  # 当前二分的右端
        # print(ll, evolvent_points[ll], evolvent_slopes[ll])
        translate_l, _ = calc_param_right(
            evolvent_points[ll], evolvent_slopes[ll])
        while True:
            translate_r, _ = calc_param_right(
                evolvent_points[r], evolvent_slopes[r])
            delta = max([abs(translate_r[i] - translate_l[i])
                         for i in range(3)])
            cur_len = math.ceil(cur_len / 2)
            # print(ll, r, delta, cur_len)
            if cur_len == 0:
                logging.error("选择渐伸线index失败")
                # print(ll, r, cur_len)
                # f_debug(5762, 5787)
                # f_debug(5762, 5786)
                # f_debug(5762, 5785)
                # for i in range(30):
                #     print(evolvent_slopes[5761 + i])
                exit(-1)
            if delta < D - radius:
                if r == point_num_all - 1:
                    pre_idx = point_num_all - 1
                    idx_list.append(pre_idx)
                    break
                else:
                    r = r + cur_len
                    # r = r + (point_num_all - 1 - r) // 2
                    if r > point_num_all - 1:
                        r = point_num_all - 1
            elif delta > D + radius:
                r = r - cur_len
                if r <= ll:
                    r = ll + 1
            else:
                pre_idx = r
                idx_list.append(pre_idx)
                break
    if not (last_idx is None):
        idx_list.append(evolvent_points.shape[0] - 1)
    return idx_list


def calc_rotate_mat(slope):
    # 计算角度
    theta_x = 0
    theta_y = math.asin(-slope[2])  # 一般都在正负90°内，不用特殊判断
    # theta_z = math.atan(slope[1] / slope[0])  # 一般不会超过90°
    if abs(slope[0]) < 1e-9:
        theta_z = 0
    else:
        theta_z = math.atan(slope[1] / slope[0])  # 一般不会超过90°
    # 断言验证，角度应该有三个方程，其中两个就是上面计算theta_y和theta_z
    assert abs(math.cos(theta_y) -
               slope[0] * math.cos(theta_z) - slope[1] * math.sin(theta_z)) < 1e-9
    return theta2mat(theta_x, theta_y, theta_z)


def calc_rotate_theta(slope):
    # 计算角度
    theta_x = 0
    theta_y = math.asin(-slope[2])  # 一般都在正负90°内，不用特殊判断
    # theta_z = math.atan(slope[1] / slope[0])  # 一般不会超过90°
    if abs(slope[0]) < 1e-9:
        theta_z = 0
    else:
        theta_z = math.atan(slope[1] / slope[0])  # 一般不会超过90°
    # 断言验证，角度应该有三个方程，其中两个就是上面计算theta_y和theta_z
    assert abs(math.cos(theta_y) -
               slope[0] * math.cos(theta_z) - slope[1] * math.sin(theta_z)) < 1e-9
    return theta_x, theta_y, theta_z


# noinspection DuplicatedCode
def calc_param_right(point, slope):
    """
        这个函数在算除了绕x轴旋转之外的五个自由度
    """
    # 计算角度
    theta_x = 0
    theta_y = math.asin(-slope[2])  # 一般都在正负90°内，不用特殊判断
    # theta_z = math.atan(slope[1] / slope[0])  # 一般不会超过90°
    if abs(slope[0]) < 1e-9:
        theta_z = 0
    else:
        theta_z = math.atan(slope[1] / slope[0])  # 一般不会超过90°
    # 断言验证，角度应该有三个方程，其中两个就是上面计算theta_y和theta_z
    test_val = abs(math.cos(theta_y) -
                   slope[0] * math.cos(theta_z) - slope[1] * math.sin(theta_z))
    # print(test_val)
    if test_val > 1e-9:
        print(f"[Error] 角度验证出错, theta_y={theta_y}, theta_z={theta_z}")
        print(f"{test_val:.20f}")
        assert False
    # 计算位移
    point_2 = np.mat([[-Dxy], [0], [0]])
    mat_2to1 = np.mat([
        [math.cos(theta_y), 0, math.sin(theta_y)],
        [0, 1, 0],
        [-math.sin(theta_y), 0, math.cos(theta_y)]
    ])
    point_1 = mat_2to1 * point_2 - np.mat([[Dyz], [0], [0]])
    mat_1to0 = np.mat([
        [math.cos(theta_z), -math.sin(theta_z), 0],
        [math.sin(theta_z), math.cos(theta_z), 0],
        [0, 0, 1]
    ])
    # print(np.mat(point).T)
    # print(point_1)
    param_translate = np.mat(point).T - mat_1to0 * point_1
    return param_translate, np.array([theta_x, theta_y, theta_z])


def gen_param_csv(*, param_list, output_path, pre_length, version):
    point_num = len(param_list)
    # 绝对参数
    with open(os.path.join(output_path, f"param_{version}_abs.csv"), "w", encoding="utf-8") as f:
        # f.write("右横向XR,右纵向YR,右升降ZR,右扭转AR,右垂摆BR,右平摆CR\n")
        for i in range(point_num):
            f.write("{},{},{},{},{},{}\n".format(*(param_list[i])))

    # 相对参数
    rel_param_list = []
    with open(os.path.join(output_path, f"param_{version}_rel.csv"), "w", encoding="utf-8") as f:
        f.write("{},{},{},{},{},{}\n".format(pre_length, 0.0, 0.0, 0.0, 0.0, 0.0))
        # f.write("右横向XR,右纵向YR,右升降ZR,右扭转AR,右垂摆BR,右平摆CR\n")
        for i in range(point_num - 1):
            t = (np.array(param_list[i + 1]) -
                 np.array(param_list[i])).tolist()
            rel_param_list.append(t)
            f.write("{},{},{},{},{},{}\n".format(*t))
    rel_param_list = [[pre_length, 0, 0, 0, 0, 0]] + rel_param_list

    sum_rel_param_list = []

    with open(os.path.join(output_path, f"param_{version}_rel_deg.csv"), "w", encoding="utf-8") as f:
        for line in rel_param_list:
            t = line[:]
            t[3] = t[3] * 180 / math.pi
            t[4] = t[4] * 180 / math.pi
            # TODO: 这里可能不应该有这个负号
            t[5] = -t[5] * 180 / math.pi
            sum_rel_param_list.append(t[:])
            # print(t)
            f.write("{},{},{},{},{},{}\n".format(*t))
    rst_sum = [0 for _ in range(6)]
    for idx, v in enumerate(sum_rel_param_list):
        for i in range(6):
            rst_sum[i] += v[i]
    # print(rst_sum)

    logging.info("生成拉弯参数的csv文件")
    return rel_param_list


def param_noise(param_list, scale):
    for i in range(1, len(param_list) - 1):
        noise = np.random.normal(0, scale, 3)
        param_list[i][0] += noise[0]
        param_list[i][1] += noise[1]
        param_list[i][5] += noise[2]
    return param_list

def idx_noise(idx_list, scale):
    '''
        Adding random normal noise to the idx list, do not change the number of steps
    '''
    l = len(idx_list)
    noise = np.random.normal(0, scale, l-2)
    # print(noise)
    for i in range(1, l-1):
        print(int(noise[i-1]))
        idx_list[i] += int(noise[i-1])
    if not all(i < j for i, j in zip(idx_list, idx_list[1:])):
        idx_list.sort()
    return idx_list