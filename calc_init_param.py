import logging
import os
import numpy as np
from core.param_util import calc_param_list
from core.param_util.param_tools import calc_param_right
from core.param_util.calc_evolvent_with_stretch import calc_evolvent_with_stretch
from core.param_util.coord_convertor import param2coord_right, theta2mat
from core.param_util.curve_tools import read_txt
from core.vector_tools import normalize_vector, calc_vector_len, calc_angle_between_vectors
import automation as at
import sys
import math


def calc_init_param(data_path, user_config):
    param_list = calc_param_list(
        recursion_path=data_path,
        strip_length=user_config["strip_length"],
        pre_length=user_config["pre_length"],
        k=user_config["k"],
        max_step_dis=user_config["max_step_dis"],
        config={"resample": False}
    )
    print(param_list)


def calc_next_idx(evolvent_points, evolvent_slopes, D, pre_idx, radius=1):
    '''
        Given previous parameter idx and next step size, calculate next idx
    '''
    point_num_all = evolvent_points.shape[0] # Total number of points
    r = point_num_all - 1 # Right side of binary search
    cur_len = point_num_all - pre_idx
    while True:
        translate_l, _ = calc_param_right(evolvent_points[pre_idx], evolvent_slopes[pre_idx])
        translate_r, _ = calc_param_right(evolvent_points[r], evolvent_slopes[r])
        delta = max([abs(translate_r[i] - translate_l[i])
                            for i in range(3)])
        cur_len = math.ceil(cur_len / 2)

        if delta < D - radius:
            if r == point_num_all - 1:
                return r
            else:
                r = r + cur_len
                if r > point_num_all - 1:
                    r = point_num_all - 1
        elif delta > D + radius:
            r = r - cur_len
            if r <= pre_idx:
                r = pre_idx + 1
        else:
            return r

def calc_next_param(recursion_path, D, strip_length, pre_length, k, pre_idx=None):

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

    if pre_idx == None:
        pre_idx = 0

    next_idx = calc_next_idx(evolvent_points, evolvent_slopes, D, pre_idx)

    next_point_0 = translated_curve_0[next_idx]
    next_point_1 = translated_curve_1[next_idx]
    initial_point_0 = translated_curve_0[0]
    initial_point_1 = translated_curve_0[1]
    translate, rotate = calc_param_right(evolvent_points[next_idx], evolvent_slopes[next_idx])
    abs_param = translate.A.reshape(3).tolist() + rotate.tolist()
    '''
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
        param2coord_right(*abs_param, 0, initial_point_0[1], initial_point_0[2]).T[0] \
        - remained_len[next_idx] * evolvent_slopes[next_idx]
    curve_2_p = \
        param2coord_right(*abs_param, 0, initial_point_1[1], initial_point_1[2]).T[0] \
        - remained_len[next_idx] * evolvent_slopes[next_idx]
    # print(remained_len[idx[i]])

    # 不加x转动时两个curve对应点的相对位置
    r_21 = normalize_vector(curve_2_p - curve_1_p)
    # 实际位置
    r_21_real = normalize_vector(next_point_1 - next_point_0)

    # 做一个叉乘，描述了不加转动时的相对位矢应该如何转到加了转动时的相对位矢
    # 如果cross和切线方向一样，说明转过去的角度是大于0的，说明是向x正方向转动，反之则是反方向转动
    cross = np.cross(r_21, r_21_real)
    if np.abs(calc_vector_len(cross)) < 1e-10:
        theta_x = 0
    else:
        theta_x = math.asin(calc_vector_len(cross))
        theta_dir = np.dot(normalize_vector(cross), evolvent_slopes[next_idx])
        if theta_dir < 0:
            theta_x = -theta_x

    abs_param[3] = theta_x
    print(theta_x)
    '''
    print("Parameter is {}".format(abs_param))
    print("Idx is {}".format(next_idx))
    return abs_param, next_idx


step_size = [3, 2, 4, 6]
pre_idx = 0
strip_length = 40
pre_length = 0.1
k = 0.05   

if __name__ == "__main__":
    mould_name = sys.argv[1]
    data_path = "./data/mould_output/" + mould_name
    
    calc_init_param(data_path, user_config={
        "strip_length": 40,
        "pre_length": 0.1,
        "max_step_dis": 8,
        "k": 0.05
    })
    
    for i in range(len(step_size)):
        D = step_size[i]
        print("Stepsize is {}".format(D))
        abs_param, pre_idx = calc_next_param(data_path, D, strip_length, pre_length, k, pre_idx)
