import logging

import numpy as np

from core.vector_tools import calc_vector_len, calc_angle_between_vectors, normalize_vector


# noinspection PyPep8Naming
def calc_evolvent_with_stretch(points: np.ndarray, length: float, total_length: float, K: float = 1.0,
                               need_remained_len=False):
    """
    计算渐伸线，返回渐伸线的点集以及每个点对应的斜率信息。
    参数:
        points - (n, 3)的ndarray
        length - 铝条长度
        K - 公式中的参数K
    返回值:
        一个二元元组，第一项为渐伸线的点集，第二项为点对应的斜率向量，两者都是(n, 3)的ndarray
    """
    logging.info("正在计算渐伸线")
    # 点作差，第一个点复制一份
    shifted_points = np.vstack([points[0:1], points[:-1]])
    # 计算两两之间的点的差
    delta = points - shifted_points
    # print(delta)
    # 各线段长度
    segment_len = np.apply_along_axis(calc_vector_len, 1, delta)
    # print(segment_len)
    # 每个采样点处的累计长度
    cumsum_len = np.cumsum(segment_len)
    # 线段总长度
    curve_len = cumsum_len[-1]
    logging.info("[info] 给定的模具特征线总长度：" + str(curve_len))
    logging.info("[info] 铝件减去左端平直部分后的长度：" + str(length))
    # 将线段总长度设为固定的长度
    curve_len = length
    # print(curve_len)
    # 各个点处的斜率，第一个点的斜率指定为x轴方向，即(1,0,0)
    evolvent_slopes = delta.copy()
    evolvent_slopes[0] = [1, 0, 0]
    evolvent_slopes = np.apply_along_axis(normalize_vector, 1, evolvent_slopes)
    shifted_evolvent_slopes = np.vstack(
        [evolvent_slopes[0:1], evolvent_slopes[:-1]]
    )
    # print(evolvent_slopes)
    # print(shifted_evolvent_slopes)

    angles = np.vectorize(
        calc_angle_between_vectors,
        signature='(n),(n)->()'
    )(evolvent_slopes, shifted_evolvent_slopes)
    # print(angles)
    # print(angles * length * K)
    # print(angles.shape)

    # 每一个点对应的delta_L
    delta_L = angles * total_length * K
    # 补拉伸的时候，改变的只有每一段时curve_len的长度，cumsum_len不会变
    cumsum_delta_L = np.cumsum(delta_L)
    logging.info("过程中的补拉长度" + str(cumsum_delta_L[-1]))

    curve_len_array = curve_len * np.ones_like(cumsum_len)
    curve_len_array += cumsum_delta_L
    logging.info("补拉后的总长度" + str(curve_len_array[-1]))

    # 求渐伸线点集
    t = np.vectorize(lambda x, y: x * y,
                     signature='(n),()->(n)')(evolvent_slopes, (curve_len_array - cumsum_len))
    # print(t)
    evolvent_points = points + t
    logging.info("模具特征线中最后一个点" + str(points[-1]))
    logging.info("渐伸线中最后一个点" + str(evolvent_points[-1]))
    logging.info("前两点的距离：" + str(calc_vector_len(points[-1] - evolvent_points[-1])))
    logging.info("渐伸线计算结束")

    if need_remained_len:
        return evolvent_points, evolvent_slopes, curve_len_array - cumsum_len

        # 下面这个好像错了
        # return evolvent_points, evolvent_slopes, curve_len - cumsum_len
    else:
        return evolvent_points, evolvent_slopes
