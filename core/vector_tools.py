import math

import numpy as np


def calc_vector_len(x: np.ndarray) -> float:
    return math.sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])


def normalize_vector(x: np.ndarray) -> np.ndarray:
    if np.abs(calc_vector_len(x)) < 1e-6:
        print("warning!", x)
        raise Exception
    ret = x / calc_vector_len(x)
    return ret


def calc_angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    cos_theta = (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
                 ) / (calc_vector_len(v1) * calc_vector_len(v2))
    if cos_theta > 1.0:
        cos_theta = 1.0
    return math.acos(cos_theta)
