import numpy as np
import math

Dyz = 76
Dxy = 205

# Dxy = 135
# Dyz = 20

# noinspection DuplicatedCode
def param2coord_right(x, y, z, theta_x, theta_y, theta_z, x_3=0, y_3=0, z_3=0):
    """
        将右边夹钳的参数转化为绝对坐标系下的坐标
        其中 x, y, z是1坐标系的原点位置
        三个theta不需要解释
        x0 y0 z0是某个点在3坐标系下的坐标
    """
    point_3 = np.mat([[x_3], [y_3], [z_3]])
    mat_3to2 = np.mat([
        [1, 0, 0],
        [0, math.cos(theta_x), -math.sin(theta_x)],
        [0, math.sin(theta_x), math.cos(theta_x)]
    ])
    point_2 = mat_3to2 * point_3 - np.mat([[Dxy], [0], [0]])
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
    point_0 = mat_1to0 * point_1 + np.mat([[x], [y], [z]])
    return point_0.A


# noinspection DuplicatedCode
def theta2vector_right(theta_x, theta_y, theta_z, x_0=1, y_0=0, z_0=0):
    vector_3 = np.mat([[x_0], [y_0], [z_0]])
    mat_3to2 = np.mat([
        [1, 0, 0],
        [0, math.cos(theta_x), -math.sin(theta_x)],
        [0, math.sin(theta_x), math.cos(theta_x)]
    ])
    vector_2 = mat_3to2 * vector_3
    # vector_2 = np.mat([[x_0], [y_0], [z_0]])
    mat_2to1 = np.mat([
        [math.cos(theta_y), 0, math.sin(theta_y)],
        [0, 1, 0],
        [-math.sin(theta_y), 0, math.cos(theta_y)]
    ])
    vector_1 = mat_2to1 * vector_2
    mat_1to0 = np.mat([
        [math.cos(theta_z), -math.sin(theta_z), 0],
        [math.sin(theta_z), math.cos(theta_z), 0],
        [0, 0, 1]
    ])
    vector_0 = mat_1to0 * vector_1
    return vector_0


def theta2mat(theta_x, theta_y, theta_z):
    mat_3to2 = np.mat([
        [1, 0, 0],
        [0, math.cos(theta_x), -math.sin(theta_x)],
        [0, math.sin(theta_x), math.cos(theta_x)]
    ])
    mat_2to1 = np.mat([
        [math.cos(theta_y), 0, math.sin(theta_y)],
        [0, 1, 0],
        [-math.sin(theta_y), 0, math.cos(theta_y)]
    ])
    mat_1to0 = np.mat([
        [math.cos(theta_z), -math.sin(theta_z), 0],
        [math.sin(theta_z), math.cos(theta_z), 0],
        [0, 0, 1]
    ])
    return mat_1to0 * mat_2to1 * mat_3to2


if __name__ == "__main__":
    pass
