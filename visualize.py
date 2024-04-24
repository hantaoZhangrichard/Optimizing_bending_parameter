import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from core.param_util.curve_tools import read_txt
from core.param_util.calc_evolvent_with_stretch import calc_evolvent_with_stretch
from calc_init_param import calc_next_idx

def read_curve(strip_length, pre_length, k):
    # curve_0 = read_txt(os.path.join(recursion_path, "feature_line_from_ug_0.txt"))
    curve_0 = read_txt("C:\Optimizing_bending_parameter\data\mould_output\\test0\\feature_line_from_ug_0.txt")
    # curve_1 = read_txt(os.path.join(recursion_path, "feature_line_from_ug_1.txt"))
    curve_1 = read_txt("C:\Optimizing_bending_parameter\data\mould_output\\test0\\feature_line_from_ug_1.txt")
    length_after_pre = strip_length + pre_length
    translate_vector = curve_0[0].copy()
    translate_vector[0] = 0
    translated_curve_0 = curve_0 - translate_vector
    translated_curve_1 = curve_1 - translate_vector

    # 生成渐伸线
    evolvent_points, evolvent_slopes, remained_len = calc_evolvent_with_stretch(
        translated_curve_0, length_after_pre - translated_curve_0[0][0], length_after_pre, k, need_remained_len=True
    )
    print(evolvent_points)
    return curve_0, evolvent_points, evolvent_slopes

def get_clamp_loc(evolvent_points, evolvent_slopes, test_name):
    data_path = "C:\Optimizing_bending_parameter\data\model\\" + test_name + "\\action_list.csv"
    action_list = pd.read_csv(data_path)
    print(action_list["0"][1])
    clamp_move = [0]
    for i in range(len(action_list)):
        pre_idx = clamp_move[-1]
        idx = calc_next_idx(evolvent_points, evolvent_slopes, action_list["0"][i], pre_idx, radius=1)
        clamp_move.append(idx)
    print(clamp_move)
    return clamp_move

def curve_visualize(curve_0, evolvent_points, clamp_move):
    x = evolvent_points[:, 0]
    y = evolvent_points[:, 1]
    z = evolvent_points[:, 2]

    x2 = curve_0[:, 0]
    y2 = curve_0[:, 1]
    z2 = curve_0[:, 2]
    # Plot the curve
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, color="blue")
    # ax.plot(x2, y2, z2, color="green")
    '''
    # Plot straight lines between highlighted points
    for i in range(len(clamp_move) - 1):
        idx1 = clamp_move[i]
        idx2 = clamp_move[i + 1]
        ax.plot([x[idx1], x[idx2]], [y[idx1], y[idx2]], [z[idx1], z[idx2]], color='red', linestyle='-', linewidth=2)
    '''
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Curve Visualization')

    # Show plot
    plt.show()


if __name__ == "__main__":
    strip_length = 40
    pre_length = 0.1
    k = 0.05
    curve_0, evolvent_points, evolvent_slopes = read_curve(strip_length, pre_length, k)
    clamp_move = get_clamp_loc(evolvent_points, evolvent_slopes, "test0_ep2")
    curve_visualize(curve_0, evolvent_points, clamp_move)