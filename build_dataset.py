import pandas as pd
import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns

def step_data_collection(start, end):
    for i in range(start, end+1):
        if i == 38 or i == 100:
            continue
        mould_name = "test" + str(i)
        cmd = ["python", "data_collection.py", mould_name]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding="utf-8")
        p.wait(2400)
        if p.poll() == 0:
            print(cmd[1], mould_name, 'Success')
        else:
            print(cmd[1], 'Failure')

def build_dataset(start, end):
    '''
        Build per-step prediction dataset.
        Input contains last step stress distribution and next step bending parameter.
        Label contains next step stress distribution.
        Start, end specify the range of tests you want to collect data for.
    '''
    dataset_X = pd.DataFrame()
    dataset_Y = pd.DataFrame()
    for i in range(start, end+1):
        mould_name = "test" + str(i)
        parameter_path = "/Optimizing_bending_parameter/data/mould_output/{}/".format(mould_name) + "param_base_rel.csv"
        if os.path.exists(parameter_path) == False:
            print("{} does not exist".format(mould_name))
            continue
        bending_parameter = pd.read_csv(parameter_path, header=None)
        # print(bending_parameter)
        step_check = True
        j = 0
        while step_check:
            step_name_x = "Step-" + str(j)
            step_name_y = "Step-" + str(j+1)
            csv_path_x = "/Optimizing_bending_parameter/data/model/{}/simulation/".format(mould_name) + "strip_mises_" + step_name_x + ".csv"
            csv_path_y = "/Optimizing_bending_parameter/data/model/{}/simulation/".format(mould_name) + "strip_mises_" + step_name_y + ".csv"
            
            if os.path.exists(csv_path_x) and os.path.exists(csv_path_y):
                df_x = pd.read_csv(csv_path_x)
                df_y = pd.read_csv(csv_path_y)
                x = df_x["S_Mises"]
                # print(len(x))
                parameter = bending_parameter.iloc[j+1] - bending_parameter.iloc[j]
                parameter = parameter[parameter!=0]
                # print(parameter)
                # print(type(parameter))
                x = pd.concat([x, parameter]).reset_index(drop=True)
                # print(x)
                
                dataset_X = pd.concat([dataset_X, x], axis=1)
                dataset_Y = pd.concat([dataset_Y, df_y["S_Mises"]], axis=1)
                j += 1
            else:
                step_check = False
        # print(dataset_X)
        print("{} finished".format(mould_name))
    
    dataset_X.to_csv("/Optimizing_bending_parameter/dataset_X.csv", index=False, mode="w", header=False)
    dataset_Y.to_csv("/Optimizing_bending_parameter/dataset_Y.csv", index=False, mode="w", header=False)
    print("Created dataset csv")

def build_springback_dataset(start, end):
    '''
        Building springback prediction dataset
        Input is the final step stress distribution (before release)
        Label is the maximum springback deviation after release
        Start, end specify the range of tests you want to collect data for
    '''
    dataset_X = pd.DataFrame()
    dataset_Y = pd.DataFrame()
    springback_set = []
    for i in range(start-1, end+1):
        mould_name = "test" + str(i)
        springback_path = "/Optimizing_bending_parameter/data/model/{}/simulation/".format(mould_name) + "springback_output.csv"
        if os.path.exists(springback_path) == False:
            print("{} does not exist".format(mould_name))
            continue
        springback = pd.read_csv(springback_path)["Springback"]
        reward = max(springback.tolist()) * 100  # Reward is the max springback deviation, so the less the better
        if reward == 0:  # If reward is 0, this means the bending process is incomplete
            continue
        springback_set.append(reward)

        j = 0

        while True:
            step_name_1 = "Step-" + str(j)
            step_name_2 = "Step-" + str(j+1)
            stress_path_1 = "/Optimizing_bending_parameter/data/model/{}/simulation/".format(mould_name) + "strip_mises_" + step_name_1 + ".csv"
            stress_path_2 = "/Optimizing_bending_parameter/data/model/{}/simulation/".format(mould_name) + "strip_mises_" + step_name_2 + ".csv"
            if os.path.exists(stress_path_2):
                j += 1
                continue
            df_x = pd.read_csv(stress_path_1)
            dataset_X = pd.concat([dataset_X, df_x["S_Mises"]], axis=1)
            break
        print("{} finished".format(mould_name))
    dataset_Y["springback"] = springback_set
    dataset_X.to_csv("/Optimizing_bending_parameter/dataset_springback_X.csv", index=False, mode="w", header=False)
    dataset_Y.to_csv("/Optimizing_bending_parameter/dataset_springback_Y.csv", index=False, mode="w", header=False)
    print("Created springback dataset csv")


if __name__ == "__main__":
    # step_data_collection(72, 100)
    # build_dataset(5, 99)
    build_springback_dataset(1, 50)